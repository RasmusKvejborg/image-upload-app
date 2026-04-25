from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


app = FastAPI(title="Local Phone Image Upload Test")

images: Dict[str, dict] = {}
upload_groups: Dict[str, dict] = {}

MODEL_NAME = "facebook/dinov2-large"
INDEX_VERSION = "furniture-focused-v1"
CATALOG_DIR = Path("images")
INDEX_PATH = Path("image_index.npz")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TOP_K_MATCHES = 5
LOW_CONFIDENCE_THRESHOLD = 91.0

processor = None
model = None
catalog_index: Optional[dict] = None


def product_name_from_path(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ")


def catalog_files() -> List[Path]:
    if not CATALOG_DIR.exists():
        return []

    return sorted(
        path
        for path in CATALOG_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def load_model():
    global processor, model

    if processor is None or model is None:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()

    return processor, model


def image_to_embedding(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    variant_embeddings = [single_image_to_embedding(variant) for variant in image_variants(image)]
    embedding = np.mean(variant_embeddings, axis=0).astype("float32")
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def single_image_to_embedding(image: Image.Image) -> np.ndarray:
    image_processor, image_model = load_model()
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = image_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0].squeeze(0).cpu().numpy()

    embedding = embedding.astype("float32")
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def image_variants(image: Image.Image) -> List[Image.Image]:
    cropped = foreground_crop(image)
    padded_crop = pad_to_square(cropped)

    variants = [image, padded_crop]
    if cropped.size != image.size:
        variants.append(cropped)
    return variants


def foreground_crop(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB")).astype("int16")
    height, width, _ = array.shape
    border = np.concatenate(
        [
            array[: max(1, height // 20), :, :].reshape(-1, 3),
            array[-max(1, height // 20) :, :, :].reshape(-1, 3),
            array[:, : max(1, width // 20), :].reshape(-1, 3),
            array[:, -max(1, width // 20) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance_from_background = np.linalg.norm(array - background, axis=2)
    channel_range = array.max(axis=2) - array.min(axis=2)
    mask = (distance_from_background > 28) | (channel_range > 35)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return image

    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    crop_width = right - left + 1
    crop_height = bottom - top + 1
    if crop_width * crop_height < width * height * 0.08:
        return image

    margin_x = int(crop_width * 0.08)
    margin_y = int(crop_height * 0.08)
    left = max(0, left - margin_x)
    right = min(width - 1, right + margin_x)
    top = max(0, top - margin_y)
    bottom = min(height - 1, bottom + margin_y)

    return image.crop((left, top, right + 1, bottom + 1))


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    size = max(width, height)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    canvas.paste(image.convert("RGB"), ((size - width) // 2, (size - height) // 2))
    return canvas


def image_bytes_to_embedding(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return image_to_embedding(image)


def cache_matches_files(files: List[Path], cached: np.lib.npyio.NpzFile) -> bool:
    required_keys = {"model_name", "index_version", "embeddings", "paths", "names", "mtimes", "sizes"}
    if not required_keys.issubset(set(cached.files)):
        return False

    return (
        str(cached["model_name"].item()) == MODEL_NAME
        and str(cached["index_version"].item()) == INDEX_VERSION
        and cached["paths"].tolist() == [str(path) for path in files]
        and cached["mtimes"].tolist() == [path.stat().st_mtime_ns for path in files]
        and cached["sizes"].tolist() == [path.stat().st_size for path in files]
    )


def build_catalog_index() -> dict:
    files = catalog_files()
    if not files:
        raise HTTPException(status_code=500, detail="No catalog images found in the images folder.")

    if INDEX_PATH.exists():
        cached = np.load(INDEX_PATH, allow_pickle=False)
        if cache_matches_files(files, cached):
            return {
                "embeddings": cached["embeddings"],
                "paths": [Path(path) for path in cached["paths"].tolist()],
                "names": cached["names"].tolist(),
            }

    embeddings = []
    names = []
    for path in files:
        with Image.open(path) as image:
            embeddings.append(image_to_embedding(image))
        names.append(product_name_from_path(path))

    index = {
        "embeddings": np.vstack(embeddings).astype("float32"),
        "paths": files,
        "names": names,
    }

    np.savez_compressed(
        INDEX_PATH,
        model_name=MODEL_NAME,
        index_version=INDEX_VERSION,
        embeddings=index["embeddings"],
        paths=np.array([str(path) for path in files]),
        names=np.array(names),
        mtimes=np.array([path.stat().st_mtime_ns for path in files], dtype=np.int64),
        sizes=np.array([path.stat().st_size for path in files], dtype=np.int64),
    )
    return index


def get_catalog_index() -> dict:
    global catalog_index

    if catalog_index is None:
        catalog_index = build_catalog_index()
    return catalog_index


def combined_embedding(image_data: List[bytes]) -> np.ndarray:
    if not image_data:
        raise HTTPException(status_code=400, detail="No images were uploaded.")

    embeddings = [image_bytes_to_embedding(data) for data in image_data]
    embedding = np.mean(embeddings, axis=0).astype("float32")
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def find_matches(image_data: List[bytes], limit: int = TOP_K_MATCHES) -> List[dict]:
    index = get_catalog_index()
    query_embedding = combined_embedding(image_data)
    scores = index["embeddings"] @ query_embedding
    top_indexes = np.argsort(scores)[::-1][:limit]

    return [
        {
            "id": int(match_index),
            "filename": index["paths"][match_index].name,
            "name": index["names"][match_index],
            "score": round(float(scores[match_index]), 4),
            "confidence_percent": round(max(0.0, min(1.0, float(scores[match_index]))) * 100, 1),
            "url": f"/catalog-image/{int(match_index)}",
        }
        for match_index in top_indexes
    ]


def group_prediction(matches: List[dict]) -> dict:
    if not matches:
        return {
            "name": "Ukendt produkt",
            "confidence_percent": 0.0,
            "low_confidence": True,
        }

    best_match = matches[0]
    confidence_percent = best_match["confidence_percent"]
    return {
        "name": best_match["name"],
        "confidence_percent": confidence_percent,
        "low_confidence": confidence_percent < LOW_CONFIDENCE_THRESHOLD,
    }


@app.post("/upload")
async def upload(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
):
    files_to_upload = files or ([file] if file is not None else [])
    if not files_to_upload:
        raise HTTPException(status_code=400, detail="No images were uploaded.")

    group_id = uuid4().hex
    uploaded = []

    for uploaded_file in files_to_upload:
        if not uploaded_file.content_type or not uploaded_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image uploads are supported.")

        image_id = uuid4().hex
        data = await uploaded_file.read()
        images[image_id] = {
            "id": image_id,
            "filename": uploaded_file.filename or f"{image_id}.jpg",
            "content_type": uploaded_file.content_type,
            "data": data,
        }
        uploaded.append(
            {
                "id": image_id,
                "filename": images[image_id]["filename"],
                "url": f"/image/{image_id}",
            }
        )

    matches = find_matches([images[image["id"]]["data"] for image in uploaded])
    prediction = group_prediction(matches)

    upload_groups[group_id] = {
        "id": group_id,
        "image_ids": [image["id"] for image in uploaded],
        "matches": matches,
        "prediction": prediction,
    }

    return {
        "group": {
            "id": group_id,
            "cover": uploaded[0],
            "images": uploaded,
            "count": len(uploaded),
            "matches": matches,
            "prediction": prediction,
        },
        "ids": [image["id"] for image in uploaded],
    }


@app.get("/images")
async def list_images():
    return [
        {
            "id": group["id"],
            "cover": {
                "id": images[group["image_ids"][0]]["id"],
                "filename": images[group["image_ids"][0]]["filename"],
                "url": f"/image/{group['image_ids'][0]}",
            },
            "images": [
                {
                    "id": image_id,
                    "filename": images[image_id]["filename"],
                    "url": f"/image/{image_id}",
                }
                for image_id in group["image_ids"]
                if image_id in images
            ],
            "count": len(group["image_ids"]),
            "matches": group.get("matches", []),
            "prediction": group.get("prediction", group_prediction(group.get("matches", []))),
        }
        for group in upload_groups.values()
        if group["image_ids"] and group["image_ids"][0] in images
    ]


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    image = images.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    return Response(content=image["data"], media_type=image["content_type"])


@app.get("/catalog-image/{catalog_id}")
async def get_catalog_image(catalog_id: int):
    index = get_catalog_index()
    if catalog_id < 0 or catalog_id >= len(index["paths"]):
        raise HTTPException(status_code=404, detail="Catalog image not found.")

    return FileResponse(index["paths"][catalog_id])


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
