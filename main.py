from io import BytesIO
import gc
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


app = FastAPI(title="Local Phone Image Upload Test")

compressed_images: OrderedDict[str, dict] = OrderedDict()
uploaded_embeddings: Dict[str, np.ndarray] = {}
upload_groups: Dict[str, dict] = {}

MODEL_NAME = "facebook/dinov2-large"
INDEX_VERSION = "furniture-focused-v1"
CATALOG_DIR = Path("images")
INDEX_PATH = Path("image_index.npz")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TOP_K_MATCHES = 5
LOW_CONFIDENCE_THRESHOLD = 91.0
MAX_COMPRESSED_IMAGES = 100
COMPRESSED_IMAGE_SIZE = (1200, 900)
TARGET_JPEG_BYTES = 100 * 1024

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


def compressed_image_from_pil(image: Image.Image) -> bytes:
    original = image.convert("RGB")
    original.thumbnail(COMPRESSED_IMAGE_SIZE, Image.Resampling.LANCZOS)

    resized = Image.new("RGB", COMPRESSED_IMAGE_SIZE, (245, 245, 245))
    offset = (
        (COMPRESSED_IMAGE_SIZE[0] - original.width) // 2,
        (COMPRESSED_IMAGE_SIZE[1] - original.height) // 2,
    )
    resized.paste(original, offset)
    original.close()

    best_data = b""
    for quality in range(88, 19, -6):
        output = BytesIO()
        resized.save(output, format="JPEG", quality=quality, optimize=True)
        data = output.getvalue()
        best_data = data
        if len(data) <= TARGET_JPEG_BYTES:
            break

    # Release the resized working copy before returning the compressed bytes.
    resized.close()
    return best_data


def store_compressed_image(image_id: str, filename: str, data: bytes) -> None:
    compressed_images[image_id] = {
        "id": image_id,
        "filename": filename,
        "content_type": "image/jpeg",
        "data": data,
    }
    compressed_images.move_to_end(image_id)

    while len(compressed_images) > MAX_COMPRESSED_IMAGES:
        compressed_images.popitem(last=False)


async def process_uploaded_image(uploaded_file: UploadFile) -> tuple[dict, np.ndarray]:
    image_id = uuid4().hex
    filename = f"{image_id}.jpg"

    # upload: read the original bytes into a short-lived local variable only.
    original_bytes = await uploaded_file.read()
    original_buffer = BytesIO(original_bytes)
    original_image = Image.open(original_buffer)
    original_image.load()

    # embedding: generate the vector immediately from the original image.
    embedding = image_to_embedding(original_image)
    uploaded_embeddings[image_id] = embedding

    # delete: the raw uploaded bytes are released before compression/storage.
    del original_bytes
    original_buffer.close()
    del original_buffer

    # compress: create the only display image we keep in memory.
    compressed_data = compressed_image_from_pil(original_image)

    # release: close and delete the decoded original image. After this point,
    # only compressed JPEG bytes and the embedding remain.
    original_image.close()
    del original_image
    gc.collect()

    # store: FIFO keeps only the latest compressed display images.
    store_compressed_image(image_id, filename, compressed_data)

    return (
        {
            "id": image_id,
            "filename": filename,
            "url": f"/image/{image_id}",
        },
        embedding,
    )


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


def combined_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise HTTPException(status_code=400, detail="No images were uploaded.")

    embedding = np.mean(embeddings, axis=0).astype("float32")
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def find_matches(embeddings: List[np.ndarray], limit: int = TOP_K_MATCHES) -> List[dict]:
    index = get_catalog_index()
    query_embedding = combined_embedding(embeddings)
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


def safe_download_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in name.lower())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:80] or "upload"


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
    group_embeddings = []

    for uploaded_file in files_to_upload:
        if not uploaded_file.content_type or not uploaded_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image uploads are supported.")

        image_info, embedding = await process_uploaded_image(uploaded_file)
        uploaded.append(image_info)
        group_embeddings.append(embedding)

    matches = find_matches(group_embeddings)
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
    groups = []
    for group in upload_groups.values():
        available_images = [
            {
                "id": image_id,
                "filename": f"{image_id}.jpg",
                "url": f"/image/{image_id}",
            }
            for image_id in group["image_ids"]
            if image_id in compressed_images
        ]
        if not available_images:
            continue

        groups.append(
            {
                "id": group["id"],
                "cover": available_images[0],
                "images": available_images,
                "count": len(group["image_ids"]),
                "matches": group.get("matches", []),
                "prediction": group.get("prediction", group_prediction(group.get("matches", []))),
            }
        )
    return groups


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    image = compressed_images.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    return Response(content=image["data"], media_type=image["content_type"])


@app.get("/download-group/{group_id}")
async def download_group(group_id: str):
    group = upload_groups.get(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Upload group not found.")

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, image_id in enumerate(group["image_ids"], 1):
            image = compressed_images.get(image_id)
            if image is None:
                continue

            original_name = Path(image["filename"]).stem
            filename = f"{index:02d}-{safe_download_name(original_name)}.jpg"
            archive.writestr(filename, image["data"])

    if not buffer.tell():
        raise HTTPException(status_code=404, detail="No compressed images available for this group.")

    buffer.seek(0)
    filename = f"{safe_download_name(group.get('prediction', {}).get('name', 'upload'))}.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/catalog-image/{catalog_id}")
async def get_catalog_image(catalog_id: int):
    index = get_catalog_index()
    if catalog_id < 0 or catalog_id >= len(index["paths"]):
        raise HTTPException(status_code=404, detail="Catalog image not found.")

    return FileResponse(index["paths"][catalog_id])


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/uploads/{group_id}")
async def upload_page(group_id: str):
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
