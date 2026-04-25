from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="Local Phone Image Upload Test")

images: Dict[str, dict] = {}


@app.post("/upload")
async def upload(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
):
    files_to_upload = files or ([file] if file is not None else [])
    if not files_to_upload:
        raise HTTPException(status_code=400, detail="No images were uploaded.")

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

    return {"images": uploaded, "ids": [image["id"] for image in uploaded]}


@app.get("/images")
async def list_images():
    return [
        {
            "id": image["id"],
            "filename": image["filename"],
            "url": f"/image/{image['id']}",
        }
        for image in images.values()
    ]


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    image = images.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    return Response(content=image["data"], media_type=image["content_type"])


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
