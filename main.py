from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import os
import cloudinary
import requests
import numpy as np
import cv2

from gradcam_utils import generate_gradcam_and_upload
from approxcam_utils import generate_approxcam_and_upload
from eigencam_utils import generate_eigencam_and_upload, generate_eigencam_pca_and_upload

app = FastAPI(title="Explainable AI API",)

# Cấu hình CORS - đơn giản và hiệu quả
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả methods
    allow_headers=["*"],  # Cho phép tất cả headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Explainable AI API"}

def handle_cam_request(
    image: UploadFile,
    model_name: str,
    cam_func,
    cam_key: str,
    **kwargs
):
    """
    This function is a placeholder for handling common logic for CAM requests.
    It can be extended to include shared functionality across different CAM endpoints.
    """
    try:
        # Lưu ảnh tạm
        temp_filename = f"temp_{uuid.uuid4().hex}.jpeg"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Sinh CAM và upload Cloudinary
        image_url = cam_func(temp_filename, model_name, **kwargs)

        os.remove(temp_filename)

        return JSONResponse(status_code=200, content={"success": True, cam_key: image_url})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/gradcam")
async def gradcam_endpoint(
    image: UploadFile = File(...),
    model_name: str = Form(...)
):
    return handle_cam_request(image, model_name, generate_gradcam_and_upload, "gradcam_url")

@app.post("/approxcam")
async def approxcam_endpoint(
    image: UploadFile = File(...),
    model_name: str = Form(...)
):
    return handle_cam_request(image, model_name, generate_approxcam_and_upload, "approxcam_url")

class EigencamRequest(BaseModel):
    cloudinary_id: str
    model_name: str

@app.post("/eigencam")
async def eigencam_endpoint(request: EigencamRequest):
    try:
        image_url = cloudinary.utils.cloudinary_url(request.cloudinary_id, secure=True)[0]

        response = requests.get(image_url)
        response.raise_for_status()

        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Cannot decode image from Cloudinary.")

        temp_filename = f"temp_{uuid.uuid4().hex}.jpeg"
        cv2.imwrite(temp_filename, image)

        result_url = generate_eigencam_and_upload(temp_filename, request.model_name)

        os.remove(temp_filename)

        return JSONResponse(status_code=200, content={"success": True, "eigencam_url": result_url})

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not download image from Cloudinary: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
