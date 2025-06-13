import torch
from torchvision import models, transforms
from PIL import Image
import cloudinary.uploader
import os
import gc
import onnxruntime as ort

# Cloudinary config (chỉ cần gọi 1 lần)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

MODEL_PATHS = {
    "resnet50_v1": "models/densenet121.pth",
    "resnet50_v2": "models/densenet121.pth",
    "resnet50": "models/densenet121.pth",
    "densenet121": "models/densenet121.pth",
}

# ONNX_PATHS = {
#     # "resnet50_v1": "models/resnet50.onnx",
#     # "resnet50_v2": "models/resnet50.onnx",
#     # "resnet50": "models/resnet50.onnx",
#     "resnet50_v1": "models/densenet121_160x160.onnx", # Temporary using this for test low performance
#     "resnet50_v2": "models/densenet121_160x160.onnx",
#     "resnet50": "models/densenet121_160x160.onnx",
#     "densenet121": "models/densenet121_160x160.onnx",
# }

# Single transform for both PyTorch and ONNX models (160x160)
TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MODELS_CACHE = {}

def get_model(model_name):
    # Convert model name to lowercase for case-insensitive matching
    model_name = model_name.lower()
    if model_name == "resnet50":
        model_name = "resnet50_v1"

    # Xóa tất cả model cũ khỏi cache để tiết kiệm RAM
    for k in list(MODELS_CACHE.keys()):
        if k != model_name:
            del MODELS_CACHE[k]
            gc.collect()

    if model_name not in MODELS_CACHE:
        if model_name == "resnet50_v1" or model_name == "resnet50_v2":
            weights = None
            model = models.resnet50(weights=weights)
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(model.fc.in_features, 2)
            )
        elif model_name == "densenet121":
            weights = None
            model = models.densenet121(weights=weights)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(model.classifier.in_features, 5)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location="cpu"))
        model.eval()
        MODELS_CACHE[model_name] = model
    return MODELS_CACHE[model_name]

ONNX_SESSIONS = {}

def get_onnx_session(model_name):
    model_name = model_name.lower()
    if model_name == "resnet50":
        model_name = "resnet50_v1"
    for k in list(ONNX_SESSIONS.keys()):
        if k != model_name:
            del ONNX_SESSIONS[k]
            gc.collect()
    if model_name not in ONNX_SESSIONS:
        # Create session with dynamic shapes support
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            ONNX_PATHS[model_name], 
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )
        
        # Check input shape
        input_meta = session.get_inputs()[0]
        print(f"ONNX model {model_name} input shape: {input_meta.shape}")
        
        ONNX_SESSIONS[model_name] = session
    return ONNX_SESSIONS[model_name]

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return TRANSFORM(image).unsqueeze(0)  # Always use 160x160
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to process image {image_path}: {str(e)}")

def upload_to_cloudinary(file_path):
    try:
        upload_result = cloudinary.uploader.upload(file_path)
        secure_url = upload_result["secure_url"]
    except Exception as e:
        raise RuntimeError(f"Failed to upload to Cloudinary: {str(e)}")
    finally:
        # Always clean up local file
        try:
            os.remove(file_path)
        except OSError:
            pass  # File might not exist or be accessible
    return secure_url

def cleanup(*args):
    for obj in args:
        del obj
    gc.collect()
