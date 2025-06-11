import torch
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.densenet import DenseNet121_Weights
from PIL import Image
import cloudinary.uploader
import os
import gc

# Cloudinary config (chỉ cần gọi 1 lần)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

MODEL_PATHS = {
    "resnet50_v1": "models/resnet50_v1.pth",
    "resnet50_v2": "models/resnet50_v2.pth",
    "resnet50": "models/resnet50_v1.pth",
    "densenet121": "models/densenet121.pth",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MODELS_CACHE = {}

def get_model(model_name):
    # Convert model name to lowercase for case-insensitive matching
    model_name = model_name.lower()
    
    if model_name not in MODELS_CACHE:
        if model_name == "resnet50_v1" or model_name == "resnet50_v2":
            weights = ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(model.fc.in_features, 2)
            )
        elif model_name == "densenet121":
            weights = DenseNet121_Weights.DEFAULT
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

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return TRANSFORM(image).unsqueeze(0)
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