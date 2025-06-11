import torch
import numpy as np
import cv2
import uuid

from common_utils import get_model, preprocess_image, upload_to_cloudinary, cleanup

def generate_approxcam(image_path, model_name, target_class_idx=None):
    """
    Generate ApproxCAM heatmap using PyTorch model (.pth) directly.
    """
    device = torch.device("cpu")
    model = get_model(model_name).to(device)
    model.eval()

    # Preprocess input
    input_tensor = preprocess_image(image_path).to(device)

    # Forward pass
    with torch.no_grad():
        feature_maps = model.features(input_tensor)
        logits = model.classifier(feature_maps.mean(dim=(2, 3)))

    # Lấy lớp mục tiêu (target_class_idx)
    if target_class_idx is None:
        target_class_idx = torch.argmax(logits, dim=1).item()

    # Tính ApproxCAM
    weights = logits[:, target_class_idx].unsqueeze(-1).unsqueeze(-1)
    approxcam = torch.sum(weights * feature_maps, dim=1).squeeze(0)

    # Chuẩn hóa heatmap
    approxcam = approxcam.cpu().numpy()
    approxcam = np.maximum(approxcam, 0)
    approxcam = approxcam / (approxcam.max() + 1e-8)

    # Đọc ảnh gốc, resize heatmap về kích thước ảnh
    img = cv2.imread(image_path)
    heatmap = cv2.resize(approxcam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    result_path = f"approxcam_{uuid.uuid4().hex}.jpeg"
    cv2.imwrite(result_path, superimposed_img)

    cleanup(model, input_tensor, feature_maps, logits, approxcam)

    return result_path

def generate_approxcam_and_upload(image_path, model_name):
    result_path = generate_approxcam(image_path, model_name)
    return upload_to_cloudinary(result_path)