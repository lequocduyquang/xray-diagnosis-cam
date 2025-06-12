import torch
import numpy as np
import cv2
import uuid
from sklearn.decomposition import PCA
from common_utils import get_model, preprocess_image, upload_to_cloudinary, cleanup, get_onnx_session

def generate_eigencam(image_path, model_name):
    """
    Generate Eigen-CAM heatmap for the given image and ONNX model.
    EigenCAM uses the principal components of the feature maps to generate class-agnostic explanations.
    """
    session = get_onnx_session(model_name)
    input_tensor = preprocess_image(image_path)  # shape (1, 3, 160, 160) - PyTorch tensor
    input_numpy = input_tensor.numpy()  # Convert to numpy array for ONNX
    input_name = session.get_inputs()[0].name

    # Forward pass
    ort_outs = session.run(None, {input_name: input_numpy})
    acts = ort_outs[0]  # Expect shape: (1, C, H, W)

    if len(acts.shape) != 4:
        raise RuntimeError("ONNX model output does not contain feature maps for EigenCAM.")

    # Reshape activations: (1, C, H, W) -> (C, H*W)
    acts_reshaped = acts.squeeze(0).reshape(acts.shape[1], -1)

    # Tính tổng tất cả activation maps (class-agnostic)
    heatmap = np.sum(acts_reshaped, axis=0)

    # Reshape heatmap về kích thước 2D ban đầu
    h, w = acts.shape[2], acts.shape[3]
    heatmap = heatmap.reshape(h, w)

    # Chuẩn hóa heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    # Visualization và lưu ảnh
    img = cv2.imread(image_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    result_path = f"eigencam_{uuid.uuid4().hex}.jpeg"
    cv2.imwrite(result_path, superimposed_img)

    cleanup(input_tensor, input_numpy, acts)

    return result_path

def generate_eigencam_pca(image_path, model_name, n_components=3):
    """
    Generate Eigen-CAM heatmap using PCA for better feature visualization.
    This version uses Principal Component Analysis to find the most important features.
    """
    device = torch.device("cpu")
    model = get_model(model_name).to(device)
    model.eval()

    # Xác định target layer
    if "resnet" in model_name:
        target_layer = model.layer4[-1]
    else:
        target_layer = model.features

    input_tensor = preprocess_image(image_path).to(device)
    
    # Hook để lấy activations
    activations = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    target_layer.register_forward_hook(forward_hook)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    acts = activations[0]  # Shape: (1, C, H, W)
    
    # Reshape activations: (1, C, H, W) -> (C, H*W)
    acts_reshaped = acts.squeeze(0).reshape(acts.shape[1], -1).detach().cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, acts_reshaped.shape[0]))
    pca_result = pca.fit_transform(acts_reshaped.T)  # Transpose to get (H*W, n_components)
    
    # Use the first principal component as heatmap
    heatmap = pca_result[:, 0].reshape(acts.shape[2], acts.shape[3])
    
    # Chuẩn hóa heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Visualization
    img = cv2.imread(image_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    result_path = f"eigencam_pca_{uuid.uuid4().hex}.jpeg"
    cv2.imwrite(result_path, superimposed_img)

    cleanup(model, input_tensor, acts)

    return result_path

def generate_eigencam_and_upload(image_path, model_name):
    result_path = generate_eigencam(image_path, model_name)
    return upload_to_cloudinary(result_path)

def generate_eigencam_pca_and_upload(image_path, model_name, n_components=3):
    result_path = generate_eigencam_pca(image_path, model_name, n_components)
    return upload_to_cloudinary(result_path)