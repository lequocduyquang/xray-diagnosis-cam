import torch
import torch.nn.functional as F
import numpy as np
import cv2
import uuid
from common_utils import get_model, preprocess_image, upload_to_cloudinary, cleanup

def generate_gradcam(image_path, model_name):
    """
    Generate Grad-CAM heatmap for the given image and model.
    """
    device = torch.device("cpu")
    model = get_model(model_name).to(device)
    model.eval()

    target_layer = model.layer4[-1] if "resnet" in model_name else model.features[-1]

    input_tensor = preprocess_image(image_path).to(device)
    input_tensor.requires_grad = True

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    grads = gradients[0]
    acts = activations[0]

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu().numpy()

    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    result_path = f"gradcam_{uuid.uuid4().hex}.jpeg"
    cv2.imwrite(result_path, superimposed_img)

    cleanup(model, input_tensor, grads, acts, heatmap)

    return result_path

def generate_gradcam_and_upload(image_path, model_name):
    result_path = generate_gradcam(image_path, model_name)
    return upload_to_cloudinary(result_path)