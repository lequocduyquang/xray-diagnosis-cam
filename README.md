# EigenCAM Implementation for X-Ray Diagnosis

## Tổng quan

EigenCAM là một phương pháp XAI (Explainable AI) class-agnostic, có nghĩa là nó không cần gradient hay class-specific information để tạo ra heatmap giải thích. Điều này làm cho EigenCAM rất hữu ích cho việc phân tích các mô hình deep learning trong chẩn đoán X-quang.

## Tính năng chính

### 1. EigenCAM Cơ bản

- **Class-agnostic**: Không cần thông tin về class cụ thể
- **Gradient-free**: Không cần tính gradient
- **Fast**: Chỉ cần 1 forward pass
- **Interpretable**: Tạo heatmap dựa trên tổng các activation maps

### 2. EigenCAM với PCA

- **Principal Component Analysis**: Sử dụng PCA để tìm các features quan trọng nhất
- **Better visualization**: Kết quả rõ ràng hơn với các principal components
- **Configurable**: Có thể điều chỉnh số lượng components

## API Endpoints

### 1. Basic EigenCAM

```http
POST /eigencam
Content-Type: multipart/form-data

Parameters:
- image: File (required) - Ảnh X-quang
- model_name: String (required) - Tên model ("resnet50_v1", "resnet50_v2", "densenet121") - case insensitive

Response:
{
  "success": true,
  "eigencam_url": "https://res.cloudinary.com/..."
}
```

### 2. EigenCAM PCA

```http
POST /eigencam-pca
Content-Type: multipart/form-data

Parameters:
- image: File (required) - Ảnh X-quang
- model_name: String (required) - Tên model (case insensitive)
- n_components: Integer (optional, default=3) - Số principal components

Response:
{
  "success": true,
  "eigencam_pca_url": "https://res.cloudinary.com/..."
}
```

## Cách hoạt động

### EigenCAM Cơ bản

1. **Extract Activations**: Lấy activation maps từ layer cuối cùng của feature extractor
2. **Sum Activation Maps**: Tính tổng tất cả activation maps
3. **Normalize**: Chuẩn hóa heatmap về range [0, 1]
4. **Visualize**: Tạo heatmap màu và overlay lên ảnh gốc

### EigenCAM PCA

1. **Extract Activations**: Lấy activation maps
2. **Reshape**: Chuyển đổi thành matrix (C, H\*W)
3. **Apply PCA**: Sử dụng Principal Component Analysis
4. **Use First PC**: Sử dụng principal component đầu tiên làm heatmap
5. **Visualize**: Tạo heatmap và overlay

## So sánh với các phương pháp khác

| Phương pháp  | Class-specific | Gradient-free | Speed  | Interpretability |
| ------------ | -------------- | ------------- | ------ | ---------------- |
| GradCAM      | ❌             | ❌            | Medium | High             |
| ApproxCAM    | ❌             | ❌            | Slow   | High             |
| EigenCAM     | ✅             | ✅            | Fast   | Medium           |
| EigenCAM PCA | ✅             | ✅            | Fast   | High             |

## Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test EigenCAM

```bash
python test_eigencam.py
```

## Sử dụng với curl

### Basic EigenCAM

```bash
curl -X POST "http://localhost:8000/eigencam" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@your_xray_image.jpg" \
  -F "model_name=resnet50_v1"
```

### EigenCAM PCA

```bash
curl -X POST "http://localhost:8000/eigencam-pca" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@your_xray_image.jpg" \
  -F "model_name=resnet50_v1" \
  -F "n_components=3"
```

## Lợi ích cho chẩn đoán X-quang

1. **Class-agnostic**: Có thể phân tích bất kỳ ảnh X-quang nào mà không cần biết kết quả chẩn đoán
2. **Fast**: Tốc độ nhanh, phù hợp cho real-time analysis
3. **Interpretable**: Giúp bác sĩ hiểu được model đang tập trung vào vùng nào
4. **Robust**: Không phụ thuộc vào gradient, ổn định hơn

## Cấu trúc code

```
├── eigencam_utils.py          # Implementation EigenCAM
├── main.py                    # FastAPI endpoints
├── common_utils.py            # Shared utilities
├── test_eigencam.py           # Test script
└── README_EIGENCAM.md         # Documentation
```

## Troubleshooting

### Lỗi thường gặp

1. **Model not found**: Đảm bảo model files tồn tại trong thư mục `models/`
2. **Memory issues**: EigenCAM sử dụng ít memory hơn GradCAM
3. **Image format**: Hỗ trợ JPEG, PNG, BMP

### Performance tips

1. **Batch processing**: Có thể xử lý nhiều ảnh cùng lúc
2. **Caching**: Models được cache để tăng tốc độ
3. **GPU support**: Có thể chuyển sang GPU nếu cần

## Tài liệu tham khảo

- [EigenCAM: Visual Explanation using Principal Components](https://arxiv.org/abs/2008.00299)
- [GradCAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Explainable AI for Medical Imaging](https://www.nature.com/articles/s41598-020-71282-6)
