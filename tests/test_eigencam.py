#!/usr/bin/env python3
"""
Test script for EigenCAM functionality
"""

import requests
import os
from PIL import Image
import numpy as np

def test_eigencam_endpoints():
    """
    Test both EigenCAM endpoints with a sample image
    """
    
    # URL của API
    base_url = "http://localhost:8000"
    
    # Test image path (you need to provide a real X-ray image)
    test_image_path = "test_xray.jpg"  # Replace with your test image
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please provide a test X-ray image to test the endpoints")
        return
    
    # Test models
    models = ["resnet50_v1", "resnet50_v2", "densenet121"]
    
    print("Testing EigenCAM endpoints...")
    print("=" * 50)
    
    for model_name in models:
        print(f"\nTesting with model: {model_name}")
        
        # Test basic EigenCAM
        print("1. Testing basic EigenCAM...")
        try:
            with open(test_image_path, "rb") as f:
                files = {"image": f}
                data = {"model_name": model_name}
                response = requests.post(f"{base_url}/eigencam", files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Basic EigenCAM successful: {result['eigencam_url']}")
            else:
                print(f"❌ Basic EigenCAM failed: {response.text}")
        except Exception as e:
            print(f"❌ Basic EigenCAM error: {str(e)}")
        
        # Test EigenCAM PCA
        print("2. Testing EigenCAM PCA...")
        try:
            with open(test_image_path, "rb") as f:
                files = {"image": f}
                data = {"model_name": model_name, "n_components": 3}
                response = requests.post(f"{base_url}/eigencam-pca", files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                print(f"✅ EigenCAM PCA successful: {result['eigencam_pca_url']}")
            else:
                print(f"❌ EigenCAM PCA failed: {response.text}")
        except Exception as e:
            print(f"❌ EigenCAM PCA error: {str(e)}")

def compare_cam_methods():
    """
    Compare different CAM methods on the same image
    """
    print("\n" + "=" * 50)
    print("Comparing CAM Methods")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    test_image_path = "test_xray.jpg"
    model_name = "resnet50_v1"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    cam_methods = [
        ("GradCAM", "/gradcam"),
        ("ApproxCAM", "/approxcam"),
        ("EigenCAM", "/eigencam"),
        ("EigenCAM PCA", "/eigencam-pca")
    ]
    
    results = {}
    
    for method_name, endpoint in cam_methods:
        print(f"\nTesting {method_name}...")
        try:
            with open(test_image_path, "rb") as f:
                files = {"image": f}
                data = {"model_name": model_name}
                if method_name == "EigenCAM PCA":
                    data["n_components"] = 3
                    
                response = requests.post(f"{base_url}{endpoint}", files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                key = list(result.keys())[1]  # Get the URL key
                results[method_name] = result[key]
                print(f"✅ {method_name} successful: {result[key]}")
            else:
                print(f"❌ {method_name} failed: {response.text}")
        except Exception as e:
            print(f"❌ {method_name} error: {str(e)}")
    
    print(f"\nSummary of results:")
    for method, url in results.items():
        print(f"{method}: {url}")

if __name__ == "__main__":
    print("EigenCAM Test Script")
    print("Make sure your FastAPI server is running on localhost:8000")
    print("You need to provide a test X-ray image named 'test_xray.jpg'")
    
    test_eigencam_endpoints()
    compare_cam_methods() 