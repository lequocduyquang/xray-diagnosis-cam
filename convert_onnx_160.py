#!/usr/bin/env python3
"""
Script to convert ONNX models to support 160x160 input size
"""

import onnx
import numpy as np
import os

def convert_onnx_to_160x160(input_path, output_path):
    """
    Convert ONNX model to support 160x160 input size
    """
    print(f"Converting {input_path} to support 160x160 input...")
    
    # Load the original model
    model = onnx.load(input_path)
    
    # Get the input node
    input_node = model.graph.input[0]
    
    # Update input shape to 160x160
    # Find the shape info
    for i, dim in enumerate(input_node.type.tensor_type.shape.dim):
        if i == 2:  # Height dimension
            dim.dim_value = 160
        elif i == 3:  # Width dimension
            dim.dim_value = 160
    
    # Update any resize operations in the model if they exist
    for node in model.graph.node:
        if node.op_type == "Resize":
            # Update resize scales if they exist
            for attr in node.attribute:
                if attr.name == "scales":
                    # Calculate new scales for 160x160
                    # Assuming original was 224x224
                    scale_h = 160.0 / 224.0
                    scale_w = 160.0 / 224.0
                    attr.floats[:] = [1.0, 1.0, scale_h, scale_w]
    
    # Save the converted model
    onnx.save(model, output_path)
    print(f"Converted model saved to {output_path}")
    
    # Verify the model
    try:
        converted_model = onnx.load(output_path)
        input_shape = converted_model.graph.input[0].type.tensor_type.shape
        print(f"Input shape: {[dim.dim_value for dim in input_shape.dim]}")
        print("✅ Model conversion successful!")
    except Exception as e:
        print(f"❌ Error verifying converted model: {e}")

def main():
    """
    Convert all ONNX models to 160x160
    """
    models_dir = "models"
    
    # List of models to convert
    models_to_convert = [
        "densenet121.onnx",
        # Add other models here if needed
    ]
    
    for model_name in models_to_convert:
        input_path = os.path.join(models_dir, model_name)
        output_path = os.path.join(models_dir, f"{model_name.replace('.onnx', '_160x160.onnx')}")
        
        if os.path.exists(input_path):
            convert_onnx_to_160x160(input_path, output_path)
        else:
            print(f"❌ Model not found: {input_path}")

if __name__ == "__main__":
    main() 