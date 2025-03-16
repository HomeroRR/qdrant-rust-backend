#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Install required packages
pip install torch transformers pillow onnx

# Run the conversion script
python convert_to_onnx.py

echo "Download and conversion complete!" 