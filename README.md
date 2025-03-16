# CLIP Image Embeddings with Qdrant

This Rust program processes images using the CLIP vision model to generate vector embeddings and stores them in a Qdrant vector database. The program is designed to handle batch processing of images and includes robust error handling and verification steps.

## Dependencies

```toml
[dependencies]
anyhow = "1.0"           # Error handling
image = "0.24"           # Image processing
ndarray = "0.15"         # N-dimensional arrays
ort = "1.16"            # ONNX Runtime
qdrant-client = "1.7"    # Qdrant client
tokio = { version = "1.0", features = ["full"] } # Async runtime
tracing = "0.1"         # Logging
tracing-subscriber = "0.3" # Logging implementation
walkdir = "2.4"         # Directory traversal
```

## Overview

The program performs the following main tasks:

1. Loads images from a specified directory
2. Generates vector embeddings using CLIP
3. Stores these embeddings in a Qdrant collection

## Prerequisites

- Rust toolchain
- Python 3.8+ (for model conversion)
- CLIP vision model in ONNX format (`models/clip_vision.onnx`)
- Access to a Qdrant instance
- Image files (JPG, JPEG, or PNG) in the `images` directory

## Model Setup

### Why Convert CLIP to ONNX?

The CLIP model was originally developed and distributed in PyTorch format. While there are several Rust machine learning frameworks available (like `tch-rs` for PyTorch bindings), converting to ONNX format offers several significant advantages:

1. **No Python Dependencies**:

   - Using PyTorch bindings in Rust would still require Python and PyTorch to be installed
   - This adds complexity to deployment and can cause version conflicts
   - ONNX format allows us to run the model with pure Rust dependencies

2. **Better Performance**:

   - ONNX Runtime is highly optimized for inference
   - Supports hardware acceleration (CPU, GPU, Metal) without additional dependencies
   - Eliminates Python interpreter overhead
   - Reduces memory usage by removing unnecessary training-related components

3. **Cross-Platform Compatibility**:

   - ONNX is a standardized format supported across many platforms
   - Easier to deploy on different operating systems
   - No need to handle platform-specific PyTorch builds

4. **Simplified Deployment**:
   - Single file containing all model weights and architecture
   - No need to manage Python environment in production
   - Smaller deployment size without PyTorch dependencies

The conversion process uses a Python script (`convert_to_onnx.py`) to:

1. Load the PyTorch CLIP model
2. Export it to ONNX format with optimized settings
3. Save it as a standalone file that our Rust program can use

The converted model preserves:

- Model architecture and weights
- Input/output specifications
- All necessary layers for inference
- Hardware acceleration capabilities

To set up the model, run:

```bash
# Install Python dependencies
pip install torch transformers pillow

# Run the conversion script
python convert_to_onnx.py
```

This will create the ONNX model file at `models/clip_vision.onnx` that our Rust program uses.

### 2. Model Architecture

The CLIP vision model:

- Takes image input of size 224x224 pixels
- Processes through a Vision Transformer (ViT) architecture
- Outputs embeddings that capture semantic image features
- Final embedding dimension: 512

The conversion process preserves:

- Model weights and architecture
- Input/output specifications
- Dynamic batch size support
- All necessary layers for inference

## How It Works

### 1. Image Processing Pipeline

The program processes images through several stages:

```rust
// 1. Image loading and preprocessing
- Resizes images to 224x224 (CLIP input size)
- Converts to RGB format
- Normalizes pixel values to [-1, 1] range

// 2. CLIP Model Processing
- Runs the image through CLIP vision model
- Performs mean pooling on the output
- Normalizes the final embedding
```

### 2. Vector Embedding Generation

The CLIP model generates embeddings in the following steps:

1. **Preprocessing**: Images are resized and normalized
2. **Model Inference**: Processed through CLIP vision model
3. **Post-processing**:
   - Mean pooling over sequence dimension
   - L2 normalization of the final embedding
   - Final embedding size: 512 dimensions

### 3. Qdrant Integration

The program interacts with Qdrant in several ways:

1. **Collection Management**:

   - Creates a collection if it doesn't exist
   - Configures for 512-dimensional vectors
   - Uses cosine similarity for distance metric

2. **Batch Processing**:

   - Processes images in batches of 25
   - Verifies successful uploads
   - Maintains point count consistency

3. **Error Handling**:
   - Verifies Qdrant connection
   - Validates collection existence
   - Confirms successful point insertions

## Usage

1. Place your images in the `images` directory
2. Ensure the CLIP model is in the `models` directory
3. Run the program:
   ```bash
   cargo run
   ```

The program will:

- Process up to 50 images (configurable)
- Generate embeddings
- Store them in Qdrant
- Provide detailed logging of the process

## Configuration

Key constants in the code:

```rust
const COLLECTION_NAME: &str = "image_embeddings";
const VECTOR_SIZE: u64 = 512;
const MAX_IMAGES: usize = 50;
```

## Error Handling

The program includes comprehensive error handling:

- Image loading/decoding errors
- Model inference errors
- Qdrant connection/upload errors
- Collection verification
- Point count validation

## Logging

Detailed logging is provided for:

- Image processing progress
- Embedding generation
- Qdrant operations
- Error conditions
- Upload verification

## Limitations

- Processes only JPG, JPEG, and PNG files
- Limited to first 50 images by default
- Requires ONNX format CLIP model
