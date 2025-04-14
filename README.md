# Image Embeddings for Qdrant

This Rust library with Python bindings generates vector embeddings from images.

The example `upload_images.rs` uses a CLIP vision model to store vector embeddings in a Qdrant vector database.

The program is designed to handle batch processing of images and includes robust error handling and verification steps.

## Usage

```python
from qdrant_embedding import ImageEmbedding

# Initialize FastEmbed models
image_model = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")

# Use FastEmbed to generate text embedding
embeddings = image_model.embed(["images/1_bathroom.jpg"])
```

## Prerequisites

- Rust toolchain >= 2021
- Python 3.8+ (pip install maturin)
- `maturin develop` (to get Python bindings)

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
fastembed = "4.6.0"     # Fastembed rust implementation
pyo3 = "0.24.0"         # To call from Python
```

## Overview

The example `upload_images.rs` performs the following main tasks:

1. Loads images from a specified directory
2. Generates vector embeddings using CLIP
3. Stores these embeddings in a Qdrant collection

### Prerequisites for the examples

- Access to a Qdrant instance (to upload vector embeddings)
- Image files (JPG, JPEG, or PNG) in the `images` directory

### Usage

1. Place your images in the `images` directory
2. Set the `QDRANT_URL` and `QDRANT_API_KEY` environment variables
3. Run the program:
   ```bash
   cargo run --example upload_images
   ```

The program will:

- Process images
- Generate embeddings
- Store them in Qdrant
- Provide detailed logging of the process

### Configuration

Key constants in the code:

```rust
const COLLECTION_NAME: &str = "image_embeddings_rust";
const VECTOR_SIZE: u64 = 512;
```

### Qdrant Integration

The program interacts with Qdrant in several ways:

1. **Collection Management**:

   - Creates a collection if it doesn't exist
   - Configures for 512-dimensional vectors
   - Uses cosine similarity for distance metric

2. **Batch Processing**:

   - Processes images in batches
   - Verifies successful uploads
   - Maintains point count consistency

3. **Error Handling**:
   - Verifies Qdrant connection
   - Validates collection existence
   - Confirms successful point insertions

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
