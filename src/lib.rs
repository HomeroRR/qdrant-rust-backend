use anyhow::Result;
use fastembed::{ImageEmbeddingModel, ImageInitOptions};
use neon::prelude::*;
use pyo3::prelude::*;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{
    collections::HashMap,
    path::PathBuf,
    time::{Duration, Instant},
};
use tracing::{error, info, warn};
use walkdir::WalkDir;

use qdrant_client::{
    qdrant::{
        vectors_config::Config as VectorConfigType, CreateCollection, Distance, PointId,
        PointStruct, UpsertPointsBuilder, Value as QdrantValue, VectorParams, Vectors,
        VectorsConfig,
    },
    Qdrant,
};

const COLLECTION_NAME: &str = "image_embeddings_rust";
const VECTOR_SIZE: u64 = 512; // CLIP embedding size (Qdrant/clip-ViT-B-32-vision)
const OPTIMAL_BATCH_SIZE: usize = 100; // Match Python's batch size for fair comparison

async fn verify_collection_count(client: &Qdrant, expected_count: Option<u64>) -> Result<u64> {
    let collection_info = client.collection_info(COLLECTION_NAME).await?;
    let points_count = collection_info
        .result
        .ok_or_else(|| anyhow::anyhow!("Failed to get collection info"))?
        .points_count
        .ok_or_else(|| anyhow::anyhow!("Failed to get points count"))?;

    if let Some(expected) = expected_count {
        if points_count != expected {
            warn!(
                "Collection count mismatch - expected: {}, actual: {}",
                expected, points_count
            );
        } else {
            info!("Collection count verified: {}", points_count);
        }
    }

    Ok(points_count)
}

pub async fn init_qdrant_collection(client: &Qdrant) -> Result<()> {
    let collections = client.list_collections().await?;

    // Check if collection exists
    let collection_exists = collections
        .collections
        .iter()
        .any(|c| c.name == COLLECTION_NAME);

    if !collection_exists {
        info!("Creating new collection: {}", COLLECTION_NAME);

        // Create collection with explicit vector configuration
        let create_collection = CreateCollection {
            collection_name: COLLECTION_NAME.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(VectorConfigType::Params(VectorParams {
                    size: VECTOR_SIZE,
                    distance: Distance::Cosine as i32,
                    on_disk: Some(false), // Keep vectors in memory for better performance
                    hnsw_config: None,    // Use default HNSW config
                    quantization_config: None, // No quantization
                    ..Default::default()
                })),
            }),
            ..Default::default()
        };

        client.create_collection(create_collection).await?;

        // Verify collection was created
        info!(
            "Collection created successfully with name: {}",
            COLLECTION_NAME
        );
        info!("Vector size configured as: {}", VECTOR_SIZE);
    } else {
        info!("Collection {} already exists", COLLECTION_NAME);
    }
    let count = verify_collection_count(client, None).await?;
    info!("Current collection size: {} points", count);

    Ok(())
}

pub async fn verify_qdrant_connection(client: &Qdrant) -> Result<()> {
    match client.health_check().await {
        Ok(_) => {
            info!("Successfully connected to Qdrant server");
            Ok(())
        }
        Err(e) => {
            error!("Failed to connect to Qdrant server: {}", e);
            Err(anyhow::anyhow!("Failed to connect to Qdrant server: {}", e))
        }
    }
}

fn collect_image_paths(image_dir: &str) -> Result<Vec<PathBuf>> {
    // Collect paths first with parallel iterator
    let image_paths: Vec<_> = WalkDir::new(image_dir)
        .into_iter()
        .par_bridge() // Use parallel iterator for directory walking
        .filter_map(|e| e.ok())
        .filter(|e| {
            if let Some(ext) = e.path().extension() {
                ["jpg", "jpeg", "png"].contains(&ext.to_str().unwrap_or(""))
            } else {
                false
            }
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    let total_images = image_paths.len();
    info!("Found {} eligible images to process", total_images);

    if total_images == 0 {
        warn!("No images found in directory: {}", image_dir);
    }
    Ok(image_paths)
}

fn create_points(
    file_names: &Vec<&str>,
    embeddings: &Vec<Vec<f32>>,
    next_point_id: &mut u64,
    points: &mut Vec<PointStruct>,
) -> Result<()> {
    info!("Initial collection count: {}", next_point_id);

    // Initialize CLIP model
    for (&file_name, embedding) in file_names.iter().zip(embeddings.iter()) {
        // Verify embedding vector
        if embedding.len() != VECTOR_SIZE as usize {
            error!(
                "Invalid embedding size for {}: expected {}, got {}",
                file_name,
                VECTOR_SIZE,
                embedding.len()
            );
            continue;
        }

        let mut payload = HashMap::<String, QdrantValue>::new();

        payload.insert(
            "filename".to_string(),
            QdrantValue {
                kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                    file_name.to_string(),
                )),
            },
        );

        let point = PointStruct {
            id: Some(PointId::from(*next_point_id)),
            payload,
            vectors: Some(Vectors {
                vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                    qdrant_client::qdrant::Vector {
                        data: embedding.clone(),
                        indices: None,       // No sparse indices
                        vector: None,        // Not using legacy vector field
                        vectors_count: None, // Not using multiple vectors
                    },
                )),
            }),
            ..Default::default()
        };

        points.push(point);

        *next_point_id += 1;
    }
    Ok(())
}

async fn verify_upsert(
    client: &Qdrant,
    expected_count: u64,
    points_len: usize,
    chunk_idx: usize,
    chunks_len: usize,
) -> Result<()> {
    let new_count = verify_collection_count(client, Some(expected_count)).await?;

    if new_count != expected_count {
        error!(
            "Upsert verification failed - Expected: {}, Actual: {}",
            expected_count, new_count
        );
    } else {
        info!(
            "Successfully upserted {} points in batch {}/{}",
            points_len,
            chunk_idx + 1,
            chunks_len
        );
    }

    Ok(())
}

pub async fn process_images(
    client: &Qdrant,
    image_dir: &str,
    model: &ImageEmbedding,
) -> Result<()> {
    let start_time = Instant::now();
    info!("Starting image processing from directory: {}", image_dir);

    // Get initial collection count
    let initial_count = verify_collection_count(client, None).await?;
    info!("Initial collection count: {}", initial_count);
    let mut next_point_id = initial_count as u64;

    // Collect image paths
    let image_paths = collect_image_paths(image_dir)?;

    // Process images in optimized batches
    let chunks: Vec<_> = image_paths.chunks(OPTIMAL_BATCH_SIZE).collect();
    let mut total_embedding_time = Duration::new(0, 0);
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let images: Vec<&str> = chunk.iter().map(|p| p.to_str().unwrap()).collect();
        let embedding_start = Instant::now();
        // Generate embeddings with the OPTIMAL_BATCH_SIZE 100
        let embeddings = model.embed(images, OPTIMAL_BATCH_SIZE)?;
        let batch_embedding_time = embedding_start.elapsed();
        total_embedding_time += batch_embedding_time;

        // Create points for batch with verification
        let file_names: Vec<&str> = chunk
            .iter()
            .map(|p| p.file_name().and_then(|n| n.to_str()).unwrap_or("unknown"))
            .collect();
        let points_len = file_names.len();
        let mut points: Vec<PointStruct> = Vec::with_capacity(points_len);
        create_points(&file_names, &embeddings, &mut next_point_id, &mut points)?;

        // Verify points before upserting
        info!(
            "Preparing to upsert {} points for batch {}/{}",
            points_len, // -> Points length: 100
            chunk_idx + 1,
            chunks.len()
        );

        // Upsert points to Qdrant
        client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points))
            .await?;

        // Verify the update was successful
        verify_upsert(client, next_point_id, points_len, chunk_idx, chunks.len()).await?;
    }

    let total_time = start_time.elapsed();

    // Print benchmark summary
    info!("\n=== BENCHMARK RESULTS ===");
    info!("Total images processed: {}", image_paths.len());
    info!("Total processing time: {:?}", total_time);
    info!(
        "Throughput: {:.2} images/second",
        image_paths.len() as f64 / total_time.as_secs_f64()
    );
    info!("\nBreakdown:");
    info!("- Total embedding time: {:?}", total_embedding_time);

    Ok(())
}

pub struct ImageEmbedding {
    model: fastembed::ImageEmbedding,
}

impl ImageEmbedding {
    pub fn new(model_name: &str) -> Result<Self> {
        let embedding_model = match model_name {
            "Qdrant/clip-ViT-B-32-vision" => ImageEmbeddingModel::ClipVitB32,
            "Qdrant/resnet50-onnx" => ImageEmbeddingModel::Resnet50,
            "Qdrant/Unicom-ViT-B-16" => ImageEmbeddingModel::UnicomVitB16,
            "Qdrant/Unicom-ViT-B-32" => ImageEmbeddingModel::UnicomVitB32,
            "nomic-ai/nomic-embed-vision-v1.5" => ImageEmbeddingModel::NomicEmbedVisionV15,
            _ => return Err(anyhow::anyhow!("Invalid model code")),
        };
        let model = fastembed::ImageEmbedding::try_new(
            ImageInitOptions::new(embedding_model).with_show_download_progress(true),
        )?;
        Ok(Self { model })
    }

    pub fn embed(&self, images: Vec<&str>, batch_size: usize) -> Result<Vec<Vec<f32>>> {
        self.model.embed(images, Some(batch_size))
    }
}

#[pyclass(name = "ImageEmbedding")]
pub struct PyImageEmbedding {
    image_embedding: ImageEmbedding,
}

#[pymethods]
impl PyImageEmbedding {
    #[new]
    pub fn new(model_name: &str) -> Self {
        let image_embedding = ImageEmbedding::new(model_name).unwrap();
        Self { image_embedding }
    }

    #[pyo3(signature = (images, batch_size=1))]
    pub fn embed(&self, images: Vec<String>, batch_size: usize) -> Vec<Vec<f32>> {
        let images: Vec<&str> = images.iter().map(|s| s.as_str()).collect();
        self.image_embedding.embed(images, batch_size).unwrap()
    }
}

#[pymodule]
fn qdrant_embedding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImageEmbedding>()?;
    Ok(())
}

pub struct JSImageEmbedding {
    image_embedding: ImageEmbedding,
}

// Needed to box `ImageEmbedding` in JavaScript
impl Finalize for JSImageEmbedding {}

// Methods exposed to JavaScript
// The `JsBox` boxed `ImageEmbedding` is expected as the `this` value on all methods except `js_new`
impl JSImageEmbedding {
    // Create a new instance of `ImageEmbedding` and place it inside a `JsBox`
    // JavaScript can hold a reference to a `JsBox`, but the contents are opaque
    fn new<'a>(mut cx: FunctionContext<'a>) -> JsResult<'a, JsBox<JSImageEmbedding>> {
        let model_name = cx.argument::<JsString>(0)?.value(&mut cx);
        let image_embedding =
            ImageEmbedding::new(&model_name).or_else(|err| cx.throw_error(err.to_string()))?;
        Ok(cx.boxed(Self { image_embedding }))
    }

    fn embed(mut cx: FunctionContext) -> JsResult<JsArray> {
        let js_images = cx.argument::<JsArray>(0)?.to_vec(&mut cx)?;
        let len = js_images.len();
        let mut images = vec![];
        for i in 0..len {
            let image = js_images
                .get(i)
                .unwrap()
                .to_string(&mut cx)
                .unwrap()
                .value(&mut cx);
            images.push(image);
        }
        let images = images.iter().map(|s| s.as_str()).collect();
        let batch_size = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;

        // Get the `this` value as a `JsBox<JSImageEmbedding>`
        let result = cx
            .this::<JsBox<JSImageEmbedding>>()?
            .image_embedding
            .embed(images, batch_size)
            .or_else(|err| cx.throw_error(err.to_string()))?;

        let array = JsArray::new(&mut cx, result.len());
        for (i, s) in result.iter().enumerate() {
            let v = JsArray::new(&mut cx, s.len());
            for (j, f) in s.iter().enumerate() {
                let f = cx.number(*f);
                v.set(&mut cx, j as u32, f)?;
            }
            array.set(&mut cx, i as u32, v)?;
        }
        Ok(array)
    }
}

#[neon::main]
// Called once when the module is loaded
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    // Export each of the Neon functions as part of the module
    cx.export_function("new", JSImageEmbedding::new)?;
    cx.export_function("embed", JSImageEmbedding::embed)?;

    Ok(())
}
