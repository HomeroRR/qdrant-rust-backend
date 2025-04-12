use anyhow::{Context, Result};
use dotenv::dotenv;
use fastembed::{ImageEmbedding, ImageEmbeddingModel, ImageInitOptions};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{
    collections::HashMap,
    env,
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

async fn init_qdrant_collection(client: &Qdrant) -> Result<()> {
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

async fn verify_qdrant_connection(client: &Qdrant) -> Result<()> {
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

async fn process_images(client: &Qdrant, image_dir: &str, model: &ImageEmbedding) -> Result<()> {
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
        let embeddings = model.embed(images, Some(OPTIMAL_BATCH_SIZE))?;
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

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenv().ok();

    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get Qdrant configuration from environment variables
    let qdrant_url = env::var("QDRANT_URL").context("QDRANT_URL environment variable not set")?;
    let qdrant_api_key =
        env::var("QDRANT_API_KEY").context("QDRANT_API_KEY environment variable not set")?;

    // Initialize Qdrant client
    let client = Qdrant::from_url(&qdrant_url)
        .api_key(qdrant_api_key)
        .build()?;

    // Verify connection to Qdrant
    verify_qdrant_connection(&client).await?;

    // Initialize collection
    init_qdrant_collection(&client).await?;

    // Load the model
    let model = ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(true),
    )?;

    // Process images
    process_images(&client, "images", &model).await?;

    info!("Image processing and embedding completed successfully!");
    Ok(())
}
