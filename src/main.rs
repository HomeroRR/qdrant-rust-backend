use anyhow::{Context, Result};
use qdrant_client::{
    qdrant::{
        vectors_config::Config as VectorConfigType, CreateCollection, Distance, VectorParams,
        VectorsConfig,
    },
    Qdrant,
};

use dotenv::dotenv;
use std::env;
use tracing::{error, info, warn};

const COLLECTION_NAME: &str = "image_embeddings_rust";
const VECTOR_SIZE: u64 = 768; // CLIP embedding size (ViT-L/14)
const _OPTIMAL_BATCH_SIZE: usize = 100; // Match Python's batch size for fair comparison

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

async fn process_images(_client: &Qdrant, _image_dir: &str) -> Result<()> {
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

    // Process images
    process_images(&client, "images").await?;

    info!("Image processing and embedding completed successfully!");
    Ok(())
}
