use anyhow::{Context, Result};
use dotenv::dotenv;
use qdrant_embedding::{
    init_qdrant_collection, process_images, verify_qdrant_connection, ImageEmbedding,
};
use std::env;
use tracing::info;

use qdrant_client::Qdrant;

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
    let model = ImageEmbedding::new("Qdrant/clip-ViT-B-32-vision")?;

    // Process images
    process_images(&client, "images", &model).await?;

    info!("Image processing and embedding completed successfully!");
    Ok(())
}
