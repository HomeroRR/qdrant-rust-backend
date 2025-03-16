use anyhow::{Result, Context};
use image::{io::Reader as ImageReader, DynamicImage};
use ndarray::{Array4, ArrayView, CowArray, IxDyn, Axis, s};
use ort::{Environment, Session, SessionBuilder, Value};
use qdrant_client::{
    prelude::*,
    qdrant::{
        CreateCollection,
        VectorParams,
        VectorsConfig,
        Distance,
        Value as QdrantValue,
        PointStruct,
        vectors_config::Config as VectorConfigType,
        WriteOrdering,
        PointId,
        shard_key::Key,
        Vectors,
        Vector,
    },
    config::QdrantConfig,
};
use std::{path::{Path, PathBuf}, sync::{Arc, Mutex}, collections::HashMap, time::{Duration, Instant}};
use walkdir::WalkDir;
use tracing::{info, warn, error, debug};
use dotenv::dotenv;
use std::env;
use rayon::prelude::*;
use tokio::sync::mpsc;
use rayon::ThreadPoolBuilder;
use num_cpus;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const COLLECTION_NAME: &str = "image_embeddings_rust";
const VECTOR_SIZE: u64 = 768; // CLIP embedding size (ViT-L/14)
const OPTIMAL_BATCH_SIZE: usize = 100;  // Match Python's batch size for fair comparison

struct ClipModel {
    session: Session,
    buffer: Array4<f32>,
}

impl ClipModel {
    fn new() -> Result<Self> {
        // Initialize ONNX Runtime environment with optimizations
        let mut builder = Environment::builder()
            .with_name("clip_optimized");
            
        // Set thread count through environment variables
        std::env::set_var("ORT_THREAD_POOL_SIZE", num_cpus::get().to_string());
        
        // Build the environment
        let env = Arc::new(builder.build()?);
        
        // Load the CLIP model with optimizations
        let session = SessionBuilder::new(&env)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_intra_threads(num_cpus::get() as i16)?
            .with_model_from_file("models/clip_vision.onnx")?;
        
        Ok(ClipModel { 
            session, 
            buffer: Array4::zeros((OPTIMAL_BATCH_SIZE, 3, 224, 224))  // Pre-allocate larger buffer
        })
    }

    fn preprocess_image(&self, image: &DynamicImage) -> Result<Array4<f32>> {
        let resize_start = Instant::now();
        // Resize image to CLIP input size (224x224)
        let image = image.resize_exact(224, 224, image::imageops::FilterType::Triangle);
        let resize_time = resize_start.elapsed();
        
        let convert_start = Instant::now();
        // Convert to RGB float tensor and normalize
        let rgb_image = image.to_rgb8();
        let mut tensor = Array4::<f32>::zeros((1, 3, 224, 224));
        
        // Get raw bytes
        let raw_pixels = rgb_image.as_raw();
        let convert_time = convert_start.elapsed();
        
        let normalize_start = Instant::now();
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let scale = _mm256_set1_ps(1.0 / 127.5);
                    let offset = _mm256_set1_ps(-1.0);
                    
                    for c in 0..3 {
                        let channel_offset = c * 224 * 224;
                        for chunk_idx in (0..224*224).step_by(8) {
                            let mut chunk = [0u8; 8];
                            for (i, &pixel) in raw_pixels[chunk_idx*3..(chunk_idx+8)*3].step_by(3).enumerate() {
                                chunk[i] = pixel;
                            }
                            
                            let pixels = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(chunk.as_ptr() as *const _)));
                            let normalized = _mm256_fmadd_ps(pixels, scale, offset);
                            
                            let y = (chunk_idx / 224) % 224;
                            let x = chunk_idx % 224;
                            for i in 0..8 {
                                if chunk_idx + i < 224*224 {
                                    tensor[[0, c, y + i/224, (x + i) % 224]] = *normalized.as_ptr().add(i) as f32;
                                }
                            }
                        }
                    }
                }
            } else {
                // Fallback to non-SIMD implementation
                raw_pixels.chunks_exact(3)
                    .enumerate()
                    .for_each(|(i, pixel)| {
                        let y = (i / 224) % 224;
                        let x = i % 224;
                        tensor[[0, 0, y, x]] = (pixel[0] as f32 / 127.5) - 1.0;
                        tensor[[0, 1, y, x]] = (pixel[1] as f32 / 127.5) - 1.0;
                        tensor[[0, 2, y, x]] = (pixel[2] as f32 / 127.5) - 1.0;
                    });
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            raw_pixels.chunks_exact(3)
                .enumerate()
                .for_each(|(i, pixel)| {
                    let y = (i / 224) % 224;
                    let x = i % 224;
                    tensor[[0, 0, y, x]] = (pixel[0] as f32 / 127.5) - 1.0;
                    tensor[[0, 1, y, x]] = (pixel[1] as f32 / 127.5) - 1.0;
                    tensor[[0, 2, y, x]] = (pixel[2] as f32 / 127.5) - 1.0;
                });
        }
        
        let normalize_time = normalize_start.elapsed();
        
        // Log preprocessing breakdown
        debug!(
            "Image preprocessing breakdown:\n\
             - Resize: {:?} ({:.1}%)\n\
             - Convert: {:?} ({:.1}%)\n\
             - Normalize: {:?} ({:.1}%)\n\
             Total: {:?}",
            resize_time,
            100.0 * resize_time.as_secs_f64() / (resize_time + convert_time + normalize_time).as_secs_f64(),
            convert_time,
            100.0 * convert_time.as_secs_f64() / (resize_time + convert_time + normalize_time).as_secs_f64(),
            normalize_time,
            100.0 * normalize_time.as_secs_f64() / (resize_time + convert_time + normalize_time).as_secs_f64(),
            resize_time + convert_time + normalize_time
        );
        
        Ok(tensor)
    }

    fn get_image_embedding(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        // Preprocess image
        let input_tensor = self.preprocess_image(img)?;
        
        // Convert to dynamic dimensions for ONNX Runtime
        let input_tensor = input_tensor.into_dyn();
        let input_tensor = CowArray::from(input_tensor);
        
        // Create input for model
        let input = Value::from_array(self.session.allocator(), &input_tensor)?;
        
        // Run inference
        let outputs = self.session.run(vec![input])?;
        
        // Get embeddings from output
        let embeddings = outputs[0].try_extract::<f32>()?;
        let embedding_slice = embeddings.view();
        
        // The output shape is [1, 75, 512] - we need to perform mean pooling over the sequence dimension
        info!("Raw embedding shape: {:?}", embedding_slice.shape());
        info!("Final embedding dimensions after mean pooling: {}", 
            embedding_slice.slice(s![0, .., ..])
                .mean_axis(Axis(0))
                .map(|arr| arr.len())
                .unwrap_or(0)
        );
        
        // Use ndarray's axis operations for mean pooling
        let final_embedding = embedding_slice
            .slice(s![0, .., ..])
            .mean_axis(Axis(0))
            .unwrap();
        
        // Normalize the final embedding
        let norm = final_embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        
        let normalized = final_embedding
            .iter()
            .map(|x| x / norm)
            .collect();
        
        Ok(normalized)
    }

    fn get_batch_embeddings(&self, images: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
        let batch_size = images.len();
        let mut batch_tensor = Array4::zeros((batch_size, 3, 224, 224));
        
        // Process all images in parallel and collect results
        let processed_tensors: Vec<_> = images.par_iter()
            .filter_map(|img| self.preprocess_image(img).ok())
            .collect();
            
        // Fill batch tensor sequentially
        for (i, processed) in processed_tensors.iter().enumerate() {
            batch_tensor.slice_mut(s![i, .., .., ..]).assign(&processed.slice(s![0, .., .., ..]));
        }
        
        // Convert to CowArray for ONNX Runtime
        let batch_tensor = CowArray::from(batch_tensor.into_dyn());
        
        // Create input for model
        let input = Value::from_array(self.session.allocator(), &batch_tensor)?;
        let outputs = self.session.run(vec![input])?;
        
        // Process results
        let embeddings = outputs[0].try_extract::<f32>()?;
        let embedding_slice = embeddings.view();
        
        // Process each embedding in parallel
        Ok((0..batch_size).into_par_iter()
            .map(|i| {
                let emb = embedding_slice.slice(s![i, .., ..]);
                let final_emb = emb.mean_axis(Axis(0)).unwrap();
                let norm = final_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                final_emb.iter().map(|x| x / norm).collect()
            })
            .collect())
    }

    fn process_batch(&mut self, images: &[&DynamicImage]) -> Result<Vec<Vec<f32>>> {
        let batch_size = images.len();
        let preprocess_start = Instant::now();
        
        // Reuse pre-allocated buffer if possible
        let mut batch_tensor = if batch_size <= OPTIMAL_BATCH_SIZE {
            self.buffer.slice_mut(s![..batch_size, .., .., ..]).to_owned()
        } else {
            Array4::zeros((batch_size, 3, 224, 224))
        };
        
        // Process images in chunks to better utilize CPU cache
        const CHUNK_SIZE: usize = 16;  // Increased chunk size for better parallelism
        let processed_tensors: Vec<_> = images.par_iter()
            .with_min_len(CHUNK_SIZE)  // Ensure minimum chunk size for parallel processing
            .map(|img| self.preprocess_image(img))
            .collect::<Result<Vec<_>>>()?;
        
        let preprocess_time = preprocess_start.elapsed();
        let tensor_copy_start = Instant::now();
            
        // Fill batch tensor with optimized sequential processing
        // Process in larger chunks for better cache locality
        const COPY_CHUNK_SIZE: usize = 32;
        for chunk_idx in 0..(processed_tensors.len() + COPY_CHUNK_SIZE - 1) / COPY_CHUNK_SIZE {
            let start_idx = chunk_idx * COPY_CHUNK_SIZE;
            let end_idx = (start_idx + COPY_CHUNK_SIZE).min(processed_tensors.len());
            
            // Process this chunk sequentially but with good cache locality
            for i in start_idx..end_idx {
                if i < batch_size {
                    batch_tensor.slice_mut(s![i, .., .., ..])
                        .assign(&processed_tensors[i].slice(s![0, .., .., ..]));
                }
            }
        }
        
        let tensor_copy_time = tensor_copy_start.elapsed();
        let inference_start = Instant::now();
        
        // Convert to CowArray and run inference
        let batch_tensor = CowArray::from(batch_tensor.into_dyn());
        let input = Value::from_array(self.session.allocator(), &batch_tensor)?;
        let outputs = self.session.run(vec![input])?;
        
        let inference_time = inference_start.elapsed();
        let postprocess_start = Instant::now();
        
        // Process embeddings in parallel with larger chunks
        let embeddings = outputs[0].try_extract::<f32>()?;
        let embedding_slice = embeddings.view();
        
        let result = (0..batch_size).into_par_iter()
            .chunks(CHUNK_SIZE)
            .flat_map(|chunk| {
                chunk.iter().map(|&i| {
                    let emb = embedding_slice.slice(s![i, .., ..]);
                    let final_emb = emb.mean_axis(Axis(0)).unwrap();
                    let norm = final_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    final_emb.iter().map(|x| x / norm).collect::<Vec<f32>>()
                }).collect::<Vec<_>>()
            })
            .collect();
            
        let postprocess_time = postprocess_start.elapsed();
        
        // Log detailed timing breakdown
        info!(
            "Batch processing breakdown (size: {}):\n\
             - Preprocessing: {:?} ({:.1}%)\n\
             - Tensor copy: {:?} ({:.1}%)\n\
             - Inference: {:?} ({:.1}%)\n\
             - Postprocessing: {:?} ({:.1}%)\n\
             Total: {:?}",
            batch_size,
            preprocess_time,
            100.0 * preprocess_time.as_secs_f64() / (preprocess_time + tensor_copy_time + inference_time + postprocess_time).as_secs_f64(),
            tensor_copy_time,
            100.0 * tensor_copy_time.as_secs_f64() / (preprocess_time + tensor_copy_time + inference_time + postprocess_time).as_secs_f64(),
            inference_time,
            100.0 * inference_time.as_secs_f64() / (preprocess_time + tensor_copy_time + inference_time + postprocess_time).as_secs_f64(),
            postprocess_time,
            100.0 * postprocess_time.as_secs_f64() / (preprocess_time + tensor_copy_time + inference_time + postprocess_time).as_secs_f64(),
            preprocess_time + tensor_copy_time + inference_time + postprocess_time
        );
        
        Ok(result)
    }
}

struct TensorPool {
    buffers: Vec<Array4<f32>>,
}

impl TensorPool {
    fn get_tensor(&mut self) -> Array4<f32> {
        self.buffers.pop().unwrap_or_else(|| Array4::zeros((1, 3, 224, 224)))
    }
    
    fn return_tensor(&mut self, tensor: Array4<f32>) {
        self.buffers.push(tensor);
    }
}

async fn verify_collection_count(client: &QdrantClient, expected_count: Option<u64>) -> Result<u64> {
    let collection_info = client.collection_info(COLLECTION_NAME).await?;
    let points_count = collection_info.result
        .ok_or_else(|| anyhow::anyhow!("Failed to get collection info"))?
        .points_count
        .ok_or_else(|| anyhow::anyhow!("Failed to get points count"))?;
    
    if let Some(expected) = expected_count {
        if points_count != expected {
            warn!("Collection count mismatch - expected: {}, actual: {}", expected, points_count);
        } else {
            info!("Collection count verified: {}", points_count);
        }
    }
    
    Ok(points_count)
}

async fn init_qdrant_collection(client: &QdrantClient) -> Result<()> {
    let collections = client.list_collections().await?;
    
    // Check if collection exists
    let collection_exists = collections.collections.iter().any(|c| c.name == COLLECTION_NAME);
    
    if !collection_exists {
        info!("Creating new collection: {}", COLLECTION_NAME);
        
        // Create collection with explicit vector configuration
        let create_collection = CreateCollection {
            collection_name: COLLECTION_NAME.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(VectorConfigType::Params(VectorParams {
                    size: VECTOR_SIZE,
                    distance: Distance::Cosine as i32,
                    on_disk: Some(false),  // Keep vectors in memory for better performance
                    hnsw_config: None,  // Use default HNSW config
                    quantization_config: None,  // No quantization
                    ..Default::default()
                })),
            }),
            ..Default::default()
        };
        
        client.create_collection(&create_collection).await?;
        
        // Verify collection was created
        let collection_info = client.collection_info(COLLECTION_NAME).await?;
        info!("Collection created successfully with name: {}", COLLECTION_NAME);
        info!("Vector size configured as: {}", VECTOR_SIZE);
    } else {
        info!("Collection {} already exists", COLLECTION_NAME);
        let count = verify_collection_count(client, None).await?;
        info!("Current collection size: {} points", count);
    }
    
    Ok(())
}

async fn verify_qdrant_connection(client: &QdrantClient) -> Result<()> {
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

async fn process_images(client: &QdrantClient, image_dir: &str) -> Result<()> {
    let start_time = Instant::now();
    info!("Starting image processing from directory: {}", image_dir);
    
    // Get initial collection count
    let initial_count = verify_collection_count(client, None).await?;
    info!("Initial collection count: {}", initial_count);
    
    // Initialize CLIP model
    let clip = Arc::new(Mutex::new(ClipModel::new()?));
    let mut point_id = initial_count as usize;
    let mut total_processed = 0;
    let mut total_embedding_time = Duration::new(0, 0);
    let mut total_upsert_time = Duration::new(0, 0);

    // Optimize thread pools
    let io_pool = Arc::new(ThreadPoolBuilder::new()
        .num_threads(16)  // Increase IO threads for better throughput
        .stack_size(2 * 1024 * 1024)  // 2MB stack size for IO threads
        .build()?);

    let processing_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())  // Use all CPU cores for processing
        .stack_size(8 * 1024 * 1024)  // 8MB stack size for processing threads
        .build()?;

    // Collect paths first with parallel iterator
    let image_paths: Vec<_> = WalkDir::new(image_dir)
        .into_iter()
        .par_bridge()  // Use parallel iterator for directory walking
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
        return Ok(());
    }

    // Process images in optimized batches
    let batch_size = OPTIMAL_BATCH_SIZE;
    let chunks: Vec<_> = image_paths.chunks(batch_size).collect();
    let clip_ref = Arc::clone(&clip);
    
    // Create channels for prefetching
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<Vec<Option<(PathBuf, DynamicImage)>>, anyhow::Error>>(2);
    let tx_clone = tx.clone();
    
    // Start prefetching first batch
    if let Some(first_chunk) = chunks.first() {
        let chunk = first_chunk.to_vec();
        let tx = tx_clone.clone();  // Clone here instead of moving
        let io_pool = Arc::clone(&io_pool);
        tokio::spawn(async move {
            let loaded = io_pool.install(|| -> Result<Vec<Option<(PathBuf, DynamicImage)>>> {
                chunk.par_iter()
                    .map(|path| -> Result<Option<(PathBuf, DynamicImage)>> {
                        let file_content = match std::fs::read(path) {
                            Ok(content) => content,
                            Err(e) => {
                                warn!("Failed to read image file {:?}: {}", path, e);
                                return Ok(None);
                            }
                        };
                        
                        match image::load_from_memory(&file_content) {
                            Ok(img) => Ok(Some((path.clone(), img))),
                            Err(e) => {
                                warn!("Failed to decode image {:?}: {}", path, e);
                                Ok(None)
                            }
                        }
                    })
                    .collect()
            });
            let _ = tx.send(loaded).await;
        });
    }
    
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let chunk_start_time = Instant::now();
        
        // Start prefetching next batch
        if chunk_idx + 1 < chunks.len() {
            let next_chunk = chunks[chunk_idx + 1].to_vec();
            let tx = tx_clone.clone();
            let io_pool = Arc::clone(&io_pool);
            tokio::spawn(async move {
                let loaded = io_pool.install(|| -> Result<Vec<Option<(PathBuf, DynamicImage)>>> {
                    next_chunk.par_iter()
                        .map(|path| -> Result<Option<(PathBuf, DynamicImage)>> {
                            let file_content = match std::fs::read(path) {
                                Ok(content) => content,
                                Err(e) => {
                                    warn!("Failed to read image file {:?}: {}", path, e);
                                    return Ok(None);
                                }
                            };
                            
                            match image::load_from_memory(&file_content) {
                                Ok(img) => Ok(Some((path.clone(), img))),
                                Err(e) => {
                                    warn!("Failed to decode image {:?}: {}", path, e);
                                    Ok(None)
                                }
                            }
                        })
                        .collect()
                });
                let _ = tx.send(loaded).await;
            });
        }
        
        // Get pre-loaded images from previous prefetch
        let loaded_images = rx.recv().await.ok_or_else(|| anyhow::anyhow!("Failed to receive prefetched images"))??;
        
        // Process valid images using processing pool
        let valid_images: Vec<(PathBuf, DynamicImage)> = loaded_images.into_iter()
            .flatten()  // Convert Vec<Option<T>> into Vec<T>
            .collect();
        
        if valid_images.is_empty() {
            continue;
        }

        // Generate embeddings in optimized batch
        let embedding_start = Instant::now();
        let embeddings = processing_pool.install(|| {
            let mut clip = clip_ref.lock().unwrap();
            let image_refs: Vec<&DynamicImage> = valid_images.iter().map(|(_, img)| img).collect();
            clip.process_batch(&image_refs)
        })?;

        // Verify embeddings
        if embeddings.is_empty() {
            warn!("No embeddings generated for batch {}", chunk_idx + 1);
            continue;
        }

        // Log embedding statistics
        info!(
            "Generated {} embeddings in batch {}. First embedding size: {}", 
            embeddings.len(),
            chunk_idx + 1,
            embeddings.first().map(|e| e.len()).unwrap_or(0)
        );

        let batch_embedding_time = embedding_start.elapsed();
        total_embedding_time += batch_embedding_time;

        // Create points for batch with verification
        let mut points: Vec<PointStruct> = Vec::with_capacity(valid_images.len());
        for ((path, _), embedding) in valid_images.iter().zip(embeddings.iter()) {
            // Verify embedding vector
            if embedding.len() != VECTOR_SIZE as usize {
                error!(
                    "Invalid embedding size for {}: expected {}, got {}", 
                    path.display(), 
                    VECTOR_SIZE, 
                    embedding.len()
                );
                continue;
            }

            let mut payload = HashMap::<String, QdrantValue>::new();
            let filename = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
                
            payload.insert(
                "filename".to_string(),
                QdrantValue {
                    kind: Some(qdrant_client::qdrant::value::Kind::StringValue(filename.clone()))
                }
            );

            let point = PointStruct {
                id: Some(PointId::from(point_id as u64)),
                payload,
                vectors: Some(Vectors {
                    vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                        qdrant_client::qdrant::Vector {
                            data: embedding.clone(),
                            indices: None,         // No sparse indices
                            vector: None,          // Not using legacy vector field
                            vectors_count: None,   // Not using multiple vectors
                        }
                    ))
                }),
                ..Default::default()
            };

            points.push(point);

            point_id += 1;
            total_processed += 1;
        }

        // Verify points before upserting
        info!(
            "Preparing to upsert {} points for batch {}/{}",
            points.len(),
            chunk_idx + 1,
            chunks.len()
        );

        // Log sample point details
        if let Some(first_point) = points.first() {
            info!(
                "Sample point details - ID: {:?}, Has vector: {}, Vector size: {}, Has payload: {}",
                first_point.id,
                first_point.vectors.is_some(),
                embeddings.first().map(|e| e.len()).unwrap_or(0),
                !first_point.payload.is_empty()
            );
        }

        // Upsert batch with detailed error handling
        let upsert_start_time = Instant::now();
        match client
            .upsert_points_blocking(
                COLLECTION_NAME,
                None as Option<Vec<Key>>,
                points.clone(),
                None as Option<WriteOrdering>
            )
            .await
        {
            Ok(response) => {
                let upsert_time = upsert_start_time.elapsed();
                total_upsert_time += upsert_time;
                
                // Verify the update was successful
                let new_count = verify_collection_count(client, Some(initial_count + total_processed as u64)).await?;
                
                if new_count != initial_count + total_processed as u64 {
                    error!(
                        "Upsert verification failed - Expected: {}, Actual: {}", 
                        initial_count + total_processed as u64,
                        new_count
                    );
                    
                    // Log the first few points for debugging
                    info!("Sample of points that should have been inserted:");
                    for (i, point) in points.iter().take(3).enumerate() {
                        info!(
                            "Point {}: ID: {:?}, Has vector: {}, Vector size: {}, Has payload: {}", 
                            i,
                            point.id,
                            point.vectors.is_some(),
                            embeddings.first().map(|e| e.len()).unwrap_or(0),
                            !point.payload.is_empty()
                        );
                    }
                } else {
                    info!(
                        "Successfully upserted {} points in batch {}/{}", 
                        points.len(),
                        chunk_idx + 1,
                        chunks.len()
                    );
                }
                
                let chunk_time = chunk_start_time.elapsed();
                info!(
                    "Batch {}/{}: Processed {} images in {:?} (load: {:?}, embedding: {:?}, upsert: {:?})", 
                    chunk_idx + 1,
                    chunks.len(),
                    valid_images.len(),
                    chunk_time,
                    chunk_time - batch_embedding_time - upsert_time,
                    batch_embedding_time,
                    upsert_time
                );
            },
            Err(e) => {
                error!("Failed to upsert batch of points: {}", e);
                error!("First point in failed batch - ID: {:?}, Vector size: {}", 
                    points.first()
                        .map(|p| p.id.clone())
                        .unwrap_or_else(|| Some(PointId::default())),
                    points.first().and_then(|p| p.vectors.as_ref()).map(|_| VECTOR_SIZE).unwrap_or(0)
                );
                return Err(e.into());
            }
        }
    }

    let total_time = start_time.elapsed();
    
    // Print benchmark summary
    info!("\n=== BENCHMARK RESULTS ===");
    info!("Total images processed: {}", total_processed);
    info!("Total processing time: {:?}", total_time);
    info!("Throughput: {:.2} images/second", total_processed as f64 / total_time.as_secs_f64());
    info!("\nBreakdown:");
    info!("- Total embedding time: {:?}", total_embedding_time);
    info!("- Total upsert time: {:?}", total_upsert_time);
    info!("- Total IO time: {:?}", total_time - total_embedding_time - total_upsert_time);
    info!("\nAverages:");
    info!("- Average time per image: {:?}", total_time.div_f64(total_processed as f64));
    info!("- Average embedding time per image: {:?}", total_embedding_time.div_f64(total_processed as f64));
    info!("- Average upsert time per batch: {:?}", total_upsert_time.div_f64((total_processed as f64 / batch_size as f64).ceil()));
    info!("\nBatch Statistics:");
    info!("- Batch size: {}", batch_size);
    info!("- Number of full batches: {}", total_processed / batch_size);
    info!("- Remaining images in last batch: {}", total_processed % batch_size);
    info!("======================\n");
    
    // Final verification with detailed logging
    let final_count = verify_collection_count(client, Some(initial_count + total_processed as u64)).await?;
    info!("Final verification - Collection count: {}, Expected: {}, Difference: {}", 
        final_count, 
        initial_count + total_processed as u64,
        (initial_count + total_processed as u64).saturating_sub(final_count)
    );
    
    if final_count < initial_count + total_processed as u64 {
        warn!("Some points may not have been successfully inserted");
    }
    
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn normalize_pixels_simd(pixels: &[u8]) -> Vec<f32> {
    use std::arch::x86_64::*;
    let mut result = Vec::with_capacity(pixels.len());
    unsafe {
        let scale = _mm256_set1_ps(1.0 / 127.5);
        let offset = _mm256_set1_ps(-1.0);
        for chunk in pixels.chunks_exact(8) {
            let pixels = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(chunk.as_ptr() as *const _)));
            let normalized = _mm256_fmadd_ps(pixels, scale, offset);
            _mm256_storeu_ps(result.as_mut_ptr().add(result.len()), normalized);
            result.set_len(result.len() + 8);
        }
    }
    result
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenv().ok();
    
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get Qdrant configuration from environment variables
    let qdrant_url = env::var("QDRANT_URL")
        .context("QDRANT_URL environment variable not set")?;
    let qdrant_api_key = env::var("QDRANT_API_KEY")
        .context("QDRANT_API_KEY environment variable not set")?;

    // Initialize Qdrant client
    let client = QdrantClient::from_url(&qdrant_url)
        .with_api_key(qdrant_api_key)
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
