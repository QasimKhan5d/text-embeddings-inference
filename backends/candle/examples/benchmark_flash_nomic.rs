use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use std::path::{Path, PathBuf};
use std::time::Instant;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Embedding, Embeddings, ModelType, Pool};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "benchmark_flash_nomic")]
#[command(about = "Benchmark FlashNomic model with correctness checking")]
struct Args {
    /// Use an existing model directory instead of downloading from HuggingFace Hub
    #[arg(long)]
    model_root: Option<PathBuf>,

    /// Save reference embeddings to file
    #[arg(long)]
    save_reference: Option<PathBuf>,

    /// Compare embeddings against reference file
    #[arg(long)]
    compare_reference: Option<PathBuf>,

    /// Batch size for benchmark
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// Sequence length for benchmark
    #[arg(long, default_value = "128")]
    seq_length: usize,

    /// Enable profiling output
    #[arg(long)]
    profile: bool,

    /// Number of warmup iterations
    #[arg(long, default_value = "3")]
    warmup: usize,

    /// Number of benchmark iterations
    #[arg(long, default_value = "50")]
    iterations: usize,

    /// Model ID
    #[arg(long, default_value = "nomic-ai/nomic-embed-text-v1.5")]
    model_id: String,

    /// Model precision
    #[arg(long, default_value = "float16")]
    dtype: String,

    /// Load real inputs from JSON file instead of synthetic
    #[arg(long)]
    real_inputs: Option<PathBuf>,
}

fn create_synthetic_batch(seq_length: usize, batch_size: usize) -> text_embeddings_backend_core::Batch {
    let mut input_ids = Vec::new();
    let mut token_type_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative_seq_lengths: Vec<u32> = vec![0];

    for b in 0..batch_size {
        for i in 0..seq_length {
            input_ids.push(((i % 30000) + 100) as u32);
            token_type_ids.push(0);
            position_ids.push(i as u32);
        }
        cumulative_seq_lengths.push(((b + 1) * seq_length) as u32);
    }

    let pooled_indices: Vec<u32> = (0..batch_size as u32).collect();

    text_embeddings_backend_core::Batch {
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length: seq_length as u32,
        pooled_indices,
        raw_indices: vec![],
    }
}

fn sort_embeddings(embeddings: Embeddings) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut pooled_embeddings = Vec::new();
    let mut raw_embeddings = Vec::new();

    for (_, embedding) in embeddings {
        match embedding {
            Embedding::Pooled(e) => pooled_embeddings.push(e),
            Embedding::All(e) => raw_embeddings.extend(e),
        }
    }

    (pooled_embeddings, raw_embeddings)
}

fn download_model(model_id: &str) -> Result<PathBuf> {
    let mut builder = ApiBuilder::from_env().with_progress(false);

    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }

    if let Some(cache_dir) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        builder = builder.with_cache_dir(cache_dir.into());
    }

    let api = builder.build()?;
    let api_repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Download required files
    let _config = api_repo.get("config.json")?;
    let _tokenizer = api_repo.get("tokenizer.json")?;

    // Try to download model files (either safetensors or pytorch)
    let model_files = api_repo.info()?;
    
    for sibling in model_files.siblings {
        if sibling.rfilename.ends_with(".safetensors") || sibling.rfilename == "pytorch_model.bin" {
            let _ = api_repo.get(&sibling.rfilename)?;
        }
    }

    // Return model root directory
    let model_root = api_repo.get("config.json")?;
    Ok(model_root
        .parent()
        .context("Could not get parent directory")?
        .to_path_buf())
}

fn save_embeddings(embeddings: &[Vec<f32>], path: &Path) -> Result<()> {
    let json = serde_json::to_string(embeddings)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn load_embeddings(path: &Path) -> Result<Vec<Vec<f32>>> {
    let json = std::fs::read_to_string(path)?;
    let embeddings: Vec<Vec<f32>> = serde_json::from_str(&json)?;
    Ok(embeddings)
}

fn embeddings_match(a: &[Vec<f32>], b: &[Vec<f32>], rtol: f32, atol: f32) -> bool {
    if a.len() != b.len() {
        eprintln!("Length mismatch: {} vs {}", a.len(), b.len());
        return false;
    }

    for (i, (emb_a, emb_b)) in a.iter().zip(b.iter()).enumerate() {
        if emb_a.len() != emb_b.len() {
            eprintln!("Embedding {} length mismatch: {} vs {}", i, emb_a.len(), emb_b.len());
            return false;
        }

        for (j, (&val_a, &val_b)) in emb_a.iter().zip(emb_b.iter()).enumerate() {
            let diff = (val_a - val_b).abs();
            let threshold = atol + rtol * val_b.abs();
            if diff > threshold {
                eprintln!(
                    "Mismatch at embedding {} position {}: {} vs {} (diff: {}, threshold: {})",
                    i, j, val_a, val_b, diff, threshold
                );
                return false;
            }
        }
    }

    true
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Resolve model path
    let model_root = if let Some(model_root) = args.model_root.clone() {
        eprintln!("Using model from --model-root: {}", model_root.display());
        model_root
    } else {
        eprintln!("Downloading model '{}' (set --model-root to skip hub download)...", args.model_id);
        let model_root = download_model(&args.model_id)?;
        eprintln!("Model ready at: {}", model_root.display());
        model_root
    };

    // Load real inputs if specified
    let real_batches: Option<Vec<text_embeddings_backend_core::Batch>> = if let Some(ref json_path) = args.real_inputs {
        let tokenizer = Tokenizer::from_file(model_root.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
        let json_content = std::fs::read_to_string(json_path)?;
        let texts: Vec<String> = serde_json::from_str(&json_content)?;
        eprintln!("Loaded {} real texts from {}", texts.len(), json_path.display());
        
        let batches: Result<Vec<_>> = texts.iter().map(|text| {
            let encoding = tokenizer.encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Failed to encode text: {:?}", e))?;
            let mut input_ids: Vec<u32> = encoding.get_ids().iter().map(|&x| x).collect();
            input_ids.truncate(args.seq_length);
            let actual_len = input_ids.len();
            while input_ids.len() < args.seq_length {
                input_ids.push(0);
            }
            
            Ok(text_embeddings_backend_core::Batch {
                input_ids,
                token_type_ids: vec![0; args.seq_length],
                position_ids: (0..args.seq_length as u32).collect(),
                cumulative_seq_lengths: vec![0, actual_len as u32],
                max_length: args.seq_length as u32,
                pooled_indices: vec![0],
                raw_indices: vec![],
            })
        }).collect();
        Some(batches?)
    } else {
        None
    };

    // Initialize backend
    let backend = CandleBackend::new(
        &model_root,
        args.dtype,
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    // Warmup
    let warmup_iters = args.warmup;
    for i in 0..warmup_iters {
        let warmup_batch = if let Some(ref batches) = real_batches {
            let batch = &batches[i % batches.len()];
            text_embeddings_backend_core::Batch {
                input_ids: batch.input_ids.clone(),
                token_type_ids: batch.token_type_ids.clone(),
                position_ids: batch.position_ids.clone(),
                cumulative_seq_lengths: batch.cumulative_seq_lengths.clone(),
                max_length: batch.max_length,
                pooled_indices: batch.pooled_indices.clone(),
                raw_indices: batch.raw_indices.clone(),
            }
        } else {
            create_synthetic_batch(args.seq_length, args.batch_size)
        };
        let _ = backend.embed(warmup_batch)?;
    }

    // Benchmark
    let iterations = args.iterations;
    let mut latencies: Vec<f64> = Vec::with_capacity(iterations);

    for i in 0..iterations {

        let bench_batch = if let Some(ref batches) = real_batches {
            let batch = &batches[i % batches.len()];
            text_embeddings_backend_core::Batch {
                input_ids: batch.input_ids.clone(),
                token_type_ids: batch.token_type_ids.clone(),
                position_ids: batch.position_ids.clone(),
                cumulative_seq_lengths: batch.cumulative_seq_lengths.clone(),
                max_length: batch.max_length,
                pooled_indices: batch.pooled_indices.clone(),
                raw_indices: batch.raw_indices.clone(),
            }
        } else {
            create_synthetic_batch(args.seq_length, args.batch_size)
        };
        let start = Instant::now();
        let result = backend.embed(bench_batch)?;
        let elapsed = start.elapsed();

        latencies.push(elapsed.as_secs_f64() * 1000.0);

        // Save embeddings on first iteration if needed
        if i == 0 {
            let (pooled, _) = sort_embeddings(result);
            
            if let Some(ref save_path) = args.save_reference {
                save_embeddings(&pooled, save_path)?;
                println!("Saved reference embeddings to {}", save_path.display());
            }

            if let Some(ref compare_path) = args.compare_reference {
                let reference = load_embeddings(compare_path)
                    .context(format!("Failed to load reference from {}", compare_path.display()))?;
                
                // Use tolerances suitable for FP16 models (rtol=1e-3, atol=1e-4)
                if embeddings_match(&pooled, &reference, 1e-3, 1e-4) {
                    println!("\n✓ CORRECTNESS CHECK PASSED");
                } else {
                    eprintln!("\n✗ CORRECTNESS CHECK FAILED");
                    std::process::exit(1);
                }
            }
        }
    }

    // Calculate statistics (skip first 5 for stability)
    let skip = 5usize.min(latencies.len().saturating_sub(1));
    let stable: Vec<f64> = latencies.iter().skip(skip).copied().collect();
    let mean = if stable.is_empty() {
        // e.g., iterations=1
        latencies.iter().copied().next().unwrap_or(f64::NAN)
    } else {
        stable.iter().sum::<f64>() / stable.len() as f64
    };

    // Output in format expected by executor
    println!("Warmup: {} iterations", warmup_iters);
    println!("Latency: {:.2} ms", mean);

    // If profiling, output profile data
    if args.profile {
        println!("\n[PROFILE_START]");
        // TODO: Add actual profiling data extraction if needed
        println!("[PROFILE_END]");
    }

    Ok(())
}

