# Nomic Embedding Optimizations

## Building

```bash
# Clone and build the optimized router
git clone https://github.com/QasimKhan5d/text-embeddings-inference
cd text-embeddings-inference
# Build with optimizations enabled
cargo build --release --features candle-cuda-nvrtc -p text-embeddings-router
```

## Running

```bash
# Start the optimized router
./target/release/text-embeddings-router \
  --model-id nomic-ai/nomic-embed-text-v1.5 \
  --dtype float16 \
  --port 8080
```

## Testing

```bash
# Send a test request
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello world"}'
```

## Requirements

- CUDA 12.4+ (NVRTC compilation)
- Compute Capability 8.0+ (Ampere or newer recommended)
- Rust 1.70+

## Validation

Run the inference-only benchmark to validate performance:

```bash
cargo build --release --features cuda,flash-attn,nvrtc-kernels -p text-embeddings-backend-candle --examples

./target/release/examples/benchmark_flash_nomic \
  --model-id nomic-ai/nomic-embed-text-v1.5 \
  --dtype float16 \
  --real-inputs nomic-evolved/data/benchmark_inputs.json \
  --warmup 10 \
  --iterations 100
```

Expected output: **~1.6ms mean latency**
