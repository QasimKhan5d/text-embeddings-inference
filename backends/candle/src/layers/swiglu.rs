use candle::{DType, Result, Tensor};

// CUDA implementation is available only when building with `--features nvrtc-kernels`.
//
// Motivation: `candle_nn::ops::swiglu` typically becomes multiple CUDA kernels (`usilu_*` + `bmul_*`)
// plus an intermediate tensor. For short sequences/batch=1 this launch overhead is significant.

#[cfg(all(feature = "cuda", feature = "nvrtc-kernels"))]
mod cuda_impl {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::LaunchConfig;
    use candle::cuda_backend::cudarc::nvrtc;
    use candle::cuda_backend::WrapErr;
    use candle::{CudaStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};
    use half::{bf16, f16};
    use lazy_static::lazy_static;
    use std::env;

    const SWIGLU_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// `silu_fwd(x) = x / (1 + exp(-x))`.

__device__ __forceinline__ __half silu_h(__half x) {
    float xf = __half2float(x);
    float yf = xf / (1.0f + expf(-xf));
    return __float2half_rn(yf);
}

__device__ __forceinline__ __nv_bfloat16 silu_bf(__nv_bfloat16 x) {
    float xf = __bfloat162float(x);
    float yf = xf / (1.0f + expf(-xf));
    return __float2bfloat16_rn(yf);
}

extern "C" __global__ void swiglu_f16(__half* out, const __half* in, int n, int inner) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int row = i / inner;
    int col = i - row * inner;
    int base = row * (inner * 2) + col;
    __half gate = in[base];
    __half up   = in[base + inner];
    __half y = silu_h(gate);
    out[i] = __float2half_rn(__half2float(y * up));
}

extern "C" __global__ void swiglu_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int n, int inner) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int row = i / inner;
    int col = i - row * inner;
    int base = row * (inner * 2) + col;
    __nv_bfloat16 gate = in[base];
    __nv_bfloat16 up   = in[base + inner];
    __nv_bfloat16 y = silu_bf(gate);
    out[i] = __float2bfloat16_rn(__bfloat162float(y * up));
}
"#;

    fn cuda_include_paths() -> Vec<String> {
        // Prefer explicit env vars, otherwise fall back to CUDA 12.4 (known-good in this environment),
        // otherwise fall back to the default CUDA symlink.
        let mut paths = Vec::new();
        if let Ok(p) = env::var("CUDA_INCLUDE_PATH") {
            paths.push(p);
        }
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            paths.push(format!("{cuda_path}/include"));
        }
        paths.push("/usr/local/cuda-12.4/include".to_string());
        paths.push("/usr/local/cuda/include".to_string());
        paths
    }

    fn compile_ptx() -> Result<nvrtc::safe::Ptx> {
        let opts = nvrtc::CompileOptions {
            use_fast_math: Some(true),
            // NVRTC does not automatically know where CUDA headers live.
            include_paths: cuda_include_paths(),
            ..Default::default()
        };
        nvrtc::safe::compile_ptx_with_opts(SWIGLU_CUDA_SRC, opts).w()
    }

    lazy_static! {
        static ref SWIGLU_PTX: nvrtc::safe::Ptx =
            compile_ptx().expect("failed to NVRTC-compile fused swiglu kernel");
    }

    #[derive(Debug, Clone)]
    pub struct SwigluFused;

    impl CustomOp1 for SwigluFused {
        fn name(&self) -> &'static str {
            "tei-swiglu-fused"
        }

        fn cpu_fwd(
            &self,
            _storage: &candle::CpuStorage,
            _layout: &Layout,
        ) -> Result<(candle::CpuStorage, Shape)> {
            candle::bail!("tei-swiglu-fused is only implemented for CUDA")
        }

        fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
            use candle::cuda_backend::cudarc::driver::LaunchAsync;

            // We require contiguous input for now (matches typical GEMM outputs).
            let (o1, o2) = layout
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::msg("swiglu_fused expects contiguous input"))?;

            let mut out_shape = layout.shape().dims().to_vec();
            let last = *out_shape
                .last()
                .ok_or_else(|| candle::Error::msg("swiglu_fused expects non-scalar input"))?;
            if last % 2 != 0 {
                candle::bail!("swiglu_fused expects last dim to be even, got {last}")
            }
            let inner = last / 2;
            *out_shape.last_mut().unwrap() = inner;
            let out_shape: Shape = out_shape.into();

            let rows = layout.shape().elem_count() / last;
            let n = rows * inner;

            let dev = storage.device().clone();
            let cfg = LaunchConfig::for_num_elems(n as u32);

            // Load the PTX once per process/device.
            if dev.get_func("tei_swiglu", "swiglu_f16").is_none()
                || dev.get_func("tei_swiglu", "swiglu_bf16").is_none()
            {
                dev.load_ptx(SWIGLU_PTX.clone(), "tei_swiglu", &["swiglu_f16", "swiglu_bf16"])
                    .w()?;
            }

            match storage.dtype() {
                DType::F16 => {
                    let inp = storage.as_cuda_slice::<f16>()?;
                    let inp = inp.slice(o1..o2);
                    // SAFETY: output written by kernel.
                    let out = unsafe { dev.alloc::<f16>(n) }.w()?;
                    let func = dev
                        .get_func("tei_swiglu", "swiglu_f16")
                        .ok_or_else(|| candle::Error::msg("missing tei_swiglu::swiglu_f16"))?;
                    // SAFETY: ffi.
                    unsafe { func.launch(cfg, (&out, &inp, n as i32, inner as i32)) }.w()?;
                    Ok((CudaStorage::wrap_cuda_slice(out, dev), out_shape))
                }
                DType::BF16 => {
                    let inp = storage.as_cuda_slice::<bf16>()?;
                    let inp = inp.slice(o1..o2);
                    // SAFETY: output written by kernel.
                    let out = unsafe { dev.alloc::<bf16>(n) }.w()?;
                    let func = dev
                        .get_func("tei_swiglu", "swiglu_bf16")
                        .ok_or_else(|| candle::Error::msg("missing tei_swiglu::swiglu_bf16"))?;
                    // SAFETY: ffi.
                    unsafe { func.launch(cfg, (&out, &inp, n as i32, inner as i32)) }.w()?;
                    Ok((CudaStorage::wrap_cuda_slice(out, dev), out_shape))
                }
                dt => candle::bail!("swiglu_fused only supports f16/bf16 (got {dt:?})"),
            }
        }
    }

    pub fn swiglu(x: &Tensor) -> Result<Tensor> {
        // Preserve dtype; output last dim is halved.
        x.apply_op1_no_bwd(&SwigluFused)
    }
}

/// Swiglu activation: `silu(gate) * up` on the last dimension split in half.
///
/// - On CUDA with `nvrtc-kernels`, this uses a fused CUDA kernel (one launch, no intermediate).
/// - Otherwise it falls back to `candle_nn::ops::swiglu`.
pub fn swiglu(x: &Tensor) -> Result<Tensor> {
    match (x.device(), x.dtype()) {
        #[cfg(all(feature = "cuda", feature = "nvrtc-kernels"))]
        (candle::Device::Cuda(_), DType::F16 | DType::BF16) => cuda_impl::swiglu(x),
        _ => candle_nn::ops::swiglu(x),
    }
}


