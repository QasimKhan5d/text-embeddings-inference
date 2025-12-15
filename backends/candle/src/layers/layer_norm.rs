use candle::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

#[cfg(feature = "cuda")]
use std::sync::OnceLock;

#[cfg(all(feature = "cuda", feature = "nvrtc-kernels"))]
mod tei_layer_norm {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
    use candle::cuda_backend::cudarc::nvrtc;
    use candle::cuda_backend::WrapErr;
    use candle::{CudaStorage, CustomOp1, DType, Layout, Result, Shape, Storage, Tensor};
    use half::{bf16, f16};
    use lazy_static::lazy_static;

    const SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Block computes 1 row (cols elements).
// Assumes x/res/gamma/beta are contiguous and row-major (rows x cols).

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <typename T, bool HasRes, bool HasBeta>
__global__ void tei_ln(T* out, const T* x, const T* res, const T* gamma, const T* beta, float eps, int rows, int cols) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    // Welford reduction for numerical stability (closer to typical LN implementations).
    extern __shared__ float sh[];
    float* sh_mean = sh;
    float* sh_m2   = sh + blockDim.x;
    int*   sh_n    = (int*)(sh + 2 * blockDim.x);  // placed after floats

    float mean = 0.f;
    float m2 = 0.f;
    int n = 0;
    int base = row * cols;
    for (int c = threadIdx.x; c < cols; c += (int)blockDim.x) {
        float v = to_f32<T>(x[base + c]);
        if (HasRes) v += to_f32<T>(res[base + c]);
        n += 1;
        float delta = v - mean;
        mean += delta / (float)n;
        float delta2 = v - mean;
        m2 += delta * delta2;
    }
    sh_mean[threadIdx.x] = mean;
    sh_m2[threadIdx.x] = m2;
    sh_n[threadIdx.x] = n;
    __syncthreads();

    // Reduce within block by combining Welford states.
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            float mean_a = sh_mean[threadIdx.x];
            float m2_a   = sh_m2[threadIdx.x];
            int n_a      = sh_n[threadIdx.x];
            float mean_b = sh_mean[threadIdx.x + stride];
            float m2_b   = sh_m2[threadIdx.x + stride];
            int n_b      = sh_n[threadIdx.x + stride];
            if (n_b > 0) {
                float delta = mean_b - mean_a;
                int n_ab = n_a + n_b;
                float mean_ab = mean_a + delta * ((float)n_b / (float)n_ab);
                float m2_ab = m2_a + m2_b + delta * delta * ((float)n_a * (float)n_b / (float)n_ab);
                sh_mean[threadIdx.x] = mean_ab;
                sh_m2[threadIdx.x] = m2_ab;
                sh_n[threadIdx.x] = n_ab;
            }
        }
        __syncthreads();
    }

    float mu = sh_mean[0];
    float var = sh_m2[0] / (float)cols;
    float inv_std = rsqrtf(var + eps);

    for (int c = threadIdx.x; c < cols; c += (int)blockDim.x) {
        float v = to_f32<T>(x[base + c]);
        if (HasRes) v += to_f32<T>(res[base + c]);
        float nrm = (v - mu) * inv_std;
        float y = nrm * to_f32<T>(gamma[c]);
        if (HasBeta) y += to_f32<T>(beta[c]);
        out[base + c] = from_f32<T>(y);
    }
}

extern "C" __global__ void tei_ln_f16(__half* out, const __half* x, const __half* gamma, const __half* beta, float eps, int rows, int cols) {
    tei_ln<__half, false, true>(out, x, nullptr, gamma, beta, eps, rows, cols);
}
extern "C" __global__ void tei_ln_f16_nobeta(__half* out, const __half* x, const __half* gamma, float eps, int rows, int cols) {
    tei_ln<__half, false, false>(out, x, nullptr, gamma, nullptr, eps, rows, cols);
}
extern "C" __global__ void tei_add_ln_f16(__half* out, const __half* x, const __half* res, const __half* gamma, const __half* beta, float eps, int rows, int cols) {
    tei_ln<__half, true, true>(out, x, res, gamma, beta, eps, rows, cols);
}
extern "C" __global__ void tei_add_ln_f16_nobeta(__half* out, const __half* x, const __half* res, const __half* gamma, float eps, int rows, int cols) {
    tei_ln<__half, true, false>(out, x, res, gamma, nullptr, eps, rows, cols);
}

extern "C" __global__ void tei_ln_bf16(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* gamma, const __nv_bfloat16* beta, float eps, int rows, int cols) {
    tei_ln<__nv_bfloat16, false, true>(out, x, nullptr, gamma, beta, eps, rows, cols);
}
extern "C" __global__ void tei_ln_bf16_nobeta(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* gamma, float eps, int rows, int cols) {
    tei_ln<__nv_bfloat16, false, false>(out, x, nullptr, gamma, nullptr, eps, rows, cols);
}
extern "C" __global__ void tei_add_ln_bf16(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* res, const __nv_bfloat16* gamma, const __nv_bfloat16* beta, float eps, int rows, int cols) {
    tei_ln<__nv_bfloat16, true, true>(out, x, res, gamma, beta, eps, rows, cols);
}
extern "C" __global__ void tei_add_ln_bf16_nobeta(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* res, const __nv_bfloat16* gamma, float eps, int rows, int cols) {
    tei_ln<__nv_bfloat16, true, false>(out, x, res, gamma, nullptr, eps, rows, cols);
}
"#;

    fn compile_ptx() -> Result<nvrtc::safe::Ptx> {
        let opts = nvrtc::CompileOptions {
            use_fast_math: Some(true),
            include_paths: vec!["/usr/local/cuda/include".to_string()],
            ..Default::default()
        };
        nvrtc::safe::compile_ptx_with_opts(SRC, opts).w()
    }

    lazy_static! {
        static ref PTX: nvrtc::safe::Ptx =
            compile_ptx().expect("failed to NVRTC-compile tei layernorm kernel");
    }

    #[derive(Debug, Clone)]
    pub struct LnFused {
        gamma: Tensor,
        beta: Option<Tensor>,
        eps: f32,
        residual: Option<Tensor>,
    }

    impl LnFused {
        pub fn new(gamma: Tensor, beta: Option<Tensor>, eps: f32, residual: Option<Tensor>) -> Self {
            Self { gamma, beta, eps, residual }
        }
    }

    impl CustomOp1 for LnFused {
        fn name(&self) -> &'static str {
            "tei-layer-norm"
        }

        fn cpu_fwd(&self, _s: &candle::CpuStorage, _l: &Layout) -> Result<(candle::CpuStorage, Shape)> {
            candle::bail!("tei-layer-norm is only implemented for CUDA")
        }

        fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
            let (o1, o2) = layout
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous input"))?;

            let shape = layout.shape();
            let (rows, cols) = shape.dims2()?;
            let rows = rows as i32;
            let cols = cols as i32;

            // gamma/beta must be 1D contiguous of size cols.
            if self.gamma.rank() != 1 || self.gamma.dims1()? as i32 != cols {
                candle::bail!("tei layernorm expects gamma shape ({cols}), got {:?}", self.gamma.shape())
            }
            if let Some(beta) = &self.beta {
                if beta.rank() != 1 || beta.dims1()? as i32 != cols {
                    candle::bail!("tei layernorm expects beta shape ({cols}), got {:?}", beta.shape())
                }
            }

            let dev = storage.device().clone();
            if dev.get_func("tei_layer_norm", "tei_ln_f16").is_none() {
                dev.load_ptx(
                    PTX.clone(),
                    "tei_layer_norm",
                    &[
                        "tei_ln_f16",
                        "tei_ln_f16_nobeta",
                        "tei_add_ln_f16",
                        "tei_add_ln_f16_nobeta",
                        "tei_ln_bf16",
                        "tei_ln_bf16_nobeta",
                        "tei_add_ln_bf16",
                        "tei_add_ln_bf16_nobeta",
                    ],
                )
                .w()?;
            }

            // Pick a block size that works well for cols=768, and use shared mem for reductions.
            let block = 256u32;
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (block, 1, 1),
                // shared: mean[block] + m2[block] (floats) + n[block] (ints)
                shared_mem_bytes: (3 * block * 4) as u32,
            };

            match storage.dtype() {
                DType::F16 => {
                    let x = storage.as_cuda_slice::<f16>()?.slice(o1..o2);
                    let (gs, gl) = self.gamma.storage_and_layout();
                    let (go1, go2) = gl
                        .contiguous_offsets()
                        .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous gamma"))?;
                    let gamma = match &*gs {
                        Storage::Cuda(s) => s.as_cuda_slice::<f16>()?.slice(go1..go2),
                        _ => candle::bail!("tei layernorm expects gamma on cuda"),
                    };
                    let out = unsafe { dev.alloc::<f16>((rows as usize) * (cols as usize)) }.w()?;

                    let has_res = self.residual.is_some();
                    let has_beta = self.beta.is_some();

                    if has_res {
                        let res_t = self.residual.as_ref().unwrap();
                        let (rs, rl) = res_t.storage_and_layout();
                        let (ro1, ro2) = rl
                            .contiguous_offsets()
                            .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous residual"))?;
                        let res = match &*rs {
                            Storage::Cuda(s) => s.as_cuda_slice::<f16>()?.slice(ro1..ro2),
                            _ => candle::bail!("tei layernorm expects residual on cuda"),
                        };
                        if has_beta {
                            let beta_t = self.beta.as_ref().unwrap();
                            let (bs, bl) = beta_t.storage_and_layout();
                            let (bo1, bo2) = bl
                                .contiguous_offsets()
                                .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous beta"))?;
                            let beta = match &*bs {
                                Storage::Cuda(s) => s.as_cuda_slice::<f16>()?.slice(bo1..bo2),
                                _ => candle::bail!("tei layernorm expects beta on cuda"),
                            };
                            let f = dev.get_func("tei_layer_norm", "tei_add_ln_f16").ok_or_else(|| candle::Error::msg("missing tei_add_ln_f16"))?;
                            unsafe { f.launch(cfg, (&out, &x, &res, &gamma, &beta, self.eps, rows, cols)) }.w()?;
                        } else {
                            let f = dev.get_func("tei_layer_norm", "tei_add_ln_f16_nobeta").ok_or_else(|| candle::Error::msg("missing tei_add_ln_f16_nobeta"))?;
                            unsafe { f.launch(cfg, (&out, &x, &res, &gamma, self.eps, rows, cols)) }.w()?;
                        }
                    } else if has_beta {
                        let beta_t = self.beta.as_ref().unwrap();
                        let (bs, bl) = beta_t.storage_and_layout();
                        let (bo1, bo2) = bl
                            .contiguous_offsets()
                            .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous beta"))?;
                        let beta = match &*bs {
                            Storage::Cuda(s) => s.as_cuda_slice::<f16>()?.slice(bo1..bo2),
                            _ => candle::bail!("tei layernorm expects beta on cuda"),
                        };
                        let f = dev.get_func("tei_layer_norm", "tei_ln_f16").ok_or_else(|| candle::Error::msg("missing tei_ln_f16"))?;
                        unsafe { f.launch(cfg, (&out, &x, &gamma, &beta, self.eps, rows, cols)) }.w()?;
                    } else {
                        let f = dev.get_func("tei_layer_norm", "tei_ln_f16_nobeta").ok_or_else(|| candle::Error::msg("missing tei_ln_f16_nobeta"))?;
                        unsafe { f.launch(cfg, (&out, &x, &gamma, self.eps, rows, cols)) }.w()?;
                    }

                    Ok((CudaStorage::wrap_cuda_slice(out, dev), shape.clone()))
                }
                DType::BF16 => {
                    let x = storage.as_cuda_slice::<bf16>()?.slice(o1..o2);
                    let (gs, gl) = self.gamma.storage_and_layout();
                    let (go1, go2) = gl
                        .contiguous_offsets()
                        .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous gamma"))?;
                    let gamma = match &*gs {
                        Storage::Cuda(s) => s.as_cuda_slice::<bf16>()?.slice(go1..go2),
                        _ => candle::bail!("tei layernorm expects gamma on cuda"),
                    };
                    let out = unsafe { dev.alloc::<bf16>((rows as usize) * (cols as usize)) }.w()?;

                    let has_res = self.residual.is_some();
                    let has_beta = self.beta.is_some();

                    if has_res {
                        let res_t = self.residual.as_ref().unwrap();
                        let (rs, rl) = res_t.storage_and_layout();
                        let (ro1, ro2) = rl
                            .contiguous_offsets()
                            .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous residual"))?;
                        let res = match &*rs {
                            Storage::Cuda(s) => s.as_cuda_slice::<bf16>()?.slice(ro1..ro2),
                            _ => candle::bail!("tei layernorm expects residual on cuda"),
                        };
                        if has_beta {
                            let beta_t = self.beta.as_ref().unwrap();
                            let (bs, bl) = beta_t.storage_and_layout();
                            let (bo1, bo2) = bl
                                .contiguous_offsets()
                                .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous beta"))?;
                            let beta = match &*bs {
                                Storage::Cuda(s) => s.as_cuda_slice::<bf16>()?.slice(bo1..bo2),
                                _ => candle::bail!("tei layernorm expects beta on cuda"),
                            };
                            let f = dev.get_func("tei_layer_norm", "tei_add_ln_bf16").ok_or_else(|| candle::Error::msg("missing tei_add_ln_bf16"))?;
                            unsafe { f.launch(cfg, (&out, &x, &res, &gamma, &beta, self.eps, rows, cols)) }.w()?;
                        } else {
                            let f = dev.get_func("tei_layer_norm", "tei_add_ln_bf16_nobeta").ok_or_else(|| candle::Error::msg("missing tei_add_ln_bf16_nobeta"))?;
                            unsafe { f.launch(cfg, (&out, &x, &res, &gamma, self.eps, rows, cols)) }.w()?;
                        }
                    } else if has_beta {
                        let beta_t = self.beta.as_ref().unwrap();
                        let (bs, bl) = beta_t.storage_and_layout();
                        let (bo1, bo2) = bl
                            .contiguous_offsets()
                            .ok_or_else(|| candle::Error::msg("tei layernorm expects contiguous beta"))?;
                        let beta = match &*bs {
                            Storage::Cuda(s) => s.as_cuda_slice::<bf16>()?.slice(bo1..bo2),
                            _ => candle::bail!("tei layernorm expects beta on cuda"),
                        };
                        let f = dev.get_func("tei_layer_norm", "tei_ln_bf16").ok_or_else(|| candle::Error::msg("missing tei_ln_bf16"))?;
                        unsafe { f.launch(cfg, (&out, &x, &gamma, &beta, self.eps, rows, cols)) }.w()?;
                    } else {
                        let f = dev.get_func("tei_layer_norm", "tei_ln_bf16_nobeta").ok_or_else(|| candle::Error::msg("missing tei_ln_bf16_nobeta"))?;
                        unsafe { f.launch(cfg, (&out, &x, &gamma, self.eps, rows, cols)) }.w()?;
                    }

                    Ok((CudaStorage::wrap_cuda_slice(out, dev), shape.clone()))
                }
                dt => candle::bail!("tei layernorm only supports f16/bf16 (got {dt:?})"),
            }
        }
    }

    pub fn layer_norm(hidden_states: &Tensor, residual: Option<&Tensor>, gamma: &Tensor, beta: Option<&Tensor>, eps: f32) -> Result<Tensor> {
        let op = LnFused::new(gamma.clone(), beta.cloned(), eps, residual.cloned());
        hidden_states.apply_op1_no_bwd(&op)
    }
}

#[cfg(feature = "cuda")]
fn tei_nvrtc_layer_norm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let Ok(v) = std::env::var("TEI_NVRTC_LAYERNORM") else {
            return false;
        };
        matches!(
            v.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}

/// CUDA LayerNorm implementation selection.
///
/// Default is **candle-layer-norm** (stable vs reference). The NVRTC `tei_layer_norm` path can be
/// enabled explicitly via `TEI_NVRTC_LAYERNORM=1` for experimentation.
#[cfg(feature = "cuda")]
fn layer_norm_cuda(
    hidden_states: &Tensor,
    residual: Option<&Tensor>,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    #[cfg(feature = "nvrtc-kernels")]
    {
        if tei_nvrtc_layer_norm_enabled() {
            return tei_layer_norm::layer_norm(hidden_states, residual, gamma, beta, eps);
        }
    }

    if let Some(residual) = residual {
        let (result, _) = candle_layer_norm::fused_add_layer_norm(
            hidden_states,
            residual,
            gamma,
            beta,
            eps,
        )?;
        Ok(result)
    } else {
        candle_layer_norm::layer_norm(hidden_states, gamma, beta, eps)
    }
}

#[derive(Debug)]
pub struct LayerNormNoBias {
    weight: Tensor,
    epsilon: f32,
    span: tracing::Span,
}

impl LayerNormNoBias {
    pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(hidden_size, "weight")
                .or_else(|_| vb.get(hidden_size, "gamma"))?,
            epsilon,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm-no-bias"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();

        match hidden_states.device() {
            Device::Cpu | Device::Metal(_) => {
                let mut hidden_states = hidden_states.clone();
                if let Some(residual) = residual {
                    hidden_states = hidden_states.add(residual)?;
                }
                let hidden_states_dtype = hidden_states.dtype();
                let internal_dtype = match hidden_states_dtype {
                    DType::F16 | DType::BF16 => DType::F32,
                    d => d,
                };
                let hidden_size = hidden_states.dim(D::Minus1)?;
                let hidden_states = hidden_states.to_dtype(internal_dtype)?;
                let mean_hidden_states =
                    (hidden_states.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states = hidden_states.broadcast_sub(&mean_hidden_states)?;
                let norm_hidden_states =
                    (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states_normed = hidden_states
                    .broadcast_div(&(norm_hidden_states + self.epsilon as f64)?.sqrt()?)?;
                let hidden_states = hidden_states_normed
                    .to_dtype(hidden_states_dtype)?
                    .broadcast_mul(&self.weight)?;

                Ok(hidden_states)
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    let original_shape = hidden_states.shape();
                    let hidden_states = hidden_states.flatten_to(D::Minus2)?;
                    let residual = match residual {
                        Some(residual) => Some(residual.flatten_to(D::Minus2)?),
                        None => None,
                    };

                    let result = match residual.as_ref() {
                        Some(residual) => layer_norm_cuda(
                            &hidden_states,
                            Some(residual),
                            &self.weight,
                            None,
                            self.epsilon,
                        ),
                        None => layer_norm_cuda(
                            &hidden_states,
                            None,
                            &self.weight,
                            None,
                            self.epsilon,
                        ),
                    }?;
                    result.reshape(original_shape)
                }
                #[cfg(not(feature = "cuda"))]
                candle::bail!("`cuda` feature is not enabled")
            }
        }
    }
}

#[derive(Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    epsilon: f32,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(hidden_size, "weight")
                .or_else(|_| vb.get(hidden_size, "gamma"))?,
            bias: vb
                .get(hidden_size, "bias")
                .or_else(|_| vb.get(hidden_size, "beta"))?,
            epsilon,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();

        match hidden_states.device() {
            Device::Cpu | Device::Metal(_) => {
                let mut hidden_states = hidden_states.clone();
                if let Some(residual) = residual {
                    hidden_states = hidden_states.add(residual)?;
                }
                let hidden_states_dtype = hidden_states.dtype();
                let internal_dtype = match hidden_states_dtype {
                    DType::F16 | DType::BF16 => DType::F32,
                    d => d,
                };
                let hidden_size = hidden_states.dim(D::Minus1)?;
                let hidden_states = hidden_states.to_dtype(internal_dtype)?;
                let mean_hidden_states =
                    (hidden_states.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states = hidden_states.broadcast_sub(&mean_hidden_states)?;
                let norm_hidden_states =
                    (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states_normed = hidden_states
                    .broadcast_div(&(norm_hidden_states + self.epsilon as f64)?.sqrt()?)?;
                let hidden_states = hidden_states_normed
                    .to_dtype(hidden_states_dtype)?
                    .broadcast_mul(&self.weight)?;

                hidden_states.broadcast_add(&self.bias)
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    let original_shape = hidden_states.shape();
                    let hidden_states = hidden_states.flatten_to(D::Minus2)?;
                    let residual = match residual {
                        Some(residual) => Some(residual.flatten_to(D::Minus2)?),
                        None => None,
                    };

                    let result = match residual.as_ref() {
                        Some(residual) => layer_norm_cuda(
                            &hidden_states,
                            Some(residual),
                            &self.weight,
                            Some(&self.bias),
                            self.epsilon,
                        ),
                        None => layer_norm_cuda(
                            &hidden_states,
                            None,
                            &self.weight,
                            Some(&self.bias),
                            self.epsilon,
                        ),
                    }?;
                    result.reshape(original_shape)
                }
                #[cfg(not(feature = "cuda"))]
                candle::bail!("`cuda` feature is not enabled")
            }
        }
    }
}
