//! Pure-Rust NCSDK v1 `.graph` (format version 2) dissector, compiler, and direct
//! 12-SHAVE `SHVMatGEMM` / `MvTensor` neural stage builder.
//!
//! Eliminates the legacy Python 3.5 / Caffe / Ubuntu 18.04 container requirement by
//! emitting hardware-verified 264-byte graph headers, 227-byte `Stage` descriptors,
//! 64-byte `op_params` blocks, and 64-byte-aligned FP16 weight/bias arenas directly.

use anyhow::{bail, Result};
use crate::v1::{f16_to_f32, f32_to_f16, GRAPH_HEADER_LEN, GRAPH_STAGE_LEN};

pub const BUFFER_NONE: u16 = 0;
pub const BUFFER_INPUT: u16 = 1;
pub const BUFFER_OUTPUT: u16 = 2;
pub const BUFFER_WEIGHTS: u16 = 3;
pub const BUFFER_SCRATCH_BASE: u16 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StageOp {
    Conv2D = 0,
    MaxPool2D = 1,
    AvgPool2D = 2,
    SoftMax = 3,
    FullyConnected = 4,
    NoneOrInput = 5,
    Relu = 6,
    DepthwiseConv2D = 8,
}

impl StageOp {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Conv2D),
            1 => Some(Self::MaxPool2D),
            2 => Some(Self::AvgPool2D),
            3 => Some(Self::SoftMax),
            4 => Some(Self::FullyConnected),
            5 => Some(Self::NoneOrInput),
            6 => Some(Self::Relu),
            8 => Some(Self::DepthwiseConv2D),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PostOp {
    None = 5,
    Relu = 6,
    ReluX = 7,
    Bias = 9,
}

#[derive(Debug, Clone)]
pub struct BufferRef {
    pub offset: u32,
    pub index: u16,
}

#[derive(Debug, Clone)]
pub struct StageRecord {
    pub name: String,
    pub op: u8,
    pub opt_mask: u32,
    pub radix_y: u8,
    pub radix_x: u8,
    pub stride_y: u8,
    pub stride_x: u8,
    pub pad_y: u8,
    pub pad_x: u8,
    pub pad_style: u8,
    pub dim_x: u32,
    pub dim_y: u32,
    pub dim_z: u32,
    pub tap_dim_x: u32,
    pub tap_dim_y: u32,
    pub tap_dim_z: u32,
    pub out_dim_x: u32,
    pub out_dim_y: u32,
    pub out_dim_z: u32,
    pub input_stride_x: u32,
    pub input_stride_y: u32,
    pub input_stride_z: u32,
    pub tap_stride_x: u32,
    pub tap_stride_y: u32,
    pub tap_stride_z: u32,
    pub out_stride_x: u32,
    pub out_stride_y: u32,
    pub out_stride_z: u32,
    pub dtype: u8,
    pub precision: u8,
    pub storage_order: u8,
    pub input: BufferRef,
    pub taps: BufferRef,
    pub bias: BufferRef,
    pub op_params: BufferRef,
    pub output: BufferRef,
    pub pre_op: u8,
    pub post_op: u8,
    pub post_param1: u32,
    pub post_stride_x: u16,
    pub post_stride_y: u16,
}

impl StageRecord {
    pub fn encode(&self) -> [u8; GRAPH_STAGE_LEN] {
        let mut b = [0u8; GRAPH_STAGE_LEN];
        let nb = self.name.as_bytes();
        let copy_len = nb.len().min(99);
        b[..copy_len].copy_from_slice(&nb[..copy_len]);

        let mut p = 100usize;
        b[p] = self.op;
        p += 1;
        b[p..p + 4].copy_from_slice(&self.opt_mask.to_le_bytes());
        p += 4;
        b[p] = self.radix_y;
        b[p + 1] = self.radix_x;
        b[p + 2] = self.stride_y;
        b[p + 3] = self.stride_x;
        b[p + 4] = self.pad_y;
        b[p + 5] = self.pad_x;
        b[p + 6] = self.pad_style;
        p += 7;

        for &d in &[
            self.dim_x,
            self.dim_y,
            self.dim_z,
            self.tap_dim_x,
            self.tap_dim_y,
            self.tap_dim_z,
            self.out_dim_x,
            self.out_dim_y,
            self.out_dim_z,
            self.input_stride_x,
            self.input_stride_y,
            self.input_stride_z,
            self.tap_stride_x,
            self.tap_stride_y,
            self.tap_stride_z,
            self.out_stride_x,
            self.out_stride_y,
            self.out_stride_z,
        ] {
            b[p..p + 4].copy_from_slice(&d.to_le_bytes());
            p += 4;
        }

        b[p] = self.dtype;
        b[p + 1] = self.precision;
        b[p + 2] = self.storage_order;
        p += 3;

        for br in [&self.input, &self.taps, &self.bias, &self.op_params, &self.output] {
            b[p..p + 4].copy_from_slice(&br.offset.to_le_bytes());
            b[p + 4..p + 6].copy_from_slice(&br.index.to_le_bytes());
            p += 6;
        }

        b[p] = self.pre_op;
        b[p + 1] = self.post_op;
        p += 2;
        b[p..p + 4].copy_from_slice(&self.post_param1.to_le_bytes());
        p += 4;
        b[p..p + 2].copy_from_slice(&self.post_stride_x.to_le_bytes());
        b[p + 2..p + 4].copy_from_slice(&self.post_stride_y.to_le_bytes());
        p += 4;
        debug_assert_eq!(p, GRAPH_STAGE_LEN);
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self> {
        if b.len() < GRAPH_STAGE_LEN {
            bail!("Stage buffer too short: {} < {}", b.len(), GRAPH_STAGE_LEN);
        }
        let name_end = b[..100].iter().position(|&c| c == 0).unwrap_or(100);
        let name = String::from_utf8_lossy(&b[..name_end]).to_string();
        let mut p = 100usize;
        let op = b[p];
        p += 1;
        let opt_mask = u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        p += 4;
        let radix_y = b[p];
        let radix_x = b[p + 1];
        let stride_y = b[p + 2];
        let stride_x = b[p + 3];
        let pad_y = b[p + 4];
        let pad_x = b[p + 5];
        let pad_style = b[p + 6];
        p += 7;

        let mut u32s = [0u32; 18];
        for slot in &mut u32s {
            *slot = u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
            p += 4;
        }
        let dtype = b[p];
        let precision = b[p + 1];
        let storage_order = b[p + 2];
        p += 3;

        let mut refs = Vec::with_capacity(5);
        for _ in 0..5 {
            let offset = u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
            let index = u16::from_le_bytes(b[p + 4..p + 6].try_into().unwrap());
            refs.push(BufferRef { offset, index });
            p += 6;
        }
        let pre_op = b[p];
        let post_op = b[p + 1];
        p += 2;
        let post_param1 = u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        p += 4;
        let post_stride_x = u16::from_le_bytes(b[p..p + 2].try_into().unwrap());
        let post_stride_y = u16::from_le_bytes(b[p + 2..p + 4].try_into().unwrap());

        Ok(Self {
            name,
            op,
            opt_mask,
            radix_y,
            radix_x,
            stride_y,
            stride_x,
            pad_y,
            pad_x,
            pad_style,
            dim_x: u32s[0],
            dim_y: u32s[1],
            dim_z: u32s[2],
            tap_dim_x: u32s[3],
            tap_dim_y: u32s[4],
            tap_dim_z: u32s[5],
            out_dim_x: u32s[6],
            out_dim_y: u32s[7],
            out_dim_z: u32s[8],
            input_stride_x: u32s[9],
            input_stride_y: u32s[10],
            input_stride_z: u32s[11],
            tap_stride_x: u32s[12],
            tap_stride_y: u32s[13],
            tap_stride_z: u32s[14],
            out_stride_x: u32s[15],
            out_stride_y: u32s[16],
            out_stride_z: u32s[17],
            dtype,
            precision,
            storage_order,
            input: refs[0].clone(),
            taps: refs[1].clone(),
            bias: refs[2].clone(),
            op_params: refs[3].clone(),
            output: refs[4].clone(),
            pre_op,
            post_op,
            post_param1,
            post_stride_x,
            post_stride_y,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DissectedGraph {
    pub file_size: u32,
    pub version: u32,
    pub nstages: u32,
    pub binary_data_offset: u32,
    pub first_shave: u16,
    pub last_shave: u16,
    pub stages: Vec<StageRecord>,
    pub weights_len: usize,
}

pub fn dissect_graph(blob: &[u8]) -> Result<DissectedGraph> {
    if blob.len() < GRAPH_HEADER_LEN {
        bail!("Graph file smaller than 264-byte header ({} bytes)", blob.len());
    }
    let file_size = u32::from_le_bytes(blob[32..36].try_into().unwrap());
    let version = u32::from_le_bytes(blob[36..40].try_into().unwrap());
    let nstages = u32::from_le_bytes(blob[240..244].try_into().unwrap());
    let binary_data_offset = u32::from_le_bytes(blob[244..248].try_into().unwrap());
    let first_shave = u16::from_le_bytes(blob[248..250].try_into().unwrap());
    let last_shave = u16::from_le_bytes(blob[250..252].try_into().unwrap());

    if version != 2 {
        bail!("Unsupported NCSDK graph version {version} (expected 2)");
    }
    if nstages == 0 || nstages > 4096 {
        bail!("Invalid stage count {nstages} in graph header");
    }
    if first_shave > last_shave || last_shave >= 12 {
        bail!("Invalid SHAVE range {first_shave}..={last_shave} in graph header");
    }
    if (binary_data_offset as usize) > blob.len() {
        bail!("binary_data_offset ({binary_data_offset}) exceeds file size ({})", blob.len());
    }

    let mut stages = Vec::with_capacity(nstages as usize);
    for i in 0..nstages as usize {
        let start = GRAPH_HEADER_LEN + i * GRAPH_STAGE_LEN;
        let end = start + GRAPH_STAGE_LEN;
        if end > blob.len() {
            bail!("Stage {i} runs past end of graph file");
        }
        stages.push(StageRecord::decode(&blob[start..end])?);
    }
    let weights_len = blob.len().saturating_sub(binary_data_offset as usize);
    Ok(DissectedGraph {
        file_size,
        version,
        nstages,
        binary_data_offset,
        first_shave,
        last_shave,
        stages,
        weights_len,
    })
}

/// Pure-Rust builder for NCSDK v1 `.graph` binaries.
pub struct GraphBuilder {
    pub num_shaves: u16,
    stages: Vec<StageRecord>,
    weights_blob: Vec<u8>,
    scratch_cursor: u32,
    next_scratch_index: u16,
}

impl GraphBuilder {
    pub fn new(num_shaves: u16) -> Self {
        let num_shaves = num_shaves.clamp(1, 12);
        Self {
            num_shaves,
            stages: Vec::new(),
            weights_blob: Vec::new(),
            scratch_cursor: 0x7200, // 64-byte aligned start matching reference mvNCCompile
            next_scratch_index: BUFFER_SCRATCH_BASE,
        }
    }

    fn align64(v: usize) -> usize {
        (v + 63) & !63
    }

    fn push_weights_aligned(&mut self, bytes: &[u8]) -> u32 {
        let aligned_len = Self::align64(self.weights_blob.len());
        if aligned_len > self.weights_blob.len() {
            self.weights_blob.resize(aligned_len, 0);
        }
        let off = self.weights_blob.len() as u32;
        self.weights_blob.extend_from_slice(bytes);
        let padded = Self::align64(self.weights_blob.len());
        self.weights_blob.resize(padded, 0);
        off
    }

    fn push_default_op_params(&mut self) -> u32 {
        let mut op_params = [0u8; 64];
        op_params[0..4].copy_from_slice(&1u32.to_le_bytes());
        self.push_weights_aligned(&op_params)
    }

    pub fn alloc_scratch(&mut self, nbytes: u32) -> BufferRef {
        let off = (self.scratch_cursor + 63) & !63;
        let idx = self.next_scratch_index;
        self.next_scratch_index += 1;
        self.scratch_cursor = off + ((nbytes + 63) & !63);
        BufferRef {
            offset: off,
            index: idx,
        }
    }

    /// Factor `m` into `(dim_x, dim_y)` closest to `sqrt(m)` so that `dim_x * dim_y == m`.
    /// This keeps the exact same contiguous `[M, K]` / `[M, N]` row-major memory layout
    /// while satisfying `OpConv::run` spatial tile bounds (`dim_x >= 8, dim_y >= 8`) when `m >= 64`.
    pub fn factor_spatial_dims(m: u32) -> (u32, u32) {
        if m == 0 {
            return (1, 1);
        }
        let root = (m as f64).sqrt() as u32;
        for d in (1..=root).rev() {
            if m % d == 0 {
                return (d, m / d);
            }
        }
        (1, m)
    }

    /// Emit the standard Stage 0 (`"data"`, `op = 5` / `kNone`) input descriptor that
    /// `mvNCCompile` places at the head of every NCSDK v1 `.graph` binary.
    pub fn add_input_data_stage(&mut self, dim_x: u32, dim_y: u32, dim_z: u32) {
        let stride_x = dim_z * 2;
        let stride_y = dim_x * stride_x;
        let stage = StageRecord {
            name: "data".to_string(),
            op: StageOp::NoneOrInput as u8,
            opt_mask: 0x8000_0000,
            radix_y: 1,
            radix_x: 1,
            stride_y: 1,
            stride_x: 1,
            pad_y: 0,
            pad_x: 0,
            pad_style: 2,
            dim_x,
            dim_y,
            dim_z,
            tap_dim_x: 1,
            tap_dim_y: dim_z,
            tap_dim_z: dim_z,
            out_dim_x: dim_x,
            out_dim_y: dim_y,
            out_dim_z: dim_z,
            input_stride_x: stride_x,
            input_stride_y: stride_y,
            input_stride_z: 2,
            tap_stride_x: stride_x,
            tap_stride_y: stride_x,
            tap_stride_z: 2,
            out_stride_x: stride_x,
            out_stride_y: stride_y,
            out_stride_z: 2,
            dtype: 2,
            precision: 2,
            storage_order: 4,
            input: BufferRef {
                offset: 0,
                index: BUFFER_INPUT,
            },
            taps: BufferRef {
                offset: 0,
                index: BUFFER_NONE,
            },
            bias: BufferRef {
                offset: 0,
                index: BUFFER_NONE,
            },
            op_params: BufferRef {
                offset: 0,
                index: BUFFER_NONE,
            },
            output: BufferRef {
                offset: 0,
                index: BUFFER_OUTPUT,
            },
            pre_op: 5,
            post_op: PostOp::None as u8,
            post_param1: 0,
            post_stride_x: 0,
            post_stride_y: 0,
        };
        self.stages.push(stage);
    }

    /// Add a general matrix multiplication / dense linear stage (`Y[M, N] = post_op(X[M, K] * W[K, N] + b[N])`),
    /// dispatched to all 12 SHAVE vector cores via `matmul::MatMulScheduler` (`SHVMatGEMM` / `SHVMatGEMV`).
    ///
    /// - `weights_kn`: row-major `[K, N]` FP16 weight matrix (`K * N` elements)
    /// - `bias_n`: optional `[N]` FP16 bias vector (`N` elements)
    pub fn add_gemm_stage(
        &mut self,
        name: &str,
        m: u32,
        k: u32,
        n: u32,
        weights_kn: &[u16],
        bias_n: Option<&[u16]>,
        post_op: PostOp,
        input_ref: BufferRef,
        output_ref: BufferRef,
    ) -> Result<()> {
        if weights_kn.len() != (k as usize) * (n as usize) {
            bail!(
                "GEMM weights length mismatch: expected K*N = {}*{}= {}, got {}",
                k,
                n,
                k * n,
                weights_kn.len()
            );
        }
        let w_bytes: Vec<u8> = weights_kn.iter().flat_map(|h| h.to_le_bytes()).collect();
        let taps_off = self.push_weights_aligned(&w_bytes);

        let bias_bytes: Vec<u8> = match bias_n {
            Some(b) => {
                if b.len() != n as usize {
                    bail!("GEMM bias length mismatch: expected N={n}, got {}", b.len());
                }
                b.iter().flat_map(|h| h.to_le_bytes()).collect()
            }
            None => vec![0u8; (n as usize) * 2],
        };
        let bias_off = self.push_weights_aligned(&bias_bytes);
        let op_params_off = self.push_default_op_params();

        let (dim_x, dim_y) = Self::factor_spatial_dims(m);
        let in_stride_x = k * 2;
        let in_stride_y = dim_x * in_stride_x;
        let out_stride_x = n * 2;
        let out_stride_y = dim_x * out_stride_x;

        // 1x1 Convolution on (dim_x, dim_y, dim_z = K) -> (out_dim_x = dim_x, out_dim_y = dim_y, out_dim_z = N)
        // maps 1:1 to `matmul::MatMulScheduler` (`SHVMatGEMM` / `SHVMatGEMV`) in FathomKey.elf
        // with identical flat [M, K] and [M, N] row-major memory layouts.
        let stage = StageRecord {
            name: name.to_string(),
            op: StageOp::Conv2D as u8,
            opt_mask: 0x8000_0000,
            radix_y: 1,
            radix_x: 1,
            stride_y: 1,
            stride_x: 1,
            pad_y: 0,
            pad_x: 0,
            pad_style: 2,
            dim_x,
            dim_y,
            dim_z: k,
            tap_dim_x: 1,
            tap_dim_y: k,
            tap_dim_z: n,
            out_dim_x: dim_x,
            out_dim_y: dim_y,
            out_dim_z: n,
            input_stride_x: in_stride_x,
            input_stride_y: in_stride_y,
            input_stride_z: 2,
            tap_stride_x: out_stride_x,
            tap_stride_y: out_stride_x,
            tap_stride_z: 2,
            out_stride_x,
            out_stride_y,
            out_stride_z: 2,
            dtype: 2,
            precision: 2,
            storage_order: 4,
            input: input_ref,
            taps: BufferRef {
                offset: taps_off,
                index: BUFFER_WEIGHTS,
            },
            bias: BufferRef {
                offset: bias_off,
                index: BUFFER_WEIGHTS,
            },
            op_params: BufferRef {
                offset: op_params_off,
                index: BUFFER_WEIGHTS,
            },
            output: output_ref,
            pre_op: 5,
            post_op: post_op as u8,
            post_param1: 0,
            post_stride_x: 0,
            post_stride_y: 0,
        };
        self.stages.push(stage);
        Ok(())
    }

    pub fn build(&self) -> Result<Vec<u8>> {
        if self.stages.is_empty() {
            bail!("Cannot build a graph with 0 stages");
        }
        let nstages = self.stages.len() as u32;
        // The firmware (LEON, SPARC) reads weights with word loads and traps on a misaligned address, so
        // the weights must start aligned. 227-byte stage records only land aligned when nstages % 4 == 0.
        let binary_data_offset = Self::align64(GRAPH_HEADER_LEN + self.stages.len() * GRAPH_STAGE_LEN) as u32;
        let total_size = (binary_data_offset as usize) + self.weights_blob.len();

        let mut blob = vec![0u8; total_size];
        // Header: filesize at 32, version=2 at 36, nstages at 240, binary_data_offset at 244, first_shave=0 at 248, last_shave=num_shaves-1 at 250
        blob[32..36].copy_from_slice(&(total_size as u32).to_le_bytes());
        blob[36..40].copy_from_slice(&2u32.to_le_bytes());
        blob[240..244].copy_from_slice(&nstages.to_le_bytes());
        blob[244..248].copy_from_slice(&binary_data_offset.to_le_bytes());
        blob[248..250].copy_from_slice(&0u16.to_le_bytes());
        blob[250..252].copy_from_slice(&(self.num_shaves - 1).to_le_bytes());

        for (i, st) in self.stages.iter().enumerate() {
            let off = GRAPH_HEADER_LEN + i * GRAPH_STAGE_LEN;
            blob[off..off + GRAPH_STAGE_LEN].copy_from_slice(&st.encode());
        }
        blob[binary_data_offset as usize..].copy_from_slice(&self.weights_blob);
        Ok(blob)
    }
}

/// Convenience helper: compile a `Y[M, N] = post_op(X[M, K] * W[K, N] + b[N])`
/// GEMM graph directly from `f32` slices, including Stage 0 (`data`) and Stage 1 (`gemm1`).
pub fn compile_gemm_f32(
    m: u32,
    k: u32,
    n: u32,
    weights_kn_f32: &[f32],
    bias_n_f32: Option<&[f32]>,
    relu: bool,
    num_shaves: u16,
) -> Result<Vec<u8>> {
    let w_f16: Vec<u16> = weights_kn_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let b_f16: Option<Vec<u16>> = bias_n_f32.map(|b| b.iter().map(|&v| f32_to_f16(v)).collect());
    let (dim_x, dim_y) = GraphBuilder::factor_spatial_dims(m);
    let mut gb = GraphBuilder::new(num_shaves);
    gb.add_input_data_stage(dim_x, dim_y, k);
    let post_op = if relu {
        PostOp::Relu
    } else if b_f16.is_some() {
        PostOp::Bias
    } else {
        PostOp::None
    };
    gb.add_gemm_stage(
        "gemm1",
        m,
        k,
        n,
        &w_f16,
        b_f16.as_deref(),
        post_op,
        BufferRef {
            offset: 0,
            index: BUFFER_INPUT,
        },
        BufferRef {
            offset: 0,
            index: BUFFER_OUTPUT,
        },
    )?;
    gb.build()
}

/// CPU FP32 reference matching the stick's FP16 input/weight/bias quantization for exact verification.
pub fn cpu_gemm_f16_ref(
    m: usize,
    k: usize,
    n: usize,
    x_mk_f16: &[u16],
    w_kn_f16: &[u16],
    b_n_f16: Option<&[u16]>,
    relu: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = match b_n_f16 {
                Some(b) => f16_to_f32(b[j]),
                None => 0.0f32,
            };
            for p in 0..k {
                let xv = f16_to_f32(x_mk_f16[i * k + p]);
                let wv = f16_to_f32(w_kn_f16[p * n + j]);
                acc += xv * wv;
            }
            if relu && acc < 0.0 {
                acc = 0.0;
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_encode_decode_roundtrip_is_exact() {
        let mut gb = GraphBuilder::new(12);
        let w = vec![f32_to_f16(0.25); 16 * 32];
        let b = vec![f32_to_f16(0.1); 32];
        gb.add_gemm_stage(
            "test_gemm",
            8,
            16,
            32,
            &w,
            Some(&b),
            PostOp::Relu,
            BufferRef { offset: 0, index: BUFFER_INPUT },
            BufferRef { offset: 0, index: BUFFER_OUTPUT },
        )
        .unwrap();
        let blob = gb.build().unwrap();
        let dg = dissect_graph(&blob).unwrap();
        assert_eq!(dg.version, 2);
        assert_eq!(dg.nstages, 1);
        assert_eq!(dg.first_shave, 0);
        assert_eq!(dg.last_shave, 11);
        assert_eq!(dg.stages[0].name, "test_gemm");
        assert_eq!(dg.stages[0].dim_x, 2);
        assert_eq!(dg.stages[0].dim_y, 4);
        assert_eq!(dg.stages[0].dim_z, 16);
        assert_eq!(dg.stages[0].out_dim_z, 32);
        assert_eq!(dg.stages[0].encode(), blob[GRAPH_HEADER_LEN..GRAPH_HEADER_LEN + GRAPH_STAGE_LEN]);
    }

    #[test]
    fn compile_gemm_f32_emits_data_and_gemm_stages_with_8x8_spatial_factors() {
        let m = 64u32;
        let k = 128u32;
        let n = 64u32;
        let w = vec![0.125f32; (k * n) as usize];
        let b = vec![0.25f32; n as usize];
        let blob = compile_gemm_f32(m, k, n, &w, Some(&b), false, 12).unwrap();
        let dg = dissect_graph(&blob).unwrap();
        assert_eq!(dg.nstages, 2);
        assert_eq!(dg.stages[0].name, "data");
        assert_eq!(dg.stages[0].op, StageOp::NoneOrInput as u8);
        assert_eq!(dg.stages[0].dim_x, 8);
        assert_eq!(dg.stages[0].dim_y, 8);
        assert_eq!(dg.stages[0].dim_z, 128);
        assert_eq!(dg.stages[1].name, "gemm1");
        assert_eq!(dg.stages[1].op, StageOp::Conv2D as u8);
        assert_eq!(dg.stages[1].dim_x, 8);
        assert_eq!(dg.stages[1].dim_y, 8);
        assert_eq!(dg.stages[1].dim_z, 128);
        assert_eq!(dg.stages[1].out_dim_x, 8);
        assert_eq!(dg.stages[1].out_dim_y, 8);
        assert_eq!(dg.stages[1].out_dim_z, 64);
        assert_eq!(dg.stages[1].post_op, PostOp::Bias as u8);
    }

    #[test]
    fn weights_start_aligned_for_any_stage_count() {
        for nstages in 1..=5u32 {
            let mut gb = GraphBuilder::new(12);
            for _ in 0..nstages {
                gb.add_input_data_stage(8, 8, 16);
            }
            let blob = gb.build().unwrap();
            let off = u32::from_le_bytes(blob[244..248].try_into().unwrap());
            assert_eq!(off % 64, 0, "{nstages} stages: weights at {off}");
            assert_eq!(u32::from_le_bytes(blob[32..36].try_into().unwrap()) as usize, blob.len());
        }
    }
}

