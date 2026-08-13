//! V9-WIRE GPU FUSE (task mandate, 2026-07-25) — the v9 eye-test hitgate
//! compose (main.rs's old `NetPresent::v9_hitgate_compose`), available two
//! ways from the SAME module so the "cpu" default and the "gpu" fused path
//! can be proven bit-comparable by a parity harness:
//!
//!   - [`compose_cpu_reference`] — the exact CPU math (relocated verbatim
//!     from main.rs, not reimplemented), still what `GAIA_V9_COMPOSE=cpu`
//!     (the IRON default) drives.
//!   - [`V9ComposePass`] — the SAME math as one WGSL compute dispatch:
//!     no GPU->CPU readback of the native AOV or the low-res split E/D
//!     trace, no CPU loop, one CPU->GPU write for the net's own raw output
//!     (unavoidable — see `rdirect_unet::run_compiled_forward`'s doc) and
//!     the composed result never leaves the GPU.
//!
//! Both read the SAME buffer layouts `gather_v9`/`gather_split`
//! (`rdirect_gather_split.wgsl`) already use: AOV is 2 vec4/px
//! (`[2i+0]=(albedo.xyz,depth)`), the split trace is 2 vec4/low-px
//! (`[2j+0]=E(rgb,count)`, `[2j+1]=D(rgb,count)`).

use bytemuck::{Pod, Zeroable};

pub const V9_COMPOSE_SHADER: &str = include_str!("rdirect_v9_compose.wgsl");

const ALBEDO_DEMOD_EPS: f32 = 1.0e-3;
const NO_HIT_SQ: f32 = 1.0e-8;

/// `rdirect.rs::low_coord` — target index -> continuous low-res coordinate.
fn low_coord(t: u32, low: u32, tgt: u32) -> f32 {
    (t as f32 + 0.5) * low as f32 / tgt as f32 - 0.5
}

/// The CPU hit-gate compose — byte-for-byte the old
/// `NetPresent::v9_hitgate_compose` body, relocated here so the "cpu" path
/// and the GPU parity harness share ONE reference instead of two
/// hand-synced copies. `aov` is `n*8` floats (2 vec4/px), `ed` is
/// `low_w*low_h*8` floats (2 vec4/low-px), `net_out` is `n*3` floats — same
/// shapes `read_buffer_f32` produces / the net forward returns.
#[allow(clippy::too_many_arguments)]
pub fn compose_cpu_reference(
    net_out: &[f32],
    aov: &[f32],
    ed: &[f32],
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    hitgate: bool,
) -> Vec<f32> {
    let mut gated = net_out.to_vec();
    if !hitgate {
        return gated;
    }
    for ty in 0..target_h {
        for tx in 0..target_w {
            let px = (ty * target_w + tx) as usize;
            let a = &aov[px * 8..px * 8 + 4]; // (albedo.xyz, depth)
            let depth = a[3];
            if depth > 0.0 {
                continue; // hit px: net's own raw output stands
            }
            let albedo = [a[0], a[1], a[2]];
            let fx = low_coord(tx, low_w, target_w);
            let fy = low_coord(ty, low_h, target_h);
            let x0 = fx.floor();
            let y0 = fy.floor();
            let dx = fx - x0;
            let dy = fy - y0;
            let clampi = |v: f32, hi: u32| -> usize { (v.max(0.0) as u32).min(hi - 1) as usize };
            let x0i = clampi(x0, low_w);
            let x1i = clampi(x0 + 1.0, low_w);
            let y0i = clampi(y0, low_h);
            let y1i = clampi(y0 + 1.0, low_h);
            let cell = |lx: usize, ly: usize| -> [f32; 3] {
                let j = ly * (low_w as usize) + lx;
                let e = &ed[j * 8..j * 8 + 4];
                let d = &ed[j * 8 + 4..j * 8 + 8];
                let ec = (e[3]).max(1.0);
                let dc = (d[3]).max(1.0);
                [e[0] / ec + d[0] / dc, e[1] / ec + d[1] / dc, e[2] / ec + d[2] / dc]
            };
            let c00 = cell(x0i, y0i);
            let c10 = cell(x1i, y0i);
            let c01 = cell(x0i, y1i);
            let c11 = cell(x1i, y1i);
            let mut composite = [0.0f32; 3];
            for c in 0..3 {
                let top = c00[c] * (1.0 - dx) + c10[c] * dx;
                let bot = c01[c] * (1.0 - dx) + c11[c] * dx;
                composite[c] = top * (1.0 - dy) + bot * dy;
            }
            let alb_sq = albedo[0] * albedo[0] + albedo[1] * albedo[1] + albedo[2] * albedo[2];
            let divisor = if alb_sq > NO_HIT_SQ {
                [albedo[0] + ALBEDO_DEMOD_EPS, albedo[1] + ALBEDO_DEMOD_EPS, albedo[2] + ALBEDO_DEMOD_EPS]
            } else {
                [1.0, 1.0, 1.0]
            };
            for c in 0..3 {
                gated[px * 3 + c] = ((composite[c] / divisor[c]).max(0.0) + 1.0).ln();
            }
        }
    }
    gated
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComposeUniform {
    n: u32,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    hitgate: u32,
    _pad0: u32,
    _pad1: u32,
}

/// The fused v9 hit-gate compose, entirely on the GPU (`GAIA_V9_COMPOSE=gpu`).
pub struct V9ComposePass {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
}

impl V9ComposePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("v9 compose"),
            source: wgpu::ShaderSource::Wgsl(V9_COMPOSE_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("v9 compose layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ComposeUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),  // net_out (read)
                storage_entry(2, true),  // aov (read)
                storage_entry(3, true),  // accum_ed (read)
                storage_entry(4, false), // gated (read_write)
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("v9 compose pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("v9 compose pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("v9_compose"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v9 compose uniform"),
            size: std::mem::size_of::<ComposeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, layout, uniform_buf }
    }

    /// Encode one fused compose dispatch. `net_out_raw` = the net's raw
    /// demod-log output already written to a GPU storage buffer (`[n,3]`),
    /// `aov` = native AOV (2 cells/px), `accum_ed` = the low-res split E/D
    /// trace (2 cells/low-px), `gated_out` = destination (`[n,3]`, same
    /// buffer `demod`/`evidence.encode_pack` read downstream).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        net_out_raw: &wgpu::Buffer,
        aov: &wgpu::Buffer,
        accum_ed: &wgpu::Buffer,
        gated_out: &wgpu::Buffer,
        n: u32,
        low_w: u32,
        low_h: u32,
        target_w: u32,
        target_h: u32,
        hitgate: bool,
    ) {
        let uniform = ComposeUniform {
            n,
            low_w,
            low_h,
            target_w,
            target_h,
            hitgate: hitgate as u32,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("v9 compose bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: net_out_raw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: aov.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: accum_ed.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gated_out.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("v9 compose pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(target_w.div_ceil(8), target_h.div_ceil(8), 1);
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
