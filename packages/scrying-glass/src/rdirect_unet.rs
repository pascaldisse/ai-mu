//! v9-body — shape-parametric MULTI-SCALE conv net, beside the existing
//! per-pixel `rdirect` MLP (`rdirect.rs` / `rdirect_live.rs`, 23-or-39-in
//! flat-vector → 5×64 ReLU → 3-out). Per HANDOFF.md → v8d CAPACITY VERDICT
//! (sealed 07-21): the per-pixel body cannot reach bar 0.035 (no spatial
//! receptive field) — this is the next body: an encoder/decoder U-Net with
//! skip connections, built on the SAME house MPSGraph tensor-path pattern
//! `rdirect_live.rs` uses (per-frame graph, `runWithMTLCommandQueue`).
//!
//! STAGE 1 ONLY (this file): shape + a perf-spike harness. NOT trained, NOT
//! wired into the live present path, NOT the ordeal. See
//! `docs/perf/2026-07-21-v9-spike.md` for the measured numbers this shape
//! produced and `examples/rdirect_v9_spike.rs` for the harness.
//!
//! IRON (never hardcode): every dimension below — scale count, per-scale
//! channel widths, kernel size, input/output channel counts, render res,
//! output res — is a field on `UnetConfig` with a documented default. The
//! graph builder (`imp::UnetLive::build`) reads ONLY from that struct; no
//! literal dimension is baked into the op graph outside it (the
//! `INPUT_FEATURES = 23` hardcode class of bug, already killed once in
//! `rdirect.rs`, is not reintroduced here).
//!
//! RESOLUTION LAW (CLAUDE.md + HANDOFF.md → RULING 07-21 + AMENDMENT): the
//! render/evidence resolution (trace buffers, the feature maps this net
//! reads) stays 640×480 by default — `UnetConfig::render_w/h`. The net's
//! OUTPUT resolution is free (`UnetConfig::output_w/h`, defaults to render
//! res) — Pleroma DRAWS at output res from world truth + evidence; it never
//! interpolates a small finished image up. When output res differs from
//! render res, the interior upsample (`resizeBilinear`) is followed by a
//! learned 3×3 conv before the graph is called done drawing at that scale —
//! never a bare resize as the last op.

/// Motion-vector channels appended to the v7 per-pixel feature set (dx, dy
/// in screen-space pixels/frame — object motion, not just camera reproject;
/// the v7 ghost gap this body targets per HANDOFF.md → DLSS-4 transfer
/// list). An IRON constant of the FEATURE CONTRACT (what a motion vector
/// is — 2 components), not a tuned architecture dimension.
pub const MOTION_VECTOR_CHANNELS: usize = 2;

/// v9's input channel count = v7's `HIST_FEATURES_SPLIT` (1-ray evidence
/// taps + gbuffer + reprojected history — the exact feature set
/// `rdirect.rs`'s `hist_features_split` already builds per pixel) plus the
/// new motion-vector channels. Treated as image channels here (one value
/// per pixel per channel) rather than a flat per-pixel vector — the whole
/// point of the conv body is that neighbouring pixels' channels are visible
/// to each other (the receptive field the per-pixel MLP never had).
pub fn default_in_channels() -> usize {
    crate::rdirect::HIST_FEATURES_SPLIT + MOTION_VECTOR_CHANNELS
}

/// Shape-parametric U-Net config. EVERY field is a dimension parameter with
/// a documented default (IRON) — `imp::UnetLive::build` reads only this
/// struct plus a random seed; nothing else sizes the graph.
#[derive(Clone, Debug)]
pub struct UnetConfig {
    /// Encoder/decoder depth. Default 3 (full res, /2, /4) — matches the
    /// HANDOFF.md v9-body spec ("default 3 scales: full/2/4").
    pub n_scales: usize,
    /// Channel width AT EACH scale (encoder side; the decoder mirrors it).
    /// `len(widths) == n_scales`, index 0 = full render res. M1-sized per
    /// `dlss4-bringup/REPORT.md` (documented DLSS-4 widths are 8-9× the
    /// WHOLE net's frame budget on ONE of 11 layers — not these widths).
    pub widths: Vec<usize>,
    /// Square conv kernel size. Default 3 (the standard U-Net 3×3).
    pub kernel: usize,
    /// Input channel count (the per-pixel feature vector, read as image
    /// channels). Default = v7's feature set + motion vectors.
    pub in_channels: usize,
    /// Output channel count (RGB demod-log radiance). Default 3 — matches
    /// `rdirect::OUTPUT_CHANNELS`.
    pub out_channels: usize,
    /// Render/evidence resolution — the trace + feature buffers this net
    /// reads. Stays 640×480 by default per CLAUDE.md → THE RESOLUTION IS
    /// 640×480 / HANDOFF.md → RULING 07-21 AMENDMENT ("render resolution"),
    /// but is a PARAMETER here, never a literal in the graph builder.
    pub render_w: usize,
    pub render_h: usize,
    /// Net OUTPUT resolution — FREE per the 07-21 amendment. Defaults to
    /// render res (no upsample by default). A larger value does NOT
    /// interpolate the render up: the decoder's last upsample step is
    /// followed by a learned conv (see module docs) before output.
    pub output_w: usize,
    pub output_h: usize,
}

impl Default for UnetConfig {
    fn default() -> Self {
        let render_w = 640usize;
        let render_h = 480usize;
        Self {
            n_scales: 3,
            widths: vec![24, 40, 64],
            kernel: 3,
            in_channels: default_in_channels(),
            out_channels: 3,
            render_w,
            render_h,
            output_w: render_w,
            output_h: render_h,
        }
    }
}

impl UnetConfig {
    /// Total learned-parameter count (weights + biases across every conv in
    /// the built graph) — for the spike table, not used by the builder.
    pub fn approx_param_count(&self) -> usize {
        let k = self.kernel;
        let mut n = 0usize;
        let mut cin = self.in_channels;
        // scale 0 stem
        n += k * k * cin * self.widths[0] + self.widths[0];
        cin = self.widths[0];
        for s in 1..self.n_scales {
            let cout = self.widths[s];
            n += k * k * cin * cout + cout; // downsample
            n += k * k * cout * cout + cout; // refine
            cin = cout;
        }
        for s in (0..self.n_scales.saturating_sub(1)).rev() {
            let cout = self.widths[s];
            let cin_up = self.widths[s + 1];
            n += k * k * cin_up * cout + cout; // post-upsample channel reduce
            n += k * k * (cout * 2) * cout + cout; // post-concat merge
        }
        if (self.output_h, self.output_w) != (self.render_h, self.render_w) {
            n += k * k * self.widths[0] * self.widths[0] + self.widths[0]; // post-resize refine
        }
        n += 1 * 1 * self.widths[0] * self.out_channels + self.out_channels; // 1x1 head
        n
    }

    /// Total input feature-map elements for one forward (batch=1).
    pub fn input_elems(&self) -> usize {
        self.render_h * self.render_w * self.in_channels
    }

    /// Total output feature-map elements for one forward (batch=1).
    pub fn output_elems(&self) -> usize {
        self.output_h * self.output_w * self.out_channels
    }
}

#[cfg(target_os = "macos")]
pub use imp::{PendingForward, UnetLive};

#[cfg(target_os = "macos")]
mod imp {
    use super::UnetConfig;
    use half::f16;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{AnyThread, Message};
    use objc2_foundation::{NSArray, NSData, NSNumber, NSString};
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLResourceOptions,
    };
    use objc2_metal_performance_shaders::{MPSCommandBuffer, MPSDataType};
    use objc2_metal_performance_shaders_graph::{
        MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphDevice, MPSGraphExecutable,
        MPSGraphPaddingStyle, MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
        MPSGraphTensorNamedDataLayout,
    };
    use std::cell::{Cell, RefCell};
    use std::ffi::c_void;
    use std::ptr::NonNull;

    /// Deterministic dependency-free PRNG (SplitMix64) — weight INIT only.
    /// Duplicated per-module by house convention (see `rdirect.rs`,
    /// `denoiser.rs`, `upscaler.rs` — each carries its own copy).
    struct SplitMix64 {
        state: u64,
    }
    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_signed_unit(&mut self) -> f32 {
            let bits = (self.next_u64() >> 40) as u32;
            let unit = (bits as f32) / (1u32 << 24) as f32;
            unit * 2.0 - 1.0
        }
    }

    /// One conv layer's graph-resident constants (fp16 storage).
    struct ConvLayer {
        weight: Retained<MPSGraphTensor>, // [kh, kw, cin, cout] HWIO
        bias: Retained<MPSGraphTensor>,   // [1, 1, 1, cout]
    }

    /// The v9 body: an MPSGraph U-Net forward built once from a `UnetConfig`
    /// plus randomly (He-)initialized weights — UNTRAINED, spike-only. Same
    /// house pattern as `rdirect_live.rs::RdirectLive` (a per-frame graph
    /// run via `runWithMTLCommandQueue_feeds_targetTensors_targetOperations`),
    /// minus the live wgpu bridge / pooled buffers (stage 2+ concern).
    pub struct UnetLive {
        graph: Retained<MPSGraph>,
        mps_device: Retained<MPSGraphDevice>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        /// V9-WIRE SPEED ROUND (2026-07-25) LEVER 2: a SECOND, DEDICATED
        /// `MTLCommandQueue` for async net submissions (`submit_gpu_bridged_async`)
        /// — the sync path (`forward_gpu_bridged`) keeps using `queue` above,
        /// byte-untouched (IRON: GAIA_V9_ASYNC=0 never even READS this field's
        /// commit behavior differently). A dedicated queue is what actually lets
        /// the GPU run the net's command buffer CONCURRENTLY with the render
        /// queue's own next-frame trace/gather submits — `RdirectLive::attach_pool`'s
        /// own S12 doc names this exact reason (`from_wgpu_queue`'s doc below
        /// flags it as the deferred "future S9 double-buffered overlap shape").
        /// No cross-queue fence is needed to READ from it (see
        /// `submit_gpu_bridged_async`'s own doc): the caller always
        /// `device.poll(wait_indefinitely())`s the gather submit before calling
        /// it, which is Metal's documented sufficient happens-before for a
        /// DIFFERENT queue's subsequent commands to see those writes.
        net_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        input: Retained<MPSGraphTensor>,
        output: Retained<MPSGraphTensor>,
        config: UnetConfig,
        /// N0-pattern GPU-ONLY split (rdirect_live.rs S3): lazily-compiled
        /// executable + fixed MTLBuffers, so `forward_gpu_ms` can read
        /// `MTLCommandBuffer` GPUStartTime/GPUEndTime instead of the honest
        /// but CPU-inflated `runWithMTLCommandQueue` wall
        /// (`forward_cpu_roundtrip`) — the exact gap n0e's own spike found
        /// (~34ms CPU encode/wait around a ~6ms GPU forward).
        gpu: RefCell<Option<GpuSplit>>,
        /// V9-WIRE ATOM (GATHER FOLLOW-UP #2, zero-copy bridge): `GpuSplit`'s
        /// `in_mtl` (fp16 input MTLBuffer) wrapped as a wgpu STORAGE buffer
        /// of `u32` (2 packed fp16 halves each), ONLY when built via
        /// `from_wgpu_queue`. `Fp16Packer` (`rdirect_gather.rs`) writes into
        /// it directly from the v9 gather's f32 output — no CPU readback, no
        /// CPU conversion loop. `None` for `from_system`/`from_weights`.
        /// SPEED ROUND: index 0 = the sync path's own buffer (untouched,
        /// same MTLBuffer `feature_buf_u32()`/set-0 always resolved to before
        /// this change); index 1 = the SECOND set the async path's `set=1`
        /// gather/pack round targets. Both `None` unless built via
        /// `from_wgpu_queue`.
        feature_buf_u32: [Option<wgpu::Buffer>; 2],
        /// V9-WIRE ATOM (GATHER FOLLOW-UP #2) probe split, mirrors
        /// `rdirect_live.rs`'s own N0.i S13 `last_commit_ms`/`last_wait_ms`
        /// house pattern: `run_compiled_forward`'s CPU wall (encode+commit,
        /// wait, output f16->f32 readout) broken into its three parts, so a
        /// caller can decompose the `forward_gpu_bridged` wall-vs-GPU-only
        /// gap `docs/perf/2026-07-21-v9-wire.md` reports instead of leaving
        /// it as one lump "glue" number.
        last_encode_commit_ms: Cell<f64>,
        last_wait_ms: Cell<f64>,
        last_readout_ms: Cell<f64>,
    }

    /// SPEED ROUND: one buffer SET's worth of the fixed input/output
    /// MTLBuffers + their MPSGraphTensorData wrappers — what used to be
    /// `GpuSplit`'s own direct fields, now duplicated ×2 (`GpuSplit::sets`)
    /// so a set-1 forward can be in flight (async, on `net_queue`) while
    /// set-0's own buffers stay untouched, and vice versa. The COMPILED
    /// `executable` is shared (MPSGraph docs: encode is stateless w.r.t.
    /// which input/output MPSGraphTensorData you pass — safe, even
    /// concurrently, across two independent buffer sets).
    struct GpuBufSet {
        in_mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
        // V9-WIRE ATOM addition: `forward_gpu_ms` only ever needed the
        // GPUStartTime/EndTime off the command buffer, so it never kept a
        // direct handle to the output MTLBuffer's own contents (only
        // `out_td`, the MPSGraphTensorData WRAPPER). `forward_gpu_with_output`
        // (below) needs to `.contents()` it directly, so it's stored here too
        // — SAME buffer `out_td` already wraps, just also held for CPU reads.
        out_mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
        in_td: Retained<MPSGraphTensorData>,
        out_td: Retained<MPSGraphTensorData>,
    }

    struct GpuSplit {
        executable: Retained<MPSGraphExecutable>,
        /// len 2 — see `GpuBufSet`'s own doc. `sets[0]` is EXACTLY what the
        /// old single-buffer `GpuSplit` held (same alloc order/shapes/order
        /// of operations in `ensure_gpu_split`), so the sync path is
        /// byte-identical.
        sets: [GpuBufSet; 2],
    }

    /// V9-WIRE SPEED ROUND (2026-07-25): a net command buffer COMMITTED (on
    /// `net_queue`) but not yet waited — `submit_gpu_bridged_async`'s return,
    /// `wait_gpu_bridged_async`'s input. Owning this is the caller's proof
    /// that `set`'s buffers are "spoken for" until the matching wait — the
    /// caller must not re-pack/re-submit the SAME `set` before waiting this.
    pub struct PendingForward {
        base: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        mps: Retained<MPSCommandBuffer>,
        set: usize,
    }

    impl UnetLive {
        /// The offline/spike path: own system Metal device + queue (mirrors
        /// `RdirectLive::from_system`).
        pub fn from_system(config: &UnetConfig, seed: u64) -> Result<Self, String> {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| "rdirect_unet: no system Metal device".to_string())?;
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| "rdirect_unet: newCommandQueue failed".to_string())?;
            Self::build(device, queue, config, seed)
        }

        fn build(
            device: Retained<ProtocolObject<dyn MTLDevice>>,
            queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
            config: &UnetConfig,
            seed: u64,
        ) -> Result<Self, String> {
            if config.n_scales == 0 {
                return Err("rdirect_unet: n_scales must be >= 1".to_string());
            }
            if config.widths.len() != config.n_scales {
                return Err(format!(
                    "rdirect_unet: widths.len()={} must equal n_scales={}",
                    config.widths.len(),
                    config.n_scales
                ));
            }
            // SAFETY: objc2 message sends to live Metal + MPSGraph objects,
            // mirroring rdirect_live.rs::RdirectLive::build exactly.
            unsafe {
                let graph = MPSGraph::new();
                let mps_device = MPSGraphDevice::deviceWithMTLDevice(&device);
                let mut rng = SplitMix64::new(seed);

                // per-scale spatial dims, ceil(/2) each step (TF_SAME stride-2
                // semantics — matches what the downsample convs below actually
                // produce, so the decoder's explicit resize targets line up).
                let mut dims = Vec::with_capacity(config.n_scales);
                dims.push((config.render_h, config.render_w));
                for s in 1..config.n_scales {
                    let (ph, pw) = dims[s - 1];
                    dims.push(((ph + 1) / 2, (pw + 1) / 2));
                }

                let in_shape = shape(&[1, config.render_h, config.render_w, config.in_channels]);
                let input = graph.placeholderWithShape_dataType_name(
                    Some(&in_shape),
                    MPSDataType::Float16,
                    None,
                );

                // ── encoder: scale-0 stem, then downsample+refine per scale ──
                let stem = conv_layer(&graph, config.in_channels, config.widths[0], config.kernel, &mut rng);
                let mut x = conv2d(&graph, &input, &stem, 1);
                x = graph.reLUWithTensor_name(&x, None);
                let mut skips: Vec<Retained<MPSGraphTensor>> = vec![x.clone()];

                for s in 1..config.n_scales {
                    let cin = config.widths[s - 1];
                    let cout = config.widths[s];
                    let down = conv_layer(&graph, cin, cout, config.kernel, &mut rng);
                    x = conv2d(&graph, &x, &down, 2);
                    x = graph.reLUWithTensor_name(&x, None);
                    let refine = conv_layer(&graph, cout, cout, config.kernel, &mut rng);
                    x = conv2d(&graph, &x, &refine, 1);
                    x = graph.reLUWithTensor_name(&x, None);
                    skips.push(x.clone());
                }

                // ── decoder: upsample (resize + learned conv), concat skip,
                // merge conv — mirrored back down to scale 0 ──
                let mut dec = skips[config.n_scales - 1].clone();
                for s in (0..config.n_scales - 1).rev() {
                    let (th, tw) = dims[s];
                    let up = resize_bilinear(&graph, &dec, th, tw);
                    let cin_up = config.widths[s + 1];
                    let cout = config.widths[s];
                    let reduce = conv_layer(&graph, cin_up, cout, config.kernel, &mut rng);
                    let mut up2 = conv2d(&graph, &up, &reduce, 1);
                    up2 = graph.reLUWithTensor_name(&up2, None);
                    let cat = graph.concatTensor_withTensor_dimension_name(&up2, &skips[s], 3, None);
                    let merge = conv_layer(&graph, cout * 2, cout, config.kernel, &mut rng);
                    dec = conv2d(&graph, &cat, &merge, 1);
                    dec = graph.reLUWithTensor_name(&dec, None);
                }

                // ── free output res: resize to output grid + one learned conv
                // (never a bare resize as the last drawing step) ──
                let mut final_feat = dec;
                if (config.output_h, config.output_w) != (config.render_h, config.render_w) {
                    final_feat = resize_bilinear(&graph, &final_feat, config.output_h, config.output_w);
                    let refine_out = conv_layer(&graph, config.widths[0], config.widths[0], config.kernel, &mut rng);
                    final_feat = conv2d(&graph, &final_feat, &refine_out, 1);
                    final_feat = graph.reLUWithTensor_name(&final_feat, None);
                }

                // ── 1x1 linear head (no ReLU — matches rdirect::Mlp's linear
                // last layer, demod-log radiance can be negative) ──
                let head = conv_layer(&graph, config.widths[0], config.out_channels, 1, &mut rng);
                let output = conv2d(&graph, &final_feat, &head, 1);

                let net_queue = device
                    .newCommandQueue()
                    .ok_or_else(|| "rdirect_unet: net_queue alloc failed".to_string())?;
                Ok(Self {
                    graph,
                    mps_device,
                    device,
                    queue,
                    net_queue,
                    input,
                    output,
                    config: config.clone(),
                    gpu: RefCell::new(None),
                    feature_buf_u32: [None, None],
                    last_encode_commit_ms: Cell::new(0.0),
                    last_wait_ms: Cell::new(0.0),
                    last_readout_ms: Cell::new(0.0),
                })
            }
        }

        pub fn config(&self) -> &UnetConfig {
            &self.config
        }

        /// PARITY / ORDEAL-DOOR / LIVE-PRESENT LOADER (V9 WIRING ATOM,
        /// 2026-07-21): builds the SAME graph shape `build` does, but every
        /// conv layer's weight/bias constants are read from a trained
        /// `super::cpu::UnetWeights` checkpoint (the CPU-trainable twin's own
        /// output — `super::cpu::deserialize_weights`) instead of He-init
        /// RNG. This is the disclosed stage-3+ gap the module doc's STAGE 1
        /// section flagged ("loading is NOT wired this stage") — now wired.
        /// The offline/spike path: own system Metal device + queue.
        pub fn from_weights(weights: &super::cpu::UnetWeights) -> Result<Self, String> {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| "rdirect_unet: no system Metal device".to_string())?;
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| "rdirect_unet: newCommandQueue failed".to_string())?;
            Self::build_from_weights(device, queue, weights)
        }

        /// V9-WIRE ATOM (GATHER FOLLOW-UP #2, 2026-07-21) — THE ZERO-COPY
        /// BRIDGE: mirrors `RdirectLive::from_wgpu_queue` exactly at the
        /// bridging step — reach the SAME Metal device/queue wgpu itself
        /// drives (the wgpu-hal Metal backdoor), instead of
        /// `from_weights`'s own separate `MTLCreateSystemDefaultDevice()`
        /// handle (the root cause `docs/perf/2026-07-21-v9-wire.md`'s
        /// §gather decomposition names for the 7.0ms readback + 11.9ms
        /// fp16-glue costs). Builds the SAME graph shape
        /// `build_from_weights` does, on the bridged device/queue, then
        /// wraps the lazily-allocated GPU-only split's OWN input MTLBuffer
        /// (`ensure_gpu_split`'s `in_mtl`, fp16) as a wgpu STORAGE buffer of
        /// `u32` (`feature_buf_u32`) — a caller's `Fp16Packer` (see
        /// `rdirect_gather.rs`) then writes the v9 gather's f32 output
        /// straight into it, GPU-side, with no CPU involvement at all.
        ///
        /// SYNCHRONIZATION: unlike `RdirectLive::attach_pool` (which opens
        /// its OWN dedicated net command queues + an `MTLSharedEvent` fence,
        /// because its S9 pipeline overlaps a net forward with the NEXT
        /// frame's trace on wgpu's queue), the SYNC path here (`forward_gpu_bridged`,
        /// `set=0`) reuses wgpu's queue directly as `self.queue` (the graph
        /// forward's own command buffer). Metal's documented guarantee —
        /// command buffers committed to ONE `MTLCommandQueue` execute in
        /// commit order — is therefore sufficient on its own: the caller's
        /// wgpu `queue.submit()` of the gather+pack commands (writing
        /// `feature_buf_u32`) MUST happen before `forward_gpu_bridged`'s own
        /// `commandBuffer()`/`commit()` call, and no explicit fence is
        /// needed. This is a real, narrower synchronization contract than
        /// `RdirectLive`'s own (documented, not silently assumed) — correct
        /// for this atom's single-shot measurement use.
        ///
        /// V9-WIRE SPEED ROUND (2026-07-25): the ASYNC path
        /// (`submit_gpu_bridged_async`/`wait_gpu_bridged_async`, `set=0/1`
        /// alternating) now exists too — the "future S9 double-buffered
        /// overlap shape" this doc used to defer. It commits on `net_queue`
        /// (a SEPARATE dedicated queue, mirroring `RdirectLive::attach_pool`'s
        /// own reasoning for real GPU concurrency with the render queue) but
        /// STILL needs no explicit fence: every caller of the async submit
        /// has ALREADY `device.poll(wait_indefinitely())`ed the gather submit
        /// (main.rs's `resolve_frame_v9_async`), which is Metal's documented
        /// sufficient happens-before for ANY other queue's later commands to
        /// observe those writes — the cross-queue case just swaps "same
        /// queue's commit order" for "CPU-observed completion" as the proof.
        pub fn from_wgpu_queue(
            wgpu_device: &wgpu::Device,
            queue: &wgpu::Queue,
            weights: &super::cpu::UnetWeights,
        ) -> Result<Self, String> {
            // SAFETY: as_hal hands the live hal Queue; we retain the raw
            // MTLCommandQueue and derive its MTLDevice. Both outlive `self`
            // (mirrors `RdirectLive::from_wgpu_queue` exactly).
            let (device, mtl_queue) = unsafe {
                queue
                    .as_hal::<wgpu::hal::api::Metal>()
                    .map(|hal_queue| {
                        let raw = hal_queue.as_raw();
                        let mtl_queue: Retained<ProtocolObject<dyn MTLCommandQueue>> =
                            raw.retain();
                        let device = mtl_queue.device();
                        (device, mtl_queue)
                    })
                    .ok_or_else(|| {
                        "rdirect_unet: wgpu is not on the Metal backend".to_string()
                    })?
            };
            let mut me = Self::build_from_weights(device.clone(), mtl_queue, weights)?;
            me.ensure_gpu_split()?;
            let n_in = me.config.input_elems();
            if n_in % 2 != 0 {
                return Err(format!(
                    "rdirect_unet: from_wgpu_queue needs an even input element count for fp16 packing (got {n_in}, in_channels={})",
                    me.config.in_channels
                ));
            }
            // SAFETY: objc2 message sends + the wgpu-hal Metal buffer
            // bridge, mirrors `RdirectLive::attach_pool`'s own MTLBuffer->
            // wgpu bridge exactly. `in_mtl` is Shared storage sized exactly
            // `n_in` fp16 elements (`ensure_gpu_split`, just ran above) —
            // `n_in/2` u32s hold that byte range exactly (n_in is even, just
            // checked). The wgpu buffer outlives nothing new: `in_mtl`
            // itself is already owned by `self.gpu`'s `GpuSplit`, retained
            // for the object's whole lifetime.
            unsafe {
                let gpu_ref = me.gpu.borrow();
                let gpu = gpu_ref.as_ref().expect("ensure_gpu_split just ran");
                let bytes = (n_in / 2) * std::mem::size_of::<u32>();
                // SPEED ROUND: bridge BOTH sets' own `in_mtl` — set 0 is the
                // pre-existing sync-path buffer (identical alloc/wrap to
                // before this change), set 1 is the async path's second slot.
                for set in 0usize..2 {
                    let hal_buf = wgpu::hal::metal::Device::buffer_from_raw(
                        gpu.sets[set].in_mtl.clone(),
                        bytes as u64,
                    );
                    let buf = wgpu_device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
                        hal_buf,
                        &wgpu::BufferDescriptor {
                            label: Some("rdirect_unet fp16 input (shared MTLBuffer)"),
                            size: bytes as u64,
                            usage: wgpu::BufferUsages::STORAGE,
                            mapped_at_creation: false,
                        },
                    );
                    me.feature_buf_u32[set] = Some(buf);
                }
                drop(gpu_ref);
            }
            Ok(me)
        }

        /// The gather-pack destination: `GpuSplit`'s fp16 input MTLBuffer
        /// (set 0, the sync path), wrapped as a wgpu STORAGE buffer of `u32`
        /// (2 packed fp16 halves each) — `Fp16Packer::encode`'s own `dst`.
        /// `None` unless built via `from_wgpu_queue`.
        pub fn feature_buf_u32(&self) -> Option<&wgpu::Buffer> {
            self.feature_buf_u32[0].as_ref()
        }

        /// SPEED ROUND: same as `feature_buf_u32()` but for an explicit
        /// buffer SET (0 or 1) — the async path's gather/pack destination
        /// alternates between the two so a still-in-flight `set`'s forward
        /// (on `net_queue`) never has its input MTLBuffer overwritten out
        /// from under it by the NEXT frame's pack.
        pub fn feature_buf_u32_set(&self, set: usize) -> Option<&wgpu::Buffer> {
            self.feature_buf_u32.get(set).and_then(|o| o.as_ref())
        }

        /// Same wgpu-queue-bridged shape as `RdirectLive::from_wgpu_queue`
        /// would offer, minus the zero-copy buffer bridge (not wired this
        /// atom — see the honest gap noted in docs/perf/2026-07-21-v9-wire.md).
        /// A straight structural mirror of `build`, one conv layer at a time,
        /// reading from `weights` in EXACTLY the order `cpu::UnetWeights::new_random`
        /// wrote them (stem -> per-scale down+down_refine -> per-scale-deepest-
        /// first up_reduce+up_merge -> optional out_refine -> head) instead of
        /// drawing from an RNG in that same order — so the byte layout the CPU
        /// twin trained is exactly the layout this graph's constants get.
        fn build_from_weights(
            device: Retained<ProtocolObject<dyn MTLDevice>>,
            queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
            weights: &super::cpu::UnetWeights,
        ) -> Result<Self, String> {
            let config = &weights.config;
            if config.n_scales == 0 {
                return Err("rdirect_unet: n_scales must be >= 1".to_string());
            }
            if config.widths.len() != config.n_scales {
                return Err(format!(
                    "rdirect_unet: widths.len()={} must equal n_scales={}",
                    config.widths.len(),
                    config.n_scales
                ));
            }
            let n_dec = config.n_scales.saturating_sub(1);
            if weights.down.len() != n_dec
                || weights.down_refine.len() != n_dec
                || weights.up_reduce.len() != n_dec
                || weights.up_merge.len() != n_dec
            {
                return Err(format!(
                    "rdirect_unet: weights layer-count mismatch for n_scales={} (down={} down_refine={} up_reduce={} up_merge={}, want {n_dec} each)",
                    config.n_scales, weights.down.len(), weights.down_refine.len(), weights.up_reduce.len(), weights.up_merge.len()
                ));
            }
            let wants_out_refine = (config.output_h, config.output_w) != (config.render_h, config.render_w);
            if wants_out_refine && weights.out_refine.is_none() {
                return Err("rdirect_unet: config wants output-res-free (output != render) but weights carry no out_refine layer".to_string());
            }
            if weights.stem.cin != config.in_channels || weights.stem.cout != config.widths[0] || weights.stem.k != config.kernel {
                return Err(format!(
                    "rdirect_unet: stem shape mismatch — weights cin={} cout={} k={}, config wants in_channels={} widths[0]={} kernel={}",
                    weights.stem.cin, weights.stem.cout, weights.stem.k, config.in_channels, config.widths[0], config.kernel
                ));
            }
            if weights.head.cin != config.widths[0] || weights.head.cout != config.out_channels {
                return Err(format!(
                    "rdirect_unet: head shape mismatch — weights cin={} cout={}, config wants widths[0]={} out_channels={}",
                    weights.head.cin, weights.head.cout, config.widths[0], config.out_channels
                ));
            }
            // SAFETY: objc2 message sends to live Metal + MPSGraph objects,
            // mirroring `build` exactly, minus the RNG.
            unsafe {
                let graph = MPSGraph::new();
                let mps_device = MPSGraphDevice::deviceWithMTLDevice(&device);

                let mut dims = Vec::with_capacity(config.n_scales);
                dims.push((config.render_h, config.render_w));
                for s in 1..config.n_scales {
                    let (ph, pw) = dims[s - 1];
                    dims.push(((ph + 1) / 2, (pw + 1) / 2));
                }

                let in_shape = shape(&[1, config.render_h, config.render_w, config.in_channels]);
                let input = graph.placeholderWithShape_dataType_name(
                    Some(&in_shape),
                    MPSDataType::Float16,
                    None,
                );

                let stem = conv_layer_from(&graph, &weights.stem);
                let mut x = conv2d(&graph, &input, &stem, 1);
                x = graph.reLUWithTensor_name(&x, None);
                let mut skips: Vec<Retained<MPSGraphTensor>> = vec![x.clone()];

                for s in 1..config.n_scales {
                    let down = conv_layer_from(&graph, &weights.down[s - 1]);
                    x = conv2d(&graph, &x, &down, 2);
                    x = graph.reLUWithTensor_name(&x, None);
                    let refine = conv_layer_from(&graph, &weights.down_refine[s - 1]);
                    x = conv2d(&graph, &x, &refine, 1);
                    x = graph.reLUWithTensor_name(&x, None);
                    skips.push(x.clone());
                }

                let mut dec = skips[config.n_scales - 1].clone();
                for (i, s) in (0..config.n_scales - 1).rev().enumerate() {
                    let (th, tw) = dims[s];
                    let up = resize_bilinear(&graph, &dec, th, tw);
                    let reduce = conv_layer_from(&graph, &weights.up_reduce[i]);
                    let mut up2 = conv2d(&graph, &up, &reduce, 1);
                    up2 = graph.reLUWithTensor_name(&up2, None);
                    let cat = graph.concatTensor_withTensor_dimension_name(&up2, &skips[s], 3, None);
                    let merge = conv_layer_from(&graph, &weights.up_merge[i]);
                    dec = conv2d(&graph, &cat, &merge, 1);
                    dec = graph.reLUWithTensor_name(&dec, None);
                }

                let mut final_feat = dec;
                if wants_out_refine {
                    final_feat = resize_bilinear(&graph, &final_feat, config.output_h, config.output_w);
                    let refine_out = conv_layer_from(&graph, weights.out_refine.as_ref().unwrap());
                    final_feat = conv2d(&graph, &final_feat, &refine_out, 1);
                    final_feat = graph.reLUWithTensor_name(&final_feat, None);
                }

                let head = conv_layer_from(&graph, &weights.head);
                let output = conv2d(&graph, &final_feat, &head, 1);

                let net_queue = device
                    .newCommandQueue()
                    .ok_or_else(|| "rdirect_unet: net_queue alloc failed".to_string())?;
                Ok(Self {
                    graph,
                    mps_device,
                    device,
                    queue,
                    net_queue,
                    input,
                    output,
                    config: config.clone(),
                    gpu: RefCell::new(None),
                    feature_buf_u32: [None, None],
                    last_encode_commit_ms: Cell::new(0.0),
                    last_wait_ms: Cell::new(0.0),
                    last_readout_ms: Cell::new(0.0),
                })
            }
        }

        /// Lazily compile the executable + allocate the fixed MTLBuffers the
        /// GPU-only split needs (once per instance). Mirrors
        /// `RdirectLive::attach_pool`'s compile call, minus the wgpu bridge.
        fn ensure_gpu_split(&self) -> Result<(), String> {
            if self.gpu.borrow().is_some() {
                return Ok(());
            }
            let in_shape = shape(&[
                1,
                self.config.render_h,
                self.config.render_w,
                self.config.in_channels,
            ]);
            let out_shape = shape(&[1, self.config.output_h, self.config.output_w, self.config.out_channels]);
            // SAFETY: objc2 message sends; buffer sizes match the shapes exactly.
            unsafe {
                let in_shaped = MPSGraphShapedType::initWithShape_dataType(
                    MPSGraphShapedType::alloc(),
                    Some(&in_shape),
                    MPSDataType::Float16,
                );
                let feeds = objc2_foundation::NSDictionary::<MPSGraphTensor, MPSGraphShapedType>::from_slices(
                    &[&*self.input],
                    &[&*in_shaped],
                );
                let targets = NSArray::from_slice(&[&*self.output]);
                let executable = self
                    .graph
                    .compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                        Some(&self.mps_device),
                        &feeds,
                        &targets,
                        None,
                        None,
                    );

                let in_bytes = self.config.input_elems() * std::mem::size_of::<half::f16>();
                let out_bytes = self.config.output_elems() * std::mem::size_of::<half::f16>();
                // SPEED ROUND: build TWO independent buffer sets (same shapes,
                // same alloc order per set as the old single-set code) so the
                // async path can have one set in flight while the other is
                // being freshly packed. Set 0's construction order is byte-
                // for-byte the old code's own order (sync path untouched).
                let build_set = |dev: &ProtocolObject<dyn MTLDevice>| -> Result<GpuBufSet, String> {
                    let in_mtl = dev
                        .newBufferWithLength_options(in_bytes, MTLResourceOptions::StorageModeShared)
                        .ok_or_else(|| "rdirect_unet: input MTLBuffer alloc failed".to_string())?;
                    let out_mtl = dev
                        .newBufferWithLength_options(out_bytes, MTLResourceOptions::StorageModeShared)
                        .ok_or_else(|| "rdirect_unet: output MTLBuffer alloc failed".to_string())?;
                    let in_td = MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                        MPSGraphTensorData::alloc(),
                        &in_mtl,
                        &in_shape,
                        MPSDataType::Float16,
                    );
                    let out_td = MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                        MPSGraphTensorData::alloc(),
                        &out_mtl,
                        &out_shape,
                        MPSDataType::Float16,
                    );
                    Ok(GpuBufSet { in_mtl, out_mtl, in_td, out_td })
                };
                let set0 = build_set(&self.device)?;
                let set1 = build_set(&self.device)?;
                *self.gpu.borrow_mut() = Some(GpuSplit { executable, sets: [set0, set1] });
            }
            Ok(())
        }

        /// N0e-PATTERN GPU-ONLY forward: writes `features_f32` into the fixed
        /// input MTLBuffer, encodes the COMPILED executable onto its OWN fresh
        /// command buffer (no queue-shared per-frame graph build), commits +
        /// waits, and returns `MTLCommandBuffer` `GPUEndTime - GPUStartTime`
        /// (ms) — the number `runWithMTLCommandQueue` (`forward_cpu_roundtrip`)
        /// hides behind its own CPU-side encode/wait cost.
        pub fn forward_gpu_ms(&self, features_f32: &[f32]) -> Result<f64, String> {
            self.ensure_gpu_split()?;
            let expected_in = self.config.input_elems();
            if features_f32.len() != expected_in {
                return Err(format!(
                    "rdirect_unet: input len {} != expected {}",
                    features_f32.len(),
                    expected_in
                ));
            }
            let gpu_ref = self.gpu.borrow();
            let gpu = gpu_ref.as_ref().expect("ensure_gpu_split just ran");
            let s0 = &gpu.sets[0];
            // SAFETY: objc2 message sends; `in_mtl` is Shared storage, sized
            // exactly to `expected_in` fp16 elements (allocated in
            // `ensure_gpu_split`), and no GPU work reads it concurrently (this
            // fn commits+waits before returning — no overlap with a prior call).
            unsafe {
                let ptr = s0.in_mtl.contents().as_ptr() as *mut half::f16;
                for (i, &v) in features_f32.iter().enumerate() {
                    *ptr.add(i) = half::f16::from_f32(v);
                }

                let base = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| "rdirect_unet: commandBuffer alloc failed".to_string())?;
                let mps = MPSCommandBuffer::commandBufferWithCommandBuffer(&base);
                let inputs = NSArray::from_slice(&[&*s0.in_td]);
                let results = NSArray::from_slice(&[&*s0.out_td]);
                let _ = gpu
                    .executable
                    .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                        &mps,
                        &inputs,
                        Some(&results),
                        None,
                    );
                mps.commit();
                mps.rootCommandBuffer().waitUntilCompleted();
                Ok((base.GPUEndTime() - base.GPUStartTime()) * 1000.0)
            }
        }

        /// V9-WIRE ATOM addition: SAME persistent-executable / fixed-buffer
        /// path `forward_gpu_ms` uses (no per-call graph recompile — sidesteps
        /// the ~34ms CPU encode/wait glue `forward_cpu_roundtrip`'s own
        /// `runWithMTLCommandQueue_feeds_targetTensors_targetOperations` call
        /// pays every invocation, per that fn's own doc), but ALSO reads back
        /// the output buffer (fp16->f32) instead of discarding it —
        /// `forward_gpu_ms` stays timing-only and untouched for its existing
        /// callers (the parity test, the spike). Returns `(wall_ms, output)`,
        /// `wall_ms` = the SAME `MTLCommandBuffer` GPUEndTime-GPUStartTime
        /// `forward_gpu_ms` reports.
        pub fn forward_gpu_with_output(&self, features_f32: &[f32]) -> Result<(f64, Vec<f32>), String> {
            self.ensure_gpu_split()?;
            let expected_in = self.config.input_elems();
            if features_f32.len() != expected_in {
                return Err(format!(
                    "rdirect_unet: input len {} != expected {}",
                    features_f32.len(),
                    expected_in
                ));
            }
            let gpu_ref = self.gpu.borrow();
            let gpu = gpu_ref.as_ref().expect("ensure_gpu_split just ran");
            // SAFETY: objc2 message sends; `in_mtl` is Shared storage, sized
            // exactly for `expected_in` fp16 elements (allocated in
            // `ensure_gpu_split`), and no GPU work reads it concurrently
            // (nothing has been submitted against it since the previous
            // forward's own wait completed).
            unsafe {
                let ptr = gpu.sets[0].in_mtl.contents().as_ptr() as *mut half::f16;
                for (i, &v) in features_f32.iter().enumerate() {
                    *ptr.add(i) = half::f16::from_f32(v);
                }
                self.run_compiled_forward(gpu, 0)
            }
        }

        /// V9-WIRE ATOM (GATHER FOLLOW-UP #2) — THE ZERO-COPY FORWARD: same
        /// persistent-compiled-executable path `forward_gpu_with_output`
        /// uses, MINUS the host f32->fp16 input-write loop — the caller has
        /// ALREADY written `feature_buf_u32()`'s backing MTLBuffer directly
        /// (a `Fp16Packer::encode` dispatch on the SAME wgpu queue this
        /// `UnetLive` was built from via `from_wgpu_queue`, `queue.submit()`-
        /// ed before this call — see `from_wgpu_queue`'s own synchronization
        /// doc for why no explicit fence is needed). Returns `(wall_ms,
        /// output)` exactly like `forward_gpu_with_output` — `wall_ms` is
        /// still GPU-only (`MTLCommandBuffer` GPUEndTime-GPUStartTime); the
        /// output readback (f16->f32, `output_elems()` — ~14x smaller than
        /// the input tensor this atom targets) is unchanged, not this atom's
        /// bottleneck.
        pub fn forward_gpu_bridged(&self) -> Result<(f64, Vec<f32>), String> {
            if self.feature_buf_u32[0].is_none() {
                return Err(
                    "rdirect_unet: forward_gpu_bridged needs from_wgpu_queue's bridge (feature_buf_u32 is None)"
                        .to_string(),
                );
            }
            self.ensure_gpu_split()?;
            let gpu_ref = self.gpu.borrow();
            let gpu = gpu_ref.as_ref().expect("ensure_gpu_split just ran");
            // SAFETY: the input MTLBuffer this forward reads (`gpu.sets[0].in_mtl`)
            // is the SAME MTLBuffer `feature_buf_u32()` wraps — the caller's
            // wgpu pack dispatch already wrote it, on the SAME MTLCommandQueue
            // `self.queue` is, submitted (and thus ordered) before this call
            // per Metal's commit-order guarantee (see `from_wgpu_queue`'s doc).
            unsafe { self.run_compiled_forward(gpu, 0) }
        }

        /// Shared tail of `forward_gpu_with_output`/`forward_gpu_bridged`:
        /// encode the compiled executable onto a fresh command buffer,
        /// commit+wait, read GPU-only ms + the f16->f32 output. Callers are
        /// responsible for the input MTLBuffer already holding this call's
        /// features (written on the CPU or bridged from wgpu). ALWAYS on
        /// `self.queue` (the sync/set-0 path) — the async path below has its
        /// OWN split submit/wait pair on `self.net_queue` instead.
        ///
        /// SAFETY: objc2 message sends; `gpu.sets[set].in_td`/`out_td` wrap
        /// `in_mtl`/`out_mtl` (Shared storage, sized exactly for
        /// `input_elems`/`output_elems` fp16 elements, `ensure_gpu_split`),
        /// and no GPU work reads/writes them concurrently (this fn
        /// commits+waits before reading `out_mtl`).
        unsafe fn run_compiled_forward(&self, gpu: &GpuSplit, set: usize) -> Result<(f64, Vec<f32>), String> {
            unsafe {
                let s = &gpu.sets[set];
                let t_a = std::time::Instant::now();
                let base = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| "rdirect_unet: commandBuffer alloc failed".to_string())?;
                let mps = MPSCommandBuffer::commandBufferWithCommandBuffer(&base);
                let inputs = NSArray::from_slice(&[&*s.in_td]);
                let results = NSArray::from_slice(&[&*s.out_td]);
                let _ = gpu
                    .executable
                    .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                        &mps,
                        &inputs,
                        Some(&results),
                        None,
                    );
                mps.commit();
                self.last_encode_commit_ms.set(t_a.elapsed().as_secs_f64() * 1000.0);

                let t_b = std::time::Instant::now();
                mps.rootCommandBuffer().waitUntilCompleted();
                self.last_wait_ms.set(t_b.elapsed().as_secs_f64() * 1000.0);
                let wall_ms = (base.GPUEndTime() - base.GPUStartTime()) * 1000.0;

                let t_c = std::time::Instant::now();
                let n_out = self.config.output_elems();
                let out_ptr = s.out_mtl.contents().as_ptr() as *const half::f16;
                let mut out = vec![0.0f32; n_out];
                for (i, o) in out.iter_mut().enumerate() {
                    *o = (*out_ptr.add(i)).to_f32();
                }
                self.last_readout_ms.set(t_c.elapsed().as_secs_f64() * 1000.0);
                Ok((wall_ms, out))
            }
        }

        /// V9-WIRE SPEED ROUND (2026-07-25) LEVER 1+2 — ASYNC SUBMIT: encode
        /// + commit the compiled executable for buffer `set` onto a FRESH
        /// command buffer on the DEDICATED `net_queue`, and return
        /// immediately WITHOUT waiting (`PendingForward`). The caller has
        /// already written `feature_buf_u32_set(set)`'s backing MTLBuffer
        /// (a wgpu pack dispatch, `device.poll(wait_indefinitely())`-
        /// confirmed complete before this call — see `from_wgpu_queue`'s
        /// updated synchronization doc for why that CPU-observed completion
        /// is sufficient even across the queue boundary, no event needed).
        /// Pair with `wait_gpu_bridged_async` to get the GPU-only ms + output.
        pub fn submit_gpu_bridged_async(&self, set: usize) -> Result<PendingForward, String> {
            if self.feature_buf_u32.get(set).and_then(|o| o.as_ref()).is_none() {
                return Err(format!(
                    "rdirect_unet: submit_gpu_bridged_async needs from_wgpu_queue's bridge for set {set}"
                ));
            }
            self.ensure_gpu_split()?;
            let gpu_ref = self.gpu.borrow();
            let gpu = gpu_ref.as_ref().expect("ensure_gpu_split just ran");
            let s = &gpu.sets[set];
            // SAFETY: objc2 message sends; `net_queue` is a distinct
            // MTLCommandQueue off the SAME MTLDevice, `s.in_td`/`out_td` wrap
            // `set`'s own fixed MTLBuffers (sized exactly, `ensure_gpu_split`).
            // The caller's contract (doc above) makes the input write visible
            // before this commits; nothing else touches `set`'s buffers until
            // `wait_gpu_bridged_async` reads `out_mtl` after waiting.
            unsafe {
                let t_a = std::time::Instant::now();
                let base = self
                    .net_queue
                    .commandBuffer()
                    .ok_or_else(|| "rdirect_unet: commandBuffer alloc failed (async)".to_string())?;
                let mps = MPSCommandBuffer::commandBufferWithCommandBuffer(&base);
                let inputs = NSArray::from_slice(&[&*s.in_td]);
                let results = NSArray::from_slice(&[&*s.out_td]);
                let _ = gpu
                    .executable
                    .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                        &mps,
                        &inputs,
                        Some(&results),
                        None,
                    );
                mps.commit();
                self.last_encode_commit_ms.set(t_a.elapsed().as_secs_f64() * 1000.0);
                Ok(PendingForward { base, mps, set })
            }
        }

        /// V9-WIRE SPEED ROUND — ASYNC WAIT: block until `pending`'s command
        /// buffer completes, then read GPU-only ms (`GPUEndTime-GPUStartTime`)
        /// + the f16->f32 output from ITS OWN set's `out_mtl`. Mirrors
        /// `run_compiled_forward`'s wait+readout tail exactly (same math, same
        /// probes) — numeric output is bit-identical to what the sync path
        /// would produce for the SAME input, since the graph/executable and
        /// its math are untouched; only the submission scheduling differs.
        pub fn wait_gpu_bridged_async(&self, pending: PendingForward) -> Result<(f64, Vec<f32>), String> {
            let gpu_ref = self.gpu.borrow();
            let gpu = gpu_ref.as_ref().expect("submit_gpu_bridged_async already ran ensure_gpu_split");
            let s = &gpu.sets[pending.set];
            // SAFETY: objc2 message sends; `pending` owns the command buffer
            // this waits+reads, its `set`'s `out_mtl` is not touched by any
            // other in-flight work (the caller alternates sets so the OTHER
            // set is the one being freshly packed while this one is pending).
            unsafe {
                let t_b = std::time::Instant::now();
                pending.mps.rootCommandBuffer().waitUntilCompleted();
                self.last_wait_ms.set(t_b.elapsed().as_secs_f64() * 1000.0);
                let wall_ms = (pending.base.GPUEndTime() - pending.base.GPUStartTime()) * 1000.0;

                let t_c = std::time::Instant::now();
                let n_out = self.config.output_elems();
                let out_ptr = s.out_mtl.contents().as_ptr() as *const half::f16;
                let mut out = vec![0.0f32; n_out];
                for (i, o) in out.iter_mut().enumerate() {
                    *o = (*out_ptr.add(i)).to_f32();
                }
                self.last_readout_ms.set(t_c.elapsed().as_secs_f64() * 1000.0);
                Ok((wall_ms, out))
            }
        }

        /// V9-WIRE ATOM (GATHER FOLLOW-UP #2) probe getters — CPU ms of the
        /// LAST `run_compiled_forward` call's three phases (encode+commit,
        /// wait, output readout). Mirrors `rdirect_live.rs`'s own
        /// `last_commit_ms`/`last_wait_ms` N0.i S13 probe pattern.
        pub fn last_encode_commit_ms(&self) -> f64 {
            self.last_encode_commit_ms.get()
        }
        pub fn last_wait_ms(&self) -> f64 {
            self.last_wait_ms.get()
        }
        pub fn last_readout_ms(&self) -> f64 {
            self.last_readout_ms.get()
        }

        /// CPU-staged forward (the honest bring-up path, mirrors
        /// `RdirectLive::forward_cpu_roundtrip`): host f32 features in
        /// (converted to fp16 for the graph), fp32 read back out. This is
        /// the wall the spike measures — the same
        /// `runWithMTLCommandQueue_feeds_targetTensors_targetOperations`
        /// call the live path's bring-up stage used before its own S1/S2
        /// zero-copy optimizations (which stage 2+ of this lane would add).
        pub fn forward_cpu_roundtrip(&self, features_f32: &[f32]) -> Result<Vec<f32>, String> {
            let expected_in = self.config.input_elems();
            if features_f32.len() != expected_in {
                return Err(format!(
                    "rdirect_unet: input len {} != expected {}",
                    features_f32.len(),
                    expected_in
                ));
            }
            let in_fp16: Vec<f16> = features_f32.iter().map(|&v| f16::from_f32(v)).collect();
            // SAFETY: objc2 message sends; buffers sized exactly to the shapes.
            unsafe {
                let bytes = std::slice::from_raw_parts(
                    in_fp16.as_ptr() as *const u8,
                    std::mem::size_of_val(in_fp16.as_slice()),
                );
                let data = NSData::with_bytes(bytes);
                let in_shape = shape(&[
                    1,
                    self.config.render_h,
                    self.config.render_w,
                    self.config.in_channels,
                ]);
                let input_data = MPSGraphTensorData::initWithDevice_data_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    &self.mps_device,
                    &data,
                    &in_shape,
                    MPSDataType::Float16,
                );

                let feeds = objc2_foundation::NSDictionary::<MPSGraphTensor, MPSGraphTensorData>::from_slices(
                    &[&*self.input],
                    &[&*input_data],
                );
                let targets = NSArray::from_slice(&[&*self.output]);

                let results = self
                    .graph
                    .runWithMTLCommandQueue_feeds_targetTensors_targetOperations(
                        &self.queue,
                        &feeds,
                        &targets,
                        None,
                    );
                let out_td = results
                    .objectForKey(&self.output)
                    .ok_or_else(|| "rdirect_unet: no result for output tensor".to_string())?;

                let ndarray = out_td.mpsndarray();
                let n_out = self.config.output_elems();
                let mut out_fp16 = vec![f16::from_f32(0.0); n_out];
                ndarray.readBytes_strideBytes(
                    NonNull::new(out_fp16.as_mut_ptr() as *mut c_void).unwrap(),
                    std::ptr::null_mut(),
                );
                Ok(out_fp16.iter().map(|v: &f16| v.to_f32()).collect())
            }
        }
    }

    /// Build one conv layer's He-init fp16 weight+bias constants.
    /// `kernel=1` legally builds a 1×1 (the output head).
    fn conv_layer(
        graph: &MPSGraph,
        cin: usize,
        cout: usize,
        kernel: usize,
        rng: &mut SplitMix64,
    ) -> ConvLayer {
        let fan_in = (kernel * kernel * cin).max(1);
        let scale = (2.0 / fan_in as f32).sqrt();
        let mut w = vec![f16::from_f32(0.0); kernel * kernel * cin * cout];
        for v in w.iter_mut() {
            *v = f16::from_f32(rng.next_signed_unit() * scale);
        }
        let b = vec![f16::from_f32(0.0); cout]; // zero-init bias (standard)
        let weight = constant_fp16(graph, &w, &[kernel, kernel, cin, cout]);
        let bias = constant_fp16(graph, &b, &[1, 1, 1, cout]);
        ConvLayer { weight, bias }
    }

    /// Build one conv layer's constants from a TRAINED `super::cpu::ConvParam`
    /// (f32 host weights, byte-compatible HWIO flatten order
    /// `((ky*k+kx)*cin+ci)*cout+co` — identical to `conv_layer`'s own RNG-fill
    /// order above), fp16-converted for the graph. Used by `build_from_weights`.
    fn conv_layer_from(graph: &MPSGraph, p: &super::cpu::ConvParam) -> ConvLayer {
        let w: Vec<f16> = p.w.iter().map(|&v| f16::from_f32(v)).collect();
        let b: Vec<f16> = p.b.iter().map(|&v| f16::from_f32(v)).collect();
        let weight = constant_fp16(graph, &w, &[p.k, p.k, p.cin, p.cout]);
        let bias = constant_fp16(graph, &b, &[1, 1, 1, p.cout]);
        ConvLayer { weight, bias }
    }

    /// 2D conv (NHWC source, HWIO weights), TF_SAME padding, groups=1, plus
    /// the bias add. ReLU (if any) is applied by the caller — the last layer
    /// of every block stays linear here so callers can choose.
    fn conv2d(
        graph: &MPSGraph,
        x: &MPSGraphTensor,
        layer: &ConvLayer,
        stride: usize,
    ) -> Retained<MPSGraphTensor> {
        // SAFETY: objc2 message sends; descriptor params are plain integers.
        unsafe {
            let desc = MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingStyle_dataLayout_weightsLayout(
                stride as _,
                stride as _,
                1,
                1,
                1,
                MPSGraphPaddingStyle::TF_SAME,
                MPSGraphTensorNamedDataLayout::NHWC,
                MPSGraphTensorNamedDataLayout::HWIO,
            ).expect("rdirect_unet: conv2d descriptor build failed");
            let conv = graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                x,
                &layer.weight,
                &desc,
                None,
            );
            graph.additionWithPrimaryTensor_secondaryTensor_name(&conv, &layer.bias, None)
        }
    }

    /// Bilinear resize to an explicit (h, w) — the interior upsample step
    /// (decoder scales, and the free-output-res refine). Never the last op
    /// in the graph on its own (see module docs / `build`'s output-res arm).
    fn resize_bilinear(
        graph: &MPSGraph,
        x: &MPSGraphTensor,
        h: usize,
        w: usize,
    ) -> Retained<MPSGraphTensor> {
        // SAFETY: objc2 message sends; size data is exactly 2 Int32s.
        unsafe {
            let size_vals: [i32; 2] = [h as i32, w as i32];
            let bytes = std::slice::from_raw_parts(
                size_vals.as_ptr() as *const u8,
                std::mem::size_of_val(&size_vals),
            );
            let data = NSData::with_bytes(bytes);
            let size_shape = shape(&[2]);
            let size_tensor =
                graph.constantWithData_shape_dataType(&data, &size_shape, MPSDataType::Int32);
            graph.resizeBilinearWithTensor_sizeTensor_centerResult_alignCorners_layout_name(
                x,
                &size_tensor,
                true,
                false,
                MPSGraphTensorNamedDataLayout::NHWC,
                None,
            )
        }
    }

    /// Build an `MPSShape` (NSArray<NSNumber>) from plain dims (no dynamic
    /// sentinel here — every forward is batch=1, fixed spatial dims, unlike
    /// `rdirect_live.rs`'s dynamic-N placeholder).
    fn shape(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
        let numbers: Vec<Retained<NSNumber>> =
            dims.iter().map(|&d| NSNumber::new_isize(d as isize)).collect();
        let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
        NSArray::from_slice(&refs)
    }

    /// A graph constant tensor from an fp16 slice with an explicit shape.
    fn constant_fp16(graph: &MPSGraph, values: &[f16], dims: &[usize]) -> Retained<MPSGraphTensor> {
        // SAFETY: bytes sized to values; shape product == values.len().
        unsafe {
            let bytes = std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values),
            );
            let data = NSData::with_bytes(bytes);
            let sh = shape(dims);
            graph.constantWithData_shape_dataType(&data, &sh, MPSDataType::Float16)
        }
    }

    // Silence unused-import lint noise on Message/NSString (kept for parity
    // with rdirect_live.rs's op-naming plumbing, used if `name` args grow).
    #[allow(unused)]
    fn _touch(_: &NSString) {}
    #[allow(unused)]
    fn _touch_msg<T: Message>(_: &T) {}
}

// ─────────────────────────────────────────────────────────────────────────
// STAGE 2 — CPU-trainable mirror of the shape above (`imp::UnetLive`'s
// MPSGraph graph, macOS-only). Pure Rust, cross-platform, no objc2: the
// house pattern for this whole codebase keeps TRAINING on a hand-rolled
// CPU forward/backward (see `rdirect.rs::Mlp` — the live MPSGraph net has
// no gradient path; `Adam`/`accumulate_backward_clamped_slice` train a
// byte-identical-shape CPU twin whose weights the live graph then loads).
// This module is that twin for the conv U-Net body: same `UnetConfig`,
// same HWIO weight layout (`(ky*k+kx)*cin+ci)*cout+co`), so a checkpoint
// trained here is byte-compatible with what `imp::UnetLive` would need to
// load (loading is NOT wired this stage — flagged in the spike doc/lane
// note, a disclosed stage-3+ gap, not hidden).
//
// Operates on whole PATCHES (`Img`, HWC f32), not per-pixel samples like
// `rdirect.rs`'s MLP trainer — convolution needs the spatial neighbourhood,
// so there is no equivalent of "subsample 5000px/pose" here. Training
// resolution is just `UnetConfig::render_w/h` set smaller for CPU wall
// budget (conv weights are resolution-independent — a checkpoint trained
// at 128×96 forwards correctly at 640×480, standard fully-convolutional
// design); the trainer (`examples/rdirect_train_v9.rs`) sets this.
// ─────────────────────────────────────────────────────────────────────────
pub mod cpu {
    use super::UnetConfig;

    /// Deterministic dependency-free PRNG (SplitMix64) — weight INIT only.
    /// Duplicated per-module by house convention (see `rdirect.rs` et al.).
    struct SplitMix64 {
        state: u64,
    }
    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_signed_unit(&mut self) -> f32 {
            let bits = (self.next_u64() >> 40) as u32;
            let unit = (bits as f32) / (1u32 << 24) as f32;
            unit * 2.0 - 1.0
        }
    }

    /// A single image/feature-map buffer, HWC layout (matches the MPSGraph
    /// side's NHWC with batch=1 dropped).
    #[derive(Clone)]
    pub struct Img {
        pub h: usize,
        pub w: usize,
        pub c: usize,
        pub data: Vec<f32>,
    }
    impl Img {
        pub fn zeros(h: usize, w: usize, c: usize) -> Self {
            Self { h, w, c, data: vec![0.0; h * w * c] }
        }
        #[inline]
        pub fn at(&self, y: usize, x: usize, ch: usize) -> f32 {
            self.data[(y * self.w + x) * self.c + ch]
        }
        #[inline]
        pub fn set(&mut self, y: usize, x: usize, ch: usize, v: f32) {
            self.data[(y * self.w + x) * self.c + ch] = v;
        }
        #[inline]
        pub fn add(&mut self, y: usize, x: usize, ch: usize, v: f32) {
            self.data[(y * self.w + x) * self.c + ch] += v;
        }
    }

    /// SAME-padded 2D conv output size (matches `MPSGraphPaddingStyle::TF_SAME`,
    /// the style `imp::conv2d` uses): ceil(in/stride).
    fn same_out_dim(in_dim: usize, stride: usize) -> usize {
        (in_dim + stride - 1) / stride
    }
    /// SAME padding split (TF convention: extra pixel goes on the "after" side).
    fn same_pad(in_dim: usize, out_dim: usize, k: usize, stride: usize) -> (i64, i64) {
        let total = ((out_dim as i64 - 1) * stride as i64 + k as i64 - in_dim as i64).max(0);
        let before = total / 2;
        (before, total - before)
    }

    /// One conv layer's learned parameters (HWIO weights — byte-compatible
    /// with `imp`'s constant layout).
    #[derive(Clone)]
    pub struct ConvParam {
        pub cin: usize,
        pub cout: usize,
        pub k: usize,
        pub stride: usize,
        pub w: Vec<f32>, // [k,k,cin,cout]
        pub b: Vec<f32>, // [cout]
    }
    impl ConvParam {
        fn new_random(cin: usize, cout: usize, k: usize, stride: usize, rng: &mut SplitMix64) -> Self {
            let fan_in = (k * k * cin).max(1);
            let scale = (2.0 / fan_in as f32).sqrt();
            let mut w = vec![0.0f32; k * k * cin * cout];
            for v in w.iter_mut() {
                *v = rng.next_signed_unit() * scale;
            }
            Self { cin, cout, k, stride, w, b: vec![0.0f32; cout] }
        }
        fn zeros_like(&self) -> Self {
            Self {
                cin: self.cin,
                cout: self.cout,
                k: self.k,
                stride: self.stride,
                w: vec![0.0; self.w.len()],
                b: vec![0.0; self.b.len()],
            }
        }
        #[inline]
        fn widx(&self, ky: usize, kx: usize, ci: usize, co: usize) -> usize {
            ((ky * self.k + kx) * self.cin + ci) * self.cout + co
        }

        /// Forward: SAME-padded conv, plus bias. Linear (no activation) —
        /// callers apply ReLU separately (`relu_forward`/`relu_backward`).
        fn forward(&self, x: &Img) -> Img {
            let oh = same_out_dim(x.h, self.stride);
            let ow = same_out_dim(x.w, self.stride);
            let (pad_top, _pad_bot) = same_pad(x.h, oh, self.k, self.stride);
            let (pad_left, _pad_right) = same_pad(x.w, ow, self.k, self.stride);
            let mut out = Img::zeros(oh, ow, self.cout);
            for oy in 0..oh {
                for ox in 0..ow {
                    for co in 0..self.cout {
                        let mut sum = self.b[co];
                        for ky in 0..self.k {
                            let iy = oy as i64 * self.stride as i64 + ky as i64 - pad_top;
                            if iy < 0 || iy >= x.h as i64 {
                                continue;
                            }
                            for kx in 0..self.k {
                                let ix = ox as i64 * self.stride as i64 + kx as i64 - pad_left;
                                if ix < 0 || ix >= x.w as i64 {
                                    continue;
                                }
                                let (iy, ix) = (iy as usize, ix as usize);
                                for ci in 0..self.cin {
                                    sum += self.w[self.widx(ky, kx, ci, co)] * x.at(iy, ix, ci);
                                }
                            }
                        }
                        out.set(oy, ox, co, sum);
                    }
                }
            }
            out
        }

        /// Backward: `dy` (grad wrt this layer's output) + the FORWARD input
        /// `x` -> (`dx`, weight/bias grads accumulated into `gw`/`gb`, which
        /// the caller zero-inits per batch and Adam-steps after).
        fn backward(&self, x: &Img, dy: &Img, gw: &mut [f32], gb: &mut [f32]) -> Img {
            let (oh, ow) = (dy.h, dy.w);
            let (pad_top, _) = same_pad(x.h, oh, self.k, self.stride);
            let (pad_left, _) = same_pad(x.w, ow, self.k, self.stride);
            let mut dx = Img::zeros(x.h, x.w, x.c);
            for oy in 0..oh {
                for ox in 0..ow {
                    for co in 0..self.cout {
                        let g = dy.at(oy, ox, co);
                        if g == 0.0 {
                            continue;
                        }
                        gb[co] += g;
                        for ky in 0..self.k {
                            let iy = oy as i64 * self.stride as i64 + ky as i64 - pad_top;
                            if iy < 0 || iy >= x.h as i64 {
                                continue;
                            }
                            for kx in 0..self.k {
                                let ix = ox as i64 * self.stride as i64 + kx as i64 - pad_left;
                                if ix < 0 || ix >= x.w as i64 {
                                    continue;
                                }
                                let (iy, ix) = (iy as usize, ix as usize);
                                for ci in 0..self.cin {
                                    let wi = self.widx(ky, kx, ci, co);
                                    gw[wi] += x.at(iy, ix, ci) * g;
                                    dx.add(iy, ix, ci, self.w[wi] * g);
                                }
                            }
                        }
                    }
                }
            }
            dx
        }
    }

    fn relu_forward(x: &Img) -> Img {
        let mut out = x.clone();
        for v in out.data.iter_mut() {
            *v = v.max(0.0);
        }
        out
    }
    /// `y` = the ALREADY-RELU'd forward output (mask = y > 0 — exact except
    /// at the measure-zero x==0 boundary).
    fn relu_backward(y: &Img, dy: &Img) -> Img {
        let mut out = dy.clone();
        for (o, &yv) in out.data.iter_mut().zip(&y.data) {
            if yv <= 0.0 {
                *o = 0.0;
            }
        }
        out
    }

    fn concat_forward(a: &Img, b: &Img) -> Img {
        assert_eq!((a.h, a.w), (b.h, b.w), "rdirect_unet::cpu concat: spatial size mismatch");
        let mut out = Img::zeros(a.h, a.w, a.c + b.c);
        for y in 0..a.h {
            for x in 0..a.w {
                for c in 0..a.c {
                    out.set(y, x, c, a.at(y, x, c));
                }
                for c in 0..b.c {
                    out.set(y, x, a.c + c, b.at(y, x, c));
                }
            }
        }
        out
    }
    fn concat_backward(dy: &Img, a_c: usize, b_c: usize) -> (Img, Img) {
        let mut da = Img::zeros(dy.h, dy.w, a_c);
        let mut db = Img::zeros(dy.h, dy.w, b_c);
        for y in 0..dy.h {
            for x in 0..dy.w {
                for c in 0..a_c {
                    da.set(y, x, c, dy.at(y, x, c));
                }
                for c in 0..b_c {
                    db.set(y, x, c, dy.at(y, x, a_c + c));
                }
            }
        }
        (da, db)
    }

    /// Bilinear resize to an explicit (h, w) — `centerResult=true,
    /// alignCorners=false`, the SAME convention `imp::resize_bilinear` asks
    /// MPSGraph for (its own doc: "same behavior as OpenCV's resize and
    /// TensorFlow V2's resize"), so a checkpoint trained against THIS
    /// implementation matches what the MPSGraph inference graph will do at
    /// forward time.
    fn resize_bilinear_forward(x: &Img, th: usize, tw: usize) -> Img {
        let mut out = Img::zeros(th, tw, x.c);
        let sy = x.h as f32 / th as f32;
        let sx = x.w as f32 / tw as f32;
        for oy in 0..th {
            let fy = (((oy as f32 + 0.5) * sy) - 0.5).clamp(0.0, (x.h.max(1) - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(x.h - 1);
            let wy = fy - y0 as f32;
            for ox in 0..tw {
                let fx = (((ox as f32 + 0.5) * sx) - 0.5).clamp(0.0, (x.w.max(1) - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(x.w - 1);
                let wx = fx - x0 as f32;
                for c in 0..x.c {
                    let v = (1.0 - wy) * (1.0 - wx) * x.at(y0, x0, c)
                        + (1.0 - wy) * wx * x.at(y0, x1, c)
                        + wy * (1.0 - wx) * x.at(y1, x0, c)
                        + wy * wx * x.at(y1, x1, c);
                    out.set(oy, ox, c, v);
                }
            }
        }
        out
    }
    fn resize_bilinear_backward(dy: &Img, src_h: usize, src_w: usize) -> Img {
        let mut dx = Img::zeros(src_h, src_w, dy.c);
        let sy = src_h as f32 / dy.h as f32;
        let sx = src_w as f32 / dy.w as f32;
        for oy in 0..dy.h {
            let fy = (((oy as f32 + 0.5) * sy) - 0.5).clamp(0.0, (src_h.max(1) - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let wy = fy - y0 as f32;
            for ox in 0..dy.w {
                let fx = (((ox as f32 + 0.5) * sx) - 0.5).clamp(0.0, (src_w.max(1) - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let wx = fx - x0 as f32;
                for c in 0..dy.c {
                    let g = dy.at(oy, ox, c);
                    dx.add(y0, x0, c, (1.0 - wy) * (1.0 - wx) * g);
                    dx.add(y0, x1, c, (1.0 - wy) * wx * g);
                    dx.add(y1, x0, c, wy * (1.0 - wx) * g);
                    dx.add(y1, x1, c, wy * wx * g);
                }
            }
        }
        dx
    }

    /// The full CPU-trainable twin of `imp::UnetLive`'s graph — same shape
    /// (`UnetConfig`), same op order (stem -> down+refine per scale ->
    /// bottleneck -> resize+reduce+concat+merge back down -> optional
    /// output-res resize+refine -> 1x1 head).
    #[derive(Clone)]
    pub struct UnetWeights {
        pub config: UnetConfig,
        pub stem: ConvParam,
        /// len == n_scales-1, encoder downsample (stride 2) per scale step.
        pub down: Vec<ConvParam>,
        /// len == n_scales-1, encoder refine (stride 1) per scale step.
        pub down_refine: Vec<ConvParam>,
        /// len == n_scales-1, decoder post-upsample channel-reduce, in
        /// DECODE ORDER (index 0 = deepest step, scale n_scales-2).
        pub up_reduce: Vec<ConvParam>,
        /// len == n_scales-1, decoder post-concat merge, same order as `up_reduce`.
        pub up_merge: Vec<ConvParam>,
        /// Some only when `config.output_h/w != config.render_h/w`.
        pub out_refine: Option<ConvParam>,
        pub head: ConvParam,
    }

    /// Everything `backward` needs that `forward` computed (cached
    /// activations, one Img per op whose output another op's backward reads).
    /// `skips[s]` = the relu'd feature at scale `s` (0 = full res .. n_scales-1
    /// = bottleneck) — cached directly (not recomputed via `relu_forward` on
    /// a pre-activation a second time) so `backward` is a straight replay.
    pub struct ForwardCache {
        input: Img,
        skips: Vec<Img>,
        /// per encoder step (index s-1, scale s=1..n_scales): (down_pre, down_relu, refine_pre)
        /// (`skips[s]` IS that step's refine_relu — not duplicated here).
        enc: Vec<(Img, Img, Img)>,
        /// per decode step, DEEPEST FIRST (index i=0 == scale n_scales-2's
        /// concat, matching `up_reduce`/`up_merge`'s own order):
        /// (resized, reduce_pre, reduce_relu, cat, merge_pre, merge_relu).
        dec: Vec<(Img, Img, Img, Img, Img, Img)>,
        /// source image the output-res resize reads (== last dec_out, or
        /// `skips[0]` if n_scales==1) — only `Some` when `out_refine` ran.
        out_pre_resize: Option<Img>,
        out_resized: Option<Img>,
        out_refine_relu: Option<Img>,
        head_input: Img,
    }

    impl UnetWeights {
        pub fn new_random(config: UnetConfig, seed: u64) -> Self {
            assert!(config.n_scales >= 1, "rdirect_unet::cpu: n_scales must be >= 1");
            assert_eq!(config.widths.len(), config.n_scales, "rdirect_unet::cpu: widths.len() must equal n_scales");
            let mut rng = SplitMix64::new(seed);
            let stem = ConvParam::new_random(config.in_channels, config.widths[0], config.kernel, 1, &mut rng);
            let mut down = Vec::with_capacity(config.n_scales.saturating_sub(1));
            let mut down_refine = Vec::with_capacity(config.n_scales.saturating_sub(1));
            for s in 1..config.n_scales {
                down.push(ConvParam::new_random(config.widths[s - 1], config.widths[s], config.kernel, 2, &mut rng));
                down_refine.push(ConvParam::new_random(config.widths[s], config.widths[s], config.kernel, 1, &mut rng));
            }
            let mut up_reduce = Vec::with_capacity(config.n_scales.saturating_sub(1));
            let mut up_merge = Vec::with_capacity(config.n_scales.saturating_sub(1));
            for s in (0..config.n_scales.saturating_sub(1)).rev() {
                let cout = config.widths[s];
                let cin_up = config.widths[s + 1];
                up_reduce.push(ConvParam::new_random(cin_up, cout, config.kernel, 1, &mut rng));
                up_merge.push(ConvParam::new_random(cout * 2, cout, config.kernel, 1, &mut rng));
            }
            let out_refine = if (config.output_h, config.output_w) != (config.render_h, config.render_w) {
                Some(ConvParam::new_random(config.widths[0], config.widths[0], config.kernel, 1, &mut rng))
            } else {
                None
            };
            let head = ConvParam::new_random(config.widths[0], config.out_channels, 1, 1, &mut rng);
            Self { config, stem, down, down_refine, up_reduce, up_merge, out_refine, head }
        }

        /// A zero-initialized, identically-shaped accumulator — the caller
        /// zeros one per batch, `backward`s every example into it, then
        /// `UnetAdam::step`s.
        pub fn zeros_like(&self) -> Self {
            Self {
                config: self.config.clone(),
                stem: self.stem.zeros_like(),
                down: self.down.iter().map(ConvParam::zeros_like).collect(),
                down_refine: self.down_refine.iter().map(ConvParam::zeros_like).collect(),
                up_reduce: self.up_reduce.iter().map(ConvParam::zeros_like).collect(),
                up_merge: self.up_merge.iter().map(ConvParam::zeros_like).collect(),
                out_refine: self.out_refine.as_ref().map(ConvParam::zeros_like),
                head: self.head.zeros_like(),
            }
        }

        /// EMA update (in place): `self = self*decay + other*(1-decay)`.
        /// Mirrors `rdirect.rs::Mlp::ema_update` — the frozen/smoothed
        /// history-source principle the v7d/v8 lineage established, applied
        /// here as the epoch-start snapshot the trainer uses for
        /// `history_forward` (see `examples/rdirect_train_v9.rs`).
        pub fn ema_update(&mut self, other: &UnetWeights, decay: f32) {
            fn ema_conv(a: &mut ConvParam, b: &ConvParam, decay: f32) {
                for (av, bv) in a.w.iter_mut().zip(&b.w) {
                    *av = *av * decay + *bv * (1.0 - decay);
                }
                for (av, bv) in a.b.iter_mut().zip(&b.b) {
                    *av = *av * decay + *bv * (1.0 - decay);
                }
            }
            ema_conv(&mut self.stem, &other.stem, decay);
            for (a, b) in self.down.iter_mut().zip(&other.down) {
                ema_conv(a, b, decay);
            }
            for (a, b) in self.down_refine.iter_mut().zip(&other.down_refine) {
                ema_conv(a, b, decay);
            }
            for (a, b) in self.up_reduce.iter_mut().zip(&other.up_reduce) {
                ema_conv(a, b, decay);
            }
            for (a, b) in self.up_merge.iter_mut().zip(&other.up_merge) {
                ema_conv(a, b, decay);
            }
            if let (Some(a), Some(b)) = (self.out_refine.as_mut(), other.out_refine.as_ref()) {
                ema_conv(a, b, decay);
            }
            ema_conv(&mut self.head, &other.head, decay);
        }

        /// Structural self-check: every conv layer's (cin,cout) matches what
        /// `other` (a CANDIDATE config, not necessarily `self.config`) says
        /// they must be — the loud-assert-on-load-mismatch pattern
        /// `forward`'s own `assert_eq!` on `in_channels` already has (see
        /// below), extended honestly to every dimension (widths per scale,
        /// n_scales-derived layer counts, out_channels, the optional
        /// out_refine's presence). Callers that overwrite a freshly-loaded
        /// checkpoint's `.config` with a DIFFERENT config (e.g. a dry-run
        /// harness pinning render/output res while keeping the checkpoint's
        /// own trained weights) must call this FIRST — a real mismatch
        /// (different `GAIA_V9_WIDTHS` between the checkpoint's training run
        /// and this run) would otherwise pair new shape metadata with
        /// old-shaped tensors and index-panic obscurely deep inside
        /// `forward`/`conv_layer` instead of here with a clear reason.
        pub fn validate_shapes_against(&self, other: &UnetConfig) -> Result<(), String> {
            if other.widths.len() != other.n_scales {
                return Err(format!("candidate config itself is inconsistent: widths.len()={} != n_scales={}", other.widths.len(), other.n_scales));
            }
            if self.stem.cin != other.in_channels || self.stem.cout != other.widths[0] {
                return Err(format!("stem {}x{} != in_channels={} widths[0]={}", self.stem.cin, self.stem.cout, other.in_channels, other.widths[0]));
            }
            let n_down = other.n_scales.saturating_sub(1);
            if self.down.len() != n_down || self.down_refine.len() != n_down || self.up_reduce.len() != n_down || self.up_merge.len() != n_down {
                return Err(format!(
                    "layer counts down={} down_refine={} up_reduce={} up_merge={} != expected n_scales-1={} (n_scales={})",
                    self.down.len(), self.down_refine.len(), self.up_reduce.len(), self.up_merge.len(), n_down, other.n_scales
                ));
            }
            for s in 1..other.n_scales {
                let i = s - 1;
                if self.down[i].cin != other.widths[s - 1] || self.down[i].cout != other.widths[s] {
                    return Err(format!("down[{i}] {}x{} != widths[{}]={} widths[{s}]={}", self.down[i].cin, self.down[i].cout, s - 1, other.widths[s - 1], other.widths[s]));
                }
                if self.down_refine[i].cin != other.widths[s] || self.down_refine[i].cout != other.widths[s] {
                    return Err(format!("down_refine[{i}] {}x{} != widths[{s}]={}", self.down_refine[i].cin, self.down_refine[i].cout, other.widths[s]));
                }
            }
            for (i, s) in (0..other.n_scales.saturating_sub(1)).rev().enumerate() {
                let cout = other.widths[s];
                let cin_up = other.widths[s + 1];
                if self.up_reduce[i].cin != cin_up || self.up_reduce[i].cout != cout {
                    return Err(format!("up_reduce[{i}] {}x{} != cin_up={cin_up} cout={cout}", self.up_reduce[i].cin, self.up_reduce[i].cout));
                }
                if self.up_merge[i].cin != cout * 2 || self.up_merge[i].cout != cout {
                    return Err(format!("up_merge[{i}] {}x{} != cin={} cout={cout}", self.up_merge[i].cin, self.up_merge[i].cout, cout * 2));
                }
            }
            let want_refine = (other.output_h, other.output_w) != (other.render_h, other.render_w);
            match (&self.out_refine, want_refine) {
                (Some(r), true) => {
                    if r.cin != other.widths[0] || r.cout != other.widths[0] {
                        return Err(format!("out_refine {}x{} != widths[0]={}", r.cin, r.cout, other.widths[0]));
                    }
                }
                (None, false) => {}
                (present, wants) => return Err(format!("out_refine presence={} != config wants={wants}", present.is_some())),
            }
            if self.head.cin != other.widths[0] || self.head.cout != other.out_channels {
                return Err(format!("head {}x{} != widths[0]={} out_channels={}", self.head.cin, self.head.cout, other.widths[0], other.out_channels));
            }
            Ok(())
        }

        /// `input` must be `config.render_h x config.render_w x
        /// config.in_channels` (HWC). Returns the output image
        /// (`config.output_h x config.output_w x config.out_channels`) plus
        /// the cache `backward` needs.
        pub fn forward(&self, input: &Img) -> (Img, ForwardCache) {
            assert_eq!((input.h, input.w, input.c), (self.config.render_h, self.config.render_w, self.config.in_channels));
            let stem_pre = self.stem.forward(input);
            let stem_relu = relu_forward(&stem_pre);

            let mut enc: Vec<(Img, Img, Img)> = Vec::with_capacity(self.config.n_scales.saturating_sub(1));
            let mut skips: Vec<Img> = vec![stem_relu.clone()];
            let mut cur = stem_relu;
            for s in 1..self.config.n_scales {
                let down_pre = self.down[s - 1].forward(&cur);
                let down_relu = relu_forward(&down_pre);
                let refine_pre = self.down_refine[s - 1].forward(&down_relu);
                let refine_relu = relu_forward(&refine_pre);
                skips.push(refine_relu.clone());
                enc.push((down_pre, down_relu, refine_pre));
                cur = refine_relu;
            }

            // Decode order: DEEPEST scale first (i=0 -> s=n_scales-2), matching
            // `up_reduce`/`up_merge`'s own construction order in `new_random`.
            let mut dec_out = skips[self.config.n_scales - 1].clone();
            let mut dec: Vec<(Img, Img, Img, Img, Img, Img)> = Vec::with_capacity(self.config.n_scales.saturating_sub(1));
            for (i, s) in (0..self.config.n_scales.saturating_sub(1)).rev().enumerate() {
                let target = &skips[s];
                let resized = resize_bilinear_forward(&dec_out, target.h, target.w);
                let reduce_pre = self.up_reduce[i].forward(&resized);
                let reduce_relu = relu_forward(&reduce_pre);
                let cat = concat_forward(&reduce_relu, target);
                let merge_pre = self.up_merge[i].forward(&cat);
                let merge_relu = relu_forward(&merge_pre);
                dec.push((resized, reduce_pre, reduce_relu, cat, merge_pre, merge_relu.clone()));
                dec_out = merge_relu;
            }

            let (out_pre_resize, out_resized, out_refine_relu, head_input) =
                if let Some(refine) = &self.out_refine {
                    let resized = resize_bilinear_forward(&dec_out, self.config.output_h, self.config.output_w);
                    let refine_pre = refine.forward(&resized);
                    let refine_relu = relu_forward(&refine_pre);
                    (Some(dec_out), Some(resized), Some(refine_relu.clone()), refine_relu)
                } else {
                    (None, None, None, dec_out)
                };

            let output = self.head.forward(&head_input);
            (
                output,
                ForwardCache { input: input.clone(), skips, enc, dec, out_pre_resize, out_resized, out_refine_relu, head_input },
            )
        }

        /// Backward: `d_output` (grad wrt the forward's return value) + the
        /// `ForwardCache` from the matching `forward` call -> gradients
        /// accumulated into `grads` (a `zeros_like(self)`-shaped
        /// accumulator the caller owns across a batch, then Adam-steps).
        /// A straight mirror of `forward`'s op order, walked in reverse —
        /// see `ForwardCache`'s field docs for the index conventions this
        /// relies on (`dec[i]` deepest-first, `skips[s]` scale s).
        pub fn backward(&self, cache: &ForwardCache, d_output: &Img, grads: &mut UnetWeights) {
            let d_head_input = self.head.backward(&cache.head_input, d_output, &mut grads.head.w, &mut grads.head.b);

            // `d_dec_out`: gradient wrt the full-res feature map that fed
            // EITHER the output-res resize (if present) or the head directly.
            let d_dec_out = if let Some(refine) = &self.out_refine {
                let d_refine_pre = relu_backward(cache.out_refine_relu.as_ref().unwrap(), &d_head_input);
                let grads_refine = grads.out_refine.as_mut().expect("grads.out_refine must exist when self.out_refine does");
                let d_resized = refine.backward(cache.out_resized.as_ref().unwrap(), &d_refine_pre, &mut grads_refine.w, &mut grads_refine.b);
                let src = cache.out_pre_resize.as_ref().unwrap();
                resize_bilinear_backward(&d_resized, src.h, src.w)
            } else {
                d_head_input
            };

            // `d_skips[s]`: total gradient into `skips[s]` from every place
            // it is READ in forward — the decoder concat at scale s<n_scales-1
            // (if any), the encoder downsample at scale s+1 (if s+1<n_scales),
            // and (for the top scale, n_scales-1) the bottleneck hand-off into
            // the decode loop / `d_dec_out` when n_scales==1. Accumulated as
            // each source is walked; every entry is `Some` by the time the
            // encoder loop below reads it (`n_scales-1` from the bottleneck
            // hand-off, every other `s` from the decoder loop's concat).
            let n_scales = self.config.n_scales;
            let n_dec = n_scales.saturating_sub(1);
            let mut d_skips: Vec<Option<Img>> = vec![None; n_scales];

            let mut d_cur = d_dec_out; // gradient wrt the CURRENT decode step's merge_relu output
            for i in (0..n_dec).rev() {
                let s = n_scales - 2 - i; // this decode step's skip scale (matches `new_random`/`forward`'s own s=(n_scales-2)-i)
                let (resized, _reduce_pre, reduce_relu, cat, _merge_pre, merge_relu) = &cache.dec[i];
                let d_merge_pre = relu_backward(merge_relu, &d_cur);
                let up_merge_grad = &mut grads.up_merge[i];
                let d_cat = self.up_merge[i].backward(cat, &d_merge_pre, &mut up_merge_grad.w, &mut up_merge_grad.b);
                let (d_reduce_relu, d_skip_s) = concat_backward(&d_cat, reduce_relu.c, cache.skips[s].c);
                accumulate(&mut d_skips[s], d_skip_s);
                let d_reduce_pre = relu_backward(reduce_relu, &d_reduce_relu);
                let up_reduce_grad = &mut grads.up_reduce[i];
                let d_resized = self.up_reduce[i].backward(resized, &d_reduce_pre, &mut up_reduce_grad.w, &mut up_reduce_grad.b);
                // the resize SOURCE is scale s+1 (bottleneck when i==0, else
                // the previous — shallower-index, deeper-scale — decode
                // step's merge_relu, whose spatial size == skips[s+1]).
                let src = &cache.skips[s + 1];
                d_cur = resize_bilinear_backward(&d_resized, src.h, src.w);
            }
            // After the loop (or immediately, if n_scales==1 and it never
            // ran), `d_cur` is the gradient into the BOTTLENECK (`skips[n_scales-1]`).
            accumulate(&mut d_skips[n_scales - 1], d_cur);

            // Encoder, reverse of its own forward loop (scale n_scales-1 down
            // to 1); each step's downsample INPUT is `skips[s-1]`, which the
            // encoder loop's own d_skips accumulation feeds into.
            for s in (1..n_scales).rev() {
                let (_down_pre, down_relu, _refine_pre) = &cache.enc[s - 1];
                let d_refine_relu = d_skips[s].take().expect("d_skips[s] must be populated before the encoder walk reaches it");
                let d_refine_pre = relu_backward(&cache.skips[s], &d_refine_relu);
                let refine_grad = &mut grads.down_refine[s - 1];
                let d_down_relu = self.down_refine[s - 1].backward(down_relu, &d_refine_pre, &mut refine_grad.w, &mut refine_grad.b);
                let d_down_pre = relu_backward(down_relu, &d_down_relu);
                let down_grad = &mut grads.down[s - 1];
                let prev = &cache.skips[s - 1];
                let d_prev = self.down[s - 1].backward(prev, &d_down_pre, &mut down_grad.w, &mut down_grad.b);
                accumulate(&mut d_skips[s - 1], d_prev);
            }
            let d_stem_relu = d_skips[0].take().expect("d_skips[0] must be populated (the stem always feeds something)");
            let d_stem_pre = relu_backward(&cache.skips[0], &d_stem_relu);
            self.stem.backward(&cache.input, &d_stem_pre, &mut grads.stem.w, &mut grads.stem.b);
        }
    }

    /// Add `b` into the accumulator slot, initializing it on first write —
    /// the shared pattern every `d_skips[s]` accumulation above uses.
    fn accumulate(slot: &mut Option<Img>, b: Img) {
        match slot {
            Some(a) => {
                for (av, bv) in a.data.iter_mut().zip(&b.data) {
                    *av += bv;
                }
            }
            None => *slot = Some(b),
        }
    }

    /// Per-parameter Adam moments, one pair of (m, v) buffers per
    /// `ConvParam`'s (w, b) — same shape as `UnetWeights`, mirrors
    /// `rdirect.rs::Adam`'s per-layer-moment-buffer pattern for the conv
    /// parameter set.
    pub struct UnetAdam {
        lr: f32,
        beta1: f32,
                beta2: f32,
        eps: f32,
        t: u64,
        m: UnetWeights,
        v: UnetWeights,
    }

    impl UnetAdam {
        pub fn new(weights: &UnetWeights, lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
            Self { lr, beta1, beta2, eps, t: 0, m: weights.zeros_like(), v: weights.zeros_like() }
        }
        pub fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
        pub fn lr(&self) -> f32 {
            self.lr
        }

        /// Adam step: `weights` updated IN PLACE from `grads` (a
        /// `zeros_like`-shaped accumulator the caller already summed a
        /// batch's `backward` calls into — NOT pre-divided by batch size;
        /// callers scale the gradient during `backward` accumulation the
        /// same way `rdirect.rs`'s per-pixel trainer does, via the `scale`
        /// argument threaded through, OR divide `grads` before calling this).
        pub fn step(&mut self, weights: &mut UnetWeights, grads: &UnetWeights) {
            self.t += 1;
            let t = self.t as i32;
            let bias1 = 1.0 - self.beta1.powi(t);
            let bias2 = 1.0 - self.beta2.powi(t);
            let (beta1, beta2, eps, lr) = (self.beta1, self.beta2, self.eps, self.lr);
            fn step_conv(p: &mut ConvParam, g: &ConvParam, m: &mut ConvParam, v: &mut ConvParam, beta1: f32, beta2: f32, eps: f32, lr: f32, bias1: f32, bias2: f32) {
                for i in 0..p.w.len() {
                    m.w[i] = beta1 * m.w[i] + (1.0 - beta1) * g.w[i];
                    v.w[i] = beta2 * v.w[i] + (1.0 - beta2) * g.w[i] * g.w[i];
                    let mhat = m.w[i] / bias1;
                    let vhat = v.w[i] / bias2;
                    p.w[i] -= lr * mhat / (vhat.sqrt() + eps);
                }
                for i in 0..p.b.len() {
                    m.b[i] = beta1 * m.b[i] + (1.0 - beta1) * g.b[i];
                    v.b[i] = beta2 * v.b[i] + (1.0 - beta2) * g.b[i] * g.b[i];
                    let mhat = m.b[i] / bias1;
                    let vhat = v.b[i] / bias2;
                    p.b[i] -= lr * mhat / (vhat.sqrt() + eps);
                }
            }
            step_conv(&mut weights.stem, &grads.stem, &mut self.m.stem, &mut self.v.stem, beta1, beta2, eps, lr, bias1, bias2);
            for i in 0..weights.down.len() {
                step_conv(&mut weights.down[i], &grads.down[i], &mut self.m.down[i], &mut self.v.down[i], beta1, beta2, eps, lr, bias1, bias2);
            }
            for i in 0..weights.down_refine.len() {
                step_conv(&mut weights.down_refine[i], &grads.down_refine[i], &mut self.m.down_refine[i], &mut self.v.down_refine[i], beta1, beta2, eps, lr, bias1, bias2);
            }
            for i in 0..weights.up_reduce.len() {
                step_conv(&mut weights.up_reduce[i], &grads.up_reduce[i], &mut self.m.up_reduce[i], &mut self.v.up_reduce[i], beta1, beta2, eps, lr, bias1, bias2);
            }
            for i in 0..weights.up_merge.len() {
                step_conv(&mut weights.up_merge[i], &grads.up_merge[i], &mut self.m.up_merge[i], &mut self.v.up_merge[i], beta1, beta2, eps, lr, bias1, bias2);
            }
            if let (Some(p), Some(g), Some(m), Some(v)) = (weights.out_refine.as_mut(), grads.out_refine.as_ref(), self.m.out_refine.as_mut(), self.v.out_refine.as_mut()) {
                step_conv(p, g, m, v, beta1, beta2, eps, lr, bias1, bias2);
            }
            step_conv(&mut weights.head, &grads.head, &mut self.m.head, &mut self.v.head, beta1, beta2, eps, lr, bias1, bias2);
        }
    }

    // ── serialization ("GAIARDR9" blob — versioned, self-describing) ──────
    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_f32_slice(buf: &mut Vec<u8>, v: &[f32]) {
        push_u32(buf, v.len() as u32);
        for &x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn push_conv(buf: &mut Vec<u8>, p: &ConvParam) {
        push_u32(buf, p.cin as u32);
        push_u32(buf, p.cout as u32);
        push_u32(buf, p.k as u32);
        push_u32(buf, p.stride as u32);
        push_f32_slice(buf, &p.w);
        push_f32_slice(buf, &p.b);
    }
    fn read_u32(cur: &mut usize, bytes: &[u8]) -> Option<u32> {
        let end = *cur + 4;
        let v = u32::from_le_bytes(bytes.get(*cur..end)?.try_into().ok()?);
        *cur = end;
        Some(v)
    }
    fn read_f32_slice(cur: &mut usize, bytes: &[u8]) -> Option<Vec<f32>> {
        let n = read_u32(cur, bytes)? as usize;
                let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let end = *cur + 4;
            out.push(f32::from_le_bytes(bytes.get(*cur..end)?.try_into().ok()?));
            *cur = end;
        }
        Some(out)
    }
    fn read_conv(cur: &mut usize, bytes: &[u8]) -> Option<ConvParam> {
        let cin = read_u32(cur, bytes)? as usize;
        let cout = read_u32(cur, bytes)? as usize;
        let k = read_u32(cur, bytes)? as usize;
        let stride = read_u32(cur, bytes)? as usize;
        let w = read_f32_slice(cur, bytes)?;
        let b = read_f32_slice(cur, bytes)?;
        Some(ConvParam { cin, cout, k, stride, w, b })
    }

    const MAGIC: &[u8; 8] = b"GAIARD9\0";

    /// Serialize a `UnetWeights` to a self-describing binary blob (config +
    /// every conv layer's dims + w/b). HWIO layout throughout — byte-
    /// compatible with what `imp::UnetLive` would read if wired to load a
    /// checkpoint (not done this stage — see the spike doc's honest gaps).
    pub fn serialize_weights(net: &UnetWeights) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        let c = &net.config;
        push_u32(&mut buf, c.n_scales as u32);
        for &w in &c.widths {
            push_u32(&mut buf, w as u32);
        }
        push_u32(&mut buf, c.kernel as u32);
        push_u32(&mut buf, c.in_channels as u32);
        push_u32(&mut buf, c.out_channels as u32);
        push_u32(&mut buf, c.render_w as u32);
        push_u32(&mut buf, c.render_h as u32);
        push_u32(&mut buf, c.output_w as u32);
        push_u32(&mut buf, c.output_h as u32);
        push_conv(&mut buf, &net.stem);
        push_u32(&mut buf, net.down.len() as u32);
        for p in &net.down {
            push_conv(&mut buf, p);
        }
        for p in &net.down_refine {
            push_conv(&mut buf, p);
        }
        for p in &net.up_reduce {
            push_conv(&mut buf, p);
        }
        for p in &net.up_merge {
            push_conv(&mut buf, p);
        }
        push_u32(&mut buf, if net.out_refine.is_some() { 1 } else { 0 });
        if let Some(p) = &net.out_refine {
            push_conv(&mut buf, p);
        }
        push_conv(&mut buf, &net.head);
        buf
    }

    pub fn deserialize_weights(bytes: &[u8]) -> Option<UnetWeights> {
        if bytes.len() < 8 || &bytes[0..8] != MAGIC {
            return None;
        }
        let mut cur = 8usize;
        let n_scales = read_u32(&mut cur, bytes)? as usize;
        let mut widths = Vec::with_capacity(n_scales);
        for _ in 0..n_scales {
            widths.push(read_u32(&mut cur, bytes)? as usize);
        }
        let kernel = read_u32(&mut cur, bytes)? as usize;
        let in_channels = read_u32(&mut cur, bytes)? as usize;
        let out_channels = read_u32(&mut cur, bytes)? as usize;
        let render_w = read_u32(&mut cur, bytes)? as usize;
        let render_h = read_u32(&mut cur, bytes)? as usize;
        let output_w = read_u32(&mut cur, bytes)? as usize;
        let output_h = read_u32(&mut cur, bytes)? as usize;
        let config = UnetConfig { n_scales, widths, kernel, in_channels, out_channels, render_w, render_h, output_w, output_h };
        let stem = read_conv(&mut cur, bytes)?;
        let n_down = read_u32(&mut cur, bytes)? as usize;
        let mut down = Vec::with_capacity(n_down);
        for _ in 0..n_down {
            down.push(read_conv(&mut cur, bytes)?);
        }
        let mut down_refine = Vec::with_capacity(n_down);
        for _ in 0..n_down {
            down_refine.push(read_conv(&mut cur, bytes)?);
        }
        let mut up_reduce = Vec::with_capacity(n_down);
        for _ in 0..n_down {
            up_reduce.push(read_conv(&mut cur, bytes)?);
        }
        let mut up_merge = Vec::with_capacity(n_down);
        for _ in 0..n_down {
            up_merge.push(read_conv(&mut cur, bytes)?);
        }
        let has_refine = read_u32(&mut cur, bytes)? != 0;
        let out_refine = if has_refine { Some(read_conv(&mut cur, bytes)?) } else { None };
        let head = read_conv(&mut cur, bytes)?;
        Some(UnetWeights { config, stem, down, down_refine, up_reduce, up_merge, out_refine, head })
    }

    pub fn weights_sha256(net: &UnetWeights) -> String {
        crate::denoiser::sha256_hex(&serialize_weights(net))
    }
}

#[cfg(test)]
mod width_serialize_roundtrip {
    //! CAPACITY ROUND (task mandate): the `.bin` header must round-trip
    //! `widths` exactly like it already round-trips `in_channels` (both are
    //! plain fields written/read generically by `serialize_weights`/
    //! `deserialize_weights` — no per-width literal in either function).
    //! Proves (a) DEFAULT widths [24,40,64] serialize/deserialize BYTE-
    //! IDENTICAL to a fresh `UnetConfig::default()` net (the IRON LAW every
    //! v9k..v9z checkpoint already depends on), and (b) CUSTOM widths (the
    //! CAPACITY ROUND's own candidate shapes) round-trip correctly —
    //! widths preserved exactly, every conv layer's shape intact, forward()
    //! output identical pre/post round-trip, and `validate_shapes_against`
    //! accepts the matching config and REJECTS a mismatched one (the loud-
    //! assert-on-load-mismatch half of the same mandate).
    use super::cpu::*;
    use super::UnetConfig;

    fn cfg(widths: Vec<usize>) -> UnetConfig {
        let n_scales = widths.len();
        UnetConfig { n_scales, widths, kernel: 3, in_channels: 7, out_channels: 3, render_w: 11, render_h: 9, output_w: 11, output_h: 9 }
    }

    fn assert_roundtrip(widths: Vec<usize>, seed: u64) {
        let config = cfg(widths.clone());
        let net = UnetWeights::new_random(config.clone(), seed);
        let bytes = serialize_weights(&net);
        let restored = deserialize_weights(&bytes).expect("deserialize");
        assert_eq!(restored.config.widths, widths, "widths did not round-trip");
        assert_eq!(restored.config.n_scales, widths.len());
        assert_eq!(restored.config.in_channels, config.in_channels);

        // byte-identical re-serialize (proves EVERY field round-tripped,
        // not just widths — a partial/lossy round-trip would diverge here).
        let bytes2 = serialize_weights(&restored);
        assert_eq!(bytes, bytes2, "re-serialized bytes diverged from the original blob");

        // forward() output identical pre/post round-trip — the shapes are
        // not just metadata-equal, the actual weight tensors moved intact.
        let mut input = Img::zeros(config.render_h, config.render_w, config.in_channels);
        let mut rng_state = seed ^ 0x1357_9BDF_2468_ACE0;
        for v in input.data.iter_mut() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (((rng_state >> 33) as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        let (out_orig, _) = net.forward(&input);
        let (out_restored, _) = restored.forward(&input);
        assert_eq!(out_orig.data, out_restored.data, "forward output diverged after round-trip");

        // loud-assert half: the restored weights validate against their OWN
        // config (self-consistent by construction) but must REJECT a
        // config with different widths (the mismatch this check exists to
        // catch — e.g. loading a checkpoint into a run with a different
        // GAIA_V9_WIDTHS).
        restored.validate_shapes_against(&restored.config).expect("self-config must validate");
        let mut bad = restored.config.clone();
        bad.widths = bad.widths.iter().map(|w| w + 1).collect();
        assert!(restored.validate_shapes_against(&bad).is_err(), "mismatched widths must be rejected, not silently accepted");
    }

    #[test]
    fn default_widths_roundtrip_byte_identical() {
        // The exact default every v9k..v9z run has shipped — this is the
        // IRON LAW half: unset GAIA_V9_WIDTHS must be provably unchanged.
        assert_roundtrip(vec![24, 40, 64], 0x00d1_5eed);
    }

    #[test]
    fn custom_widths_roundtrip() {
        // CAPACITY ROUND candidate shapes (STEP 2's own bench configs) —
        // proves the format is genuinely width-generic, not just tested at
        // the one shape every prior run happened to use.
        assert_roundtrip(vec![28, 48, 80], 0x2846_8000_1111);
        assert_roundtrip(vec![32, 56, 96], 0x3256_9600_2222);
    }

    #[test]
    fn asymmetric_scale_count_roundtrips() {
        // n_scales != 3 (the default n_down / stem/head plumbing must not
        // secretly assume exactly 2 down-steps).
        assert_roundtrip(vec![8, 16], 0xAAAA_BBBB);
        assert_roundtrip(vec![6, 10, 14, 22], 0xCCCC_DDDD);
    }
}

#[cfg(test)]
mod cpu_grad_check {
    //! Finite-difference gradient check on `cpu::UnetWeights::backward` —
    //! the hand-rolled conv/resize/concat backward above is exactly the
    //! class of code that silently produces a plausible-looking but WRONG
    //! gradient (a real risk already caught once while writing this file:
    //! an earlier draft had stub/placeholder backward math that compiled
    //! clean but was not analytically checked). This test is that check:
    //! random tiny net + random input -> compare `backward`'s analytic
    //! d(loss)/d(param) against central-difference numeric gradients on a
    //! sample of weights from EVERY layer kind (stem, down, down_refine,
    //! up_reduce, up_merge, out_refine, head) and on the input itself.
    use super::cpu::*;
    use super::UnetConfig;

    fn small_config(output_free: bool) -> UnetConfig {
        UnetConfig {
            n_scales: 3,
            widths: vec![3, 4, 5],
            kernel: 3,
            in_channels: 5,
            out_channels: 2,
            render_w: 9,
            render_h: 7,
            output_w: if output_free { 13 } else { 9 },
            output_h: if output_free { 11 } else { 7 },
        }
    }

    fn sum_sq_loss(out: &Img) -> f64 {
        // A simple, everywhere-differentiable scalar loss: 0.5 * sum(out^2)
        // against a fixed target, so d(loss)/d(out) = (out - target) is
        // exactly what `backward`'s `d_output` argument expects.
        out.data.iter().map(|&v| 0.5 * (v as f64) * (v as f64)).sum()
    }
    /// d(0.5*sum(v^2))/dv = v — the loss's own gradient IS the value.
    fn d_loss_d_out(out: &Img) -> Img {
        out.clone()
    }

    fn run_check(output_free: bool, seed: u64) {
        let config = small_config(output_free);
        let mut net = UnetWeights::new_random(config.clone(), seed);
        let mut input = Img::zeros(config.render_h, config.render_w, config.in_channels);
        let mut rng_state = seed ^ 0xABCD_EF01_2345_6789;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((rng_state >> 33) as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        for v in input.data.iter_mut() {
            *v = next();
        }

        let (out0, cache) = net.forward(&input);
        let d_out = d_loss_d_out(&out0);
        let mut grads = net.zeros_like();
        net.backward(&cache, &d_out, &mut grads);

        const EPS: f32 = 1.0e-3;
        const REL_TOL: f64 = 0.03; // 3% — fp32 forward/backward, small net
        const ABS_TOL: f64 = 2.0e-3;

        // Mutate `net`'s own storage in place, forward, finite-diff, compare
        // to `grads`' matching slot (`grads` was computed ONCE above, before
        // any mutation, and is only ever read from here on — the ground
        // truth). One macro call per layer kind (small, explicit — no
        // generic reflection over the struct).
        let mut n_checked = 0usize;
        let mut max_rel_err = 0.0f64;

        // `$field` is a direct field-path TOKEN (e.g. `net.stem.w`), spliced
        // verbatim at each use site inside the macro body — NOT a closure.
        // (An earlier draft passed a `|n: &mut UnetWeights| -> &mut [f32]`
        // closure here; called immediately it LOOKS fine, but plain closure
        // literals infer one concrete elided lifetime for their single
        // signature, and requiring that same signature to also satisfy the
        // macro's later `net.forward(&input)` immutable reborrow does not
        // typecheck — "lifetime may not live long enough". Splicing the
        // field path directly sidesteps the closure entirely: each `$field[i]`
        // is its own independent borrow, released before the next statement,
        // same as the closure version intended but without the HRTB trap.)
        macro_rules! check_slice {
            ($field:expr, $grad_slice:expr, $label:expr) => {{
                let g: &[f32] = $grad_slice;
                let n = g.len();
                let stride = (n / 6).max(1); // sample ~6 elements, not every one (speed)
                let mut i = 0usize;
                while i < n {
                    let orig = $field[i];
                    $field[i] = orig + EPS;
                    let (out_p, _) = net.forward(&input);
                    let lp = sum_sq_loss(&out_p);
                    $field[i] = orig - EPS;
                    let (out_m, _) = net.forward(&input);
                    let lm = sum_sq_loss(&out_m);
                    $field[i] = orig;
                    let numeric = (lp - lm) / (2.0 * EPS as f64);
                    let analytic = g[i] as f64;
                    let err = (numeric - analytic).abs();
                    let rel = err / numeric.abs().max(ABS_TOL);
                    max_rel_err = max_rel_err.max(rel);
                    assert!(
                        err < ABS_TOL || rel < REL_TOL,
                        "{} idx={i}: numeric={numeric:.6} analytic={analytic:.6} err={err:.6} rel={rel:.4}",
                        $label
                    );
                    n_checked += 1;
                    i += stride;
                }
            }};
        }

        // SAFETY of the finite-diff loop above: we mutate `net` (not
        // `grads`) in place per-element, restore it immediately after each
        // probe, and only ever read `grads` (computed ONCE, before any
        // mutation) — so `grads` stays the single ground truth throughout.
        check_slice!(net.stem.w, &grads.stem.w, "stem.w");
        check_slice!(net.stem.b, &grads.stem.b, "stem.b");
        for i in 0..net.down.len() {
            check_slice!(net.down[i].w, &grads.down[i].w, "down.w");
            check_slice!(net.down_refine[i].w, &grads.down_refine[i].w, "down_refine.w");
            check_slice!(net.up_reduce[i].w, &grads.up_reduce[i].w, "up_reduce.w");
            check_slice!(net.up_merge[i].w, &grads.up_merge[i].w, "up_merge.w");
        }
        if net.out_refine.is_some() {
            check_slice!(
                net.out_refine.as_mut().unwrap().w,
                grads.out_refine.as_ref().unwrap().w.as_slice(),
                "out_refine.w"
            );
        }
        check_slice!(net.head.w, &grads.head.w, "head.w");

        eprintln!("[cpu_grad_check] output_free={output_free} seed={seed:#x}: {n_checked} params checked, max_rel_err={max_rel_err:.4}");
        assert!(n_checked > 20, "grad check sampled too few params ({n_checked}) to be meaningful");
    }

    #[test]
    fn gradients_match_finite_difference_same_res() {
        run_check(false, 0x1234_5678);
    }

    #[test]
    fn gradients_match_finite_difference_free_output_res() {
        run_check(true, 0x9999_1111);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// V9 WIRING ATOM, ITEM 1 (2026-07-21) — BODY-GENERIC PARITY ORDEAL: the CPU
// twin (`cpu::UnetWeights::forward`, already grad-checked above) vs the
// live GPU tensor-path forward (`imp::UnetLive::from_weights` ->
// `forward_cpu_roundtrip`), SAME trained weights, SAME input, asserted near.
// House pattern: `tests/rdirect_live_ordeals.rs`'s N0 GATE 1 (the v7 MLP's
// own CPU-vs-live parity gate, prior S1/S2 parity work on branch
// neural-live) — same shape of test, different body.
//
// TOLERANCE DERIVATION (done BEFORE running, not fit to the result after —
// canon law: "canon reds = re-derive by hand, never bump-to-match"):
//
//   The v7 MLP's live GEMM path (rdirect_live.rs) stores weights/activations
//   in fp32 (`MPSDataType::Float32`, confirmed by grep on that file) — its
//   own N0 GATE 1 derives a ~1e-3 envelope purely from f32 GEMM
//   reassociation, and MEASURES ~1.6e-7 actual (a ~6000x margin — the bound
//   there is generous on purpose, a wiring-error trip wire, not a tight
//   estimate).
//
//   `rdirect_unet.rs`'s graph is DIFFERENT in kind: the module doc states
//   plainly "Weight/activation storage is fp16 (`half::f16`) end to end on
//   the graph" — every weight/bias constant AND (per `build_from_weights`'s
//   `conv_layer_from`) every intermediate the graph produces is fp16-typed.
//   IEEE-754 binary16 has 1 implicit + 10 explicit mantissa bits (11 bits
//   total precision) -> unit roundoff u16 = 2^-11 ~= 4.883e-4 (the max
//   relative error a single correctly-rounded fp16 value can carry).
//
//   Apple does not publish (and this repo does not have on file) the exact
//   internal accumulation precision MPSGraph's fp16 conv/resize/concat ops
//   use on this silicon ("fp16 storage, fp32 accumulate" is the common
//   tensor-core convention industry-wide, but is NOT confirmed here for
//   this graph — flagged UNVERIFIED, not assumed). The conservative bound
//   below therefore assumes the WORST CASE the module doc's own words
//   support: every op boundary along the deepest path re-rounds to fp16.
//
//   Deepest-path op count for C-med (n_scales=3, same-res output): stem(1)
//   + per extra scale (down + down_refine = 2) x (n_scales-1=2) = 4, +
//   per decode step (resize + up_reduce + concat + up_merge = effectively
//   2 CONV ops, resize/concat carry no weights but still round their
//   fp16 output) x (n_scales-1=2) ~= 4, + head(1) = ~10 op boundaries;
//   +2 more (resize + out_refine conv) when output res is free (~12).
//   Worst-case LINEAR sum (no cancellation, matching the conservative
//   spirit of the v7 gate's own "worst-case reassociation" framing):
//     12 ops x u16 (4.883e-4) ~= 5.86e-3.
//
//   REL_TOL is set at 1.0e-2 (~1.7x that worst-case linear-sum estimate,
//   leaving headroom for the UNVERIFIED accumulation-precision assumption
//   without being so loose a real wiring bug (wrong layer order, transposed
//   weight layout, wrong stride) would slip under it — a mis-wired graph
//   produces O(1) relative garbage, not a few-percent drift. ABS_TOL is set
//   at 5.0e-3 as the floor for near-zero (post-ReLU-dead / near-bias-only)
//   channels where a relative measure is meaningless. Actuals are printed
//   either way so a future tightening has real numbers to tighten against.
#[cfg(all(test, target_os = "macos"))]
mod gpu_cpu_parity {
    use super::cpu::UnetWeights;
    use super::{UnetConfig, UnetLive};
    use wgpu::util::DeviceExt;

    const REL_TOL: f32 = 1.0e-2; // derived above: ~1.7x the worst-case fp16 linear-sum bound (5.86e-3)
    const ABS_TOL: f32 = 5.0e-3; // floor for near-zero channels (relative measure meaningless there)

    /// Deterministic dependency-free PRNG (SplitMix64) — INPUT data only
    /// (house per-module duplication convention, matches `cpu_grad_check`).
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_unit(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let bits = (z >> 40) as u32;
            ((bits as f32) / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// C-med shape (`docs/perf/2026-07-21-v9-spike.md`'s chosen config,
    /// `UnetConfig::default()`'s own widths/n_scales/kernel/in_channels/
    /// out_channels) at a SMALL render res for test wall-time (fully-
    /// convolutional -- resolution-independent per the module's own STAGE 2
    /// doc comment) -- `output_free` exercises the resize+refine arm too.
    fn small_cmed_config(output_free: bool) -> UnetConfig {
        let d = UnetConfig::default(); // C-med widths/n_scales/kernel/channels
        UnetConfig {
            render_w: 20,
            render_h: 16,
            output_w: if output_free { 28 } else { 20 },
            output_h: if output_free { 24 } else { 16 },
            ..d
        }
    }

    fn run_parity(output_free: bool, seed: u64) {
        let config = small_cmed_config(output_free);
        let weights = UnetWeights::new_random(config.clone(), seed);

        let mut rng = SplitMix64(seed ^ 0x9999_AAAA_5555_1111);
        let n = config.render_h * config.render_w * config.in_channels;
        let mut input = super::cpu::Img::zeros(config.render_h, config.render_w, config.in_channels);
        for v in input.data.iter_mut() {
            *v = rng.next_unit();
        }
        assert_eq!(input.data.len(), n);

        let (cpu_out, _cache) = weights.forward(&input);

        let live = match UnetLive::from_weights(&weights) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[gpu_cpu_parity] SKIP -- no live Metal path: {e}");
                return;
            }
        };
        let gpu_out = live
            .forward_cpu_roundtrip(&input.data)
            .expect("live GPU forward ran");
        assert_eq!(gpu_out.len(), cpu_out.data.len(), "output element count");

        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        let mut sum_abs = 0f64;
        for i in 0..cpu_out.data.len() {
            let a = (gpu_out[i] - cpu_out.data[i]).abs();
            let denom = cpu_out.data[i].abs().max(1e-3);
            max_abs = max_abs.max(a);
            max_rel = max_rel.max(a / denom);
            sum_abs += a as f64;
        }
        let mean_abs = sum_abs / cpu_out.data.len() as f64;
        eprintln!(
            "[gpu_cpu_parity] output_free={output_free} seed={seed:#x} n={}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} mean_abs={mean_abs:.3e} (bar REL_TOL={REL_TOL:.1e} ABS_TOL={ABS_TOL:.1e})",
            cpu_out.data.len()
        );
        assert!(
            max_abs < ABS_TOL || max_rel < REL_TOL,
            "gpu_cpu_parity output_free={output_free}: max_abs={max_abs:.3e} (bar {ABS_TOL:.1e}) max_rel={max_rel:.3e} (bar {REL_TOL:.1e})"
        );

        // Determinism: same weights + same input twice -> byte-identical GPU output.
        let gpu_out2 = live.forward_cpu_roundtrip(&input.data).expect("second forward");
        assert_eq!(gpu_out, gpu_out2, "live U-Net forward is not deterministic run-to-run");

        // V9-WIRE ATOM (GATHER FOLLOW-UP) addition: `forward_gpu_with_output`
        // (the persistent-compiled-executable path the frame-wall harness's
        // fast-forward row uses) must agree with `forward_cpu_roundtrip`
        // (this test's own already-proven-vs-CPU path, SAME `self.graph`/
        // `self.output`, just encoded via `MPSGraphExecutable` instead of
        // `runWithMTLCommandQueue_feeds_targetTensors_targetOperations`) to
        // the SAME bar this test already applies against the CPU twin —
        // isolates "does the fast encode path reproduce the SAME graph's
        // answer" from the CPU-vs-GPU fp16 tolerance already derived above.
        let (_fast_gpu_ms, fast_out) =
            live.forward_gpu_with_output(&input.data).expect("forward_gpu_with_output ran");
        assert_eq!(fast_out.len(), cpu_out.data.len(), "forward_gpu_with_output element count");
        let mut max_abs_fast = 0f32;
        let mut max_rel_fast = 0f32;
        for i in 0..cpu_out.data.len() {
            let a = (fast_out[i] - cpu_out.data[i]).abs();
            let denom = cpu_out.data[i].abs().max(1e-3);
            max_abs_fast = max_abs_fast.max(a);
            max_rel_fast = max_rel_fast.max(a / denom);
        }
        eprintln!(
            "[gpu_cpu_parity] forward_gpu_with_output vs cpu twin: max_abs={max_abs_fast:.3e} max_rel={max_rel_fast:.3e}"
        );
        assert!(
            max_abs_fast < ABS_TOL || max_rel_fast < REL_TOL,
            "forward_gpu_with_output vs cpu twin output_free={output_free}: max_abs={max_abs_fast:.3e} (bar {ABS_TOL:.1e}) max_rel={max_rel_fast:.3e} (bar {REL_TOL:.1e})"
        );

        // V9-WIRE ATOM (ZERO-COPY BRIDGE, 2026-07-21) addition:
        // `UnetLive::from_wgpu_queue` + `Fp16Packer` + `forward_gpu_bridged`
        // (the whole point of this atom — the SAME feature values reach the
        // graph via a GPU pack dispatch into the bridged MTLBuffer instead of
        // a CPU scalar `f16::from_f32` loop) must ALSO agree with the CPU
        // twin to the SAME bar. `headless_device` (integrator.rs) gives an
        // independent wgpu Metal device/queue — exercises the real
        // `queue.as_hal::<Metal>()` bridge path, not a mock.
        match crate::integrator::headless_device() {
            None => {
                eprintln!("[gpu_cpu_parity] SKIP bridged path -- no wgpu Metal adapter");
            }
            Some((device, queue)) => {
                let bridged = match UnetLive::from_wgpu_queue(&device, &queue, &weights) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("[gpu_cpu_parity] SKIP bridged path -- from_wgpu_queue: {e}");
                        return;
                    }
                };
                let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gpu_cpu_parity bridged src (f32 feats)"),
                    contents: bytemuck::cast_slice(&input.data),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let dst = bridged
                    .feature_buf_u32()
                    .expect("from_wgpu_queue built the bridge");
                let packer = crate::rdirect_gather::Fp16Packer::new(&device);
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpu_cpu_parity bridged pack"),
                });
                packer.encode(&device, &mut enc, &src, dst, input.data.len());
                queue.submit(Some(enc.finish()));
                let _ = device.poll(wgpu::PollType::wait_indefinitely());

                let (_bridged_ms, bridged_out) =
                    bridged.forward_gpu_bridged().expect("forward_gpu_bridged ran");
                assert_eq!(bridged_out.len(), cpu_out.data.len(), "forward_gpu_bridged element count");
                let mut max_abs_br = 0f32;
                let mut max_rel_br = 0f32;
                for i in 0..cpu_out.data.len() {
                    let a = (bridged_out[i] - cpu_out.data[i]).abs();
                    let denom = cpu_out.data[i].abs().max(1e-3);
                    max_abs_br = max_abs_br.max(a);
                    max_rel_br = max_rel_br.max(a / denom);
                }
                eprintln!(
                    "[gpu_cpu_parity] forward_gpu_bridged (zero-copy) vs cpu twin: max_abs={max_abs_br:.3e} max_rel={max_rel_br:.3e}"
                );
                assert!(
                    max_abs_br < ABS_TOL || max_rel_br < REL_TOL,
                    "forward_gpu_bridged vs cpu twin output_free={output_free}: max_abs={max_abs_br:.3e} (bar {ABS_TOL:.1e}) max_rel={max_rel_br:.3e} (bar {REL_TOL:.1e})"
                );
            }
        }
    }

    #[test]
    fn cpu_twin_matches_gpu_tensor_path_same_res() {
        run_parity(false, 0x7E51_C0DE);
    }

    #[test]
    fn cpu_twin_matches_gpu_tensor_path_free_output_res() {
        run_parity(true, 0xF4EE_0057);
    }

    /// V9-WIRE SPEED ROUND (2026-07-25) NUMERIC GUARD: `submit_gpu_bridged_async`
    /// + `wait_gpu_bridged_async` (set 1, the dedicated `net_queue`) must
    /// reproduce EXACTLY (bit-for-bit -- same compiled executable, same input
    /// values, no algorithm change) what `forward_gpu_bridged` (set 0, the
    /// sync path's own `queue`) produces for the IDENTICAL input feature
    /// tensor. This isolates the async submission MECHANICS (dedicated
    /// queue, deferred wait, second buffer set) from any scene/RNG noise a
    /// live capture would carry -- the task mandate's own "parity check vs
    /// sync path, max abs diff 0 or fp-justified" applied at the tensor
    /// level, where it can be checked exactly instead of visually.
    #[test]
    fn async_bridge_matches_sync_bridge_same_input() {
        let config = small_cmed_config(false);
        let weights = UnetWeights::new_random(config.clone(), 0xA5A5_1234_5EED_0001);
        let mut rng = SplitMix64(0xA5A5_1234_5678_9ABC);
        let n = config.render_h * config.render_w * config.in_channels;
        let mut input = super::cpu::Img::zeros(config.render_h, config.render_w, config.in_channels);
        for v in input.data.iter_mut() {
            *v = rng.next_unit();
        }
        assert_eq!(input.data.len(), n);

        let Some((device, queue)) = crate::integrator::headless_device() else {
            eprintln!("[async_bridge_parity] SKIP -- no wgpu Metal adapter");
            return;
        };
        let bridged = match UnetLive::from_wgpu_queue(&device, &queue, &weights) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[async_bridge_parity] SKIP -- from_wgpu_queue: {e}");
                return;
            }
        };
        let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("async_bridge_parity src (f32 feats)"),
            contents: bytemuck::cast_slice(&input.data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let packer = crate::rdirect_gather::Fp16Packer::new(&device);

        // Pack the SAME input into set 0 (sync) AND set 1 (async).
        for set in 0..2usize {
            let dst = bridged
                .feature_buf_u32_set(set)
                .unwrap_or_else(|| panic!("from_wgpu_queue built set {set}'s bridge"));
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("async_bridge_parity pack"),
            });
            packer.encode(&device, &mut enc, &src, dst, input.data.len());
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }

        let (_sync_ms, sync_out) = bridged.forward_gpu_bridged().expect("sync forward (set 0) ran");
        let pending = bridged
            .submit_gpu_bridged_async(1)
            .expect("submit_gpu_bridged_async(1) ran");
        let (_async_ms, async_out) = bridged
            .wait_gpu_bridged_async(pending)
            .expect("wait_gpu_bridged_async(1) ran");

        assert_eq!(sync_out.len(), async_out.len(), "output element count");
        let mut max_abs = 0f32;
        let mut n_diff = 0usize;
        for i in 0..sync_out.len() {
            let a = (sync_out[i] - async_out[i]).abs();
            if a > 0.0 { n_diff += 1; }
            max_abs = max_abs.max(a);
        }
        eprintln!(
            "[async_bridge_parity] n={} max_abs_diff={max_abs:.6e} n_differing_elems={n_diff}",
            sync_out.len()
        );
        assert_eq!(sync_out, async_out, "async (set 1, net_queue) output must be bit-identical to sync (set 0, queue) for the SAME input");
    }
}
