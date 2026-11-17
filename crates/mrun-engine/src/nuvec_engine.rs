//! `NuvecEngine`: the [`Engine`] contract v2 (D028) over nuvec, for attention-only and hybrid
//! (attention + Gated DeltaNet) families. `ScriptedEngine` in `mrun` is the behavioural
//! reference; both allocate and account alike.
//!
//! - The bookkeeping (length, pending step, capacity, limit, segments, checkpoints, bytes) is the
//!   shared [`Ledger`]: every operation is planned and checked (`NeedsMemory` included) before
//!   anything changes, and a move that fails half-way poisons the engine until `reset_all`.
//! - Allocation follows [`grown_capacity`](crate::grown_capacity); a cancelled or failed chunk
//!   gives its growth back, and a cut releases capacity beyond one growth step.
//! - `prefill_chunk` keeps at most two layers in flight: layer i is queued before layer i − 1 is
//!   awaited, then cancellation is checked, so the GPU never idles between layers.
//! - Decode is pipelined like mlx-lm's `generate_step`: `sample` feeds the sampled token while it
//!   is still lazy and only then reads it back, so the next forward runs while the host
//!   detokenizes.
//! - Recurrent (position-bound) state of hybrids cannot be cut: a cancelled chunk and a discarded
//!   pipelined step restore the handles taken before them (the model never writes into its
//!   inputs); `trim` allows only the live length, the length before a pending step, or 0;
//!   checkpoints share the evaluated handles; `detach_tail` moves the live recurrent state into
//!   the segment, after which prefill and forward fail with `NoRecurrentState` until `restore` or
//!   `attach`.
//! - Memory (D029, D048): the buffer cache is limited and cleared after growth, moves, drops, each
//!   prefill chunk and every 256 decode positions; weights are measured after warm-up and wired;
//!   scratch is (warm-up peak above the weights and the warm-up's own state +
//!   [`analytic_scratch`]) × 1.25, its position-dependent terms at the capacity-limited depth.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::budget::{self, BudgetError, DeviceMemory, GIB, INITIAL_STATE_TOKENS, Overrides};
use crate::calibration;
use crate::engine::{
    CancelToken, Checkpoint, CheckpointId, ChunkProgress, Engine, EngineError, ImageSpan,
    LayerKind, MAX_IMAGE_TOKENS, MAX_LANES, MIN_LANE_TOKENS, MemoryBudget, ModelInfo, Pick, Pixels,
    Sampling, Segment, SegmentId, Step, TimeLen, TokenMask, chunk_len, verify_serially,
};
use crate::identity::{self, IdentityError};
use crate::ledger::Ledger;
use crate::load::{self, Weights};
use crate::models::dflash::{self, DFlash, DraftContext};
use crate::models::{self, FamilyConfig, LoadError, Model, Part, Pos, Run};
use crate::sample;
use crate::state::{self, KvTail, LayerKv, LayerState, Recurrent, RecurrentState, SeqState};
use crate::tensor::{self, Array, Pipe};

/// The buffer cache limit; counted inside scratch.
pub const CACHE_LIMIT: u64 = 256 << 20;
/// Safety factor on scratch.
const SCRATCH_MARGIN: f64 = 1.25;
/// Decode returns freed buffers to the system every this many positions.
const DECODE_CLEAR_INTERVAL: u32 = 256;

/// How to load a model.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Content identity of the model directory ([`identity::identify`]).
    pub identity: String,
    pub overrides: Overrides,
    /// Memory to plan with when the device reports none (CPU).
    pub fallback_device: Option<DeviceMemory>,
    /// Run on the CPU even when a Metal device is present.
    pub cpu: bool,
    /// Where the `chunk_len` calibration is persisted ([`calibration::default_path`] in mrun);
    /// `None` calibrates without storing. CPU engines are not calibrated.
    pub calibration: Option<PathBuf>,
    /// The layers to load (026: one Mac's part of a pool); `None` loads all of them.
    pub part: Option<Part>,
    /// Lanes to run (011, 1–4); `None` decides from the context capacity. A part of a pool runs
    /// one.
    pub lanes: Option<usize>,
    /// Skip the vision tower even when the model has one (009 `--vision off`).
    pub no_vision: bool,
    /// A DFlash drafter's directory to load beside the model (030); ignored for a part of a
    /// pool.
    pub drafter: Option<PathBuf>,
}

/// Where an injected failure strikes (test hook for FR-028/SC-012, [`NuvecEngine::fail_next`]).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailPoint {
    /// After layer `i` of the next prefill chunk was queued.
    PrefillLayer(usize),
    /// After the next forward's graph was built.
    Forward,
    /// At the next sample's readback of the token.
    SampleReadback,
    /// Half-way through the next `detach_tail`, after the plan.
    Detach,
    /// Half-way through the next `attach`, after the plan.
    Attach,
}

#[derive(Debug, thiserror::Error)]
pub enum NuvecLoadError {
    #[error(transparent)]
    Model(#[from] LoadError),
    #[error(transparent)]
    Budget(#[from] BudgetError),
    #[error(transparent)]
    Identity(#[from] IdentityError),
    #[error(transparent)]
    Tensor(#[from] tensor::Error),
    #[error("cannot read {path}: {message}")]
    Config { path: String, message: String },
    #[error("the device reports no memory and none was given")]
    NoDevice,
    #[error(
        "{requested} lanes need {} tokens of context ({} each), but this budget holds {capacity}; {fits} {}",
        u64::from(MIN_LANE_TOKENS) * *requested as u64,
        MIN_LANE_TOKENS,
        if *fits == 1 { "lane fits" } else { "lanes fit" }
    )]
    Lanes {
        requested: usize,
        capacity: u32,
        fits: usize,
    },
}

/// A step fed before the caller asked for it: its token, step id, logits and (hybrids) the
/// recurrent state before it, restored if the step is discarded.
#[derive(Debug)]
struct Pending {
    token: u32,
    id: u64,
    logits: Array,
    before: Option<RecurrentState>,
}

#[derive(Debug)]
struct StoredSegment {
    tail: KvTail,
    /// Hybrids: the recurrent state at the segment's end.
    recurrent: Option<RecurrentState>,
    /// The token ids of the segment's positions (for the repetition penalty).
    ids: Vec<u32>,
    /// The drafter's context for the segment's last positions (030).
    draft: Option<DraftContext>,
}

/// A lane's own state while another lane is selected (011); the ledger parks its live fields.
#[derive(Debug)]
struct LaneSlot {
    seq: SeqState,
    step: Option<(u64, Array)>,
    draws: u64,
    ids: Vec<u32>,
    verified: Vec<RecurrentState>,
    draft: Option<DraftContext>,
}

/// A prefill chunk run a few layers at a time (011): the hidden state between layers, the next
/// layer, the layer still being awaited, and what undoes the chunk.
#[derive(Debug)]
struct InFlight {
    lane: usize,
    tokens: Vec<u32>,
    x: Array,
    next: usize,
    /// The chunk's first slot, its delta and rotation runs (009).
    at: u32,
    delta: i32,
    runs: Vec<Run>,
    previous: Option<Array>,
    capacity: u32,
    start: Option<RecurrentState>,
    /// The drafter's taps of the layers run so far (030).
    taps: Vec<Array>,
}

pub struct NuvecEngine {
    info: ModelInfo,
    model: Box<dyn Model>,
    seq: SeqState,
    /// Checkpoints hold the recurrent state at their position (empty for attention-only
    /// models).
    ledger: Ledger<Pending, StoredSegment, RecurrentState>,
    /// Logits of the current step.
    step: Option<(u64, Array)>,
    /// Samples drawn since the last `set_limit` (seeded sampling is reproducible per request).
    draws: u64,
    /// The token id at each live position, for the repetition penalty. Positions are recorded
    /// as they are fed; entries past `len()` (a rolled-back pending step) are stale and ignored.
    ids: Vec<u32>,
    /// An injected failure (test hook).
    fail: Option<FailPoint>,
    /// The layers this engine runs (all, or its part of a pool).
    part: Part,
    /// The ring to the pool's other parts (026); `None` alone.
    pipe: Option<Pipe>,
    /// Hybrids after a verify (006): the recurrent state after each position from the verify's
    /// start (index 0: before `last`), for exact trims while the ledger's window is open.
    verified: Vec<RecurrentState>,
    /// The selected lane (011); the fields above (`seq`, `step`, `draws`, `ids`, `verified`) are
    /// its.
    lane: usize,
    /// Every lane's slot but the selected one's (`None` at the selected index).
    parked: Vec<Option<LaneSlot>>,
    /// The chunk in flight, if any.
    /// Each lane's chunk in flight (011: a shorter prompt may take over between layers).
    chunks: Vec<Option<InFlight>>,
    /// The vision tower (009) and its encoder scratch at the image cap.
    vision: Option<models::vision::VisionTower>,
    encoder: u64,
    /// The DFlash drafter (030), the layers it reads (`tapped[k]`: layer `k`'s output) and the
    /// selected lane's context (the other lanes' are parked with them).
    drafter: Option<DFlash>,
    tapped: Vec<bool>,
    draft: Option<DraftContext>,
}

impl std::fmt::Debug for NuvecEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NuvecEngine")
            .field("family", &self.info.family)
            .field("len", &self.ledger.len())
            .field("capacity", &self.seq.capacity())
            .finish_non_exhaustive()
    }
}

/// The random key of draw `draw` (counted from 1): from the seed, or from the clock.
fn key_at(seed: Option<u64>, draw: u64) -> tensor::Result<Array> {
    let base = seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    tensor::random_key(base.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(draw))
}

/// A verify pass's logits rows, the recurrent states after each position, and the drafter's
/// taps (030).
type Verified = (Array, Vec<RecurrentState>, Vec<Array>);

/// Feeds `features` of positions from `at` to `ctx` (030). The drafter only proposes, so a
/// failure clears its context instead of failing the step.
fn feed_drafter(
    drafter: Option<&DFlash>,
    ctx: Option<&mut DraftContext>,
    features: tensor::Result<Array>,
    at: u32,
) {
    let (Some(drafter), Some(ctx)) = (drafter, ctx) else {
        return;
    };
    if features.and_then(|f| drafter.extend(ctx, &f, at)).is_err() {
        ctx.clear();
    }
}

/// The name of a drafter directory: `org/name` from the store's `org--name/<commit>` layout,
/// else the directory's own name.
fn drafter_name(dir: &Path) -> String {
    let parent = dir
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .filter(|n| n.contains("--"));
    match parent {
        Some(repo) => repo.replacen("--", "/", 1),
        None => dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("drafter")
            .to_owned(),
    }
}

fn backend(error: tensor::Error) -> EngineError {
    EngineError::Backend(error.0)
}

fn injected() -> EngineError {
    EngineError::Backend("injected failure".into())
}

impl NuvecEngine {
    /// Loads the model in `dir`: family and layout from `config.json`, the weights (read and
    /// measured by a one-token warm-up), then the memory budget.
    pub fn load(dir: &Path, options: LoadOptions) -> Result<Self, NuvecLoadError> {
        if options.cpu {
            tensor::use_device(tensor::Device::Cpu);
        }
        let config_path = dir.join("config.json");
        let config: serde_json::Value = std::fs::read_to_string(&config_path)
            .map_err(|e| e.to_string())
            .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()))
            .map_err(|message| NuvecLoadError::Config {
                path: config_path.display().to_string(),
                message,
            })?;
        let family = FamilyConfig::from_config(&config)?;
        // Quantization mode, bits, group sizes and per-path overrides are refused from
        // `config.json` alone, before any safetensors file is opened (FR-002, SC-010).
        load::Quantization::from_config(&config)?;
        if let Some(text) = config.get("text_config").filter(|t| t.is_object()) {
            load::Quantization::from_config(text)?;
        }
        let eos = identity::eos_union(dir, &[])?;
        let mut info = family.model_info(options.identity.clone(), eos);

        let device = match tensor::device_memory() {
            Some((physical, working)) if !options.cpu => DeviceMemory {
                physical,
                recommended_working_set: working,
                available: crate::sysmem::available(),
                estimated: false,
            },
            _ => options.fallback_device.ok_or(NuvecLoadError::NoDevice)?,
        };
        let layers = info.layer_kinds.len();
        let part = options.part.unwrap_or(Part::whole(layers));
        if part.first >= part.end || part.end > layers {
            return Err(LoadError::InvalidConfig {
                key: "layers".into(),
                reason: format!(
                    "{}..{} is not a range of the {layers} layers",
                    part.first, part.end
                ),
            }
            .into());
        }
        // The vision tower (009): whole models of the Qwen3.5 families that ship one.
        let whole_model = part == Part::whole(layers);
        let vision_on = !options.no_vision
            && whole_model
            && config["vision_config"].is_object()
            && matches!(family.family.name, "qwen3_5" | "qwen3_5_moe");
        let keep = |name: &str| {
            (vision_on && models::vision::is_vision_tensor(name))
                || (!load::unused_tensor(name) && part.keeps(name))
        };
        let weights_bytes = load::weights_bytes_of(dir, keep);
        let plan = budget::plan(device, options.overrides)?;
        // Refuse early, before reading gigabytes, when the files alone cannot fit.
        budget::check_fits(&plan, weights_bytes, 0, &info)?;
        tensor::memory::set_cache_limit(CACHE_LIMIT);

        let text_config = config
            .get("text_config")
            .filter(|t| t.is_object())
            .unwrap_or(&config);
        let mut weights = Weights::load(dir, keep)?;
        let vision = if vision_on {
            Some(models::vision::VisionTower::load(
                &mut weights,
                &config["vision_config"],
            )?)
        } else {
            None
        };
        info.vision = vision.as_ref().map(|v| {
            let c = v.config();
            crate::engine::Vision {
                bytes: v.bytes(),
                patch: c.patch as u32,
                merge: c.merge as u32,
                temporal: c.temporal as u32,
            }
        });
        let model = models::build(family.family.name, &config, text_config, weights, part)?;
        if model.layer_kinds() != info.layer_kinds.as_slice() {
            return Err(LoadError::InvalidConfig {
                key: "layer_types".into(),
                reason: "disagrees with full_attention_interval".into(),
            }
            .into());
        }
        // The drafter (030) reads the whole model's layers on this Mac.
        let drafter = match options.drafter.as_deref().filter(|_| whole_model) {
            Some(dir) => {
                let target = dflash::Target {
                    hidden: model.hidden(),
                    vocab: info.vocab,
                    layers,
                };
                Some((dir, DFlash::load(dir, target)?))
            }
            None => None,
        };
        info.drafter = drafter.as_ref().map(|(dir, d)| crate::engine::Drafter {
            name: drafter_name(dir),
            bytes: d.bytes(),
            context_bytes: d.config().context_bytes(),
            window: d.config().keep() + 1,
        });
        let drafter = drafter.map(|(_, d)| d);
        let tapped: Vec<bool> = match &drafter {
            Some(d) => (0..layers).map(|i| d.taps().contains(&i)).collect(),
            None => Vec::new(),
        };
        // The state's real dtype decides its size (a config's `torch_dtype` may disagree); a part
        // holds the state of its own layers only.
        let attention = part_kinds(&info.layer_kinds, part)
            .filter(|k| *k == LayerKind::FullAttention)
            .count() as u64;
        info.kv_bytes_per_token = attention
            * 2
            * u64::from(info.kv_heads)
            * u64::from(info.head_dim)
            * model.state_dtype().size();
        info.recurrent_bytes = model.recurrent_bytes();
        let linear: Vec<bool> = part_kinds(&info.layer_kinds, part)
            .map(|k| k == LayerKind::Linear)
            .collect();
        let seq = SeqState::new(&linear, info.kv_heads, info.head_dim, model.state_dtype());
        // Refuse before the warm-up reads the weights when the analytic scratch cannot fit.
        let encoder = vision
            .as_ref()
            .map_or(0, |v| v.scratch(u64::from(MAX_IMAGE_TOKENS) * 4));
        let initial = analytic_scratch(
            model.as_ref(),
            &info,
            part,
            INITIAL_STATE_TOKENS as u32,
            encoder,
        );
        budget::check_fits(&plan, weights_bytes, initial, &info)?;
        let mut engine = Self {
            ledger: Ledger::new(&info, plan),
            info,
            model,
            seq,
            step: None,
            draws: 0,
            ids: Vec::new(),
            fail: None,
            part,
            pipe: None,
            verified: Vec::new(),
            lane: 0,
            parked: vec![None],
            chunks: vec![None],
            vision,
            encoder,
            draft: drafter.as_ref().map(DFlash::context),
            drafter,
            tapped,
        };
        let (weights, peak) = engine.warm_up()?;
        let scratch = |depth: u32| {
            let analytic = analytic_scratch(
                engine.model.as_ref(),
                &engine.info,
                engine.part,
                depth,
                engine.encoder,
            );
            ((peak + analytic) as f64 * SCRATCH_MARGIN) as u64
        };
        let plan = match options.overrides.context {
            Some(context) => budget::for_context(plan, context, weights, &scratch, &engine.info)?,
            None => plan,
        };
        // Lanes (011): each beyond the first holds back its live position-bound state and adds a
        // decode row and a paused chunk's hidden state to the scratch.
        let budget_for = |lanes: usize| {
            let rows = (lanes - 1) as u64;
            // A paused chunk of another lane holds its hidden state between layers, and the lane
            // two more recurrent states: one a pipelined step rolls back to, one its chunk does.
            let paused = u64::from(chunk_len(0)) * engine.model.hidden() as u64 * 4
                + 2 * engine.info.recurrent_bytes;
            // Each lane beyond the first also holds its drafter context (030); the warm-up's peak
            // counted the first one.
            let contexts = rows * engine.info.drafter.as_ref().map_or(0, |d| d.context_bytes);
            let scratch = |depth: u32| {
                let row = engine.model.chunk_scratch(1, depth) + paused;
                scratch(depth) + ((rows * row) as f64 * SCRATCH_MARGIN) as u64 + contexts
            };
            budget::with_state_capacity(plan, weights, &scratch, &engine.info)
                .map(|b| budget::with_lanes(b, lanes, &engine.info))
        };
        let whole = engine.part == Part::whole(engine.info.layers() as usize);
        let fits = |b: &MemoryBudget, lanes: usize| {
            lanes == 1 || b.kv_capacity_tokens as u64 >= lanes as u64 * u64::from(MIN_LANE_TOKENS)
        };
        let (lanes, budget) = match options.lanes.filter(|_| whole) {
            Some(requested) => {
                let requested = requested.clamp(1, MAX_LANES);
                let budget = budget_for(requested)?;
                if !fits(&budget, requested) {
                    let fits = (1..requested)
                        .rev()
                        .find(|&n| budget_for(n).is_ok_and(|b| fits(&b, n)))
                        .unwrap_or(1);
                    return Err(NuvecLoadError::Lanes {
                        requested,
                        capacity: budget.kv_capacity_tokens,
                        fits,
                    });
                }
                (requested, budget)
            }
            None if whole => match budget_for(2) {
                Ok(two) if fits(&two, 2) => (2, two),
                _ => (1, budget_for(1)?),
            },
            None => (1, budget_for(1)?),
        };
        engine.ledger.budget = budget;
        engine.set_lanes(lanes);
        tensor::memory::set_memory_limit(budget.budget);
        // D062: wire the weights only when they fit what was available before the load;
        // wiring into a system already short of memory starves everything else.
        if budget::may_wire(device, weights) {
            tensor::memory::set_wired_limit(weights.min(device.recommended_working_set));
        } else {
            eprintln!(
                "note: weights not wired: {} of weights exceed the {} available (after a {} margin for other apps); macOS may page them out under pressure",
                budget::gib(weights),
                budget::gib(device.available_room().unwrap_or(0)),
                budget::gib(budget::margin(device.physical))
            );
        }
        engine.info.numerics = if options.cpu {
            crate::engine::Numerics {
                nax: false,
                gpu_arch: "cpu".into(),
            }
        } else {
            tensor::numerics()
        };
        if tensor::on_gpu() {
            let key = calibration::Key {
                device: format!(
                    "{}, {} GiB",
                    tensor::gpu_info_string(c"device_name").unwrap_or_else(|| "unknown GPU".into()),
                    device.physical.div_ceil(GIB)
                ),
                family: engine.info.family.clone(),
                quantization: calibration::quantization_key(&config),
                numerics: engine.info.numerics.clone(),
                identity: engine.info.identity.clone(),
            };
            // A stored calibration is reused when it covers this capacity (a smaller budget
            // timed fewer depths).
            let deepest = calibration::depths(engine.ledger.budget.kv_capacity_tokens)
                .last()
                .copied()
                .unwrap_or(0);
            let stored = options
                .calibration
                .as_deref()
                .and_then(|path| calibration::load(path, &key))
                .filter(|t| t.points().last().is_some_and(|&(d, _)| d >= deepest));
            engine.info.time_len = match stored {
                Some(time_len) => time_len,
                None => {
                    let time_len = engine.calibrate()?;
                    if let Some(path) = &options.calibration {
                        // Not fatal: the next start calibrates again.
                        let _ = calibration::store(path, &key, &time_len);
                    }
                    time_len
                }
            };
        }
        Ok(engine)
    }

    /// `prefill_chunk` that also stores checkpoints of the position-bound state at the
    /// chunk-relative positions `at` (ascending, inside `(0, tokens.len())`), captured by running
    /// the Gated DeltaNet recurrence in segments that thread the state (003 SC-009; projections
    /// and convolution unchanged). `Ok(None)` when cancelled (nothing changed, as
    /// `prefill_chunk`). Attention-only models store zero-byte checkpoints.
    ///
    /// Measurement only (003 T040, `tests/segmented_scan.rs`): segmenting leaves the chunk's
    /// outputs bit-identical, and on the CPU the captured state equals a chunk end's; on Metal it
    /// does not (projections and attention of a chunk of another length differ in the last bits),
    /// so it is not part of the [`Engine`] contract and the cache must keep chunk ends.
    #[doc(hidden)]
    pub fn prefill_chunk_with_checkpoints(
        &mut self,
        tokens: &[u32],
        cancel: &CancelToken,
        at: &[u32],
    ) -> Result<Option<Vec<Checkpoint>>, EngineError> {
        if at.windows(2).any(|w| w[0] >= w[1])
            || at.first().is_some_and(|&a| a == 0)
            || at.last().is_some_and(|&a| a as usize >= tokens.len())
        {
            let len = self.ledger.len();
            return Err(EngineError::InvalidRange {
                start: len + at.first().copied().unwrap_or(0),
                end: len + at.last().copied().unwrap_or(0),
                len: len + tokens.len() as u32,
            });
        }
        let growth = self
            .ledger
            .plan_append_with_checkpoints(tokens.len() as u32, at.len() as u32)?;
        let capacity = self.grow(growth)?;
        self.step = None;
        let start = self.hybrid().then(|| self.seq.recurrent());
        let splits: Vec<i32> = if self.hybrid() {
            at.iter().map(|&a| a as i32).collect()
        } else {
            Vec::new()
        };
        let mut captured = Vec::new();
        let outcome = self.run_chunk(tokens, cancel, &splits, &mut captured);
        if !matches!(outcome, Ok(true)) {
            if let Some(start) = start {
                self.seq.set_recurrent(start);
            }
            self.shrink_back(capacity)?;
            tensor::memory::clear_cache();
            return outcome.map(|_| None);
        }
        let base = self.ledger.len();
        self.ledger.advance(tokens.len() as u32);
        self.record(base, tokens);
        let bytes = self.info.recurrent_bytes;
        let mut states = captured.into_iter();
        let checkpoints = at
            .iter()
            .map(|&a| {
                let state = if self.hybrid() {
                    states.next().unwrap_or_default()
                } else {
                    Vec::new()
                };
                self.ledger.commit_checkpoint_at(base + a, bytes, state)
            })
            .collect();
        tensor::memory::clear_cache();
        Ok(Some(checkpoints))
    }

    /// The selected lane's drafter context: its first and end position (030 tests).
    #[doc(hidden)]
    pub fn draft_span(&self) -> Option<(u32, u32)> {
        self.draft.as_ref().map(DraftContext::span)
    }

    /// Runs `lanes` empty lanes whatever the capacity (tests: the tiny fixtures hold fewer than
    /// [`MIN_LANE_TOKENS`] positions).
    #[doc(hidden)]
    pub fn force_lanes(&mut self, lanes: usize) {
        self.reset_all();
        self.set_lanes(lanes.clamp(1, MAX_LANES));
    }

    /// The arrays of checkpoint `id` as f32 values (conv tail, then delta state, per linear
    /// layer), for bitwise comparisons in tests.
    #[doc(hidden)]
    pub fn checkpoint_values(&self, id: CheckpointId) -> Result<Vec<Vec<f32>>, EngineError> {
        let state = self
            .ledger
            .checkpoint_data(id)
            .ok_or(EngineError::UnknownCheckpoint(id))?;
        let mut out = Vec::new();
        for r in state {
            for a in r.conv.iter().chain(&r.state) {
                out.push(
                    tensor::astype(a, tensor::Dtype::Float32)
                        .and_then(|a| a.to_vec_f32())
                        .map_err(backend)?,
                );
            }
        }
        Ok(out)
    }

    /// Arms a failure at `at` for the next call that reaches it (test hook, FR-028/SC-012).
    #[doc(hidden)]
    pub fn fail_next(&mut self, at: FailPoint) {
        self.fail = Some(at);
    }

    /// Whether an armed failure strikes at `at` (disarming it).
    fn strikes(&mut self, at: FailPoint) -> bool {
        if self.fail == Some(at) {
            self.fail = None;
            return true;
        }
        false
    }

    /// The `chunk_len` time bound (FR-007, D050): one attention layer at each calibration depth
    /// and one linear layer (depth-independent), the slower of the two per chunk length.
    fn calibrate(&self) -> tensor::Result<TimeLen> {
        let kinds = &self.info.layer_kinds;
        let first = |kind: LayerKind| (self.part.first..self.part.end).find(|&i| kinds[i] == kind);
        let attention = first(LayerKind::FullAttention);
        let linear = first(LayerKind::Linear);
        let mut linear_ms: HashMap<u32, f64> = HashMap::new();
        let depths = calibration::depths(self.ledger.budget.kv_capacity_tokens);
        let debug = std::env::var_os("MRUN_CALIBRATION_DEBUG").is_some();
        let result = calibration::search(&depths, |depth, len| {
            let mut ms: f64 = 0.0;
            if let Some(i) = attention {
                let t = self.time_layer(i, len, depth)?;
                if debug {
                    eprintln!(
                        "calibration: attention layer {i}, {len} tokens at {depth}: {t:.1} ms"
                    );
                }
                ms = ms.max(t);
            }
            if let Some(i) = linear {
                let t = match linear_ms.get(&len) {
                    Some(t) => *t,
                    None => {
                        let t = self.time_layer(i, len, 0)?;
                        if debug {
                            eprintln!("calibration: linear layer {i}, {len} tokens: {t:.1} ms");
                        }
                        linear_ms.insert(len, t);
                        t
                    }
                };
                ms = ms.max(t);
            }
            Ok(ms)
        });
        tensor::memory::clear_cache();
        result
    }

    /// Milliseconds layer `i` takes for a `len`-token chunk at `depth`: the fastest of three
    /// timed runs after an untimed one that compiles the kernels (D059), on zeroed state.
    fn time_layer(&self, i: usize, len: u32, depth: u32) -> tensor::Result<f64> {
        let ids = Array::from_u32(&vec![0; len as usize], &[1, len as i32]);
        let x = if self.part.head {
            self.model.embed(&ids)?
        } else {
            tensor::zeros(
                &[1, len as i32, self.model.hidden()],
                self.model.state_dtype(),
            )?
        };
        x.eval()?;
        let mut best = f64::MAX;
        for run in 0..4 {
            let mut state = match self.info.layer_kinds[i] {
                LayerKind::FullAttention => LayerState::Kv(LayerKv::zeroed(
                    self.info.kv_heads,
                    self.info.head_dim,
                    depth + len,
                    self.model.state_dtype(),
                )?),
                LayerKind::Linear => LayerState::Recurrent(Recurrent::default()),
            };
            let start = Instant::now();
            let out = self.model.layer(i, &x, &mut state, Pos::at(depth as i32))?;
            let mut outputs = state.arrays();
            outputs.push(&out);
            tensor::eval(&outputs)?;
            if run > 0 {
                best = best.min(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        Ok(best)
    }

    /// A chunk and one decode step through every layer: reads all weights and compiles the
    /// kernels. Returns the bytes the allocator holds afterwards without sequence state (the
    /// weights), and the peak above them and the warm-up's own state during the warm-up.
    fn warm_up(&mut self) -> Result<(u64, u64), NuvecLoadError> {
        // A multi-token chunk and one decode step: both paths compile their kernels and
        // compiled ops here instead of on the first request (≈ 1 s on Qwen3.6).
        let backend = |e: EngineError| tensor::Error(e.to_string());
        // The budget is computed from the warm-up's measurements: allow it 256 positions.
        let planned = self.ledger.budget;
        self.ledger.budget.kv_capacity_tokens = 256;
        self.ledger.budget.state_capacity_bytes = u64::MAX;
        self.ledger.set_limit(256).map_err(backend)?;
        tensor::memory::reset_peak();
        self.prefill_chunk(&[0; 16], &CancelToken::new())
            .map_err(backend)?;
        let step = self.forward(0).map_err(backend)?;
        self.current_logits(&step).map_err(backend)?.eval()?;
        // A draft (030): its kernels compile here and its scratch counts in the peak.
        if self.drafter.is_some() {
            let max = self.max_draft();
            self.draft(0, max).map_err(backend)?;
        }
        // The state is accounted for separately; the peak counts it once already.
        let state = self.ledger.allocated();
        self.reset();
        self.ledger.set_limit(0).map_err(backend)?;
        self.ledger.budget = planned;
        tensor::memory::clear_cache();
        let weights = tensor::memory::active();
        let peak = tensor::memory::peak().saturating_sub(weights + state);
        Ok((weights, peak))
    }

    fn hybrid(&self) -> bool {
        self.ledger.hybrid()
    }

    /// Gives the engine `lanes` lanes, all empty, lane 0 selected.
    fn set_lanes(&mut self, lanes: usize) {
        self.select_lane(0);
        self.chunks = (0..lanes).map(|_| None).collect();
        self.parked = (0..lanes)
            .map(|i| {
                (i != 0).then(|| LaneSlot {
                    seq: self.seq.fresh(),
                    step: None,
                    draws: 0,
                    ids: Vec::new(),
                    verified: Vec::new(),
                    draft: self.drafter.as_ref().map(DFlash::context),
                })
            })
            .collect();
        self.ledger.set_lanes(lanes);
    }

    fn select_lane(&mut self, lane: usize) {
        if lane == self.lane {
            return;
        }
        let incoming = self.parked[lane].take().expect("a parked lane");
        let outgoing = LaneSlot {
            seq: std::mem::replace(&mut self.seq, incoming.seq),
            step: std::mem::replace(&mut self.step, incoming.step),
            draws: std::mem::replace(&mut self.draws, incoming.draws),
            ids: std::mem::replace(&mut self.ids, incoming.ids),
            verified: std::mem::replace(&mut self.verified, incoming.verified),
            draft: std::mem::replace(&mut self.draft, incoming.draft),
        };
        self.parked[self.lane] = Some(outgoing);
        self.lane = lane;
        self.ledger.select(lane);
    }

    /// Takes lane `lane`'s sequence state out (for a batched step); [`Self::put_seq`] returns it.
    fn take_seq(&mut self, lane: usize) -> SeqState {
        let fresh = self.seq.fresh();
        if lane == self.lane {
            std::mem::replace(&mut self.seq, fresh)
        } else {
            let slot = self.parked[lane].as_mut().expect("a parked lane");
            std::mem::replace(&mut slot.seq, fresh)
        }
    }

    fn put_seq(&mut self, lane: usize, seq: SeqState) {
        if lane == self.lane {
            self.seq = seq;
        } else {
            self.parked[lane].as_mut().expect("a parked lane").seq = seq;
        }
    }

    /// One decode step of `lanes` (distinct), fed `ids` `[1, B]` at `offsets`: the logits of
    /// every row `[B, vocab]`, each row as the lane's own step computes it.
    fn feed_lanes(
        &mut self,
        lanes: &[usize],
        ids: &Array,
        offsets: &[i32],
    ) -> Result<Array, EngineError> {
        let mut seqs: Vec<SeqState> = lanes.iter().map(|&l| self.take_seq(l)).collect();
        let positions: Vec<Pos<'static>> = seqs
            .iter()
            .zip(offsets)
            .map(|(s, &at)| Pos {
                at,
                delta: s.media.delta(at as u32),
                runs: &[],
            })
            .collect();
        let first = self.part.first;
        let model = &self.model;
        let tapped = &self.tapped;
        let mut taps = Vec::new();
        let out = tensor::with_row_exact(|| {
            let mut x = model.embed(ids)?;
            for k in 0..seqs[0].layers.len() {
                let mut states: Vec<&mut LayerState> =
                    seqs.iter_mut().map(|s| &mut s.layers[k]).collect();
                x = model.layer_lanes(first + k, &x, &mut states, &positions)?;
                if tapped.get(k) == Some(&true) {
                    taps.push(x.clone());
                }
            }
            model.logits_rows(&x)
        })
        .map_err(backend);
        for (&lane, seq) in lanes.iter().zip(seqs) {
            self.put_seq(lane, seq);
        }
        if out.is_ok() && !taps.is_empty() {
            let features = tensor::concatenate(&taps.iter().collect::<Vec<_>>(), 2);
            for (b, (&lane, &at)) in lanes.iter().zip(offsets).enumerate() {
                let row = features.as_ref().map_err(Clone::clone).and_then(|f| {
                    let shape = f.shape();
                    tensor::slice(f, &[0, b as i32, 0], &[1, b as i32 + 1, shape[2]])
                });
                let mut ctx = self.lane_draft(lane).take();
                feed_drafter(self.drafter.as_ref(), ctx.as_mut(), row, at as u32);
                *self.lane_draft(lane) = ctx;
            }
        }
        out
    }

    /// Lane `lane`'s drafter context slot (030).
    fn lane_draft(&mut self, lane: usize) -> &mut Option<DraftContext> {
        if lane == self.lane {
            &mut self.draft
        } else {
            &mut self.parked[lane].as_mut().expect("a parked lane").draft
        }
    }

    /// Feeds the selected lane's drafter the concatenated `taps` of positions from `at` (030).
    fn feed_draft(&mut self, taps: Vec<Array>, at: u32) {
        if taps.is_empty() {
            return;
        }
        let features = tensor::concatenate(&taps.iter().collect::<Vec<_>>(), 2);
        feed_drafter(self.drafter.as_ref(), self.draft.as_mut(), features, at);
    }

    /// Row `r` of `[B, vocab]` logits as `[1, vocab]`.
    fn row(logits: &Array, r: i32) -> Result<Array, EngineError> {
        let vocab = logits.dim(-1);
        tensor::slice(logits, &[r, 0], &[r + 1, vocab]).map_err(backend)
    }

    /// Undoes the chunk in flight on its lane.
    fn undo_chunk(&mut self, chunk: InFlight) -> Result<(), EngineError> {
        self.select_lane(chunk.lane);
        if let Some(start) = chunk.start {
            self.seq.set_recurrent(start);
        }
        let undone = self.shrink_back(chunk.capacity);
        tensor::memory::clear_cache();
        undone
    }

    /// Drops a pending step, un-feeding its token.
    fn roll_back(&mut self) {
        if let Some(pending) = self.ledger.roll_back()
            && let Some(before) = pending.before
        {
            self.seq.set_recurrent(before);
        }
    }

    /// Records `tokens` as the ids of the positions from `start` (dropping any after).
    fn record(&mut self, start: u32, tokens: &[u32]) {
        self.ids.truncate(start as usize);
        debug_assert_eq!(self.ids.len(), start as usize, "token ids out of step");
        self.ids.extend_from_slice(tokens);
    }

    /// The last [`sample::REPETITION_CONTEXT`] token ids of the live sequence as a `[1, n]`
    /// array, when `sampling` has a repetition penalty (`None` otherwise: nothing is built).
    fn recent_ids(&self, sampling: &Sampling) -> Option<Array> {
        if sampling.repetition_penalty == 1.0 {
            return None;
        }
        let live = &self.ids[..(self.ledger.len() as usize).min(self.ids.len())];
        let recent = &live[live.len().saturating_sub(sample::REPETITION_CONTEXT)..];
        (!recent.is_empty()).then(|| Array::from_u32(recent, &[1, recent.len() as i32]))
    }

    /// Rolls back a pending step and grows to `capacity` (planned by `plan_append`). Returns the
    /// capacity before, to give the growth back if the work fails.
    fn grow(&mut self, capacity: Option<u32>) -> Result<u32, EngineError> {
        self.roll_back();
        let before = self.seq.capacity();
        if let Some(capacity) = capacity {
            let len = self.ledger.len();
            if let Err(e) = self.seq.resize(len, capacity) {
                return Err(self.ledger.poison(format!("growth: {}", e.0)));
            }
            self.ledger.set_capacity(capacity);
        }
        Ok(before)
    }

    /// Gives back growth after a cancelled or failed append.
    fn shrink_back(&mut self, capacity: u32) -> Result<(), EngineError> {
        if self.seq.capacity() == capacity {
            return Ok(());
        }
        let len = self.ledger.len();
        match self.seq.resize(len, capacity) {
            Ok(()) => {
                self.ledger.set_capacity(capacity);
                Ok(())
            }
            Err(e) => Err(self.ledger.poison(format!("undoing growth: {}", e.0))),
        }
    }

    /// Runs `ids` (`[1, 1]`, possibly lazy) through the model at `len()` and returns the
    /// logits; the caller counts the position once the step is committed.
    fn feed(&mut self, ids: &Array) -> Result<Array, EngineError> {
        let mut x = self.input(ids)?;
        let pos = self.text_pos();
        let mut taps = Vec::new();
        for (k, layer) in self.seq.layers.iter_mut().enumerate() {
            x = self
                .model
                .layer(self.part.first + k, &x, layer, pos)
                .map_err(backend)?;
            if self.tapped.get(k) == Some(&true) {
                taps.push(x.clone());
            }
        }
        // The drafter sees the position at once (a rolled-back step is cut on the next feed).
        self.feed_draft(taps, pos.at as u32);
        // A step's hidden state goes round the ring: back to the head for the logits.
        match (&self.pipe, self.part.head) {
            (None, _) if !self.part.head => Ok(x),
            (None, _) => self.model.logits(&x),
            // `recv_like` takes the send for its shape only, not as a dependency: the send is
            // evaluated first, or it would never run.
            (Some(pipe), true) => pipe
                .send(&x)
                .and_then(|sent| sent.eval().map(|()| sent))
                .and_then(|sent| pipe.recv_like(&sent))
                .and_then(|back| self.model.logits(&back)),
            (Some(pipe), false) => pipe.send(&x),
        }
        .map_err(backend)
    }

    /// The input of this part's first layer: the embedded `ids`, or (a part without the head)
    /// the hidden state from the previous part, or zeros before a ring is attached (warm-up).
    fn input(&self, ids: &Array) -> Result<Array, EngineError> {
        if self.part.head {
            return self.model.embed(ids).map_err(backend);
        }
        let shape = [1, ids.dim(1), self.model.hidden()];
        match &self.pipe {
            Some(pipe) => pipe.recv(&shape, self.model.state_dtype()),
            None => tensor::zeros(&shape, self.model.state_dtype()),
        }
        .map_err(backend)
    }

    /// Where the next text row sits: slot `len()`, its delta after the images so far.
    fn text_pos(&self) -> Pos<'static> {
        let at = self.ledger.len();
        Pos {
            at: at as i32,
            delta: self.seq.media.delta(at),
            runs: &[],
        }
    }

    /// The input of a prefill chunk at `len()` (009): token embeddings, with the rows of images
    /// replaced by their features (media keys are ids at or above the vocabulary).
    fn chunk_input(&self, tokens: &[u32]) -> Result<Array, EngineError> {
        let at = self.ledger.len();
        let len = tokens.len() as i32;
        let rows = self.seq.media.rows(at, tokens.len() as u32);
        if rows.is_empty() {
            return self.input(&Array::from_u32(tokens, &[1, len]));
        }
        if !self.part.head {
            return Err(EngineError::Backend(
                "images need the whole model on one Mac".into(),
            ));
        }
        let vocab = self.info.vocab;
        let text: Vec<u32> = tokens
            .iter()
            .map(|&t| if t >= vocab { 0 } else { t })
            .collect();
        let x = self.input(&Array::from_u32(&text, &[1, len]))?;
        let hidden = self.model.hidden();
        let mut pieces = Vec::new();
        let mut row = 0i32;
        for (r, span, first, count) in rows {
            let (r, first, count) = (r as i32, first as i32, count as i32);
            if r > row {
                pieces.push(tensor::slice(&x, &[0, row, 0], &[1, r, hidden]).map_err(backend)?);
            }
            let features = self.seq.media.spans[span]
                .features
                .as_ref()
                .ok_or_else(|| EngineError::Backend("an image reached prefill unencoded".into()))?;
            let rows = tensor::slice(features, &[first, 0], &[first + count, hidden])
                .and_then(|f| tensor::reshape(&f, &[1, count, hidden]))
                .and_then(|f| tensor::astype(&f, x.dtype()))
                .map_err(backend)?;
            pieces.push(rows);
            row = r + count;
        }
        if row < len {
            pieces.push(tensor::slice(&x, &[0, row, 0], &[1, len, hidden]).map_err(backend)?);
        }
        tensor::concatenate(&pieces.iter().collect::<Vec<_>>(), 1).map_err(backend)
    }

    /// The layers this engine runs.
    pub fn part(&self) -> Part {
        self.part
    }

    /// Attaches (or detaches) the ring to the pool's other parts (026). Decode is not pipelined
    /// while a ring is attached: every step crosses every part.
    pub fn set_pipe(&mut self, pipe: Option<Pipe>) {
        self.pipe = pipe;
    }

    pub fn pipe(&self) -> Option<&Pipe> {
        self.pipe.as_ref()
    }

    /// Whether `n` positions can be appended now, without appending them (026: a pool's host
    /// checks before its stages start receiving).
    pub fn check_append(&self, n: u32) -> Result<(), EngineError> {
        self.ledger.plan_append(n).map(|_| ())
    }

    /// How many positions' worth of state this engine's budget holds.
    pub fn state_tokens(&self) -> u64 {
        self.ledger.budget.state_capacity_bytes / self.info.kv_bytes_per_token.max(1)
    }

    /// Lowers the state budget to `tokens` positions' worth (026: a pool's host never plans
    /// state one of its stages cannot hold, so the stages' moves succeed whenever the host's do).
    pub fn cap_state_tokens(&mut self, tokens: u64) {
        let budget = &mut self.ledger.budget;
        let bytes = tokens.saturating_mul(self.info.kv_bytes_per_token.max(1));
        budget.state_capacity_bytes = budget.state_capacity_bytes.min(bytes);
        budget.kv_capacity_tokens = budget
            .kv_capacity_tokens
            .min(u32::try_from(tokens).unwrap_or(u32::MAX));
    }

    /// After a position was committed: as mlx-lm, return decode's freed buffers every 256.
    /// Drops a verify's states and closes its trim window: the next state change makes them
    /// stale, and they count against the budget only while it is open (III).
    fn end_verify(&mut self) {
        if !self.verified.is_empty() || self.ledger.window().is_some() {
            self.verified.clear();
            self.ledger.close_window();
        }
    }

    fn after_decode_step(&self) {
        if self.ledger.len().is_multiple_of(DECODE_CLEAR_INTERVAL) {
            tensor::memory::clear_cache();
        }
    }

    /// Random key for the next draw: from the seed and the draw count, or from the clock.
    fn key(&mut self, seed: Option<u64>) -> tensor::Result<Array> {
        self.draws += 1;
        key_at(seed, self.draws)
    }

    /// The token ids of positions before `end` that a repetition penalty sees (as
    /// [`Self::recent_ids`] at that length).
    fn recent_before(&self, sampling: &Sampling, end: usize) -> Option<Array> {
        if sampling.repetition_penalty == 1.0 {
            return None;
        }
        let live = &self.ids[..end.min(self.ids.len())];
        let recent = &live[live.len().saturating_sub(sample::REPETITION_CONTEXT)..];
        (!recent.is_empty()).then(|| Array::from_u32(recent, &[1, recent.len() as i32]))
    }

    /// The verify pass (006): `tokens` through every layer at `len()` as decode steps would run
    /// them, row by row exact; returns the logits of every row and the recurrent state after each
    /// position but the last (the last is live).
    fn run_verify(&mut self, tokens: &[u32]) -> Result<Verified, EngineError> {
        let pos = self.text_pos();
        let ids = Array::from_u32(tokens, &[1, tokens.len() as i32]);
        tensor::with_row_exact(|| {
            let mut x = self.model.embed(&ids).map_err(backend)?;
            let mut states = vec![Vec::new(); tokens.len() - 1];
            let mut taps = Vec::new();
            for (k, layer) in self.seq.layers.iter_mut().enumerate() {
                let mut at = Vec::new();
                x = self
                    .model
                    .layer_verify(self.part.first + k, &x, layer, pos, &mut at)
                    .map_err(backend)?;
                for (state, r) in states.iter_mut().zip(at) {
                    state.push(r);
                }
                if self.tapped.get(k) == Some(&true) {
                    taps.push(x.clone());
                }
            }
            let rows = self.model.logits_rows(&x).map_err(backend)?;
            let mut outputs: Vec<&Array> = vec![&rows];
            for layer in &self.seq.layers {
                outputs.extend(layer.arrays());
            }
            outputs.extend(
                states
                    .iter()
                    .flatten()
                    .flat_map(|r| r.conv.iter().chain(&r.state)),
            );
            tensor::eval(&outputs).map_err(backend)?;
            Ok((rows, states, taps))
        })
    }

    fn current_logits(&self, step: &Step) -> Result<&Array, EngineError> {
        match &self.step {
            Some((id, logits)) if *id == step.id() => Ok(logits),
            _ => Err(EngineError::InvalidStep),
        }
    }

    /// The prefill layer loop over `tokens` at `len()`; `Ok(false)` when cancelled. With
    /// `splits` (chunk-relative), linear layers run their recurrence in segments and the
    /// recurrent state after each split is pushed to `captured` (per split, in layer order).
    fn run_chunk(
        &mut self,
        tokens: &[u32],
        cancel: &CancelToken,
        splits: &[i32],
        captured: &mut Vec<RecurrentState>,
    ) -> Result<bool, EngineError> {
        let mut x = self.chunk_input(tokens)?;
        let at = self.ledger.len();
        let runs = self.seq.media.runs(at, tokens.len() as u32);
        let delta = self.seq.media.delta(at);
        let mut previous: Option<Array> = None;
        captured.clear();
        captured.resize_with(splits.len(), Vec::new);
        for k in 0..self.seq.layers.len() {
            let i = self.part.first + k;
            let layer = &mut self.seq.layers[k];
            let mut at_splits = Vec::new();
            let pos = Pos {
                at: at as i32,
                delta,
                runs: &runs,
            };
            x = if splits.is_empty() {
                self.model.layer(i, &x, layer, pos)
            } else {
                self.model
                    .layer_split(i, &x, layer, pos, splits, &mut at_splits)
            }
            .map_err(backend)?;
            let mut outputs = layer.arrays();
            outputs.push(&x);
            for r in &at_splits {
                outputs.extend(r.conv.iter().chain(&r.state));
            }
            tensor::async_eval(&outputs).map_err(backend)?;
            if self.strikes(FailPoint::PrefillLayer(i)) {
                return Err(injected());
            }
            for (states, r) in captured.iter_mut().zip(at_splits) {
                states.push(r);
            }
            // Layer i is queued; wait for layer i − 1, then check for cancellation.
            if let Some(done) = previous.replace(x.clone()) {
                done.eval().map_err(backend)?;
                if cancel.is_cancelled() {
                    return Ok(false);
                }
            }
        }
        // A prefill chunk's hidden state goes on to the next part; the last part's is not needed.
        if let Some(pipe) = self.pipe.as_ref().filter(|p| !p.last()) {
            x = pipe.send(&x).map_err(backend)?;
        }
        x.eval().map_err(backend)?;
        Ok(true)
    }
}

/// Scratch beyond the warm-up peak (FR-017) for sequences up to `depth` positions: the family's
/// largest chunk term over the chunk schedule up to `depth`, one attention layer at `depth` (the
/// transient of growth and moves), one recurrent state (rollback copy) and the cache limit.
fn analytic_scratch(
    model: &dyn Model,
    info: &ModelInfo,
    part: Part,
    depth: u32,
    encoder: u64,
) -> u64 {
    let depth = depth.min(info.max_position);
    let mut at = 0u32;
    // Encoding an image (009) runs between chunks: its activations never meet a chunk's.
    let mut chunk = encoder;
    while at < depth {
        let len = chunk_len(at);
        chunk = chunk.max(model.chunk_scratch(len, at));
        at = at.saturating_add(len);
    }
    let attention = part_kinds(&info.layer_kinds, part)
        .filter(|k| *k == LayerKind::FullAttention)
        .count() as u64;
    let layer = (info.kv_bytes_per_token / attention.max(1)) * u64::from(depth);
    chunk + layer + info.recurrent_bytes + CACHE_LIMIT
}

/// The kinds of `part`'s layers.
fn part_kinds(kinds: &[LayerKind], part: Part) -> impl Iterator<Item = LayerKind> + '_ {
    kinds[part.first..part.end].iter().copied()
}

impl Engine for NuvecEngine {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn budget(&self) -> &MemoryBudget {
        &self.ledger.budget
    }

    fn len(&self) -> u32 {
        self.ledger.len()
    }

    fn allocated(&self) -> u64 {
        self.ledger.allocated()
    }

    fn set_limit(&mut self, total: u32) -> Result<(), EngineError> {
        self.end_verify();
        self.ledger.set_limit(total)?;
        self.draws = 0;
        Ok(())
    }

    fn set_media(&mut self, spans: &[ImageSpan]) -> Result<(), EngineError> {
        if !spans.is_empty() && self.vision.is_none() {
            return Err(EngineError::Backend("this model reads no images".into()));
        }
        let len = self.ledger.len();
        let old = std::mem::take(&mut self.seq.media.spans);
        self.seq.media.spans = spans
            .iter()
            .map(|s| state::MediaSpan {
                start: s.start,
                height: s.height,
                width: s.width,
                key: s.key,
                features: old
                    .iter()
                    .find(|o| o.key == s.key && o.start == s.start)
                    .and_then(|o| o.features.clone()),
            })
            .collect();
        self.seq.media.committed(len);
        Ok(())
    }

    fn needs_image(&self, key: u64) -> bool {
        let len = self.ledger.len();
        self.seq
            .media
            .spans
            .iter()
            .any(|s| s.key == key && s.end() > len && s.features.is_none())
    }

    fn encode_image(
        &mut self,
        key: u64,
        pixels: &Pixels,
        cancel: &CancelToken,
    ) -> Result<bool, EngineError> {
        let Some(tower) = &self.vision else {
            return Err(EngineError::Backend("this model reads no images".into()));
        };
        let c = tower.config();
        let (gh, gw) = (pixels.grid_h as i32, pixels.grid_w as i32);
        let width = c.patch_values();
        if pixels.data.len() != (gh * gw * width) as usize || gh % c.merge != 0 || gw % c.merge != 0
        {
            return Err(EngineError::Backend(
                "image patches do not match their grid".into(),
            ));
        }
        let tokens = (gh / c.merge) * (gw / c.merge);
        if tokens as u32 > MAX_IMAGE_TOKENS {
            return Err(EngineError::Backend("image above the token cap".into()));
        }
        let patches = Array::from_f32(&pixels.data, &[gh * gw, width]);
        let Some(features) = tower.encode(&patches, gh, gw, cancel).map_err(backend)? else {
            return Ok(false);
        };
        for s in self.seq.media.spans.iter_mut().filter(|s| s.key == key) {
            if (s.height * s.width) as i32 != tokens {
                return Err(EngineError::Backend("image span and grid disagree".into()));
            }
            s.features = Some(features.clone());
        }
        Ok(true)
    }

    fn prefill_chunk(&mut self, tokens: &[u32], cancel: &CancelToken) -> Result<bool, EngineError> {
        if tokens.is_empty() {
            self.end_verify();
            return Ok(true);
        }
        self.begin_chunk(tokens)?;
        match self.advance_chunk(usize::MAX, cancel)? {
            ChunkProgress::Done => Ok(true),
            ChunkProgress::Cancelled => Ok(false),
            ChunkProgress::More => unreachable!("every layer ran"),
        }
    }

    fn forward(&mut self, last: u32) -> Result<Step, EngineError> {
        self.end_verify();
        self.ledger.usable()?;
        if let Some(pending) = self.ledger.reuse_pending(|p| p.token == last) {
            self.step = Some((pending.id, pending.logits));
            return Ok(Step::new(pending.id));
        }
        let growth = self.ledger.plan_append(1)?;
        let capacity = self.grow(growth)?;
        self.step = None;
        let before = self.hybrid().then(|| self.seq.recurrent());
        // With a ring attached the step finishes here: what is sent next on the ring must not
        // overtake the step's own transfers.
        let ring = self.pipe.is_some();
        let fed = self
            .feed(&Array::from_u32(&[last], &[1, 1]))
            .and_then(|logits| {
                if ring {
                    logits.eval().map_err(backend)
                } else {
                    tensor::async_eval(&[&logits]).map_err(backend)
                }
                .map(|()| logits)
            })
            .and_then(|logits| {
                if self.strikes(FailPoint::Forward) {
                    Err(injected())
                } else {
                    Ok(logits)
                }
            });
        let logits = match fed {
            Ok(logits) => logits,
            Err(e) => {
                if let Some(before) = before {
                    self.seq.set_recurrent(before);
                }
                self.shrink_back(capacity)?;
                return Err(e);
            }
        };
        self.ledger.advance(1);
        self.record(self.ledger.len() - 1, &[last]);
        self.after_decode_step();
        let id = self.ledger.id();
        self.step = Some((id, logits));
        Ok(Step::new(id))
    }

    fn sample(
        &mut self,
        step: Step,
        sampling: &Sampling,
        mask: Option<&TokenMask>,
    ) -> Result<u32, EngineError> {
        self.ledger.usable()?;
        let logits = self.current_logits(&step)?.clone();
        if let Some(mask) = mask {
            mask.check_width(self.info.vocab)?;
        }
        self.step = None;
        let mut key = if sampling.is_greedy() {
            None
        } else {
            Some(self.key(sampling.seed).map_err(backend)?)
        };
        let recent = self.recent_ids(sampling);
        let token = sample::sample(&logits, sampling, mask, recent.as_ref(), || {
            key.take().ok_or_else(|| tensor::Error("no key".into()))
        })
        .map_err(backend)?
        .ok_or(EngineError::EmptyMask)?;
        // The token goes to the GPU on its own first: encoded together with the next forward it
        // would share a command buffer with that forward's first kernels, and reading it back
        // would wait for those too (≈ 10 % of decode speed on Qwen3.6). mlx-lm gets the same
        // effect by sampling inside the previous step.
        tensor::async_eval(&[&token]).map_err(backend)?;
        // Pipeline: feed the sampled token while it is still lazy, when the allocation has room.
        // Nothing is recorded until the token is read back; a failure restores the state.
        let before = self.hybrid().then(|| self.seq.recurrent());
        let next = if self.ledger.has_room() && self.pipe.is_none() {
            let fed = tensor::reshape(&token, &[1, 1])
                .map_err(backend)
                .and_then(|ids| self.feed(&ids))
                .and_then(|next| tensor::async_eval(&[&next]).map_err(backend).map(|()| next));
            Some(fed)
        } else {
            None
        };
        let value = if self.strikes(FailPoint::SampleReadback) {
            Err(injected())
        } else {
            token.item_u32().map_err(backend)
        };
        match (value, next) {
            (Ok(value), Some(Ok(logits))) => {
                let id = self.ledger.id();
                self.ledger.push_pending(Pending {
                    token: value,
                    id,
                    logits,
                    before,
                });
                self.record(self.ledger.len() - 1, &[value]);
                self.after_decode_step();
                Ok(value)
            }
            (Ok(value), None) => Ok(value),
            (Err(e), _) | (_, Some(Err(e))) => {
                if let Some(before) = before {
                    self.seq.set_recurrent(before);
                }
                Err(e)
            }
        }
    }

    fn max_draft(&self) -> u32 {
        // Products and attention of up to `ROW_INDEPENDENT_ROWS` rows are row-exact on the GPU and
        // the CPU (nuvec 0.9.0); pools verify nothing.
        if self.pipe.is_some() || self.part != Part::whole(self.model.layers()) {
            return 0;
        }
        // nuvec keeps float32 attention at head sizes 192 and 256 serial-exact for 32 / gqa rows
        // only; the query heads are not known here, so the smallest such width.
        let rows = if self.model.state_dtype() == tensor::Dtype::Float32
            && matches!(self.info.head_dim, 192 | 256)
        {
            2
        } else {
            tensor::ROW_INDEPENDENT_ROWS as u32
        };
        rows - 1
    }

    fn verify(
        &mut self,
        last: u32,
        draft: &[u32],
        sampling: &Sampling,
    ) -> Result<Vec<u32>, EngineError> {
        self.ledger.usable()?;
        let max = self.max_draft();
        if max > 0 && draft.len() > max as usize {
            return Err(EngineError::DraftTooLong {
                draft: draft.len() as u32,
                max,
            });
        }
        if max == 0 {
            return verify_serially(self, last, draft, sampling);
        }
        let hybrid = self.hybrid();
        // Shorten the draft to what the reservation and the free memory hold (the recurrent
        // state after each position is kept until the next call).
        let mut n = draft.len();
        let growth = loop {
            if n == 0 {
                return verify_serially(self, last, &[], sampling);
            }
            let positions = n as u32 + 1;
            let states = if hybrid { positions } else { 0 };
            match self.ledger.plan_append_with_checkpoints(positions, states) {
                Ok(growth) => break growth,
                Err(EngineError::NeedsMemory { .. } | EngineError::ReservationExceeded { .. }) => {
                    n -= 1;
                }
                Err(e) => return Err(e),
            }
        };
        let draft = &draft[..n];
        self.verified.clear();
        // A pending step of `last` already holds row 0 (without growth, which rolls it back).
        let pending = match self.ledger.pending() {
            Some(p) if growth.is_none() && p.token == last => {
                Some((p.logits.clone(), p.before.clone()))
            }
            _ => None,
        };
        let reused = pending.is_some();
        let capacity = if reused {
            self.seq.capacity()
        } else {
            self.grow(growth)?
        };
        self.step = None;
        let start = if reused {
            self.ledger.len() - 1
        } else {
            self.ledger.len()
        };
        let fed: Vec<u32> = if reused {
            draft.to_vec()
        } else {
            [&[last][..], draft].concat()
        };
        let at_pass = hybrid.then(|| self.seq.recurrent());
        let run_at = self.ledger.len();
        let (rows, captured, taps) = match self.run_verify(&fed) {
            Ok(pass) => pass,
            Err(e) => {
                if let Some(state) = at_pass {
                    self.seq.set_recurrent(state);
                }
                if !reused {
                    self.shrink_back(capacity)?;
                }
                return Err(e);
            }
        };
        // Every row sampled as `sample` samples its position: its repetition window and draw.
        let mut logits = Vec::with_capacity(draft.len() + 1);
        if let Some((row0, _)) = &pending {
            logits.push(row0.clone());
        }
        let vocab = rows.dim(-1);
        for r in 0..fed.len() as i32 {
            logits.push(tensor::slice(&rows, &[r, 0], &[r + 1, vocab]).map_err(backend)?);
        }
        self.record(start, &[&[last][..], draft].concat());
        let mut samples = Vec::with_capacity(logits.len());
        for (r, row) in logits.iter().enumerate() {
            let recent = self.recent_before(sampling, start as usize + r + 1);
            let draw = self.draws + r as u64 + 1;
            let token = sample::sample(row, sampling, None, recent.as_ref(), || {
                key_at(sampling.seed, draw)
            })
            .map_err(backend)?
            .ok_or(EngineError::EmptyMask)?;
            samples.push(tensor::reshape(&token, &[1]).map_err(backend)?);
        }
        let samples = tensor::concatenate(&samples.iter().collect::<Vec<_>>(), 0)
            .and_then(|all| all.to_vec_u32())
            .map_err(backend)?;
        let kept = draft
            .iter()
            .zip(&samples)
            .take_while(|(d, s)| d == s)
            .count();
        let next = samples[kept];
        if !sampling.is_greedy() {
            self.draws += kept as u64 + 1;
        }
        // Commit `last` and the kept drafts; the state after them becomes live.
        if reused {
            self.ledger.reuse_pending(|_| true);
            self.ledger.advance(kept as u32);
        } else {
            self.ledger.advance(kept as u32 + 1);
        }
        self.record(start, &[&[last][..], &draft[..kept]].concat());
        // The drafter sees the committed rows (a reused pending step's row is already in it).
        let rows_kept = if reused { kept } else { kept + 1 };
        if rows_kept > 0 && !taps.is_empty() {
            let features = tensor::concatenate(&taps.iter().collect::<Vec<_>>(), 2).and_then(|f| {
                let width = f.dim(2);
                tensor::slice(&f, &[0, 0, 0], &[1, rows_kept as i32, width])
            });
            feed_drafter(self.drafter.as_ref(), self.draft.as_mut(), features, run_at);
        }
        if let Some(at_pass) = at_pass {
            let live = self.seq.recurrent();
            let mut states = Vec::with_capacity(draft.len() + 2);
            if let Some((_, before)) = pending {
                states.push(before.expect("a hybrid's pending step keeps its state"));
            }
            states.push(at_pass);
            states.extend(captured);
            states.push(live);
            states.truncate(kept + 2);
            self.seq.set_recurrent(states[kept + 1].clone());
            self.verified = states;
        }
        // Pipeline the final sample as `sample` does. With a drafter (030) the next round's
        // verify takes it as its first row instead: a pass of its own would cost a whole step
        // per round, and the drafter drafts from the positions before it.
        if self.ledger.has_room() && self.pipe.is_none() && self.drafter.is_none() {
            let before = hybrid.then(|| self.seq.recurrent());
            let fed = self
                .feed(&Array::from_u32(&[next], &[1, 1]))
                .and_then(|logits| {
                    tensor::async_eval(&[&logits])
                        .map_err(backend)
                        .map(|()| logits)
                });
            match fed {
                Ok(logits) => {
                    let id = self.ledger.id();
                    self.ledger.push_pending(Pending {
                        token: next,
                        id,
                        logits,
                        before,
                    });
                    self.record(self.ledger.len() - 1, &[next]);
                }
                Err(e) => {
                    if let Some(before) = before {
                        self.seq.set_recurrent(before);
                    }
                    return Err(e);
                }
            }
        }
        self.ledger.open_window(start);
        self.after_decode_step();
        let mut committed = draft[..kept].to_vec();
        committed.push(next);
        Ok(committed)
    }

    fn draft(&mut self, last: u32, max: u32) -> Result<Vec<u32>, EngineError> {
        let (Some(drafter), Some(ctx)) = (&self.drafter, &self.draft) else {
            return Ok(Vec::new());
        };
        // `last` sits at the pending step's position when it was fed already, else next.
        let at = match self.ledger.pending() {
            Some(p) if p.token == last => self.ledger.len() - 1,
            _ => self.ledger.len(),
        };
        drafter
            .draft(ctx, self.model.as_ref(), last, at, max as usize)
            .map_err(backend)
    }

    fn top_logprobs(&mut self, step: &Step, n: usize) -> Result<Vec<(u32, f32)>, EngineError> {
        let logits = self.current_logits(step)?;
        sample::top_logprobs(logits, n).map_err(backend)
    }

    fn trim(&mut self, len: u32) -> Result<(), EngineError> {
        let window = self.ledger.window();
        let cut = self.ledger.trim(len)?;
        self.ids.truncate(len as usize);
        if let Some(before) = cut.rolled_back.and_then(|p| p.before) {
            self.seq.set_recurrent(before);
        }
        // Inside a verify's window: the state after exactly `len` positions.
        if let Some(state) = window
            .and_then(|start| len.checked_sub(start))
            .and_then(|i| self.verified.get(i as usize))
        {
            self.seq.set_recurrent(state.clone());
        }
        if cut.cut_step {
            self.step = None;
        }
        if cut.zero_recurrent {
            drop(self.seq.take_recurrent());
        }
        if let Some(capacity) = cut.shrink
            && let Err(e) = self.seq.resize(len, capacity)
        {
            return Err(self.ledger.poison(format!("shrink: {}", e.0)));
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError> {
        self.end_verify();
        let bytes = self.ledger.plan_checkpoint()?;
        let recurrent = if self.hybrid() {
            // Before the pending step's rollback: the state at the confirmed length.
            let state = match self.ledger.pending() {
                Some(Pending {
                    before: Some(before),
                    ..
                }) => before.clone(),
                _ => self.seq.recurrent(),
            };
            state::eval(&state).map_err(backend)?;
            state
        } else {
            Vec::new()
        };
        self.roll_back();
        Ok(self.ledger.commit_checkpoint(bytes, recurrent))
    }

    fn restore(&mut self, id: CheckpointId) -> Result<(), EngineError> {
        self.end_verify();
        let position = self.ledger.plan_restore(id)?;
        let recurrent = self.ledger.checkpoint_data(id).cloned();
        self.roll_back();
        self.step = None;
        self.ledger.commit_restore(position);
        self.ids.truncate(position as usize);
        if self.hybrid()
            && let Some(recurrent) = recurrent
        {
            self.seq.set_recurrent(recurrent);
        }
        Ok(())
    }

    fn drop_checkpoint(&mut self, id: CheckpointId) {
        if self.ledger.drop_checkpoint(id) {
            tensor::memory::clear_cache();
        }
    }

    fn detach_tail(&mut self, at: u32) -> Result<Segment, EngineError> {
        self.end_verify();
        let extra = self.draft.as_ref().map_or(0, |c| c.tail_bytes(at));
        let plan = self.ledger.plan_detach_with(at, extra)?;
        self.roll_back();
        self.step = None;
        if self.strikes(FailPoint::Detach) {
            return Err(self.ledger.poison("detach: injected failure"));
        }
        let tail = match self.seq.detach(plan.start, plan.end, plan.capacity) {
            Ok(tail) => tail,
            Err(e) => return Err(self.ledger.poison(format!("detach: {}", e.0))),
        };
        let recurrent = self.hybrid().then(|| self.seq.take_recurrent());
        self.ids.truncate(plan.end as usize);
        let ids = self
            .ids
            .split_off((plan.start as usize).min(self.ids.len()));
        // The drafter's context for the segment's positions goes with it; a failure only costs
        // drafts.
        let draft = self.draft.as_mut().map(|c| {
            c.split_off(plan.start).unwrap_or_else(|_| {
                c.clear();
                c.split_off(plan.start).expect("an empty context splits")
            })
        });
        Ok(self.ledger.commit_detach(
            plan,
            StoredSegment {
                tail,
                recurrent,
                ids,
                draft,
            },
        ))
    }

    fn attach(&mut self, id: SegmentId) -> Result<(), EngineError> {
        self.end_verify();
        let plan = self.ledger.plan_attach(id)?;
        let stored = self.ledger.take_segment(id).expect("planned");
        self.roll_back();
        self.step = None;
        if self.strikes(FailPoint::Attach) {
            drop(stored);
            return Err(self.ledger.poison("attach: injected failure"));
        }
        if let Err(e) = self
            .seq
            .attach(plan.segment.start, stored.tail, plan.capacity)
        {
            return Err(self.ledger.poison(format!("attach: {}", e.0)));
        }
        if let Some(recurrent) = stored.recurrent {
            self.seq.set_recurrent(recurrent);
        }
        self.record(plan.segment.start, &stored.ids);
        if let (Some(ctx), Some(part)) = (self.draft.as_mut(), stored.draft)
            && ctx.append(part).is_err()
        {
            ctx.clear();
        }
        self.ledger.commit_attach(plan);
        Ok(())
    }

    fn drop_segment(&mut self, id: SegmentId) {
        if self.ledger.drop_segment(id) {
            tensor::memory::clear_cache();
        }
    }

    fn reset(&mut self) {
        self.end_verify();
        if let Some(ctx) = self.draft.as_mut() {
            ctx.clear();
        }
        self.seq.clear();
        self.ledger.reset();
        self.step = None;
        self.ids.clear();
    }

    fn reset_all(&mut self) {
        self.chunks.iter_mut().for_each(|c| *c = None);
        for lane in 0..self.parked.len() {
            self.select_lane(lane);
            self.end_verify();
            self.reset();
            self.draws = 0;
            self.verified.clear();
        }
        self.select_lane(0);
        self.ledger.reset_all();
        tensor::memory::clear_cache();
    }

    fn lanes(&self) -> usize {
        self.parked.len()
    }

    fn lane(&self) -> usize {
        self.lane
    }

    fn select(&mut self, lane: usize) {
        self.select_lane(lane);
    }

    fn begin_chunk(&mut self, tokens: &[u32]) -> Result<(), EngineError> {
        assert!(
            self.chunks[self.lane].is_none(),
            "one chunk in flight per lane"
        );
        self.end_verify();
        let growth = self.ledger.plan_append(tokens.len() as u32)?;
        let capacity = self.grow(growth)?;
        self.step = None;
        // The recurrent arrays from the chunk start stay alive until the chunk commits; on a
        // cancel or an error they come back. Attention positions written past `len` are
        // overwritten by the next prefill.
        let start = self.hybrid().then(|| self.seq.recurrent());
        let x = match self.chunk_input(tokens) {
            Ok(x) => x,
            Err(e) => {
                if let Some(start) = start {
                    self.seq.set_recurrent(start);
                }
                self.shrink_back(capacity)?;
                return Err(e);
            }
        };
        self.chunks[self.lane] = Some(InFlight {
            lane: self.lane,
            tokens: tokens.to_vec(),
            x,
            next: 0,
            at: self.ledger.len(),
            delta: self.seq.media.delta(self.ledger.len()),
            runs: self.seq.media.runs(self.ledger.len(), tokens.len() as u32),
            previous: None,
            capacity,
            start,
            taps: Vec::new(),
        });
        Ok(())
    }

    fn advance_chunk(
        &mut self,
        layers: usize,
        cancel: &CancelToken,
    ) -> Result<ChunkProgress, EngineError> {
        let Some(mut chunk) = self.chunks[self.lane].take() else {
            return Err(EngineError::Backend("no chunk in flight".into()));
        };
        // Another lane's chunk may still have its last layer running: let it finish first, so
        // one chunk layer's activations are in flight at a time, as scratch counts them (an
        // error there shows when that lane advances again).
        for other in self.chunks.iter().flatten() {
            if let Some(layer) = &other.previous {
                let _ = layer.eval();
            }
        }
        self.select_lane(chunk.lane);
        let total = self.seq.layers.len();
        let stop = chunk.next.saturating_add(layers).min(total);
        while chunk.next < stop {
            let k = chunk.next;
            let i = self.part.first + k;
            let layer = &mut self.seq.layers[k];
            let ran = self
                .model
                .layer(
                    i,
                    &chunk.x,
                    layer,
                    Pos {
                        at: chunk.at as i32,
                        delta: chunk.delta,
                        runs: &chunk.runs,
                    },
                )
                .and_then(|x| {
                    let mut outputs = layer.arrays();
                    outputs.push(&x);
                    tensor::async_eval(&outputs).map(|()| x)
                })
                .map_err(backend)
                .and_then(|x| {
                    if self.strikes(FailPoint::PrefillLayer(i)) {
                        Err(injected())
                    } else {
                        Ok(x)
                    }
                });
            let x = match ran {
                Ok(x) => x,
                Err(e) => {
                    self.undo_chunk(chunk)?;
                    return Err(e);
                }
            };
            if self.tapped.get(k) == Some(&true) {
                chunk.taps.push(x.clone());
            }
            chunk.x = x;
            chunk.next += 1;
            // Layer k is queued; wait for layer k − 1, then check for cancellation.
            if let Some(done) = chunk.previous.replace(chunk.x.clone()) {
                if let Err(e) = done.eval() {
                    self.undo_chunk(chunk)?;
                    return Err(backend(e));
                }
                if cancel.is_cancelled() {
                    self.undo_chunk(chunk)?;
                    return Ok(ChunkProgress::Cancelled);
                }
            }
        }
        if chunk.next < total {
            self.chunks[self.lane] = Some(chunk);
            return Ok(ChunkProgress::More);
        }
        // A prefill chunk's hidden state goes on to the next part; the last part's is not needed.
        let mut x = chunk.x.clone();
        if let Some(pipe) = self.pipe.as_ref().filter(|p| !p.last()) {
            x = match pipe.send(&x) {
                Ok(x) => x,
                Err(e) => {
                    self.undo_chunk(chunk)?;
                    return Err(backend(e));
                }
            };
        }
        if let Err(e) = x.eval() {
            self.undo_chunk(chunk)?;
            return Err(backend(e));
        }
        let start = self.ledger.len();
        self.ledger.advance(chunk.tokens.len() as u32);
        self.record(start, &chunk.tokens);
        self.feed_draft(std::mem::take(&mut chunk.taps), start);
        tensor::memory::clear_cache();
        Ok(ChunkProgress::Done)
    }

    fn abort_chunk(&mut self) {
        if let Some(chunk) = self.chunks[self.lane].take() {
            let selected = self.lane;
            let _ = self.undo_chunk(chunk);
            self.select_lane(selected);
        }
    }

    fn forward_lanes(&mut self, rows: &[(usize, u32)]) -> Result<Vec<Step>, (usize, EngineError)> {
        let selected = self.lane;
        if rows.len() == 1 || self.pipe.is_some() {
            let mut steps = Vec::with_capacity(rows.len());
            for (i, &(lane, last)) in rows.iter().enumerate() {
                self.select_lane(lane);
                match self.forward(last) {
                    Ok(step) => steps.push(step),
                    Err(e) => {
                        self.select_lane(selected);
                        return Err((i, e));
                    }
                }
            }
            self.select_lane(selected);
            return Ok(steps);
        }
        // Each row: its pending step when it fed `last`, else growth for one more position.
        struct Fresh {
            row: usize,
            lane: usize,
            last: u32,
            capacity: u32,
            before: Option<RecurrentState>,
            offset: i32,
        }
        let mut steps: Vec<Option<Step>> = (0..rows.len()).map(|_| None).collect();
        let mut fresh: Vec<Fresh> = Vec::new();
        let undo = |engine: &mut Self, fresh: Vec<Fresh>| {
            for f in fresh {
                engine.select_lane(f.lane);
                if let Some(before) = f.before {
                    engine.seq.set_recurrent(before);
                }
                let _ = engine.shrink_back(f.capacity);
            }
        };
        for (i, &(lane, last)) in rows.iter().enumerate() {
            self.select_lane(lane);
            self.end_verify();
            let planned = self.ledger.usable().and_then(|()| {
                if let Some(pending) = self.ledger.reuse_pending(|p| p.token == last) {
                    self.step = Some((pending.id, pending.logits));
                    return Ok(None);
                }
                let growth = self.ledger.plan_append(1)?;
                self.grow(growth).map(Some)
            });
            match planned {
                Ok(None) => {
                    let id = self.step.as_ref().expect("reused").0;
                    steps[i] = Some(Step::new(id));
                }
                Ok(Some(capacity)) => {
                    self.step = None;
                    fresh.push(Fresh {
                        row: i,
                        lane,
                        last,
                        capacity,
                        before: self.hybrid().then(|| self.seq.recurrent()),
                        offset: self.ledger.len() as i32,
                    });
                }
                Err(e) => {
                    undo(self, fresh);
                    self.select_lane(selected);
                    return Err((i, e));
                }
            }
        }
        if !fresh.is_empty() {
            let lanes: Vec<usize> = fresh.iter().map(|f| f.lane).collect();
            let tokens: Vec<u32> = fresh.iter().map(|f| f.last).collect();
            let offsets: Vec<i32> = fresh.iter().map(|f| f.offset).collect();
            let ids = Array::from_u32(&tokens, &[1, tokens.len() as i32]);
            let fed = self.feed_lanes(&lanes, &ids, &offsets).and_then(|logits| {
                tensor::async_eval(&[&logits])
                    .map_err(backend)
                    .map(|()| logits)
            });
            let logits = match fed {
                Ok(logits) => logits,
                Err(e) => {
                    let row = fresh[0].row;
                    undo(self, fresh);
                    self.select_lane(selected);
                    return Err((row, e));
                }
            };
            for (r, f) in fresh.iter().enumerate() {
                self.select_lane(f.lane);
                let row = match Self::row(&logits, r as i32) {
                    Ok(row) => row,
                    Err(e) => {
                        self.select_lane(selected);
                        return Err((f.row, e));
                    }
                };
                self.ledger.advance(1);
                self.record(self.ledger.len() - 1, &[f.last]);
                self.after_decode_step();
                let id = self.ledger.id();
                self.step = Some((id, row));
                steps[f.row] = Some(Step::new(id));
            }
        }
        self.select_lane(selected);
        Ok(steps.into_iter().map(|s| s.expect("every row")).collect())
    }

    fn sample_lanes(&mut self, picks: Vec<Pick<'_>>) -> Vec<Result<u32, EngineError>> {
        let selected = self.lane;
        if picks.len() == 1 || self.pipe.is_some() {
            let out = picks
                .into_iter()
                .map(|p| {
                    self.select_lane(p.lane);
                    self.sample(p.step, p.sampling, p.mask)
                })
                .collect();
            self.select_lane(selected);
            return out;
        }
        // The tokens, lazily, each as `sample` draws it.
        let mut tokens: Vec<Result<Array, EngineError>> = Vec::with_capacity(picks.len());
        for p in &picks {
            self.select_lane(p.lane);
            let drawn = (|| {
                self.ledger.usable()?;
                let logits = self.current_logits(&p.step)?.clone();
                if let Some(mask) = p.mask {
                    mask.check_width(self.info.vocab)?;
                }
                self.step = None;
                let mut key = if p.sampling.is_greedy() {
                    None
                } else {
                    Some(self.key(p.sampling.seed).map_err(backend)?)
                };
                let recent = self.recent_ids(p.sampling);
                sample::sample(&logits, p.sampling, p.mask, recent.as_ref(), || {
                    key.take().ok_or_else(|| tensor::Error("no key".into()))
                })
                .map_err(backend)?
                .ok_or(EngineError::EmptyMask)
            })();
            tokens.push(drawn);
        }
        let drawn: Vec<&Array> = tokens.iter().filter_map(|t| t.as_ref().ok()).collect();
        if let Err(e) = tensor::async_eval(&drawn) {
            self.select_lane(selected);
            return picks.iter().map(|_| Err(backend(e.clone()))).collect();
        }
        // Pipeline: one batched step of the sampled tokens of every lane with room.
        let mut fed: Vec<(usize, Option<RecurrentState>)> = Vec::new();
        let mut offsets = Vec::new();
        let mut ids = Vec::new();
        for (i, p) in picks.iter().enumerate() {
            let Ok(token) = &tokens[i] else { continue };
            self.select_lane(p.lane);
            if self.ledger.has_room() {
                fed.push((i, self.hybrid().then(|| self.seq.recurrent())));
                offsets.push(self.ledger.len() as i32);
                ids.push(tensor::reshape(token, &[1, 1]));
            }
        }
        let next = if fed.is_empty() {
            None
        } else {
            let lanes: Vec<usize> = fed.iter().map(|&(i, _)| picks[i].lane).collect();
            let batched = ids
                .into_iter()
                .collect::<tensor::Result<Vec<_>>>()
                .and_then(|ids| tensor::concatenate(&ids.iter().collect::<Vec<_>>(), 1))
                .map_err(backend)
                .and_then(|ids| self.feed_lanes(&lanes, &ids, &offsets))
                .and_then(|next| tensor::async_eval(&[&next]).map_err(backend).map(|()| next));
            Some(batched)
        };
        let mut out = Vec::with_capacity(picks.len());
        for (i, p) in picks.iter().enumerate() {
            self.select_lane(p.lane);
            let slot = fed.iter().position(|&(j, _)| j == i);
            let value = match &tokens[i] {
                Ok(token) if self.strikes(FailPoint::SampleReadback) => {
                    let _ = token;
                    Err(injected())
                }
                Ok(token) => token.item_u32().map_err(backend),
                Err(e) => Err(e.clone()),
            };
            let before = slot.and_then(|s| fed[s].1.clone());
            match (value, slot, &next) {
                (Ok(value), Some(r), Some(Ok(logits))) => match Self::row(logits, r as i32) {
                    Ok(row) => {
                        let id = self.ledger.id();
                        self.ledger.push_pending(Pending {
                            token: value,
                            id,
                            logits: row,
                            before,
                        });
                        self.record(self.ledger.len() - 1, &[value]);
                        self.after_decode_step();
                        out.push(Ok(value));
                    }
                    Err(e) => {
                        if let Some(before) = before {
                            self.seq.set_recurrent(before);
                        }
                        out.push(Err(e));
                    }
                },
                (Ok(value), None, _) => out.push(Ok(value)),
                (Ok(_), Some(_), Some(Err(e))) => {
                    if let Some(before) = before {
                        self.seq.set_recurrent(before);
                    }
                    out.push(Err(e.clone()));
                }
                (Err(e), _, _) => {
                    if let Some(before) = before {
                        self.seq.set_recurrent(before);
                    }
                    out.push(Err(e));
                }
                (Ok(_), Some(_), None) => unreachable!("fed rows have a step"),
            }
        }
        self.select_lane(selected);
        out
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use crate::testing;

    /// The token ids behind the repetition penalty follow every move of the live sequence.
    #[test]
    fn token_ids_follow_the_live_sequence() {
        let dir = tempfile::tempdir().unwrap();
        testing::write_tiny_qwen3(dir.path(), 64).unwrap();
        let mut engine = testing::load_cpu(dir.path(), None).unwrap();
        let live = |e: &NuvecEngine| e.ids[..e.ledger.len() as usize].to_vec();
        let penalty = Sampling {
            repetition_penalty: 1.3,
            ..Sampling::GREEDY
        };
        engine.set_limit(64).unwrap();
        assert!(
            engine
                .prefill_chunk(&[5, 6, 7], &CancelToken::new())
                .unwrap()
        );
        let step = engine.forward(8).unwrap();
        assert_eq!(live(&engine), [5, 6, 7, 8]);
        // Sampling feeds the token as a pending step; it is recorded at once.
        let a = engine.sample(step, &penalty, None).unwrap();
        assert_eq!(live(&engine), [5, 6, 7, 8, a]);
        // Forward with another token rolls the pending one back.
        let other = (a + 1) % 64;
        let step = engine.forward(other).unwrap();
        assert_eq!(live(&engine), [5, 6, 7, 8, other]);
        let b = engine.sample(step, &Sampling::GREEDY, None).unwrap();
        engine.trim(engine.len()).unwrap();
        assert_eq!(live(&engine), [5, 6, 7, 8, other, b]);
        let segment = engine.detach_tail(2).unwrap();
        assert_eq!(live(&engine), [5, 6]);
        engine.attach(segment.id).unwrap();
        assert_eq!(live(&engine), [5, 6, 7, 8, other, b]);
        engine.trim(3).unwrap();
        assert_eq!(live(&engine), [5, 6, 7]);
        // Only the last REPETITION_CONTEXT ids are passed.
        let many: Vec<u32> = (0..30).collect();
        assert!(engine.prefill_chunk(&many, &CancelToken::new()).unwrap());
        let recent = engine.recent_ids(&penalty).unwrap().to_vec_u32().unwrap();
        assert_eq!(recent, &many[10..]);
        assert!(engine.recent_ids(&Sampling::GREEDY).is_none());
        engine.reset();
        assert!(engine.ids.is_empty() && engine.recent_ids(&penalty).is_none());
    }
}
