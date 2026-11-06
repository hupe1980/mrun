//! The pool's plan (026 FR-004, R4): which contiguous layers each Mac runs, from each Mac's own
//! budget (III per Mac), the context the pool leaves, and an honest speed range.
//!
//! Layers start in proportion to each Mac's room after its fixed costs (the host also carries
//! the embeddings and the head), then move one at a time from the Mac that limits the context to
//! a neighbour while the context grows. Decoding is memory-bound: a token costs each Mac the
//! bytes its layers read at its memory bandwidth, plus one link delay per hop around the ring.

use super::link::Link;

const GIB: u64 = 1 << 30;
/// Working memory each Mac keeps beyond weights and state (activations, the allocator's cache).
pub const STAGE_SCRATCH: u64 = 3 * GIB / 2;
/// The shortest context a pool is offered with (a coding agent needs more; the plan says so).
pub const MIN_CONTEXT: u32 = 8192;
/// Most Macs a pool plans for.
pub const MAX_MACS: usize = 8;
/// Share of the memory bandwidth decoding reaches.
const BANDWIDTH_SHARE: f64 = 0.75;

/// What one layer costs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayerCost {
    pub weights: u64,
    /// Bytes read per decoded token (a mixture of experts reads its active experts only).
    pub active: u64,
    /// State per token (attention layers).
    pub kv_per_token: u64,
    /// State per sequence (linear layers).
    pub recurrent: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelCost {
    pub name: String,
    pub layers: Vec<LayerCost>,
    /// Embeddings, final norm and head, on the host.
    pub head: u64,
    /// Bytes the head reads per token.
    pub head_active: u64,
    pub max_position: u32,
}

/// One Mac as the plan sees it.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct MacSpec {
    pub mac_id: String,
    pub name: String,
    pub chip: String,
    /// This Mac's own budget (computed on that Mac, D062).
    pub budget: u64,
    pub gpu_family: String,
    pub power: String,
    /// The link from the host (none for the host).
    pub link: Option<Link>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Stage {
    pub mac_id: String,
    pub name: String,
    /// `[first, end)`, 0-based.
    pub first: usize,
    pub end: usize,
    pub weights_bytes: u64,
    pub state_bytes: u64,
    pub scratch_bytes: u64,
    pub budget: u64,
    pub fits: bool,
    /// Bytes missing on this Mac when it does not fit.
    pub short_bytes: u64,
    pub ms_per_token: f64,
    /// The link from the host (none for the host).
    pub link: Option<Link>,
    /// What this Mac downloads first: its layers' files when it lacks them.
    pub download_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Pace {
    /// `stage` or `link`.
    pub kind: &'static str,
    pub mac_id: String,
    pub name: String,
    /// Share of a token's time.
    pub share: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Plan {
    pub model: String,
    pub fits: bool,
    pub stages: Vec<Stage>,
    pub context_tokens: u32,
    pub speed: Speed,
    pub pace: Option<Pace>,
    pub numerics_match: bool,
    /// Why it does not fit, in a sentence.
    pub refusal: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Speed {
    pub low: f64,
    pub high: f64,
}

/// Memory bandwidth in GB/s of a chip as `sysctl machdep.cpu.brand_string` names it (Apple's
/// figures; the base chip when the bin is unknown).
pub fn bandwidth_gbs(chip: &str) -> f64 {
    let chip = chip.trim_start_matches("Apple ").to_ascii_lowercase();
    let generation = ["m1", "m2", "m3", "m4", "m5"]
        .iter()
        .position(|g| chip.starts_with(g))
        .unwrap_or(0);
    let tier = if chip.contains("ultra") {
        3
    } else if chip.contains("max") {
        2
    } else if chip.contains("pro") {
        1
    } else {
        0
    };
    const TABLE: [[f64; 4]; 5] = [
        [68.0, 200.0, 400.0, 800.0],
        [100.0, 200.0, 400.0, 800.0],
        [100.0, 150.0, 300.0, 800.0],
        [120.0, 273.0, 410.0, 819.0],
        [153.0, 307.0, 460.0, 1000.0],
    ];
    TABLE[generation][tier]
}

/// What `layers` of `model` hold on a Mac: weights, state at `context` tokens.
fn load(model: &ModelCost, first: usize, end: usize, host: bool) -> (u64, u64, u64) {
    let layers = &model.layers[first..end];
    let weights = layers.iter().map(|l| l.weights).sum::<u64>() + if host { model.head } else { 0 };
    let kv = layers.iter().map(|l| l.kv_per_token).sum();
    let recurrent = layers.iter().map(|l| l.recurrent).sum();
    (weights, kv, recurrent)
}

/// The context `macs` leave with `bounds` (stage `i` runs `bounds[i]..bounds[i + 1]`), and how
/// far the tightest Mac is short of memory (0 when all fit).
fn context(model: &ModelCost, macs: &[MacSpec], bounds: &[usize]) -> (u64, u64) {
    let mut context = u64::from(model.max_position);
    let mut short = 0u64;
    for (i, mac) in macs.iter().enumerate() {
        let (weights, kv, recurrent) = load(model, bounds[i], bounds[i + 1], i == 0);
        let fixed = weights + recurrent + STAGE_SCRATCH;
        match mac.budget.checked_sub(fixed) {
            Some(free) if kv > 0 => context = context.min(free / kv),
            Some(_) => {}
            None => {
                short = short.max(fixed - mac.budget);
                context = 0;
            }
        }
    }
    (context, short)
}

/// Initial bounds: layers in proportion to each Mac's room after its fixed costs.
fn proportional(model: &ModelCost, macs: &[MacSpec]) -> Vec<usize> {
    let n = model.layers.len();
    let room: Vec<f64> = macs
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let fixed = STAGE_SCRATCH + if i == 0 { model.head } else { 0 };
            m.budget.saturating_sub(fixed).max(1) as f64
        })
        .collect();
    let total: f64 = room.iter().sum();
    let weights: u64 = model.layers.iter().map(|l| l.weights).sum::<u64>().max(1);
    let mut bounds = vec![0];
    let mut acc = 0.0;
    let mut cum = 0u64;
    let mut layer = 0;
    for share in &room[..macs.len() - 1] {
        acc += share / total;
        while layer < n && (cum as f64) < acc * weights as f64 {
            cum += model.layers[layer].weights;
            layer += 1;
        }
        // Every Mac keeps at least one layer.
        let at = layer.clamp(bounds.last().unwrap() + 1, n - (macs.len() - bounds.len()));
        bounds.push(at);
        layer = at;
        cum = model.layers[..at].iter().map(|l| l.weights).sum();
    }
    bounds.push(n);
    bounds
}

/// Moves single layers across stage boundaries while that grows the context (or shrinks the
/// shortfall).
fn balance(model: &ModelCost, macs: &[MacSpec], mut bounds: Vec<usize>) -> Vec<usize> {
    let score = |b: &[usize]| {
        let (ctx, short) = context(model, macs, b);
        (ctx, std::cmp::Reverse(short))
    };
    let mut best = score(&bounds);
    for _ in 0..model.layers.len() * macs.len() {
        let mut improved = false;
        for k in 1..macs.len() {
            for delta in [-1i64, 1] {
                let at = bounds[k] as i64 + delta;
                if at <= bounds[k - 1] as i64 || at >= bounds[k + 1] as i64 {
                    continue;
                }
                let mut next = bounds.clone();
                next[k] = at as usize;
                let s = score(&next);
                if s > best {
                    best = s;
                    bounds = next;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    bounds
}

/// The plan for `model` on `macs` (the host first, members in stage order).
/// What a Mac holds of a model: all of it, a part (layers `[first, end)`), or nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Holds {
    Whole,
    Part([usize; 2]),
    #[default]
    Nothing,
}

impl Holds {
    pub fn covers(self, first: usize, end: usize) -> bool {
        match self {
            Self::Whole => true,
            Self::Part([a, b]) => a <= first && end <= b,
            Self::Nothing => false,
        }
    }
}

/// Fills each stage's download: its layers' weights on a Mac that lacks them (`holds` per
/// stage, the host's first).
pub fn with_downloads(plan: &mut Plan, model: &ModelCost, holds: &[Holds]) {
    for (stage, holds) in plan.stages.iter_mut().zip(holds) {
        stage.download_bytes = if holds.covers(stage.first, stage.end) {
            0
        } else {
            model.layers[stage.first..stage.end]
                .iter()
                .map(|l| l.weights)
                .sum()
        };
    }
}

pub fn plan(model: &ModelCost, macs: &[MacSpec]) -> Plan {
    let refuse = |why: String| Plan {
        model: model.name.clone(),
        fits: false,
        stages: Vec::new(),
        context_tokens: 0,
        speed: Speed {
            low: 0.0,
            high: 0.0,
        },
        pace: None,
        numerics_match: true,
        refusal: Some(why),
    };
    if macs.is_empty() || macs.len() > MAX_MACS {
        return refuse(format!("a pool plans for 1 to {MAX_MACS} Macs"));
    }
    if model.layers.len() < macs.len() {
        return refuse(format!(
            "{} has {} layers, fewer than the {} Macs",
            model.name,
            model.layers.len(),
            macs.len()
        ));
    }
    let bounds = balance(model, macs, proportional(model, macs));
    let (context_tokens, _) = context(model, macs, &bounds);
    let context_tokens = context_tokens.min(u64::from(model.max_position)) as u32;
    let mut stages = Vec::new();
    let mut parts: Vec<(&'static str, String, String, f64)> = Vec::new();
    for (i, mac) in macs.iter().enumerate() {
        let (first, end) = (bounds[i], bounds[i + 1]);
        let (weights, kv, recurrent) = load(model, first, end, i == 0);
        let state = kv * u64::from(context_tokens) + recurrent;
        let need = weights + state + STAGE_SCRATCH;
        let active = model.layers[first..end]
            .iter()
            .map(|l| l.active)
            .sum::<u64>()
            + if i == 0 { model.head_active } else { 0 };
        let ms = active as f64 / (bandwidth_gbs(&mac.chip) * 1e9 * BANDWIDTH_SHARE) * 1e3;
        parts.push(("stage", mac.mac_id.clone(), mac.name.clone(), ms));
        stages.push(Stage {
            mac_id: mac.mac_id.clone(),
            name: mac.name.clone(),
            first,
            end,
            weights_bytes: weights,
            state_bytes: state,
            scratch_bytes: STAGE_SCRATCH,
            budget: mac.budget,
            fits: need <= mac.budget,
            short_bytes: need.saturating_sub(mac.budget),
            ms_per_token: ms,
            link: mac.link,
            download_bytes: 0,
        });
    }
    // One hop per Mac around the ring; a hop between two members is taken as the slower of
    // their links to the host.
    let hop = |i: usize, p90: bool| {
        let rtt = |m: &MacSpec| {
            m.link
                .map_or(0.0, |l| if p90 { l.rtt_ms_p90 } else { l.rtt_ms_p50 })
        };
        let next = &macs[(i + 1) % macs.len()];
        rtt(&macs[i]).max(rtt(next)) / 2.0
    };
    let compute: f64 = parts.iter().map(|p| p.3).sum();
    let links = |p90| {
        (0..macs.len())
            .map(|i| if macs.len() > 1 { hop(i, p90) } else { 0.0 })
            .sum::<f64>()
    };
    let tok_s = |ms: f64| if ms > 0.0 { 1000.0 / ms } else { 0.0 };
    let speed = Speed {
        low: tok_s(compute + links(true)),
        high: tok_s(compute + links(false)),
    };
    for (i, mac) in macs.iter().enumerate().skip(1) {
        let ms = hop(i - 1, false) + hop(i, false);
        parts.push(("link", mac.mac_id.clone(), mac.name.clone(), ms / 2.0));
    }
    let total: f64 = compute + links(false);
    let pace = (macs.len() > 1)
        .then(|| {
            parts
                .iter()
                .max_by(|a, b| a.3.total_cmp(&b.3))
                .map(|(kind, mac_id, name, ms)| Pace {
                    kind,
                    mac_id: mac_id.clone(),
                    name: name.clone(),
                    share: if total > 0.0 { ms / total } else { 0.0 },
                })
        })
        .flatten();
    let fits = stages.iter().all(|s| s.fits) && context_tokens >= MIN_CONTEXT;
    let refusal = (!fits).then(|| match stages.iter().find(|s| !s.fits) {
        Some(s) => format!(
            "{} does not fit: {} needs {:.1} GB more",
            model.name,
            s.name,
            s.short_bytes as f64 / 1e9
        ),
        None => format!(
            "{} leaves {context_tokens} tokens of context, fewer than {MIN_CONTEXT}",
            model.name
        ),
    });
    Plan {
        model: model.name.clone(),
        fits,
        stages,
        context_tokens,
        speed,
        pace,
        numerics_match: macs.iter().all(|m| m.gpu_family == macs[0].gpu_family),
        refusal,
    }
}

#[cfg(test)]
mod tests {
    use super::super::link::Kind;
    use super::*;

    /// A 94-layer, 125 GB model (Qwen3-235B-A22B 4-bit-like: MoE, 8 % active).
    fn big() -> ModelCost {
        let layer = LayerCost {
            weights: 1_300_000_000,
            active: 110_000_000,
            kv_per_token: 2 * 4 * 128 * 2,
            recurrent: 0,
        };
        ModelCost {
            name: "qwen3-235b".into(),
            layers: vec![layer; 94],
            head: 1_200_000_000,
            head_active: 600_000_000,
            max_position: 262_144,
        }
    }

    fn mac(id: &str, gib: u64, chip: &str, link: Option<(Kind, f64)>) -> MacSpec {
        MacSpec {
            mac_id: id.into(),
            name: format!("{id}'s Mac"),
            chip: chip.into(),
            budget: gib * GIB,
            gpu_family: "applegpu_g16s".into(),
            power: "adapter".into(),
            link: link.map(|(kind, rtt)| Link {
                kind,
                rtt_ms_p50: rtt,
                rtt_ms_p90: rtt * 1.5,
                mbps: 1000.0,
            }),
        }
    }

    #[test]
    fn a_mac_downloads_only_the_layers_it_lacks() {
        let macs = [
            mac("ben", 80, "Apple M4 Max", None),
            mac("anna", 80, "Apple M4 Pro", Some((Kind::Thunderbolt, 0.2))),
        ];
        let mut p = plan(&big(), &macs);
        let (first, end) = (p.stages[1].first, p.stages[1].end);
        with_downloads(&mut p, &big(), &[Holds::Whole, Holds::Nothing]);
        assert_eq!(p.stages[0].download_bytes, 0);
        assert_eq!(
            p.stages[1].download_bytes,
            (end - first) as u64 * 1_300_000_000
        );
        with_downloads(&mut p, &big(), &[Holds::Whole, Holds::Part([first, 94])]);
        assert_eq!(
            p.stages[1].download_bytes, 0,
            "a part that covers its layers"
        );
        assert!(!Holds::Part([first + 1, 94]).covers(first, end));
        assert_eq!(
            serde_json::to_value(Holds::Part([3, 7])).unwrap(),
            serde_json::json!({"part": [3, 7]})
        );
    }

    #[test]
    fn a_model_too_big_for_one_mac_fits_four() {
        let one = plan(&big(), &[mac("ben", 40, "Apple M4 Max", None)]);
        assert!(!one.fits);
        assert!(
            one.refusal
                .unwrap()
                .contains("does not fit: ben's Mac needs")
        );
        let macs = [
            mac("ben", 40, "Apple M4 Max", None),
            mac("anna", 40, "Apple M4 Pro", Some((Kind::Thunderbolt, 0.2))),
            mac("chris", 40, "Apple M4 Pro", Some((Kind::Ethernet, 0.4))),
            mac("dana", 20, "Apple M3", Some((Kind::WiFi, 6.0))),
        ];
        let p = plan(&big(), &macs);
        assert!(p.fits, "{p:?}");
        // Contiguous, complete, every Mac some layers; the smaller Mac fewer.
        assert_eq!(p.stages[0].first, 0);
        assert_eq!(p.stages.last().unwrap().end, 94);
        for w in p.stages.windows(2) {
            assert_eq!(w[0].end, w[1].first);
        }
        assert!(p.stages.iter().all(|s| s.end > s.first && s.fits));
        let layers = |i: usize| p.stages[i].end - p.stages[i].first;
        assert!(layers(3) < layers(1), "{p:?}");
        assert!(p.context_tokens >= MIN_CONTEXT);
        // Dana's Wi-Fi or her slower chip sets the pace.
        assert_eq!(p.pace.as_ref().unwrap().mac_id, "dana");
        assert!(p.speed.low < p.speed.high && p.speed.low > 0.0);
    }

    #[test]
    fn the_pool_refuses_what_does_not_fit_and_too_many_macs() {
        let macs = [
            mac("ben", 32, "Apple M4 Pro", None),
            mac("anna", 32, "Apple M4 Pro", Some((Kind::Ethernet, 0.4))),
        ];
        let p = plan(&big(), &macs);
        assert!(!p.fits);
        assert!(p.stages.iter().any(|s| s.short_bytes > 0));
        let nine: Vec<MacSpec> = (0..9)
            .map(|i| mac(&i.to_string(), 64, "Apple M4", None))
            .collect();
        assert!(plan(&big(), &nine).refusal.unwrap().contains("1 to 8 Macs"));
    }

    #[test]
    fn mixed_gpu_families_are_flagged_and_bandwidth_follows_the_chip() {
        let mut macs = [
            mac("ben", 96, "Apple M4 Max", None),
            mac("anna", 96, "Apple M1 Pro", Some((Kind::Ethernet, 0.4))),
        ];
        macs[1].gpu_family = "applegpu_g13s".into();
        assert!(!plan(&big(), &macs).numerics_match);
        assert_eq!(bandwidth_gbs("Apple M4 Pro"), 273.0);
        assert_eq!(bandwidth_gbs("Apple M1"), 68.0);
        assert_eq!(bandwidth_gbs("Apple M2 Ultra"), 800.0);
    }

    #[test]
    fn two_equal_macs_split_evenly() {
        let macs = [
            mac("ben", 96, "Apple M4 Pro", None),
            mac("anna", 96, "Apple M4 Pro", Some((Kind::Thunderbolt, 0.2))),
        ];
        let p = plan(&big(), &macs);
        let (a, b) = (p.stages[0].end, 94 - p.stages[0].end);
        assert!(a.abs_diff(b) <= 2, "{a} / {b}");
    }
}
