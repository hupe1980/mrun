//! Curated models for remote mode (013 FR-006, research R12): weights size (Hugging Face API,
//! 2026-10-19), an estimate of KV bytes per token, the context vLLM is started with, and the
//! vLLM tool and reasoning parsers. The fit rule: weights + 64K tokens of KV + 2 GiB of
//! activations within 90 % of the GPU's memory (vLLM's `--gpu-memory-utilization 0.9`).

use mrun_runpod::{CloudType, GpuType};

/// Tokens of KV every curated model must hold (Claude Code's and OpenCode's minimum, 004).
pub const FIT_TOKENS: u64 = 65_536;
const GIB: f64 = (1u64 << 30) as f64;
const OVERHEAD_GIB: f64 = 2.0;
const UTILIZATION: f64 = 0.9;

/// One curated model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Offer {
    pub repo: &'static str,
    pub weights_bytes: u64,
    /// Full-attention KV bytes per token (estimate from the model config; hybrid models keep a
    /// fixed recurrent state besides).
    pub kv_bytes_per_token: u64,
    pub max_model_len: u32,
    pub tool_parser: &'static str,
    pub reasoning_parser: Option<&'static str>,
    pub note: &'static str,
    /// The pod image when the model needs another vLLM build than `VLLM_IMAGE` (pinned by
    /// digest, CUDA 13.0 like it).
    pub image: Option<&'static str>,
    /// Weights vLLM keeps in host RAM instead of on the GPU (an offloaded embedding table).
    pub host_bytes: u64,
    /// Least host RAM of the pod in GB (0: RunPod's default).
    pub min_ram_gb: u32,
    /// Environment of vLLM.
    pub vllm_env: &'static [(&'static str, &'static str)],
    /// vLLM flags after mrun's own (a repeated flag overrides mrun's).
    pub vllm_args: &'static [&'static str],
    /// The weights need a Blackwell GPU (NVFP4 kernels).
    pub blackwell: bool,
}

/// The stub "model" of CPU test pods (`cloud up --cpu`); also the defaults of the fields most
/// offers do not set.
pub const STUB: Offer = Offer {
    repo: "stub",
    weights_bytes: 0,
    kv_bytes_per_token: 0,
    max_model_len: 65_536,
    tool_parser: "hermes",
    reasoning_parser: None,
    note: "test pods only: a fake vLLM inside the agent",
    image: None,
    host_bytes: 0,
    min_ram_gb: 0,
    vllm_env: &[],
    vllm_args: &[],
    blackwell: false,
};

pub const OFFERS: [Offer; 6] = [
    Offer {
        repo: "Qwen/Qwen3.6-35B-A3B-FP8",
        weights_bytes: 37_460_000_000,
        kv_bytes_per_token: 20_480,
        max_model_len: 131_072,
        tool_parser: "qwen3_coder",
        reasoning_parser: Some("qwen3"),
        note: "the local qwen3.6 at full precision and speed",
        ..STUB
    },
    Offer {
        repo: "Qwen/Qwen3.8-27B-FP8",
        weights_bytes: 30_870_000_000,
        kv_bytes_per_token: 65_536,
        max_model_len: 131_072,
        tool_parser: "qwen3_coder",
        reasoning_parser: Some("qwen3"),
        note: "dense 27B; 15 tok/s on the M4 Pro, full speed here",
        ..STUB
    },
    Offer {
        repo: "Qwen/Qwen3-Coder-Next-FP8",
        weights_bytes: 80_380_000_000,
        kv_bytes_per_token: 24_576,
        // 64K is what passed the client gate (R15); a 131K start failed in engine init on a
        // 96 GB card (2026-10-19, root cause not captured).
        max_model_len: 65_536,
        tool_parser: "qwen3_coder",
        reasoning_parser: None,
        note: "80B coder; does not fit a 48 GB Mac",
        ..STUB
    },
    Offer {
        repo: "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        weights_bytes: 31_180_000_000,
        kv_bytes_per_token: 98_304,
        max_model_len: 65_536,
        tool_parser: "qwen3_coder",
        reasoning_parser: None,
        note: "30B MoE coder",
        ..STUB
    },
    Offer {
        repo: "Qwen/Qwen3-4B-Instruct-2507",
        weights_bytes: 8_040_000_000,
        kv_bytes_per_token: 147_456,
        max_model_len: 65_536,
        tool_parser: "hermes",
        reasoning_parser: None,
        note: "small and cheap: for trying remote mode",
        ..STUB
    },
    // 125B / 6B active with a 51B n-gram table (BF16, 102.4 GB of the 186.4 GB) that vLLM keeps
    // in host RAM, so about 84 GB stays on the GPU. Experts NVFP4 (Blackwell only), the rest BF16.
    // Card (2026-09-06): one RTX PRO 6000, 88.8 GiB at 0.92 utilization, 32K context measured;
    // 64K here is not yet measured. KV: 12 full-attention layers, 2 KV heads × 256.
    Offer {
        repo: "primitive-ai/Qwen3.8-Flash-Next-NVFP4",
        weights_bytes: 186_395_827_182,
        kv_bytes_per_token: 24_576,
        max_model_len: 65_536,
        tool_parser: "qwen3_coder",
        reasoning_parser: Some("qwen3"),
        note: "Qwen3.8 Flash Next, 4-bit experts; Blackwell, n-gram table in host RAM",
        image: Some(
            "vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8",
        ),
        host_bytes: 102_400_491_520,
        min_ram_gb: 120,
        vllm_env: &[
            ("VLLM_PLE_CPU_OFFLOAD", "1"),
            ("VLLM_PLE_OFFLOAD_READY_TIMEOUT", "1800"),
        ],
        vllm_args: &[
            "--distributed-executor-backend",
            "mp",
            "--gpu-memory-utilization",
            "0.92",
        ],
        blackwell: true,
    },
];

/// The curated offer for `repo` (case-insensitive).
pub fn find(repo: &str) -> Option<Offer> {
    OFFERS
        .iter()
        .copied()
        .find(|o| o.repo.eq_ignore_ascii_case(repo.trim()))
}

impl Offer {
    /// GiB of GPU memory the model needs (weights on the GPU + 64K KV + overhead, at 90 %
    /// utilization).
    pub fn min_vram_gib(&self) -> f64 {
        let need = (self.weights_bytes - self.host_bytes) as f64 / GIB
            + (FIT_TOKENS * self.kv_bytes_per_token) as f64 / GIB
            + OVERHEAD_GIB;
        need / UTILIZATION
    }

    pub fn fits(&self, gpu: &GpuType) -> bool {
        f64::from(gpu.memory_gb) >= self.min_vram_gib() && (!self.blackwell || is_blackwell(gpu))
    }

    /// Disk for the weights (container disk without a volume): weights + 20 %, at least 20 GB.
    pub fn disk_gb(&self) -> u32 {
        ((self.weights_bytes as f64 * 1.2 / 1e9).ceil() as u32).max(20)
    }
}

/// A Blackwell GPU (B200, B300, RTX PRO 6000, RTX 50 series), by RunPod's type id.
fn is_blackwell(gpu: &GpuType) -> bool {
    ["B200", "B300", "Blackwell", "RTX 50"]
        .iter()
        .any(|name| gpu.id.contains(name))
}

/// GPU types that fit `offer` and are offered in `cloud`: in stock first, then cheapest.
pub fn fitting<'a>(
    offer: &Offer,
    gpus: &'a [GpuType],
    cloud: CloudType,
) -> Vec<(&'a GpuType, f64)> {
    let mut out: Vec<(&GpuType, f64)> = gpus
        .iter()
        .filter(|g| g.is_nvidia() && offer.fits(g))
        .filter_map(|g| g.price(cloud).map(|p| (g, p)))
        .collect();
    // Stock is unknown when no type reports any (an older API answer): price alone then.
    let any_stock = out.iter().any(|(g, _)| g.stock(cloud).is_some());
    out.sort_by(|a, b| {
        let sold_out = |g: &GpuType| any_stock && g.stock(cloud).is_none();
        sold_out(a.0).cmp(&sold_out(b.0)).then(a.1.total_cmp(&b.1))
    });
    out
}

/// In-stock alternatives for `offer` after `failed` (in `cloud`) could not be rented: every GPU
/// that fits, in either cloud, that reports stock, cheapest first, without the one that failed.
/// Same-class cards (other variants of a 96 GB card, the other cloud) usually come first because
/// they cost less than the next larger GPU.
pub fn alternatives<'a>(
    offer: &Offer,
    gpus: &'a [GpuType],
    failed: &str,
    cloud: CloudType,
) -> Vec<(&'a GpuType, CloudType, f64)> {
    let mut out: Vec<(&GpuType, CloudType, f64)> = [CloudType::Secure, CloudType::Community]
        .into_iter()
        .flat_map(|c| {
            gpus.iter()
                .filter(move |g| g.is_nvidia() && offer.fits(g) && g.stock(c).is_some())
                .filter(move |g| !(c == cloud && (g.id == failed || g.display_name == failed)))
                .filter_map(move |g| g.price(c).map(|p| (g, c, p)))
        })
        .collect();
    out.sort_by(|a, b| a.2.total_cmp(&b.2));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, memory_gb: u32, secure: f64, community: f64) -> GpuType {
        GpuType {
            id: format!("NVIDIA {name}"),
            display_name: name.into(),
            memory_gb,
            secure_cloud: secure > 0.0,
            community_cloud: community > 0.0,
            secure_price: secure,
            community_price: community,
            ..GpuType::default()
        }
    }

    #[test]
    fn fit_needs_weights_plus_64k_tokens_of_kv() {
        let small = find("qwen/qwen3-4b-instruct-2507").unwrap();
        let big = find("Qwen/Qwen3-Coder-Next-FP8").unwrap();
        let qwen36 = find("Qwen/Qwen3.6-35B-A3B-FP8").unwrap();
        let gpus = [
            gpu("RTX A5000", 24, 0.27, 0.16),
            gpu("L40S", 48, 1.09, 0.79),
            gpu("H100 SXM", 80, 3.99, 2.69),
            gpu("H100 NVL", 94, 3.19, 2.59),
            gpu("RTX 4080", 16, 0.0, 0.27),
        ];
        assert!(small.fits(&gpus[0]) && !small.fits(&gpus[4]));
        assert!(qwen36.fits(&gpus[1]) && !qwen36.fits(&gpus[0]));
        assert!(
            !big.fits(&gpus[2]),
            "80 GB is too small for the 80B coder at FP8"
        );
        assert!(big.fits(&gpus[3]));
        let cheapest = fitting(&qwen36, &gpus, CloudType::Community);
        assert_eq!(cheapest[0].0.display_name, "L40S");
        let secure = fitting(&small, &gpus, CloudType::Secure);
        assert_eq!(secure[0].0.display_name, "RTX A5000");
        assert!(secure.iter().all(|(g, _)| g.display_name != "RTX 4080"));
        let amd = GpuType {
            id: "AMD Instinct MI300X OAM".into(),
            ..gpu("MI300X", 192, 2.39, 0.0)
        };
        assert!(
            fitting(&big, &[amd], CloudType::Secure).is_empty(),
            "CUDA image: NVIDIA only"
        );
        // In stock beats cheaper but sold out.
        let mut stocked = gpus.clone();
        stocked[2].community_stock = Some(mrun_runpod::Stock {
            stock_status: Some("Low".into()),
        });
        let first = fitting(&small, &stocked, CloudType::Community)[0].0;
        assert_eq!(first.display_name, "H100 SXM");
        assert!(find("unknown/model").is_none());
        assert_eq!(small.disk_gb(), 20);
        assert_eq!(qwen36.disk_gb(), 45);
    }

    #[test]
    fn flash_next_fits_one_blackwell_card_with_the_ngram_table_in_host_ram() {
        let flash = find("primitive-ai/Qwen3.8-Flash-Next-NVFP4").unwrap();
        let pro = GpuType {
            id: "NVIDIA RTX PRO 6000 Blackwell Server Edition".into(),
            ..gpu("RTX PRO 6000", 96, 2.49, 0.0)
        };
        let h200 = gpu("H200", 141, 5.29, 0.0);
        let h100 = gpu("H100 SXM", 80, 3.99, 0.0);
        assert!(flash.fits(&pro), "{:.1} GiB", flash.min_vram_gib());
        assert!(!flash.fits(&h200), "NVFP4 experts need Blackwell");
        assert!(!flash.fits(&h100));
        assert_eq!(flash.disk_gb(), 224, "the download holds the whole table");
    }

    #[test]
    fn sold_out_alternatives_include_the_same_card_in_the_other_cloud() {
        use mrun_runpod::Stock;
        let stocked = |mut g: GpuType, secure: Option<&str>, community: Option<&str>| {
            g.secure_stock = secure.map(|s| Stock {
                stock_status: Some(s.into()),
            });
            g.community_stock = community.map(|s| Stock {
                stock_status: Some(s.into()),
            });
            g
        };
        let offer = Offer {
            weights_bytes: 80_400_000_000,
            kv_bytes_per_token: 24_576,
            ..STUB
        };
        let gpus = vec![
            stocked(gpu("RTX PRO 6000", 96, 2.49, 1.69), Some("Medium"), None),
            stocked(gpu("RTX PRO 6000 WK", 96, 2.59, 1.69), None, Some("Low")),
            stocked(gpu("H200 SXM", 141, 5.29, 3.59), Some("Medium"), None),
            stocked(gpu("RTX 4090", 24, 0.89, 0.34), Some("High"), Some("High")),
        ];
        let alt = alternatives(&offer, &gpus, "NVIDIA RTX PRO 6000", CloudType::Secure);
        let names: Vec<_> = alt
            .iter()
            .map(|(g, c, _)| (g.display_name.as_str(), *c))
            .collect();
        assert_eq!(
            names,
            [
                ("RTX PRO 6000 WK", CloudType::Community),
                ("H200 SXM", CloudType::Secure)
            ]
        );
    }
}
