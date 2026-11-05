//! Mapped weights (spec 022 FR-011, SC-002): nuvec maps a safetensors file whose tensors sit on
//! their element size and copies one whose data starts off it, as MLX's unpadded headers leave
//! every mlx-community checkpoint. The same tiny model is loaded on the GPU as committed (MLX's
//! layout) and with its header padded with spaces (what `mrun pull` writes): the copied load
//! allocates its weights, the mapped one does not.

#![cfg(feature = "engine")]

use mrun_engine::budget::{DeviceMemory, GIB};
use mrun_engine::tensor::{self, memory};
use mrun_engine::testing::fixture;
use mrun_engine::{LoadOptions, NuvecEngine};
use std::fs;
use std::path::Path;

/// Copies `src` to `dst` with the safetensors header padded so the data starts on 16 bytes.
fn realigned(src: &Path, dst: &Path) {
    let bytes = fs::read(src).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let pad = (16 - (8 + len) % 16) % 16;
    assert!(pad > 0, "the fixture is unpadded, as MLX writes it");
    let mut out = (len + pad).to_le_bytes().to_vec();
    out.extend_from_slice(&bytes[8..8 + len as usize]);
    out.extend(std::iter::repeat_n(b' ', pad as usize));
    out.extend_from_slice(&bytes[8 + len as usize..]);
    fs::write(dst, out).unwrap();
}

/// Bytes the allocator holds after loading `dir` on the GPU (mappings excluded).
fn allocated_after_load(dir: &Path) -> u64 {
    tensor::memory::clear_cache();
    let before = memory::active_allocated();
    let engine = NuvecEngine::load(
        dir,
        LoadOptions {
            identity: "mapped".into(),
            cpu: false,
            fallback_device: Some(DeviceMemory {
                physical: 16 * GIB,
                recommended_working_set: 16 * GIB,
                available: None,
                estimated: false,
            }),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let held = memory::active_allocated().saturating_sub(before);
    drop(engine);
    held
}

#[test]
fn realigned_weights_are_mapped_and_unaligned_ones_copied() {
    if !tensor::metal_available() {
        eprintln!("skipped: no Metal device");
        return;
    }
    let source = fixture("tiny_qwen3_5");
    let root = tempfile::tempdir().unwrap();
    let (copied, mapped) = (root.path().join("copied"), root.path().join("mapped"));
    for dir in [&copied, &mapped] {
        fs::create_dir_all(dir).unwrap();
        fs::copy(source.join("config.json"), dir.join("config.json")).unwrap();
    }
    let weights = source.join("model.safetensors");
    fs::copy(&weights, copied.join("model.safetensors")).unwrap();
    realigned(&weights, &mapped.join("model.safetensors"));

    let bytes = fs::read(&weights).unwrap();
    let data = bytes.len() as u64 - 8 - u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let (copied, mapped) = (allocated_after_load(&copied), allocated_after_load(&mapped));
    // Everything but the weights (state, scratch) is the same in both loads.
    assert!(
        copied >= mapped + data * 9 / 10,
        "copied load holds {copied} B, mapped {mapped} B, weights {data} B"
    );
}
