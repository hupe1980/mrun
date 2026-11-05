//! Safetensors files laid out for mapping (spec 022 R1): MLX writes the header unpadded, so the
//! data starts at an odd offset and no tensor sits on its element size; nuvec then copies the
//! weights instead of mapping them. Padding the JSON header with spaces (0x20, which the format
//! allows) so that `8 + header length` is a multiple of [`ALIGN`] aligns every tensor of the
//! shipped checkpoints without moving a tensor byte relative to the others.
//!
//! A realigned file is `u64 LE (L + pad) ‖ header ‖ pad × 0x20 ‖ data`; [`original_sha256`]
//! hashes the original `u64 LE (L) ‖ header ‖ data`, so the Hub's hash still verifies it.

use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

/// Alignment of the data section: covers every dtype the engine loads (at most 8 bytes).
pub const ALIGN: u64 = 16;
/// The largest pad [`pad_for`] returns.
pub const MAX_PAD: u64 = ALIGN - 1;
/// Headers larger than this are refused (safetensors' own limit is 100 MB).
const MAX_HEADER: u64 = 100 << 20;

/// Spaces to append to a header of `header_len` bytes so the data starts on [`ALIGN`].
pub fn pad_for(header_len: u64) -> u64 {
    (ALIGN - (8 + header_len) % ALIGN) % ALIGN
}

/// The header length a safetensors file declares in its first 8 bytes.
fn header_len(file: &mut fs::File) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    let len = u64::from_le_bytes(bytes);
    if len > MAX_HEADER {
        return Err(invalid(&format!("safetensors header of {len} bytes")));
    }
    Ok(len)
}

/// The pad `realign` would add to the file at `path` named `name`: 0 for other files and for a
/// header that cannot be read (the file is placed as it is; loading it reports the problem).
pub fn pad_of(path: &Path, name: &str) -> io::Result<u64> {
    if !name.ends_with(".safetensors") {
        return Ok(0);
    }
    Ok(header_len(&mut fs::File::open(path)?).map_or(0, pad_for))
}

/// Writes `src` to `dst` laid out for mapping and returns the pad. Files that are not
/// `*.safetensors`, or need no pad, are copied unchanged (a clone on APFS: no extra disk).
pub fn realign(src: &Path, dst: &Path, name: &str) -> io::Result<u32> {
    if !name.ends_with(".safetensors") {
        fs::copy(src, dst)?;
        return Ok(0);
    }
    let mut input = fs::File::open(src)?;
    let Ok(len) = header_len(&mut input) else {
        drop(input);
        fs::copy(src, dst)?;
        return Ok(0);
    };
    let pad = pad_for(len);
    if pad == 0 {
        drop(input);
        fs::copy(src, dst)?;
        return Ok(0);
    }
    let mut out = io::BufWriter::new(fs::File::create(dst)?);
    out.write_all(&(len + pad).to_le_bytes())?;
    io::copy(&mut (&mut input).take(len), &mut out)?;
    out.write_all(&vec![b' '; pad as usize])?;
    io::copy(&mut input, &mut out)?;
    out.into_inner()
        .map_err(io::IntoInnerError::into_error)?
        .sync_all()?;
    Ok(pad as u32)
}

/// Feeds `each` the bytes of the original file behind `path`, written with `pad` (0: the file as
/// it is), in chunks.
pub fn original_bytes(path: &Path, pad: u32, mut each: impl FnMut(&[u8])) -> io::Result<()> {
    let mut file = fs::File::open(path)?;
    let mut rest: Box<dyn Read> = if pad > 0 {
        let len = header_len(&mut file)?
            .checked_sub(u64::from(pad))
            .ok_or_else(|| invalid("header shorter than its pad"))?;
        each(&len.to_le_bytes());
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)?;
        each(&header);
        let mut spaces = vec![0u8; pad as usize];
        file.read_exact(&mut spaces)?;
        if spaces.iter().any(|&b| b != b' ') {
            return Err(invalid("padding is not spaces"));
        }
        Box::new(file)
    } else {
        Box::new(file)
    };
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        match rest.read(&mut buffer)? {
            0 => return Ok(()),
            n => each(&buffer[..n]),
        }
    }
}

/// sha256 of the original file behind `path`, written with `pad`.
pub fn original_sha256(path: &Path, pad: u32) -> io::Result<String> {
    let mut hasher = Sha256::new();
    original_bytes(path, pad, |chunk| hasher.update(chunk))?;
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors file with `header` (JSON plus `extra` spaces) and `data`.
    fn file(header: &str, extra: usize, data: &[u8]) -> Vec<u8> {
        let header = format!("{header}{}", " ".repeat(extra));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    fn sha(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    #[test]
    fn pads_put_the_data_on_the_alignment() {
        for len in 0..64 {
            let pad = pad_for(len);
            assert!(pad <= MAX_PAD);
            assert_eq!((8 + len + pad) % ALIGN, 0, "header of {len} bytes");
        }
    }

    #[test]
    fn realigned_files_keep_every_tensor_byte_and_the_hub_hash() {
        let dir = tempfile::tempdir().unwrap();
        // Two bf16 tensors at relative offsets 0 and 16, as MLX writes them; the unpadded header
        // leaves the data off the alignment.
        let header = r#"{"a":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]},"b":{"dtype":"BF16","shape":[8],"data_offsets":[16,32]}}"#;
        let data: Vec<u8> = (0..32).collect();
        for extra in [0, 3] {
            let original = file(header, extra, &data);
            let src = dir.path().join("src.safetensors");
            fs::write(&src, &original).unwrap();
            let dst = dir.path().join("model.safetensors");
            let pad = realign(&src, &dst, "model.safetensors").unwrap();
            let out = fs::read(&dst).unwrap();
            let start = 8 + u64::from_le_bytes(out[..8].try_into().unwrap()) as usize;
            assert_eq!(start as u64 % ALIGN, 0);
            assert_eq!(&out[start..], &data[..], "tensor bytes unchanged");
            assert_eq!(out.len(), original.len() + pad as usize);
            // Header JSON (and any spaces it had) unchanged, then only spaces.
            let declared = original.len() - data.len() - 8;
            assert_eq!(&out[8..8 + declared], &original[8..8 + declared]);
            assert!(out[8 + declared..start].iter().all(|&b| b == b' '));
            assert_eq!(original_sha256(&dst, pad).unwrap(), sha(&original));
        }
    }

    #[test]
    fn other_files_and_aligned_ones_are_copied_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::write(&src, b"{\"model_type\":\"qwen3\"}").unwrap();
        let dst = dir.path().join("config.json");
        assert_eq!(realign(&src, &dst, "config.json").unwrap(), 0);
        assert_eq!(fs::read(&dst).unwrap(), fs::read(&src).unwrap());

        let header = "{}"; // 8 + 2 + 6 spaces = 16
        let aligned = file(header, 6, &[1, 2, 3, 4]);
        fs::write(&src, &aligned).unwrap();
        let dst = dir.path().join("w.safetensors");
        assert_eq!(realign(&src, &dst, "w.safetensors").unwrap(), 0);
        assert_eq!(fs::read(&dst).unwrap(), aligned);
        assert_eq!(original_sha256(&dst, 0).unwrap(), sha(&aligned));
    }

    #[test]
    fn unreadable_headers_are_placed_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::write(&src, vec![7u8; 4096]).unwrap(); // declares a header of ~500 PB
        let dst = dir.path().join("x.safetensors");
        assert_eq!(pad_of(&src, "x.safetensors").unwrap(), 0);
        assert_eq!(realign(&src, &dst, "x.safetensors").unwrap(), 0);
        assert_eq!(fs::read(&dst).unwrap(), vec![7u8; 4096]);
    }

    #[test]
    fn a_changed_byte_changes_the_original_hash() {
        let dir = tempfile::tempdir().unwrap();
        let original = file(
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
            0,
            &[9; 4],
        );
        let src = dir.path().join("src.safetensors");
        fs::write(&src, &original).unwrap();
        let dst = dir.path().join("m.safetensors");
        let pad = realign(&src, &dst, "m.safetensors").unwrap();
        assert!(pad > 0);
        let good = original_sha256(&dst, pad).unwrap();
        let mut bytes = fs::read(&dst).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1; // a data byte
        fs::write(&dst, &bytes).unwrap();
        assert_ne!(original_sha256(&dst, pad).unwrap(), good);
        bytes[last] ^= 1;
        let space = bytes.len() - 5; // the last pad byte (data is 4 bytes)
        bytes[space] = b'x';
        fs::write(&dst, &bytes).unwrap();
        assert!(
            original_sha256(&dst, pad).is_err(),
            "a pad byte that is not a space"
        );
    }
}
