//! Images in prompts (009): what arrives inline (PNG, JPEG, WebP, GIF), checked, sized by mrun's
//! own rule `resize_v1` and turned into the patches the vision tower reads. Sizing needs only the
//! header, so an image whose state is reused is never decoded.

use mrun_engine::{MAX_IMAGE_TOKENS, MIN_IMAGE_TOKENS, Pixels, Vision};
use sha2::{Digest, Sha256};
use std::io::Cursor;
use std::sync::Arc;

/// The most bytes one image may have as sent.
pub const MAX_IMAGE_BYTES: usize = 20 << 20;
/// The most pixels one image may have decoded.
pub const MAX_IMAGE_PIXELS: u64 = 50_000_000;
/// Width to height (or back) beyond which an image is refused (Qwen2-VL's limit).
const MAX_ASPECT: u32 = 200;
/// The sizing rule's name; it is part of every image's digest.
pub const RESIZE: &str = "resize_v1";

/// An image as a request carries it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub bytes: Arc<Vec<u8>>,
    /// Width and height as sent.
    pub width: u32,
    pub height: u32,
    /// Its token grid (`resize_v1`): rows × columns, one token per 32 px square.
    pub rows: u32,
    pub columns: u32,
    /// SHA-256 of the bytes, the rule and the grid: two images share state only when equal.
    pub digest: [u8; 32],
}

impl Image {
    /// Checks `bytes` and sizes them for a tower with `token_px`-pixel tokens.
    pub fn new(bytes: Vec<u8>, token_px: u32) -> Result<Self, String> {
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(format!(
                "image is {} MiB; at most 20 MiB",
                bytes.len().div_ceil(1 << 20)
            ));
        }
        let reader = image::ImageReader::new(Cursor::new(&bytes[..]))
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        match reader.format() {
            Some(
                image::ImageFormat::Png
                | image::ImageFormat::Jpeg
                | image::ImageFormat::WebP
                | image::ImageFormat::Gif,
            ) => {}
            _ => return Err("not a PNG, JPEG, WebP or GIF image".into()),
        }
        let (width, height) = reader
            .into_dimensions()
            .map_err(|e| format!("unreadable image: {e}"))?;
        if width == 0 || height == 0 {
            return Err("image has no pixels".into());
        }
        if u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS {
            return Err(format!(
                "image is {width} × {height}; at most 50 megapixels"
            ));
        }
        if width.max(height) / width.min(height).max(1) > MAX_ASPECT {
            return Err(format!("image is {width} × {height}; too narrow to read"));
        }
        let (rows, columns) = grid(width, height, token_px);
        let mut hash = Sha256::new();
        hash.update(&bytes);
        hash.update(RESIZE.as_bytes());
        hash.update(rows.to_le_bytes());
        hash.update(columns.to_le_bytes());
        Ok(Self {
            bytes: Arc::new(bytes),
            width,
            height,
            rows,
            columns,
            digest: hash.finalize().into(),
        })
    }

    /// Tokens the image takes in the prompt.
    pub fn tokens(&self) -> u32 {
        self.rows * self.columns
    }

    /// Decodes, resizes (Catmull-Rom) and cuts the image into the tower's patches.
    pub fn pixels(&self, vision: &Vision) -> Result<Pixels, String> {
        let mut reader = image::ImageReader::new(Cursor::new(&self.bytes[..]))
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        let mut limits = image::Limits::default();
        limits.max_alloc = Some(MAX_IMAGE_PIXELS * 8);
        reader.limits(limits);
        let decoded = reader
            .decode()
            .map_err(|e| format!("unreadable image: {e}"))?;
        let (h, w) = (
            self.rows * vision.token_px(),
            self.columns * vision.token_px(),
        );
        let resized = decoded
            .resize_exact(w, h, image::imageops::FilterType::CatmullRom)
            .into_rgba8();
        let p = vision.patch as usize;
        let t = vision.temporal as usize;
        let (gh, gw) = (h / vision.patch, w / vision.patch);
        let m = vision.merge;
        let mut data = Vec::with_capacity((gh * gw) as usize * t * p * p * 3);
        // Merge order: 2 × 2 blocks row by row, the patches of a block row by row.
        for br in 0..gh / m {
            for bc in 0..gw / m {
                for r in 0..m {
                    for c in 0..m {
                        let (py, px) = ((br * m + r) as usize * p, (bc * m + c) as usize * p);
                        for _ in 0..t {
                            for y in 0..p {
                                for x in 0..p {
                                    let [r8, g8, b8, a8] =
                                        resized.get_pixel((px + x) as u32, (py + y) as u32).0;
                                    let a = f32::from(a8) / 255.0;
                                    for v in [r8, g8, b8] {
                                        // Transparent parts are white; then [0, 1] → [-1, 1].
                                        let on_white = f32::from(v) / 255.0 * a + (1.0 - a);
                                        data.push((on_white - 0.5) / 0.5);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Pixels {
            data,
            grid_h: gh,
            grid_w: gw,
        })
    }
}

/// `resize_v1`: both sides rounded to whole tokens, aspect kept, between `MIN_IMAGE_TOKENS` and
/// `MAX_IMAGE_TOKENS` tokens (Qwen2-VL's `smart_resize` with those bounds).
pub fn grid(width: u32, height: u32, token_px: u32) -> (u32, u32) {
    let f = f64::from(token_px);
    let (h, w) = (f64::from(height), f64::from(width));
    let max = f64::from(MAX_IMAGE_TOKENS) * f * f;
    let min = f64::from(MIN_IMAGE_TOKENS) * f * f;
    let round = |v: f64| ((v / f).round() * f).max(f);
    let (mut hb, mut wb) = (round(h), round(w));
    if hb * wb > max {
        let beta = (h * w / max).sqrt();
        hb = ((h / beta / f).floor() * f).max(f);
        wb = ((w / beta / f).floor() * f).max(f);
    } else if hb * wb < min {
        let beta = (min / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    ((hb / f) as u32, (wb / f) as u32)
}

/// The bytes of a `data:` URL (base64 only).
pub fn data_url(url: &str) -> Option<Result<Vec<u8>, String>> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if !meta.ends_with(";base64") {
        return Some(Err("data URL is not base64".into()));
    }
    Some(base64_decode(data))
}

pub fn base64_decode(data: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .map_err(|_| "image data is not valid base64".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(width, height, image::Rgba(rgba));
        let mut out = Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    const VISION: Vision = Vision {
        bytes: 0,
        patch: 16,
        merge: 2,
        temporal: 2,
    };

    #[test]
    fn sizes_round_to_tokens_within_the_bounds() {
        // A Claude-size screenshot: 1,568 × 992 → 49 × 31 tokens.
        assert_eq!(grid(1568, 992, 32), (31, 49));
        // Large images shrink below the cap, aspect kept.
        let (r, c) = grid(4000, 3000, 32);
        assert!(
            r * c <= MAX_IMAGE_TOKENS && r * c > MAX_IMAGE_TOKENS * 9 / 10,
            "{r}×{c}"
        );
        // Tiny images grow to the floor.
        let (r, c) = grid(10, 10, 32);
        assert!(r * c >= MIN_IMAGE_TOKENS, "{r}×{c}");
    }

    #[test]
    fn an_image_is_checked_sized_and_digested() {
        let red = Image::new(png(64, 96, [255, 0, 0, 255]), 32).unwrap();
        assert_eq!((red.width, red.height), (64, 96));
        assert_eq!(red.tokens(), red.rows * red.columns);
        let again = Image::new(png(64, 96, [255, 0, 0, 255]), 32).unwrap();
        assert_eq!(red.digest, again.digest);
        let blue = Image::new(png(64, 96, [0, 0, 255, 255]), 32).unwrap();
        assert_ne!(red.digest, blue.digest);
        assert!(Image::new(b"not an image".to_vec(), 32).is_err());
        assert!(Image::new(vec![0; MAX_IMAGE_BYTES + 1], 32).is_err());
    }

    #[test]
    fn patches_follow_the_grid_and_transparency_is_white() {
        let clear = Image::new(png(64, 64, [0, 0, 0, 0]), 32).unwrap();
        let px = clear.pixels(&VISION).unwrap();
        assert_eq!((px.grid_h, px.grid_w), (clear.rows * 2, clear.columns * 2));
        assert_eq!(
            px.data.len(),
            (px.grid_h * px.grid_w) as usize * 2 * 16 * 16 * 3
        );
        assert!(
            px.data.iter().all(|v| (*v - 1.0).abs() < 1e-6),
            "white is 1.0"
        );
        let red = Image::new(png(64, 64, [255, 0, 0, 255]), 32).unwrap();
        let px = red.pixels(&VISION).unwrap();
        assert_eq!(&px.data[..3], &[1.0, -1.0, -1.0]);
    }

    #[test]
    fn data_urls_are_base64_only() {
        assert_eq!(
            data_url("data:image/png;base64,aGk=").unwrap().unwrap(),
            b"hi"
        );
        assert!(data_url("data:image/png,raw").unwrap().is_err());
        assert!(data_url("https://example.com/a.png").is_none());
    }
}
