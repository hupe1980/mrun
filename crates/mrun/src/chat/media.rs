//! Image tokens (009). The template writes one `<|image_pad|>` per image; after rendering each
//! becomes the image's tokens. Their ids are keys above every vocabulary: a block of consecutive
//! ids per image digest, handed out once per process, so equal ids mean the same image and the
//! prefix cache needs nothing new to stay exact (FR-006).

use super::image::Image;
use mrun_engine::ImageSpan;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The first key; ids from here up are images, never text (vocabularies end far below).
pub const FIRST_KEY: u32 = 1 << 30;

static KEYS: Mutex<Option<Registry>> = Mutex::new(None);

struct Registry {
    next: u32,
    blocks: HashMap<[u8; 32], u32>,
}

/// The first key of `image`'s block (the same for every request with these bytes).
pub fn key(image: &Image) -> Result<u32, String> {
    let mut keys = KEYS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let reg = keys.get_or_insert_with(|| Registry {
        next: FIRST_KEY,
        blocks: HashMap::new(),
    });
    if let Some(&k) = reg.blocks.get(&image.digest) {
        return Ok(k);
    }
    let k = reg.next;
    reg.next = k
        .checked_add(image.tokens())
        .ok_or("too many distinct images for this process; restart mrun")?;
    reg.blocks.insert(image.digest, k);
    Ok(k)
}

/// Whether `token` is an image token rather than text.
pub fn is_image(token: u32) -> bool {
    token >= FIRST_KEY
}

/// One image of a rendered prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub start: u32,
    pub image: Arc<Image>,
    pub key: u32,
}

impl Placed {
    pub fn span(&self) -> ImageSpan {
        ImageSpan {
            start: self.start,
            height: self.image.rows,
            width: self.image.columns,
            key: u64::from(self.key),
        }
    }
}

/// A prompt whose image places hold their images' tokens.
#[derive(Debug)]
pub struct Expanded {
    pub tokens: Vec<u32>,
    pub placed: Vec<Placed>,
    /// Each pad's index before expansion and the tokens it added.
    added: Vec<(u32, u32)>,
}

impl Expanded {
    /// Where the token at `old` (before expansion) is now.
    pub fn map(&self, old: u32) -> u32 {
        old + self
            .added
            .iter()
            .filter(|(at, _)| *at < old)
            .map(|(_, add)| add)
            .sum::<u32>()
    }
}

/// Replaces each `pad` in `tokens` by the next image's keys; the images in template order.
pub fn expand(tokens: &[u32], pad: u32, images: &[Arc<Image>]) -> Result<Expanded, String> {
    let pads: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == pad)
        .map(|(i, _)| i)
        .collect();
    if pads.len() != images.len() {
        return Err(format!(
            "the prompt has {} image places for {} images",
            pads.len(),
            images.len()
        ));
    }
    let mut out = Vec::with_capacity(tokens.len());
    let mut placed = Vec::with_capacity(images.len());
    let mut extra: Vec<(u32, u32)> = Vec::new();
    let mut next = 0;
    for (i, &t) in tokens.iter().enumerate() {
        if t == pad {
            let image = images[next].clone();
            next += 1;
            let k = key(&image)?;
            let start = out.len() as u32;
            out.extend(k..k + image.tokens());
            extra.push((i as u32, image.tokens() - 1));
            placed.push(Placed {
                start,
                image,
                key: k,
            });
        } else {
            out.push(t);
        }
    }
    Ok(Expanded {
        tokens: out,
        placed,
        added: extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(seed: u8, rows: u32, columns: u32) -> Arc<Image> {
        Arc::new(Image {
            bytes: Arc::new(vec![seed]),
            width: columns * 32,
            height: rows * 32,
            rows,
            columns,
            digest: [seed; 32],
        })
    }

    #[test]
    fn pads_become_each_images_own_keys() {
        let (a, b) = (image(1, 2, 2), image(2, 1, 3));
        let e = expand(&[5, 9, 6, 9, 7], 9, &[a.clone(), b.clone()]).unwrap();
        let (tokens, placed) = (e.tokens.clone(), e.placed.clone());
        let map = |old| e.map(old);
        assert_eq!(tokens.len(), 5 - 2 + 4 + 3);
        assert_eq!(placed[0].start, 1);
        assert_eq!(placed[1].start, 6);
        assert!(tokens[1..5].iter().all(|t| is_image(*t)));
        assert_eq!(tokens[5], 6);
        assert_eq!(
            map(2),
            5,
            "text after the first image moves by its 3 extra tokens"
        );
        assert_eq!(map(4), 9);
        // The same bytes get the same keys in the next request; other bytes never.
        let again = expand(&[9], 9, &[a]).unwrap().tokens;
        assert_eq!(&again[..], &tokens[1..5]);
        let other = expand(&[9], 9, &[image(3, 2, 2)]).unwrap().tokens;
        assert!(other.iter().all(|t| !tokens.contains(t)));
        assert!(expand(&[9, 9], 9, &[b]).is_err());
    }
}
