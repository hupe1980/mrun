//! Sampling from one position's logits, following mlx-lm `sample_utils` (0.32): log-probs in the
//! logits dtype, a token mask as `where(mask, logprobs, −inf)`, then top-p, min-p and top-k
//! filters, and greedy argmax or categorical sampling at `1 / temperature`. A repetition penalty
//! (mlx-lm `make_repetition_penalty`) runs on the raw logits first, as mlx-lm's logits
//! processors run before its sampler.

use crate::engine::{Sampling, TokenMask};
use crate::tensor::{self, Array, Dtype};

/// `logits − logsumexp(logits)` over the vocabulary (`[1, V]`).
pub fn logprobs(logits: &Array) -> tensor::Result<Array> {
    tensor::subtract(logits, &tensor::logsumexp(logits, -1, true)?)
}

fn scalar(value: f32, like: &Array) -> tensor::Result<Array> {
    tensor::astype(&Array::scalar_f32(value), like.dtype())
}

/// How many of the most recent token ids the repetition penalty looks at (mlx-lm's default
/// `repetition_context_size`).
pub const REPETITION_CONTEXT: usize = 20;

/// mlx-lm `make_repetition_penalty`: the logits of the `recent` ids (`[1, n]`, at most
/// [`REPETITION_CONTEXT`]) are multiplied by `penalty` where negative and divided by it
/// otherwise, in the logits dtype; repeated ids get the same value.
fn repetition_penalty(logits: &Array, recent: &Array, penalty: f32) -> tensor::Result<Array> {
    let selected = tensor::take_along_axis(logits, recent, -1)?;
    let factor = scalar(penalty, &selected)?;
    let positive = tensor::greater_equal(&selected, &scalar(0.0, &selected)?)?;
    let penalized = tensor::where_(
        &positive,
        &tensor::divide(&selected, &factor)?,
        &tensor::multiply(&selected, &factor)?,
    )?;
    tensor::put_along_axis(logits, recent, &penalized, -1)
}

/// mlx-lm `apply_top_p`: keep the most likely tokens whose mass reaches `p` (mass summed in f32).
fn top_p(logprobs: &Array, p: f32) -> tensor::Result<Array> {
    let sorted = tensor::sort(logprobs, -1)?;
    let probs = tensor::exp(&tensor::astype(&sorted, Dtype::Float32)?)?;
    let above = tensor::cumsum(&probs, -1, true, false)?;
    let first = |a: &Array| tensor::slice(a, &[0, 0], &[1, 1]);
    let total = tensor::add(&first(&above)?, &first(&probs)?)?;
    let cut = tensor::multiply(&total, &Array::scalar_f32(p))?;
    let dropped = tensor::sum(
        &tensor::astype(&tensor::greater_equal(&above, &cut)?, Dtype::Uint32)?,
        -1,
        true,
    )?;
    let threshold = tensor::take_along_axis(&sorted, &dropped, -1)?;
    let keep = tensor::greater_equal(logprobs, &threshold)?;
    tensor::where_(&keep, logprobs, &scalar(f32::NEG_INFINITY, logprobs)?)
}

/// mlx-lm `apply_min_p`: drop tokens below `min_p × max probability`.
fn min_p(logprobs: &Array, min_p: f32) -> tensor::Result<Array> {
    let top = tensor::max(logprobs, -1, true)?;
    let floor = tensor::add(&top, &scalar(min_p.ln(), &top)?)?;
    let keep = tensor::greater_equal(logprobs, &floor)?;
    tensor::where_(&keep, logprobs, &scalar(f32::NEG_INFINITY, logprobs)?)
}

/// mlx-lm `apply_top_k`: keep the `k` most likely tokens (`k` at least the vocabulary keeps
/// all).
fn top_k(logprobs: &Array, k: u32) -> tensor::Result<Array> {
    let vocab = logprobs.dim(-1);
    let Ok(k) = i32::try_from(k) else {
        return Ok(logprobs.clone());
    };
    if k >= vocab {
        return Ok(logprobs.clone());
    }
    let order = tensor::argpartition(&tensor::negative(logprobs)?, k - 1, -1)?;
    let drop = tensor::slice(&order, &[0, k], &[1, vocab])?;
    let fill = tensor::reshape(&scalar(f32::NEG_INFINITY, logprobs)?, &[1, 1])?;
    tensor::put_along_axis(logprobs, &drop, &fill, -1)
}

/// The next token from `logits` (`[1, V]`) as a lazy `[1]` array, so the caller can feed it
/// before reading it back (pipelining). `None` when `mask` allows nothing. A mask narrower than
/// `V` (a tokenizer vocabulary below the padded embedding) disallows the rest; a wider one is an
/// error.
///
/// `recent` holds the sequence's last token ids (`[1, n]` u32, prompt and generated, at most
/// [`REPETITION_CONTEXT`]) for `sampling.repetition_penalty`; with a penalty of 1 or no ids it
/// is not touched and no operation is added (bit-identical to sampling without it).
pub fn sample(
    logits: &Array,
    sampling: &Sampling,
    mask: Option<&TokenMask>,
    recent: Option<&Array>,
    key: impl FnOnce() -> tensor::Result<Array>,
) -> tensor::Result<Option<Array>> {
    let penalized;
    let logits = match recent {
        Some(recent) if sampling.repetition_penalty != 1.0 && recent.size() > 0 => {
            penalized = repetition_penalty(logits, recent, sampling.repetition_penalty)?;
            &penalized
        }
        _ => logits,
    };
    let mut lp = logprobs(logits)?;
    if let Some(mask) = mask {
        if mask.is_none() {
            return Ok(None);
        }
        let width = lp.dim(-1);
        let flags = mask.to_flags(width as u32).ok_or_else(|| {
            tensor::Error(format!(
                "a token mask over {} tokens is wider than the {width} logits",
                mask.vocab()
            ))
        })?;
        let allowed = Array::from_bool(&flags, &[1, width]);
        lp = tensor::where_(&allowed, &lp, &scalar(f32::NEG_INFINITY, &lp)?)?;
    }
    if sampling.is_greedy() {
        return tensor::argmax(&lp, -1, false).map(Some);
    }
    if sampling.top_p > 0.0 && sampling.top_p < 1.0 {
        lp = top_p(&lp, sampling.top_p)?;
    }
    if sampling.min_p > 0.0 {
        lp = min_p(&lp, sampling.min_p)?;
    }
    if sampling.top_k > 0 {
        lp = top_k(&lp, sampling.top_k)?;
    }
    let scaled = tensor::multiply(&lp, &scalar(1.0 / sampling.temperature, &lp)?)?;
    tensor::categorical(&scaled, -1, &key()?).map(Some)
}

/// The `n` most likely tokens of `logits` with their log-probabilities, most likely first. The
/// selection runs on the device; only `n` pairs are read back.
pub fn top_logprobs(logits: &Array, n: usize) -> tensor::Result<Vec<(u32, f32)>> {
    let lp = logprobs(logits)?;
    let vocab = lp.dim(-1);
    let n = (n as i32).clamp(1, vocab);
    let ids = if n == vocab {
        tensor::argsort(&tensor::negative(&lp)?, -1)?
    } else {
        let part = tensor::argpartition(&tensor::negative(&lp)?, n - 1, -1)?;
        tensor::slice(&part, &[0, 0], &[1, n])?
    };
    let values = tensor::take_along_axis(&lp, &ids, -1)?;
    let mut pairs: Vec<(u32, f32)> = ids
        .to_vec_u32()?
        .into_iter()
        .zip(values.to_vec_f32()?)
        .collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Device, use_device};

    fn logits(values: &[f32]) -> Array {
        Array::from_f32(values, &[1, values.len() as i32])
    }

    fn token(values: &[f32], sampling: Sampling, mask: Option<&TokenMask>, seed: u64) -> u32 {
        sample(&logits(values), &sampling, mask, None, || {
            tensor::random_key(seed)
        })
        .unwrap()
        .unwrap()
        .item_u32()
        .unwrap()
    }

    #[test]
    fn greedy_takes_the_argmax_and_respects_the_mask() {
        use_device(Device::Cpu);
        let values = [0.0, 5.0, 1.0, 4.0];
        assert_eq!(token(&values, Sampling::GREEDY, None, 0), 1);
        let mut mask = TokenMask::none_allowed(4);
        mask.allow(2);
        mask.allow(3);
        assert_eq!(token(&values, Sampling::GREEDY, Some(&mask), 0), 3);
        let none = TokenMask::none_allowed(4);
        let sampled = sample(
            &logits(&values),
            &Sampling::GREEDY,
            Some(&none),
            None,
            || unreachable!(),
        );
        assert!(sampled.unwrap().is_none());
    }

    #[test]
    fn a_tokenizer_wide_mask_pads_to_the_logits() {
        use_device(Device::Cpu);
        // Logits for 6 positions (padded embedding), a mask over the 4 tokenizer tokens.
        let values = [0.0, 1.0, 2.0, 3.0, 9.0, 9.0];
        let mut mask = TokenMask::none_allowed(4);
        mask.allow(1);
        mask.allow(2);
        assert_eq!(token(&values, Sampling::GREEDY, Some(&mask), 0), 2);
        let hot = Sampling {
            temperature: 1.0,
            ..Sampling::GREEDY
        };
        for seed in 0..16 {
            assert!([1, 2].contains(&token(&values, hot, Some(&mask), seed)));
        }
        let mut wide = TokenMask::none_allowed(7);
        wide.allow(0);
        assert!(
            sample(
                &logits(&values),
                &Sampling::GREEDY,
                Some(&wide),
                None,
                || unreachable!()
            )
            .is_err()
        );
        // top_k beyond the vocabulary (or i32) keeps every token.
        let k = Sampling {
            top_k: u32::MAX,
            ..hot
        };
        assert!(token(&values, k, None, 1) < 6);
    }

    #[test]
    fn filters_leave_only_the_expected_tokens() {
        use_device(Device::Cpu);
        let values = [0.0, 8.0, 7.9, -5.0, 7.8];
        let hot = Sampling {
            temperature: 1.0,
            ..Sampling::GREEDY
        };
        for seed in 0..32 {
            let k = token(&values, Sampling { top_k: 2, ..hot }, None, seed);
            assert!(k == 1 || k == 2, "top-k gave {k}");
            let p = token(&values, Sampling { top_p: 0.3, ..hot }, None, seed);
            assert_eq!(p, 1, "top-p");
            let m = token(&values, Sampling { min_p: 0.5, ..hot }, None, seed);
            assert!([1, 2, 4].contains(&m), "min-p gave {m}");
        }
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        use_device(Device::Cpu);
        let values: Vec<f32> = (0..50).map(|i| (i % 7) as f32 * 0.3).collect();
        let hot = Sampling {
            temperature: 1.0,
            ..Sampling::GREEDY
        };
        let a: Vec<u32> = (0..10).map(|s| token(&values, hot, None, s)).collect();
        let b: Vec<u32> = (0..10).map(|s| token(&values, hot, None, s)).collect();
        assert_eq!(a, b);
    }

    fn penalized(values: &[f32], sampling: Sampling, recent: &[u32], seed: u64) -> Array {
        let ids = Array::from_u32(recent, &[1, recent.len() as i32]);
        sample(&logits(values), &sampling, None, Some(&ids), || {
            tensor::random_key(seed)
        })
        .unwrap()
        .unwrap()
    }

    #[test]
    fn a_neutral_repetition_penalty_changes_nothing() {
        use_device(Device::Cpu);
        let values: Vec<f32> = (0..50)
            .map(|i| ((i * 13) % 17) as f32 * 0.37 - 3.0)
            .collect();
        let recent = [3, 7, 7, 12, 40];
        for sampling in [
            Sampling::GREEDY,
            Sampling {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 20,
                min_p: 0.05,
                ..Sampling::GREEDY
            },
        ] {
            assert_eq!(sampling.repetition_penalty, 1.0);
            for seed in 0..16 {
                let with = penalized(&values, sampling, &recent, seed)
                    .item_u32()
                    .unwrap();
                assert_eq!(with, token(&values, sampling, None, seed));
            }
        }
        // The penalty rule itself at 1 leaves the logits bit-identical.
        let l = logits(&values);
        let ids = Array::from_u32(&recent, &[1, recent.len() as i32]);
        let same = repetition_penalty(&l, &ids, 1.0).unwrap();
        let bits = |a: &Array| -> Vec<u32> {
            a.to_vec_f32()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        assert_eq!(bits(&same), bits(&l));
    }

    #[test]
    fn a_repetition_penalty_lowers_recent_tokens() {
        use_device(Device::Cpu);
        let values = [0.0, 5.0, 1.0, 4.0, -2.0];
        let greedy = |penalty: f32, recent: &[u32]| {
            let s = Sampling {
                repetition_penalty: penalty,
                ..Sampling::GREEDY
            };
            penalized(&values, s, recent, 0).item_u32().unwrap()
        };
        assert_eq!(greedy(1.0, &[1]), 1);
        // 5 / 1.5 = 3.33 < 4: greedy picks the next token.
        assert_eq!(greedy(1.5, &[1]), 3);
        // A repeated id is penalized once; the penalty also lowers a negative logit (× penalty).
        assert_eq!(greedy(1.5, &[1, 1, 1]), 3);
        let l = logits(&values);
        let ids = Array::from_u32(&[1, 4, 1], &[1, 3]);
        let out = repetition_penalty(&l, &ids, 2.0)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(out, [0.0, 2.5, 1.0, 4.0, -4.0]);
        // Sampled: the penalized token's probability drops.
        let hot = Sampling {
            temperature: 1.0,
            repetition_penalty: 3.0,
            ..Sampling::GREEDY
        };
        let picks = |s: Sampling| {
            (0..200)
                .filter(|&seed| penalized(&values, s, &[1], seed).item_u32().unwrap() == 1)
                .count()
        };
        assert!(
            picks(hot)
                < picks(Sampling {
                    repetition_penalty: 1.0,
                    ..hot
                })
        );
    }

    #[test]
    fn top_logprobs_are_sorted() {
        use_device(Device::Cpu);
        let top = top_logprobs(&logits(&[0.0, 3.0, 1.0, 2.0]), 2).unwrap();
        assert_eq!(top.iter().map(|t| t.0).collect::<Vec<_>>(), [1, 3]);
        assert!(top[0].1 > top[1].1 && top[0].1 < 0.0);
        assert_eq!(top_logprobs(&logits(&[0.0, 3.0]), 9).unwrap().len(), 2);
    }

    #[test]
    fn mask_flags_expand_word_by_word() {
        let mut mask = TokenMask::none_allowed(70);
        for t in [0, 63, 64, 69] {
            mask.allow(t);
        }
        let flags = mask.to_flags(70).unwrap();
        assert_eq!(flags.len(), 70);
        assert_eq!(mask.to_flags(69), None);
        let padded = mask.to_flags(200).unwrap();
        assert_eq!(padded.len(), 200);
        assert!(padded[69] && !padded[70..].iter().any(|&f| f));
        let set: Vec<usize> = flags
            .iter()
            .enumerate()
            .filter(|f| *f.1)
            .map(|f| f.0)
            .collect();
        assert_eq!(set, [0, 63, 64, 69]);
    }
}
