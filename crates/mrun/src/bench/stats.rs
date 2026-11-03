//! Statistics of a run (005 FR-004, D038).
//!
//! - A cell's summary: min, median and max always; nearest-rank p95 and the interquartile range
//!   only from 20 samples, where they mean something.
//! - Comparisons: mrun's speed-up over another system with a bootstrap 95 % interval. Agent cells
//!   pair turns (the same turn of the same trace on both systems) and summarize the per-turn
//!   ratios by their geometric mean; synthetic cells, which have no pairing, compare medians and
//!   resample each side on its own. Resampling is seeded, so the same samples give the same
//!   interval.

/// Samples needed before p95 and IQR are reported.
pub const MIN_FOR_SPREAD: usize = 20;
/// Bootstrap resamples per interval.
pub const RESAMPLES: u32 = 2_000;
/// Seed of every bootstrap (fixed: the same samples give the same interval).
pub const SEED: u64 = 0x2026_1015;
/// Two-sided confidence level of the intervals.
pub const LEVEL: f64 = 0.95;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub min: f64,
    pub median: f64,
    pub max: f64,
    pub p95: Option<f64>,
    pub iqr: Option<f64>,
}

/// Nearest-rank percentile of sorted samples (`p` in 0–100).
fn nearest_rank(sorted: &[f64], p: f64) -> f64 {
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn sorted(samples: &[f64]) -> Vec<f64> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

pub fn median(samples: &[f64]) -> Option<f64> {
    (!samples.is_empty()).then(|| median_of_sorted(&sorted(samples)))
}

pub fn summarize(samples: &[f64]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let sorted = sorted(samples);
    let n = sorted.len();
    let spread = n >= MIN_FOR_SPREAD;
    Some(Summary {
        min: sorted[0],
        median: median_of_sorted(&sorted),
        max: sorted[n - 1],
        p95: spread.then(|| nearest_rank(&sorted, 95.0)),
        iqr: spread.then(|| nearest_rank(&sorted, 75.0) - nearest_rank(&sorted, 25.0)),
    })
}

/// Geometric mean of positive, finite values; `None` when there is none.
pub fn geometric_mean(values: &[f64]) -> Option<f64> {
    let logs: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|v| v.ln())
        .collect();
    (!logs.is_empty() && logs.len() == values.len())
        .then(|| (logs.iter().sum::<f64>() / logs.len() as f64).exp())
}

/// A point estimate with its bootstrap interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub estimate: f64,
    pub low: f64,
    pub high: f64,
}

/// SplitMix64: a small, seeded source for resampling indices.
#[derive(Debug)]
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Linear-interpolated quantile (`q` in 0–1) of sorted values.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn percentile_interval(estimate: f64, mut stats: Vec<f64>) -> Interval {
    stats.sort_by(f64::total_cmp);
    let tail = (1.0 - LEVEL) / 2.0;
    Interval {
        estimate,
        low: quantile(&stats, tail),
        high: quantile(&stats, 1.0 - tail),
    }
}

/// Paired comparison: `ratios[i]` is mrun's speed-up on turn `i`. The estimate is their geometric
/// mean; the interval resamples turns with replacement.
pub fn paired(ratios: &[f64], resamples: u32, seed: u64) -> Option<Interval> {
    let estimate = geometric_mean(ratios)?;
    let logs: Vec<f64> = ratios.iter().map(|r| r.ln()).collect();
    let n = logs.len();
    let mut rng = SplitMix(seed);
    let stats = (0..resamples.max(1))
        .map(|_| {
            let sum: f64 = (0..n).map(|_| logs[rng.below(n)]).sum();
            (sum / n as f64).exp()
        })
        .collect();
    Some(percentile_interval(estimate, stats))
}

/// Unpaired comparison of medians: `numerator / denominator`, each side resampled on its own.
pub fn ratio_of_medians(
    numerator: &[f64],
    denominator: &[f64],
    resamples: u32,
    seed: u64,
) -> Option<Interval> {
    let positive = |s: &[f64]| !s.is_empty() && s.iter().all(|v| v.is_finite() && *v > 0.0);
    if !positive(numerator) || !positive(denominator) {
        return None;
    }
    let estimate = median(numerator)? / median(denominator)?;
    let mut rng = SplitMix(seed);
    let mut resample = |s: &[f64]| -> f64 {
        let drawn: Vec<f64> = (0..s.len()).map(|_| s[rng.below(s.len())]).collect();
        median_of_sorted(&sorted(&drawn))
    };
    let stats = (0..resamples.max(1))
        .map(|_| resample(numerator) / resample(denominator))
        .collect();
    Some(percentile_interval(estimate, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_samples_have_no_spread() {
        let s = summarize(&[3.0, 1.0, 2.0, 10.0, 4.0]).unwrap();
        assert_eq!((s.min, s.median, s.max), (1.0, 3.0, 10.0));
        assert_eq!((s.p95, s.iqr), (None, None));
        assert_eq!(summarize(&[2.0, 4.0]).unwrap().median, 3.0);
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn twenty_samples_get_nearest_rank_p95_and_iqr() {
        let samples: Vec<f64> = (1..=20).map(f64::from).collect();
        let s = summarize(&samples).unwrap();
        assert_eq!(s.p95, Some(19.0));
        assert_eq!(s.iqr, Some(15.0 - 5.0));
        assert_eq!(s.median, 10.5);
    }

    #[test]
    fn geometric_mean_needs_positive_values() {
        assert!((geometric_mean(&[2.0, 8.0]).unwrap() - 4.0).abs() < 1e-12);
        assert!(geometric_mean(&[]).is_none());
        assert!(geometric_mean(&[1.0, 0.0]).is_none());
        assert!(geometric_mean(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn constant_ratios_give_a_point_interval() {
        let i = paired(&[2.0; 10], RESAMPLES, SEED).unwrap();
        assert!((i.estimate - 2.0).abs() < 1e-12);
        assert!((i.low - 2.0).abs() < 1e-12 && (i.high - 2.0).abs() < 1e-12);
    }

    #[test]
    fn paired_interval_brackets_the_estimate_and_is_reproducible() {
        let ratios: Vec<f64> = (0..40)
            .map(|i| 1.2 + 0.05 * f64::from(i % 7) - 0.1)
            .collect();
        let a = paired(&ratios, RESAMPLES, SEED).unwrap();
        let b = paired(&ratios, RESAMPLES, SEED).unwrap();
        assert_eq!(a, b, "seeded");
        assert!(a.low <= a.estimate && a.estimate <= a.high);
        assert!(
            a.low > 1.0,
            "a consistent 10-40 % speed-up is not noise: {a:?}"
        );
        // A different seed moves the bounds a little, never the estimate.
        let c = paired(&ratios, RESAMPLES, SEED + 1).unwrap();
        assert_eq!(c.estimate, a.estimate);
        assert!((c.low - a.low).abs() < 0.02);
    }

    #[test]
    fn noisy_ratios_around_one_include_one() {
        let ratios = [0.7, 1.4, 0.8, 1.3, 1.0, 0.9, 1.1, 0.75, 1.35, 1.05];
        let i = paired(&ratios, RESAMPLES, SEED).unwrap();
        assert!(i.low < 1.0 && i.high > 1.0, "{i:?}");
        assert!(paired(&[], RESAMPLES, SEED).is_none());
        assert!(paired(&[1.0, -1.0], RESAMPLES, SEED).is_none());
    }

    #[test]
    fn ratio_of_medians_is_unpaired() {
        let fast = [10.0, 11.0, 9.0, 10.5, 9.5];
        let slow = [20.0, 21.0, 19.0, 22.0, 18.0];
        let i = ratio_of_medians(&slow, &fast, RESAMPLES, SEED).unwrap();
        assert!((i.estimate - 2.0).abs() < 1e-12);
        assert!(i.low > 1.5 && i.high < 2.6, "{i:?}");
        assert!(ratio_of_medians(&[], &fast, RESAMPLES, SEED).is_none());
        assert!(ratio_of_medians(&slow, &[0.0], RESAMPLES, SEED).is_none());
    }

    #[test]
    fn quantile_interpolates() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&s, 0.0), 1.0);
        assert_eq!(quantile(&s, 0.5), 3.0);
        assert_eq!(quantile(&s, 0.125), 1.5);
        assert_eq!(quantile(&s, 1.0), 5.0);
    }
}
