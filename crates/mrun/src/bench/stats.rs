//! Summary statistics of a cell's samples (005 FR-004): min, median and max always; nearest-rank
//! p95 and the interquartile range only from 20 samples, where they mean something.

/// Samples needed before p95 and IQR are reported.
pub const MIN_FOR_SPREAD: usize = 20;

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

pub fn summarize(samples: &[f64]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    let spread = n >= MIN_FOR_SPREAD;
    Some(Summary {
        min: sorted[0],
        median,
        max: sorted[n - 1],
        p95: spread.then(|| nearest_rank(&sorted, 95.0)),
        iqr: spread.then(|| nearest_rank(&sorted, 75.0) - nearest_rank(&sorted, 25.0)),
    })
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
}
