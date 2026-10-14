use super::{Observations, Sample, Severity};
use std::time::{Duration, Instant};

pub fn collect(
    clock: impl Fn() -> Duration,
    mut read: impl FnMut(u32) -> Option<u64>,
) -> Observations {
    let mut sample = |kind| {
        let at = clock();
        read(kind).map(|value| Sample { at, value })
    };
    Observations {
        physical: sample(0),
        available: sample(1),
        device_recommended: sample(2),
        device_allocated: sample(3),
        footprint: sample(4),
        rss: sample(5),
        swap: sample(6),
        pageouts: sample(7),
        pressure: sample(8).and_then(|observation| {
            let value = match observation.value {
                0 => Severity::Nominal,
                1 => Severity::Elevated,
                2 => Severity::Critical,
                _ => return None,
            };
            Some(Sample {
                at: observation.at,
                value,
            })
        }),
    }
}

pub fn system(epoch: Instant) -> (Observations, Duration) {
    let observations = collect(|| epoch.elapsed(), |_| None);
    (observations, epoch.elapsed())
}

pub fn require_complete(observations: &Observations) -> Result<(), super::Error> {
    if observations.physical.is_none()
        || observations.available.is_none()
        || observations.device_recommended.is_none()
        || observations.device_allocated.is_none()
        || observations.footprint.is_none()
        || observations.rss.is_none()
        || observations.swap.is_none()
        || observations.pageouts.is_none()
        || observations.pressure.is_none()
    {
        return Err(super::Error::Missing);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn observations_have_independent_times_and_missing_or_unknown_values_stay_missing() {
        let tick = Cell::new(0);
        let clock = || {
            let previous = tick.get();
            tick.set(previous + 1);
            Duration::from_millis(previous)
        };
        let output = collect(clock, |kind| {
            if kind == 8 {
                Some(0)
            } else {
                Some(u64::from(kind))
            }
        });
        assert_eq!(output.physical.unwrap().at, Duration::ZERO);
        assert_eq!(output.available.unwrap().at, Duration::from_millis(1));
        assert_eq!(output.pressure.unwrap().at, Duration::from_millis(8));
        let missing = collect(|| Duration::ZERO, |_| None);
        assert_eq!(
            require_complete(&missing),
            Err(super::super::Error::Missing)
        );
        require_complete(&output).unwrap();
        assert!(
            missing.physical.is_none() && missing.available.is_none() && missing.pressure.is_none()
        );
        let unknown = collect(|| Duration::ZERO, |_| Some(3));
        assert!(unknown.pressure.is_none());
    }

    #[test]
    fn system_observations_without_a_backend_cannot_authorize_load() {
        let (observations, _) = system(Instant::now());
        assert_eq!(
            require_complete(&observations),
            Err(super::super::Error::Missing)
        );
        assert!(observations.available.is_none());
        assert!(observations.device_recommended.is_none());
    }
}
