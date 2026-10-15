//! Configuration sources (004 FR-010): defaults, then `MRUN_<FLAG>` environment variables, then
//! flags; no config file in 0.1. Flags and their variables are declared on the clap arguments;
//! this module catches misspelled variables, which would otherwise be ignored silently.

/// Every `MRUN_*` variable mrun reads (`--record` and `--allow-over` have none by design).
pub const KNOWN: [&str; 12] = [
    "MRUN_HOME",
    "MRUN_PORT",
    "MRUN_MODEL",
    "MRUN_MAX_OUTPUT",
    "MRUN_QUEUE",
    "MRUN_QUEUE_WAIT_S",
    "MRUN_MEMORY_FRACTION",
    "MRUN_RESERVE_GIB",
    "MRUN_NO_CACHE",
    "MRUN_REJECT_TOOL_IMAGES",
    "MRUN_DETACH",
    // Set by `mrun launch codex` for Codex's provider key.
    "MRUN_API_KEY",
];

/// Warnings for unknown `MRUN_*` names, with the closest known one.
pub fn unknown_variables(names: impl Iterator<Item = String>) -> Vec<String> {
    names
        .filter(|name| name.starts_with("MRUN_") && !KNOWN.contains(&name.as_str()))
        .map(|name| {
            let closest = KNOWN
                .iter()
                .min_by_key(|known| distance(&name, known))
                .filter(|known| distance(&name, known) <= 3);
            match closest {
                Some(known) => {
                    format!("warning: unknown variable {name} is ignored; did you mean {known}?")
                }
                None => format!("warning: unknown variable {name} is ignored"),
            }
        })
        .collect()
}

/// Levenshtein distance.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let next = (row[j + 1] + 1)
                .min(row[j] + 1)
                .min(prev + usize::from(ca != *cb));
            prev = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misspelled_variables_get_a_suggestion() {
        let names = ["MRUN_PROT", "MRUN_PORT", "MRUN_ENTIRELY_ELSE", "PATH"]
            .map(String::from)
            .into_iter();
        let warnings = unknown_variables(names);
        assert_eq!(
            warnings,
            vec![
                "warning: unknown variable MRUN_PROT is ignored; did you mean MRUN_PORT?",
                "warning: unknown variable MRUN_ENTIRELY_ELSE is ignored",
            ]
        );
    }
}
