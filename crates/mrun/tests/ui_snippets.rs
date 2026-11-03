//! 020 SC-005: every API path the "Your code" snippets (`ui/src/lib/snippets.ts`) call is a
//! route the server serves (the `.route("…"` literals of `src/api/mod.rs`).

use std::collections::BTreeSet;
use std::path::Path;

fn read(relative: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(relative))
        .unwrap_or_else(|e| panic!("{relative}: {e}"))
}

/// Every `/v1/<segment>…` path in `text`, up to the first character a path does not hold.
fn api_paths(text: &str) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    let mut rest = text;
    while let Some(at) = rest.find("/v1/") {
        let tail = &rest[at..];
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || "/_-".contains(c)))
            .unwrap_or(tail.len());
        let path = tail[..end].trim_end_matches('/');
        if path.len() > "/v1".len() {
            paths.insert(path.to_string());
        }
        rest = &rest[at + end.max(1)..];
    }
    paths
}

/// The paths of the server's `.route("…"` calls.
fn routes(source: &str) -> BTreeSet<String> {
    source
        .split(".route(")
        .skip(1)
        .filter_map(|call| {
            let call = call.trim_start();
            let literal = call.strip_prefix('"')?;
            Some(literal[..literal.find('"')?].to_string())
        })
        .collect()
}

#[test]
fn every_snippet_path_is_a_served_route() {
    let served = routes(&read("src/api/mod.rs"));
    assert!(
        served.contains("/v1/messages") && served.contains("/v1/models"),
        "{served:?}"
    );
    let used = api_paths(&read("ui/src/lib/snippets.ts"));
    // The three APIs of research R5.
    for path in ["/v1/messages", "/v1/chat/completions", "/v1/responses"] {
        assert!(used.contains(path), "snippets.ts has no {path}: {used:?}");
    }
    for path in &used {
        assert!(served.contains(path), "{path} is not served: {served:?}");
    }
}

#[test]
fn paths_are_read_up_to_their_end() {
    let text = r#"curl ${base}/v1/messages \
path: "/v1/chat/completions" base_url="${base}/v1", `${base}/v1/responses`"#;
    assert_eq!(
        api_paths(text).into_iter().collect::<Vec<_>>(),
        ["/v1/chat/completions", "/v1/messages", "/v1/responses"]
    );
}
