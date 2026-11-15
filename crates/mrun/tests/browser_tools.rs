//! `mrun mcp serve browser`: pageon's tools from the mrun binary, with its sandbox started as
//! `mrun content` (the server starts the caged content process before it answers).
#![cfg(all(feature = "browser", any(target_os = "macos", target_os = "linux")))]

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[test]
fn the_browser_lists_its_tools_from_a_caged_content_process() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_mrun"))
        .args(["mcp", "serve", "browser"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-06-18","capabilities":{{}},"clientInfo":{{"name":"t","version":"0"}}}}}}"#
    )
    .unwrap();
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":2,"method":"tools/list"}}"#).unwrap();
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"browser_open","arguments":{{"url":"http://127.0.0.1:9/"}}}}}}"#
    )
    .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut next =
        || serde_json::from_str::<serde_json::Value>(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(next()["result"]["serverInfo"]["name"], "pageon");
    let tools = next();
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    for name in [
        "browser_open",
        "browser_click",
        "browser_type",
        "browser_screenshot",
    ] {
        assert!(names.contains(&name), "{names:?}");
    }
    let refused = next();
    assert_eq!(refused["result"]["isError"], true);
    assert!(
        refused["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("private network"),
        "{refused}"
    );
    drop(stdin);
    assert!(child.wait().unwrap().success());
}
