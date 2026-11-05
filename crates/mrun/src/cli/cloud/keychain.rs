//! The RunPod key in the macOS Keychain (024): a generic password, service `mrun`, account
//! `runpod`, through `/usr/bin/security`. The key goes to `security` on its stdin (`security -i`),
//! never in argv, where `ps` would show it. `MRUN_KEYCHAIN_STUB` (a directory) replaces the
//! Keychain in tests.

use std::io::Write;
use std::process::{Command, Stdio};

const SERVICE: &str = "mrun";
const ACCOUNT: &str = "runpod";
const SECURITY: &str = "/usr/bin/security";

fn stub() -> Option<std::path::PathBuf> {
    std::env::var_os("MRUN_KEYCHAIN_STUB")
        .filter(|d| !d.is_empty())
        .map(|d| std::path::PathBuf::from(d).join(format!("{SERVICE}-{ACCOUNT}")))
}

/// The stored key, if any.
pub fn find() -> Option<String> {
    find_at(stub())
}

/// Stores `key`, replacing a stored one.
pub fn store(key: &str) -> Result<(), String> {
    store_at(stub(), key)
}

/// Deletes the stored key; `false` when none was stored.
pub fn delete() -> Result<bool, String> {
    delete_at(stub())
}

fn find_at(stub: Option<std::path::PathBuf>) -> Option<String> {
    if let Some(path) = stub {
        return std::fs::read_to_string(path)
            .ok()
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty());
    }
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = Command::new(SECURITY)
        .args(["find-generic-password", "-s", SERVICE, "-a", ACCOUNT, "-w"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let key = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !key.is_empty()).then_some(key)
}

fn store_at(stub: Option<std::path::PathBuf>, key: &str) -> Result<(), String> {
    if key
        .chars()
        .any(|c| c.is_whitespace() || c == '"' || c == '\\')
    {
        return Err("a RunPod key has no spaces, quotes or backslashes".into());
    }
    if let Some(path) = stub {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        return std::fs::write(path, key).map_err(|e| e.to_string());
    }
    if !cfg!(target_os = "macos") {
        return Err("the Keychain exists on macOS only; set RUNPOD_API_KEY instead".into());
    }
    let mut child = Command::new(SECURITY)
        .arg("-i")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run {SECURITY}: {e}"))?;
    let command = format!(
        "add-generic-password -U -s {SERVICE} -a {ACCOUNT} -l \"mrun: RunPod API key\" -w \"{key}\"\n"
    );
    child
        .stdin
        .take()
        .ok_or("no stdin for security")?
        .write_all(command.as_bytes())
        .map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let errors = String::from_utf8_lossy(&out.stderr);
    if out.status.success() && errors.trim().is_empty() {
        Ok(())
    } else {
        Err(format!("the Keychain refused the key: {}", errors.trim()))
    }
}

fn delete_at(stub: Option<std::path::PathBuf>) -> Result<bool, String> {
    if let Some(path) = stub {
        return match std::fs::remove_file(path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.to_string()),
        };
    }
    if !cfg!(target_os = "macos") {
        return Ok(false);
    }
    let found = find_at(None).is_some();
    if found {
        let status = Command::new(SECURITY)
            .args(["delete-generic-password", "-s", SERVICE, "-a", ACCOUNT])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("cannot run {SECURITY}: {e}"))?;
        if !status.success() {
            return Err("the Keychain did not delete the key".into());
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip_through_the_stub() {
        let dir = tempfile::tempdir().unwrap();
        let at = || Some(dir.path().join("mrun-runpod"));
        assert_eq!(find_at(at()), None);
        assert_eq!(delete_at(at()), Ok(false));
        store_at(at(), "rpa_ABC123").unwrap();
        assert_eq!(find_at(at()).as_deref(), Some("rpa_ABC123"));
        store_at(at(), "rpa_NEW").unwrap();
        assert_eq!(find_at(at()).as_deref(), Some("rpa_NEW"));
        assert!(store_at(at(), "has space").is_err());
        assert!(store_at(at(), "quote\"d").is_err());
        assert_eq!(delete_at(at()), Ok(true));
        assert_eq!(find_at(at()), None);
    }
}
