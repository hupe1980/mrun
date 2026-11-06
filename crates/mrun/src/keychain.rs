//! mrun's secrets in the macOS Keychain: generic passwords of service `mrun`, through
//! `/usr/bin/security`: the RunPod key (024, account `runpod`) and the Mac pool's identity, pins
//! and membership secrets (026). A value goes to `security` on its stdin (`security -i`), never
//! in argv, where `ps` would show it. `MRUN_KEYCHAIN_STUB` (a directory) replaces the Keychain in
//! tests.

use std::io::Write;
use std::process::{Command, Stdio};

const SERVICE: &str = "mrun";
const ACCOUNT: &str = "runpod";
const SECURITY: &str = "/usr/bin/security";

/// The stored RunPod key, if any.
pub fn find() -> Option<String> {
    find_item(ACCOUNT)
}

/// Stores the RunPod key, replacing a stored one.
pub fn store(key: &str) -> Result<(), String> {
    if key
        .chars()
        .any(|c| c.is_whitespace() || c == '"' || c == '\\')
    {
        return Err("a RunPod key has no spaces, quotes or backslashes".into());
    }
    store_item(ACCOUNT, "mrun: RunPod API key", key)
}

/// Deletes the stored RunPod key; `false` when none was stored.
pub fn delete() -> Result<bool, String> {
    delete_item(ACCOUNT)
}

/// The value stored under `account`, if any.
pub fn find_item(account: &str) -> Option<String> {
    Vault::system().find(account)
}

/// Stores `value` (no whitespace, quotes or backslashes: hex for binary data) under `account`
/// with the Keychain label `label`.
pub fn store_item(account: &str, label: &str, value: &str) -> Result<(), String> {
    Vault::system().store(account, label, value)
}

/// Deletes `account`'s value; `false` when none was stored.
pub fn delete_item(account: &str) -> Result<bool, String> {
    Vault::system().delete(account)
}

/// Where secrets go: the Keychain (or `MRUN_KEYCHAIN_STUB`), or a directory of its own (tests
/// that run several Macs' servers in one process).
#[derive(Clone, Debug, Default)]
pub struct Vault {
    dir: Option<std::path::PathBuf>,
}

impl Vault {
    pub fn system() -> Self {
        Self {
            dir: std::env::var_os("MRUN_KEYCHAIN_STUB")
                .filter(|d| !d.is_empty())
                .map(std::path::PathBuf::from),
        }
    }

    pub fn dir(dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            dir: Some(dir.into()),
        }
    }

    fn at(&self, account: &str) -> Option<std::path::PathBuf> {
        self.dir
            .as_ref()
            .map(|d| d.join(format!("{SERVICE}-{account}")))
    }

    pub fn find(&self, account: &str) -> Option<String> {
        find_at(self.at(account), account)
    }

    pub fn store(&self, account: &str, label: &str, value: &str) -> Result<(), String> {
        store_at(self.at(account), account, label, value)
    }

    pub fn delete(&self, account: &str) -> Result<bool, String> {
        delete_at(self.at(account), account)
    }
}

fn find_at(stub: Option<std::path::PathBuf>, account: &str) -> Option<String> {
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
        .args(["find-generic-password", "-s", SERVICE, "-a", account, "-w"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let key = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !key.is_empty()).then_some(key)
}

fn store_at(
    stub: Option<std::path::PathBuf>,
    account: &str,
    label: &str,
    key: &str,
) -> Result<(), String> {
    let plain = |t: &str| {
        !t.chars()
            .any(|c| c.is_whitespace() || c == '"' || c == '\\')
    };
    if !plain(key) || !plain(account) || label.contains(['"', '\\', '\n']) {
        return Err("a Keychain value has no spaces, quotes or backslashes".into());
    }
    if let Some(path) = stub {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        return std::fs::write(path, key).map_err(|e| e.to_string());
    }
    if !cfg!(target_os = "macos") {
        return Err("the Keychain exists on macOS only".into());
    }
    let mut child = Command::new(SECURITY)
        .arg("-i")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run {SECURITY}: {e}"))?;
    let command =
        format!("add-generic-password -U -s {SERVICE} -a {account} -l \"{label}\" -w \"{key}\"\n");
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

fn delete_at(stub: Option<std::path::PathBuf>, account: &str) -> Result<bool, String> {
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
    let found = find_at(None, account).is_some();
    if found {
        let status = Command::new(SECURITY)
            .args(["delete-generic-password", "-s", SERVICE, "-a", account])
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
        let store = |key| store_at(at(), "runpod", "label", key);
        assert_eq!(find_at(at(), "runpod"), None);
        assert_eq!(delete_at(at(), "runpod"), Ok(false));
        store("rpa_ABC123").unwrap();
        assert_eq!(find_at(at(), "runpod").as_deref(), Some("rpa_ABC123"));
        store("rpa_NEW").unwrap();
        assert_eq!(find_at(at(), "runpod").as_deref(), Some("rpa_NEW"));
        assert!(store("has space").is_err());
        assert!(store("quote\"d").is_err());
        assert_eq!(delete_at(at(), "runpod"), Ok(true));
        assert_eq!(find_at(at(), "runpod"), None);
    }
}
