//! What the window owns on disk (018 data model): `ui.json` (its preferences), kept chats, the
//! login item, and the single-instance record. Nothing here is server configuration, which
//! stays D066's `config.toml` (never written by mrun).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The window's preferences (`MRUN_HOME/ui.json`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// `None` until the first chat asks (FR-010).
    pub keep_chats: Option<bool>,
    pub menu_bar: bool,
    pub open_at_login: bool,
    /// The model the window last served, used by Start.
    pub last_model: Option<String>,
    /// The window's last frame, written by the shell only (019 FR-010).
    pub window: Option<Frame>,
    /// Models the user added to Models' cloud section (024).
    pub cloud_models: Vec<CloudModel>,
}

/// A model added to the cloud section: the repo and what `pull --dry-run` said of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CloudModel {
    pub repo: String,
    #[serde(default)]
    pub weights_bytes: u64,
    #[serde(default)]
    pub family: Option<String>,
}

/// A window frame in logical points (origin top-left, as the shell's monitors report it).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// The window's smallest size; a saved frame below it grows to it.
pub const MIN_WIDTH: f64 = 800.0;
pub const MIN_HEIGHT: f64 = 560.0;
/// How much of a monitor a restored frame must cover in each direction (data model).
const MIN_VISIBLE: f64 = 100.0;

/// The saved frame to restore, at least the minimum size, when it covers at least
/// [`MIN_VISIBLE`] × [`MIN_VISIBLE`] of one monitor's work area; `None` (default size,
/// centred) when it would open off screen, say after a display was unplugged.
pub fn restore_frame(saved: Option<Frame>, monitors: &[Frame]) -> Option<Frame> {
    let saved = saved?;
    if ![saved.x, saved.y, saved.width, saved.height]
        .iter()
        .all(|v| v.is_finite())
    {
        return None;
    }
    let frame = Frame {
        width: saved.width.max(MIN_WIDTH),
        height: saved.height.max(MIN_HEIGHT),
        ..saved
    };
    let overlap = |m: &Frame| {
        let w = (frame.x + frame.width).min(m.x + m.width) - frame.x.max(m.x);
        let h = (frame.y + frame.height).min(m.y + m.height) - frame.y.max(m.y);
        w >= MIN_VISIBLE && h >= MIN_VISIBLE
    };
    monitors.iter().any(overlap).then_some(frame)
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            keep_chats: None,
            menu_bar: true,
            open_at_login: false,
            last_model: None,
            window: None,
            cloud_models: Vec::new(),
        }
    }
}

/// Files under one mrun home.
#[derive(Debug)]
pub struct Store {
    home: PathBuf,
    prefs: Mutex<Prefs>,
}

impl Store {
    /// Reads `ui.json`; an unreadable file is renamed `ui.json.bad` and the defaults apply.
    pub fn open(home: &Path) -> Self {
        let path = home.join("ui.json");
        let prefs = match fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|_| {
                let _ = fs::rename(&path, home.join("ui.json.bad"));
                Prefs::default()
            }),
            Err(_) => Prefs::default(),
        };
        let prefs = Prefs {
            open_at_login: login_item(home).is_file(),
            ..prefs
        };
        Self {
            home: home.to_path_buf(),
            prefs: Mutex::new(prefs),
        }
    }

    pub fn prefs(&self) -> Prefs {
        self.prefs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Applies `patch` (fields of [`Prefs`] but `window`, which only [`Store::set_window`]
    /// writes); `open_at_login` adds or removes the login item.
    pub fn update(&self, patch: &Value) -> io::Result<Prefs> {
        let mut prefs = self
            .prefs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut merged = serde_json::to_value(&*prefs).unwrap_or_default();
        if let (Value::Object(base), Value::Object(fields)) = (&mut merged, patch) {
            for (key, value) in fields {
                if base.contains_key(key) && key != "window" {
                    base.insert(key.clone(), value.clone());
                }
            }
        }
        let next: Prefs = serde_json::from_value(merged)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        if next.open_at_login != prefs.open_at_login {
            set_login_item(&self.home, next.open_at_login)?;
        }
        self.write(&next)?;
        *prefs = next.clone();
        Ok(next)
    }

    /// Saves the window's frame (the shell, on move, resize and close).
    pub fn set_window(&self, frame: Frame) -> io::Result<()> {
        let mut prefs = self
            .prefs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let next = Prefs {
            window: Some(frame),
            ..prefs.clone()
        };
        self.write(&next)?;
        *prefs = next;
        Ok(())
    }

    fn write(&self, prefs: &Prefs) -> io::Result<()> {
        fs::create_dir_all(&self.home)?;
        write_private(
            &self.home.join("ui.json"),
            serde_json::to_string_pretty(prefs)?.as_bytes(),
        )
    }

    fn chats_dir(&self) -> PathBuf {
        self.home.join("chats")
    }

    fn keeping(&self) -> bool {
        self.prefs().keep_chats == Some(true)
    }

    /// `[{id, title, updated}]`, newest first; empty unless chats are kept.
    pub fn chats(&self) -> Vec<Value> {
        let mut list: Vec<Value> = fs::read_dir(self.chats_dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| fs::read_to_string(e.path()).ok())
            .filter_map(|text| serde_json::from_str::<Value>(&text).ok())
            .map(|c| serde_json::json!({"id": c["id"], "title": c["title"], "updated": c["updated"]}))
            .collect();
        list.sort_by(|a, b| b["updated"].as_str().cmp(&a["updated"].as_str()));
        list
    }

    pub fn chat(&self, id: &str) -> Option<Value> {
        let path = self.chats_dir().join(format!("{}.json", valid_id(id)?));
        serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
    }

    /// Writes a chat; refused (`Ok(false)`) unless keeping chats is on (PR-5).
    pub fn save_chat(&self, id: &str, chat: &Value) -> io::Result<bool> {
        let Some(id) = valid_id(id) else {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "bad chat id"));
        };
        if !self.keeping() {
            return Ok(false);
        }
        fs::create_dir_all(self.chats_dir())?;
        write_private(
            &self.chats_dir().join(format!("{id}.json")),
            serde_json::to_string(chat)?.as_bytes(),
        )?;
        Ok(true)
    }

    pub fn delete_chat(&self, id: &str) -> io::Result<()> {
        if let Some(id) = valid_id(id) {
            match fs::remove_file(self.chats_dir().join(format!("{id}.json"))) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
                _ => {}
            }
        }
        Ok(())
    }

    /// Removes every kept chat; returns how many there were.
    pub fn delete_chats(&self) -> io::Result<usize> {
        let count = fs::read_dir(self.chats_dir())
            .map(|d| d.flatten().count())
            .unwrap_or(0);
        match fs::remove_dir_all(self.chats_dir()) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(count),
        }
    }
}

/// Chat ids are 16 lowercase hex characters (data model); anything else never reaches a path.
fn valid_id(id: &str) -> Option<&str> {
    (id.len() == 16
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
    .then_some(id)
}

/// Writes `bytes` readable by the user only (chats are the user's words).
fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600))?;
    }
    fs::rename(tmp, path)
}

/// `~/Library/LaunchAgents/dev.mrun.ui.plist` (the user's, next to `MRUN_HOME`'s parent in tests).
fn login_item(home: &Path) -> PathBuf {
    let base = std::env::var_os("HOME").map_or_else(|| home.join(".."), PathBuf::from);
    base.join("Library/LaunchAgents/dev.mrun.ui.plist")
}

fn set_login_item(home: &Path, on: bool) -> io::Result<()> {
    let path = login_item(home);
    if !on {
        return match fs::remove_file(&path) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        };
    }
    let exe = std::env::current_exe()?;
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.mrun.ui</string>
  <key>ProgramArguments</key>
  <array><string>{}</string><string>ui</string><string>--background</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"#,
        xml_escape(&exe.display().to_string())
    );
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, plist)
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The running window's record (`MRUN_HOME/ui.pid`: pid, port, token), readable by the user
/// only, so a second `mrun ui` can bring it forward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance {
    pub pid: u32,
    pub port: u16,
    pub token: String,
}

pub fn instance_file(home: &Path) -> PathBuf {
    home.join("ui.pid")
}

pub fn write_instance(home: &Path, instance: &Instance) -> io::Result<()> {
    fs::create_dir_all(home)?;
    write_private(
        &instance_file(home),
        format!("{} {} {}\n", instance.pid, instance.port, instance.token).as_bytes(),
    )
}

/// The recorded instance, if its process is alive and its port answers.
pub fn running_instance(home: &Path) -> Option<Instance> {
    let text = fs::read_to_string(instance_file(home)).ok()?;
    let mut parts = text.split_whitespace();
    let instance = Instance {
        pid: parts.next()?.parse().ok()?,
        port: parts.next()?.parse().ok()?,
        token: parts.next()?.to_string(),
    };
    let alive = std::process::Command::new("kill")
        .args(["-0", &instance.pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let answers = std::net::TcpStream::connect_timeout(
        &([127, 0, 0, 1], instance.port).into(),
        std::time::Duration::from_millis(300),
    )
    .is_ok();
    (alive && answers).then_some(instance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chats_are_written_only_when_kept() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path());
        let chat = serde_json::json!({"id": "0123456789abcdef", "title": "t", "updated": "2026"});
        assert!(!store.save_chat("0123456789abcdef", &chat).unwrap());
        assert!(store.chats().is_empty());
        store
            .update(&serde_json::json!({"keep_chats": true}))
            .unwrap();
        assert!(store.save_chat("0123456789abcdef", &chat).unwrap());
        assert_eq!(store.chats().len(), 1);
        assert_eq!(store.chat("0123456789abcdef").unwrap()["title"], "t");
        assert_eq!(store.delete_chats().unwrap(), 1);
        assert!(store.chats().is_empty());
    }

    #[test]
    fn ids_never_escape_the_chats_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path());
        store
            .update(&serde_json::json!({"keep_chats": true}))
            .unwrap();
        let chat = serde_json::json!({});
        assert!(store.save_chat("../../etc/passwd", &chat).is_err());
        assert!(store.chat("../ui").is_none());
    }

    #[test]
    fn added_cloud_models_are_kept() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path());
        store
            .update(&serde_json::json!({"cloud_models": [
                {"repo": "org/name", "weights_bytes": 30, "family": "qwen3"}
            ]}))
            .unwrap();
        let kept = Store::open(dir.path()).prefs().cloud_models;
        assert_eq!(kept.len(), 1);
        assert_eq!(
            (kept[0].repo.as_str(), kept[0].weights_bytes),
            ("org/name", 30)
        );
    }

    #[test]
    fn a_bad_prefs_file_is_set_aside() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("ui.json"), "{not json").unwrap();
        let store = Store::open(dir.path());
        assert_eq!(store.prefs().keep_chats, None);
        assert!(dir.path().join("ui.json.bad").is_file());
    }

    #[test]
    fn the_window_frame_is_kept_and_only_the_shell_writes_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path());
        let frame = Frame {
            x: 120.0,
            y: 80.0,
            width: 1100.0,
            height: 720.0,
        };
        store.set_window(frame).unwrap();
        assert_eq!(Store::open(dir.path()).prefs().window, Some(frame));
        let prefs = store
            .update(&serde_json::json!({
                "menu_bar": false,
                "window": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            }))
            .unwrap();
        assert!(!prefs.menu_bar);
        assert_eq!(prefs.window, Some(frame), "update leaves the frame alone");
        assert!(store.update(&serde_json::json!({"window": "bad"})).is_ok());
        assert_eq!(Store::open(dir.path()).prefs().window, Some(frame));
    }

    #[test]
    fn a_frame_is_restored_only_on_screen() {
        let screen = Frame {
            x: 0.0,
            y: 25.0,
            width: 1512.0,
            height: 920.0,
        };
        let at = |x, y, width, height| Frame {
            x,
            y,
            width,
            height,
        };
        let saved = at(100.0, 100.0, 1000.0, 700.0);
        assert_eq!(restore_frame(Some(saved), &[screen]), Some(saved));
        assert_eq!(restore_frame(None, &[screen]), None);
        // Small frames grow to the minimum size.
        assert_eq!(
            restore_frame(Some(at(10.0, 40.0, 300.0, 200.0)), &[screen]),
            Some(at(10.0, 40.0, MIN_WIDTH, MIN_HEIGHT))
        );
        // On an unplugged display, or just a sliver on this one: the default instead.
        assert_eq!(
            restore_frame(Some(at(3000.0, 100.0, 1000.0, 700.0)), &[screen]),
            None
        );
        assert_eq!(
            restore_frame(Some(at(1450.0, 100.0, 1000.0, 700.0)), &[screen]),
            None
        );
        assert_eq!(
            restore_frame(Some(at(f64::NAN, 0.0, 1000.0, 700.0)), &[screen]),
            None
        );
        assert_eq!(restore_frame(Some(saved), &[]), None);
        // A second display to the left (negative x) counts.
        let left = at(-1920.0, 0.0, 1920.0, 1080.0);
        let there = at(-1500.0, 200.0, 1000.0, 700.0);
        assert_eq!(restore_frame(Some(there), &[screen, left]), Some(there));
    }

    #[test]
    fn unknown_pref_fields_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path());
        let prefs = store
            .update(&serde_json::json!({"menu_bar": false, "evil": 1}))
            .unwrap();
        assert!(!prefs.menu_bar);
        let text = fs::read_to_string(dir.path().join("ui.json")).unwrap();
        assert!(!text.contains("evil"));
    }
}
