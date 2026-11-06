//! Finding pools on the local network (026 R5): a host announces `_mrun-pool._tcp` with its
//! pool's name, its own name and id, member count, planned model and protocol version, only
//! while it hosts; a Mac looking for pools browses for a moment and announces nothing.

use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const SERVICE: &str = "_mrun-pool._tcp.local.";
/// The pool protocol; a Mac with another refuses to join (E063).
pub const PROTOCOL: u32 = 1;

/// A pool as announced.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Announced {
    pub name: String,
    pub host: String,
    pub host_id: String,
    pub members: u32,
    pub model: Option<String>,
    pub protocol: u32,
    /// `host:port` of the host's pool port.
    pub addresses: Vec<String>,
}

impl Announced {
    pub fn compatible(&self) -> bool {
        self.protocol == PROTOCOL
    }

    pub fn txt(&self) -> HashMap<String, String> {
        let mut txt = HashMap::from([
            ("name".to_string(), self.name.clone()),
            ("host".to_string(), self.host.clone()),
            ("id".to_string(), self.host_id.clone()),
            ("members".to_string(), self.members.to_string()),
            ("v".to_string(), self.protocol.to_string()),
        ]);
        if let Some(model) = &self.model {
            txt.insert("model".into(), model.clone());
        }
        txt
    }

    pub fn from_txt(txt: &HashMap<String, String>, addresses: Vec<String>) -> Option<Self> {
        Some(Self {
            name: txt.get("name")?.clone(),
            host: txt.get("host")?.clone(),
            host_id: txt.get("id")?.clone(),
            members: txt.get("members")?.parse().ok()?,
            model: txt.get("model").cloned().filter(|m| !m.is_empty()),
            protocol: txt.get("v")?.parse().ok()?,
            addresses,
        })
    }
}

/// A running announcement; dropped, it is withdrawn.
pub struct Announcement {
    daemon: mdns_sd::ServiceDaemon,
    fullname: String,
}

impl std::fmt::Debug for Announcement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Announcement({})", self.fullname)
    }
}

impl Announcement {
    pub fn start(pool: &Announced, port: u16) -> Result<Self, String> {
        let daemon = mdns_sd::ServiceDaemon::new().map_err(|e| e.to_string())?;
        let instance = format!("{}-{}", pool.host_id, port);
        let host = format!("mrun-{}.local.", pool.host_id);
        let info = mdns_sd::ServiceInfo::new(SERVICE, &instance, &host, "", port, pool.txt())
            .map_err(|e| e.to_string())?
            .enable_addr_auto();
        let fullname = info.get_fullname().to_string();
        daemon.register(info).map_err(|e| e.to_string())?;
        Ok(Self { daemon, fullname })
    }
}

impl Drop for Announcement {
    fn drop(&mut self) {
        let _ = self.daemon.unregister(&self.fullname);
        let _ = self.daemon.shutdown();
    }
}

/// The pools announced on the local network, browsing for `wait`.
pub fn browse(wait: Duration) -> Result<Vec<Announced>, String> {
    let daemon = mdns_sd::ServiceDaemon::new().map_err(|e| e.to_string())?;
    let events = daemon.browse(SERVICE).map_err(|e| e.to_string())?;
    let deadline = Instant::now() + wait;
    let mut found: Vec<Announced> = Vec::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        let Ok(event) = events.recv_timeout(left) else {
            break;
        };
        if let mdns_sd::ServiceEvent::ServiceResolved(service) = event {
            let txt: HashMap<String, String> = service
                .txt_properties
                .iter()
                .map(|p| (p.key().to_string(), p.val_str().to_string()))
                .collect();
            let addresses = service
                .get_addresses_v4()
                .into_iter()
                .map(|ip| format!("{ip}:{}", service.get_port()))
                .collect();
            if let Some(pool) = Announced::from_txt(&txt, addresses)
                && !found.iter().any(|p| p.host_id == pool.host_id)
            {
                found.push(pool);
            }
        }
    }
    let _ = daemon.stop_browse(SERVICE);
    let _ = daemon.shutdown();
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> Announced {
        Announced {
            name: "Ben's pool".into(),
            host: "Ben's MacBook Pro".into(),
            host_id: "0123456789abcdef".into(),
            members: 3,
            model: Some("qwen3.8-235b".into()),
            protocol: PROTOCOL,
            addresses: vec!["10.0.0.2:8485".into()],
        }
    }

    #[test]
    fn announcements_carry_only_the_pool_and_round_trip() {
        let p = pool();
        let txt = p.txt();
        let mut keys: Vec<_> = txt.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["host", "id", "members", "model", "name", "v"]);
        assert_eq!(
            Announced::from_txt(&txt, p.addresses.clone()),
            Some(p.clone())
        );
        let mut old = txt.clone();
        old.insert("v".into(), "0".into());
        assert!(!Announced::from_txt(&old, vec![]).unwrap().compatible());
        old.remove("name");
        assert_eq!(Announced::from_txt(&old, vec![]), None);
    }

    /// Needs multicast on this machine's network: run by hand.
    #[test]
    #[ignore]
    fn a_pool_announced_here_is_found() {
        let p = pool();
        let _on = Announcement::start(&p, 48485).unwrap();
        let found = browse(Duration::from_secs(3)).unwrap();
        assert!(found.iter().any(|f| f.host_id == p.host_id), "{found:?}");
    }
}
