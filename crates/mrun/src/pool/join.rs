//! Joining a pool (026 US1, R5): numeric comparison, as Bluetooth pairing. The member says hello
//! over TLS with a fresh nonce, the host answers with its own; both show
//! `6 digits of SHA-256(host certificate ‖ member certificate ‖ host nonce ‖ member nonce)`. A
//! Mac in the middle has other certificates, so the codes differ. The host admits, the member
//! confirms, and the host hands over the membership secret on the confirmed connection. A join
//! not decided within 2 minutes expires; nothing is stored until both said yes.

use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

/// How long a join may wait for both answers.
pub const JOIN_WINDOW: Duration = Duration::from_secs(120);

/// The six digits both Macs show.
pub fn code(
    host_cert: &[u8],
    member_cert: &[u8],
    host_nonce: &[u8],
    member_nonce: &[u8],
) -> String {
    let digest = Sha256::new()
        .chain_update((host_cert.len() as u64).to_be_bytes())
        .chain_update(host_cert)
        .chain_update((member_cert.len() as u64).to_be_bytes())
        .chain_update(member_cert)
        .chain_update(host_nonce)
        .chain_update(member_nonce)
        .finalize();
    let n = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) % 1_000_000;
    format!("{n:06}")
}

/// `482 913`: the code as both Macs print it.
pub fn spaced(code: &str) -> String {
    match code.len() {
        6 => format!("{} {}", &code[..3], &code[3..]),
        _ => code.to_string(),
    }
}

/// What the joining Mac says about itself.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Hello {
    pub name: String,
    pub chip: String,
    pub memory_bytes: u64,
    /// `adapter` or `battery`.
    pub power: String,
    pub protocol: u32,
    /// Hex, 16 bytes.
    pub nonce: String,
    /// The joining Mac's pool port, where the host reaches it.
    pub port: u16,
}

/// The host's answer to a hello.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Welcome {
    pub pool: String,
    pub host: String,
    pub host_id: String,
    pub protocol: u32,
    pub nonce: String,
    /// The pool's model traffic is not encrypted (its host's choice; constitution IV).
    #[serde(default)]
    pub unencrypted: bool,
}

/// How a join ended without a membership.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Refused {
    Declined,
    Expired,
}

/// A join the host has not decided yet.
#[derive(Debug)]
pub struct Waiting {
    pub mac_id: String,
    pub certificate: Vec<u8>,
    pub hello: Hello,
    pub code: String,
    pub since: Instant,
    /// `Some(true)` admitted, `Some(false)` declined.
    pub decision: tokio::sync::watch::Sender<Option<bool>>,
}

impl Waiting {
    pub fn expired(&self, now: Instant) -> bool {
        now.duration_since(self.since) > JOIN_WINDOW
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "mac_id": self.mac_id,
            "name": self.hello.name,
            "chip": self.hello.chip,
            "memory_bytes": self.hello.memory_bytes,
            "power": self.hello.power,
            "code": self.code,
            "waiting_s": self.since.elapsed().as_secs(),
        })
    }
}

/// Waits until the host decides `waiting`'s join, or its window ends.
pub async fn decided(
    mut decision: tokio::sync::watch::Receiver<Option<bool>>,
    since: Instant,
) -> Result<(), Refused> {
    let left = JOIN_WINDOW.saturating_sub(since.elapsed());
    let wait = decision.wait_for(Option::is_some);
    match tokio::time::timeout(left, wait).await {
        Ok(Ok(answer)) if *answer == Some(true) => Ok(()),
        Ok(Ok(_)) | Ok(Err(_)) => Err(Refused::Declined),
        Err(_) => Err(Refused::Expired),
    }
}

/// 16 random bytes, hex.
pub fn nonce() -> std::io::Result<String> {
    crate::remote::random_hex(16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_sides_compute_the_same_code_and_a_substitute_changes_it() {
        let (host, member, mallory) = (
            b"host cert".as_slice(),
            b"member cert".as_slice(),
            b"m".as_slice(),
        );
        let c = code(host, member, b"n1", b"n2");
        assert_eq!(c.len(), 6);
        assert!(c.chars().all(|d| d.is_ascii_digit()));
        assert_eq!(code(host, member, b"n1", b"n2"), c);
        assert_ne!(
            code(mallory, member, b"n1", b"n2"),
            c,
            "a Mac in the middle shows another code"
        );
        assert_ne!(code(host, mallory, b"n1", b"n2"), c);
        assert_ne!(
            code(host, member, b"n1", b"n3"),
            c,
            "fresh nonces, fresh code"
        );
        // Length prefixes: moving bytes between the certificates changes the code.
        assert_ne!(code(b"ab", b"c", b"", b""), code(b"a", b"bc", b"", b""));
        assert_eq!(spaced("482913"), "482 913");
    }

    #[tokio::test]
    async fn a_join_is_admitted_declined_or_expires() {
        let (tx, rx) = tokio::sync::watch::channel(None);
        tx.send_replace(Some(true));
        assert_eq!(decided(rx, Instant::now()).await, Ok(()));
        let (tx, rx) = tokio::sync::watch::channel(None);
        tx.send_replace(Some(false));
        assert_eq!(decided(rx, Instant::now()).await, Err(Refused::Declined));
        let (_tx, rx) = tokio::sync::watch::channel(None);
        let long_ago = Instant::now() - JOIN_WINDOW - Duration::from_secs(1);
        assert_eq!(decided(rx, long_ago).await, Err(Refused::Expired));
    }
}
