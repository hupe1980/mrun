//! This Mac's identity in pools and the Macs it is a member with (026 data model).
//!
//! The identity is a self-signed certificate made on first use; its `mac_id` is the first 16 hex
//! digits of the certificate's SHA-256. The key, the other Macs' pinned certificates and each
//! membership's secret live in the Keychain (`pool-identity`, `pool-pin-<mac_id>`,
//! `pool-secret-<mac_id>`); names and roles in `$MRUN_HOME/pool/members.toml`. Nothing secret is
//! written to the file, shown or logged.

use crate::keychain::Vault;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const IDENTITY: &str = "pool-identity";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    (text.len().is_multiple_of(2))
        .then(|| {
            (0..text.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
                .collect()
        })
        .flatten()
}

/// The id of the Mac holding `certificate` (DER).
pub fn mac_id(certificate: &[u8]) -> String {
    hex(&Sha256::digest(certificate))[..16].to_string()
}

/// This Mac's certificate and key (DER, PKCS #8).
#[derive(Clone)]
pub struct Identity {
    pub certificate: Vec<u8>,
    pub key: Vec<u8>,
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Identity({})", self.mac_id())
    }
}

impl Identity {
    /// The stored identity, or a new one, stored.
    pub fn load_or_create(vault: &Vault) -> Result<Self, String> {
        if let Some(stored) = vault.find(IDENTITY)
            && let Some((cert, key)) = stored.split_once('.')
            && let (Some(certificate), Some(key)) = (unhex(cert), unhex(key))
        {
            return Ok(Self { certificate, key });
        }
        let made = rcgen::generate_simple_self_signed(vec!["mrun-pool".to_string()])
            .map_err(|e| format!("cannot make this Mac's pool certificate: {e}"))?;
        let identity = Self {
            certificate: made.cert.der().to_vec(),
            key: made.signing_key.serialize_der(),
        };
        let value = format!("{}.{}", hex(&identity.certificate), hex(&identity.key));
        vault.store(IDENTITY, "mrun: this Mac's pool identity", &value)?;
        Ok(identity)
    }

    pub fn mac_id(&self) -> String {
        mac_id(&self.certificate)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// This Mac admitted the other into its pool.
    Host,
    /// This Mac joined the other's pool.
    Member,
}

/// Another Mac this one is paired with in a pool.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Membership {
    pub mac_id: String,
    pub name: String,
    pub chip: String,
    pub memory_bytes: u64,
    pub role: Role,
    /// RFC 3339.
    pub joined_at: String,
    /// Where the other Mac's pool port was last reached (`host:port`), to rejoin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
struct File {
    #[serde(default, rename = "member")]
    members: Vec<Membership>,
}

/// The memberships of this Mac, with their pins and secrets in the Keychain.
#[derive(Debug)]
pub struct Members {
    path: PathBuf,
    vault: Vault,
}

impl Members {
    pub fn at(home: &Path, vault: Vault) -> Self {
        Self {
            path: home.join("pool").join("members.toml"),
            vault,
        }
    }

    fn read(&self) -> File {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|text| toml::from_str(&text).ok())
            .unwrap_or_default()
    }

    fn write(&self, file: &File) -> Result<(), String> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let text = toml::to_string(file).map_err(|e| e.to_string())?;
        std::fs::write(&self.path, text).map_err(|e| e.to_string())
    }

    pub fn list(&self) -> Vec<Membership> {
        self.read().members
    }

    pub fn get(&self, mac_id: &str) -> Option<Membership> {
        self.list().into_iter().find(|m| m.mac_id == mac_id)
    }

    /// Records `membership` with the other Mac's certificate and the membership's secret.
    pub fn add(
        &self,
        membership: Membership,
        certificate: &[u8],
        secret: &[u8; 32],
    ) -> Result<(), String> {
        let id = &membership.mac_id;
        self.vault.store(
            &format!("pool-pin-{id}"),
            &format!(
                "mrun: pool certificate of {}",
                membership.name.replace('"', "")
            ),
            &hex(certificate),
        )?;
        self.vault.store(
            &format!("pool-secret-{id}"),
            &format!(
                "mrun: pool secret with {}",
                membership.name.replace('"', "")
            ),
            &hex(secret),
        )?;
        let mut file = self.read();
        file.members.retain(|m| m.mac_id != membership.mac_id);
        file.members.push(membership);
        self.write(&file)
    }

    /// The pinned certificate of `mac_id`.
    pub fn pin(&self, mac_id: &str) -> Option<Vec<u8>> {
        self.vault
            .find(&format!("pool-pin-{mac_id}"))
            .and_then(|h| unhex(&h))
    }

    /// The membership secret with `mac_id`.
    pub fn secret(&self, mac_id: &str) -> Option<[u8; 32]> {
        self.vault
            .find(&format!("pool-secret-{mac_id}"))
            .and_then(|h| unhex(&h))
            .and_then(|b| b.try_into().ok())
    }

    /// Forgets `mac_id`: the file entry, its pin and the secret. `false` when it was unknown.
    pub fn remove(&self, mac_id: &str) -> Result<bool, String> {
        let mut file = self.read();
        let before = file.members.len();
        file.members.retain(|m| m.mac_id != mac_id);
        let pin = self.vault.delete(&format!("pool-pin-{mac_id}"))?;
        let secret = self.vault.delete(&format!("pool-secret-{mac_id}"))?;
        self.write(&file)?;
        Ok(file.members.len() < before || pin || secret)
    }
}

/// 32 random bytes.
pub fn random_secret() -> std::io::Result<[u8; 32]> {
    use std::io::Read;
    let mut buf = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memberships_round_trip_without_secrets_in_the_file() {
        let home = tempfile::tempdir().unwrap();
        let vault = Vault::dir(home.path().join("keychain"));
        let members = Members::at(home.path(), vault.clone());
        let identity = Identity::load_or_create(&vault).unwrap();
        assert_eq!(identity.mac_id().len(), 16);
        let again = Identity::load_or_create(&vault).unwrap();
        assert_eq!(again.certificate, identity.certificate, "made once");
        assert!(!format!("{identity:?}").contains(&hex(&identity.key)[..8]));

        let other = rcgen::generate_simple_self_signed(vec!["x".into()]).unwrap();
        let cert = other.cert.der().to_vec();
        let id = mac_id(&cert);
        let secret = random_secret().unwrap();
        let membership = Membership {
            mac_id: id.clone(),
            name: "Anna's MacBook Pro".into(),
            chip: "Apple M4 Pro".into(),
            memory_bytes: 48 << 30,
            role: Role::Host,
            joined_at: "2026-11-05T10:00:00Z".into(),
            address: None,
        };
        members.add(membership.clone(), &cert, &secret).unwrap();
        assert_eq!(members.get(&id), Some(membership));
        assert_eq!(members.pin(&id), Some(cert));
        assert_eq!(members.secret(&id), Some(secret));
        let text = std::fs::read_to_string(home.path().join("pool/members.toml")).unwrap();
        assert!(
            !text.contains(&hex(&secret)),
            "no secret in the file: {text}"
        );
        assert!(members.remove(&id).unwrap());
        assert!(members.get(&id).is_none() && members.pin(&id).is_none());
        assert!(members.secret(&id).is_none());
        assert!(!members.remove(&id).unwrap());
    }

    #[test]
    fn hex_round_trips() {
        assert_eq!(unhex(&hex(&[0, 1, 254, 255])), Some(vec![0, 1, 254, 255]));
        assert_eq!(unhex("abc"), None);
        assert_eq!(unhex("zz"), None);
    }
}
