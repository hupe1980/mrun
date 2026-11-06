//! Mac pool (026): one model as a pipeline across several Macs, colleagues' included. The host
//! holds the endpoint, tokenizer and prefix cache; every Mac runs a contiguous range of layers.
//!
//! This module holds the pool's lifecycle: hosting (announced, accepting joins on the pool port)
//! or being a member, the join itself ([`join`]), and leaving. Without a pool nothing listens
//! beyond loopback (FR-001). The pool port serves TLS between mrun servers only ([`tls`]); the
//! loopback routes under `/mrun/pool` are what `mrun pool` and the window call.

pub mod discovery;
#[cfg(feature = "engine")]
pub mod engine;
pub mod identity;
pub mod join;
pub mod link;
pub mod plan;
pub mod relay;
#[cfg(feature = "engine")]
pub mod serve;
pub mod tls;

use crate::keychain::Vault;
use axum::Json;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use identity::{Identity, Members, Membership, Role};
use join::{Hello, Refused, Waiting, Welcome};
use serde_json::{Value, json};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// The pool port's default.
pub const DEFAULT_PORT: u16 = 8485;

/// The header that names the member a relayed request came from, set by the host's pool port
/// from the member's certificate.
pub const MEMBER_HEADER: &str = "x-mrun-member";

/// Where this Mac's pool lives and how it presents itself.
#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub home: PathBuf,
    pub vault: Vault,
    /// The pool port (0: any, tests).
    pub port: u16,
    /// Where the pool port listens: every interface, or loopback (tests).
    pub bind: IpAddr,
    /// Announce on the local network (off in tests that run several Macs in one process).
    pub announce: bool,
    pub name: String,
    pub chip: String,
    pub memory_bytes: u64,
}

impl PoolConfig {
    /// This Mac, with its name, chip and memory.
    pub fn system(port: u16) -> Self {
        let machine = crate::bench::machine::machine();
        Self {
            home: crate::cli::mrun_home(),
            vault: Vault::system(),
            port,
            bind: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            announce: true,
            name: computer_name().unwrap_or_else(|| "This Mac".into()),
            chip: machine.chip,
            memory_bytes: machine.memory_bytes,
        }
    }
}

/// The Mac's name as Sharing settings show it.
fn computer_name() -> Option<String> {
    let out = std::process::Command::new("/usr/sbin/scutil")
        .args(["--get", "ComputerName"])
        .output()
        .ok()?;
    let name = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !name.is_empty()).then_some(name)
}

/// A Mac in this pool, as the host sees it.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Member {
    pub mac_id: String,
    pub name: String,
    pub chip: String,
    pub memory_bytes: u64,
    pub power: String,
    pub address: String,
    /// `joined`, a stage state (`loading`, `connecting`, `serving`, `failed`), or `lost`.
    pub state: String,
    /// Heartbeats missed in a row.
    #[serde(skip)]
    pub missed: u32,
}

#[derive(Debug)]
struct Hosting {
    name: String,
    /// The model the pool serves.
    model: Option<String>,
    /// Why the pool cannot serve now (a member lost), if so.
    degraded: Option<String>,
    /// The plan of the model it serves.
    plan: Option<Value>,
    /// While degraded because the rest cannot hold the model: whom the pool waits for, and the
    /// largest pulled model that fits the Macs answering (`{wait, smaller}`).
    recovery: Option<Value>,
    /// The lost Mac the pool waits for (its return serves the model again).
    waiting_for: Option<String>,
    /// Serving again on the Macs answering now (after a leave, a removal or a loss).
    replanning: bool,
    /// The model's ring traffic is not encrypted (the host chose so when starting it).
    unencrypted: bool,
    /// The server the pool's model serves on, kept to serve again when a Mac leaves.
    #[cfg_attr(not(feature = "engine"), allow(dead_code))]
    server: Option<crate::cli::serve::ServerState>,
    members: Vec<Member>,
    waiting: Vec<Waiting>,
    announcement: Option<discovery::Announcement>,
}

/// The pool port while this Mac hosts or is a member.
#[derive(Debug)]
struct Listening {
    addr: SocketAddr,
    server: tokio::task::JoinHandle<()>,
}

#[derive(Debug)]
struct Joined {
    pool: String,
    /// The pool's model traffic is not encrypted.
    unencrypted: bool,
    host: Membership,
    address: String,
}

#[derive(Debug)]
struct Joining {
    welcome: Welcome,
    address: String,
    certificate: Vec<u8>,
    client: reqwest::Client,
    code: String,
}

#[derive(Debug, Default)]
struct Inner {
    listening: Option<Listening>,
    hosting: Option<Hosting>,
    joined: Option<Joined>,
    joining: Option<Joining>,
}

/// A pool error for the loopback routes and the CLI: an E06x code and a sentence.
#[derive(Debug)]
pub struct PoolError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl PoolError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn usage(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, "E061", message)
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "E001", message)
    }
}

impl IntoResponse for PoolError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({"error": {"code": self.code, "message": self.message}})),
        )
            .into_response()
    }
}

type Result<T> = std::result::Result<T, PoolError>;

/// This Mac's side of every pool.
#[derive(Debug)]
pub struct Pool {
    config: PoolConfig,
    /// The ring while the pool serves (closed when a member is lost, so waiting transfers fail).
    #[cfg(feature = "engine")]
    ring: Arc<Mutex<Option<mrun_engine::Pipe>>>,
    /// Why the ring was closed (the lost member), for the requests that fail on it.
    #[cfg(feature = "engine")]
    ring_reason: engine::RingReason,
    members: Members,
    /// This Mac's API server: its loopback port and request history.
    server: Mutex<Option<(u16, Arc<crate::api::status::Tracker>)>>,
    /// Replaces this Mac's served model (a member lending its memory drops its own).
    ready: Mutex<Option<Arc<tokio::sync::watch::Sender<crate::cli::serve::Readiness>>>>,
    /// This Mac's server settings (its own model comes back with them).
    args: Mutex<Option<Arc<crate::cli::serve::ServeArgs>>>,
    /// The model this Mac served before the pool took its memory, to load again after.
    own: Mutex<Option<String>>,
    /// This Mac's part while it lends its memory to a pool.
    #[cfg(feature = "engine")]
    stage: Arc<serve::Stage>,
    identity: Mutex<Option<Identity>>,
    inner: Mutex<Inner>,
    /// This pool itself, for work it starts in the background.
    #[cfg_attr(not(feature = "engine"), allow(dead_code))]
    this: std::sync::Weak<Pool>,
}

impl Pool {
    pub fn new(config: PoolConfig) -> Arc<Self> {
        Arc::new_cyclic(|this| Self {
            this: this.clone(),
            #[cfg(feature = "engine")]
            ring: Arc::default(),
            #[cfg(feature = "engine")]
            ring_reason: Arc::default(),
            members: Members::at(&config.home, config.vault.clone()),
            server: Mutex::new(None),
            ready: Mutex::new(None),
            args: Mutex::new(None),
            own: Mutex::new(None),
            #[cfg(feature = "engine")]
            stage: Arc::default(),
            config,
            identity: Mutex::new(None),
            inner: Mutex::default(),
        })
    }

    #[cfg_attr(not(feature = "engine"), allow(dead_code))]
    fn me(&self) -> Option<Arc<Self>> {
        self.this.upgrade()
    }

    /// Closes the ring the pool serves on (every Mac's part stops; requests on it fail with
    /// `reason`).
    #[cfg(feature = "engine")]
    fn close_ring(&self, reason: &str) {
        if let Some(ring) = self
            .ring
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            *self
                .ring_reason
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(reason.to_string());
            ring.close();
        }
    }

    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// This Mac's identity, made on first use.
    pub fn identity(&self) -> Result<Identity> {
        let mut slot = self.identity.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(identity) = slot.as_ref() {
            return Ok(identity.clone());
        }
        let identity = Identity::load_or_create(&self.config.vault).map_err(PoolError::internal)?;
        *slot = Some(identity.clone());
        Ok(identity)
    }

    pub fn memberships(&self) -> &Members {
        &self.members
    }

    /// Records this Mac's API server, which relayed member requests go to.
    pub fn attach_server(
        &self,
        port: u16,
        tracker: Arc<crate::api::status::Tracker>,
        ready: Arc<tokio::sync::watch::Sender<crate::cli::serve::Readiness>>,
        args: Arc<crate::cli::serve::ServeArgs>,
    ) {
        *self.server.lock().unwrap_or_else(PoisonError::into_inner) = Some((port, tracker));
        *self.ready.lock().unwrap_or_else(PoisonError::into_inner) = Some(ready);
        *self.args.lock().unwrap_or_else(PoisonError::into_inner) = Some(args);
    }

    /// Whether this Mac lends its memory to a pool's model now (it stays awake meanwhile).
    pub fn lending(&self) -> bool {
        #[cfg(feature = "engine")]
        return matches!(self.stage.get().state, "loading" | "connecting" | "serving");
        #[cfg(not(feature = "engine"))]
        false
    }

    /// Drops this Mac's own model (its memory goes to the pool) and remembers it, to load it
    /// again when the pool is over; returns the model's name.
    pub fn drop_own_model(&self) -> Option<String> {
        let ready = self
            .ready
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()?;
        let before = ready.send_replace(crate::cli::serve::Readiness::Loading);
        let crate::cli::serve::Readiness::Ready(loaded) = before else {
            return None;
        };
        let mut own = self.own.lock().unwrap_or_else(PoisonError::into_inner);
        // A pool's own model (re-planned) is not this Mac's.
        if !self.serves_pool() {
            own.get_or_insert_with(|| loaded.model_id.clone());
        }
        Some(loaded.model_id.clone())
    }

    /// Whether this Mac's server serves the pool's model now (it hosts one).
    fn serves_pool(&self) -> bool {
        self.inner()
            .hosting
            .as_ref()
            .is_some_and(|h| h.model.is_some())
    }

    /// The pool is over for this Mac: its part stops and the model it served before comes back
    /// (or the server says it has none).
    pub fn restore_own_model(&self) {
        #[cfg(feature = "engine")]
        self.stage.stop();
        let ready = self
            .ready
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let args = self
            .args
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let own = self
            .own
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let (Some(ready), Some(args), Some(name)) = (ready, args, own) else {
            return;
        };
        ready.send_replace(crate::cli::serve::Readiness::Loading);
        std::thread::spawn(move || {
            let state = match crate::cli::serve::load_named(&args, &name) {
                Ok(loaded) => {
                    loaded.handle.queue().set_capacity(args.queue);
                    crate::cli::serve::Readiness::Ready(Arc::new(loaded))
                }
                Err(error) => crate::cli::serve::Readiness::Failed(error),
            };
            ready.send_replace(state);
        });
    }

    /// Whether this server relays to a remote GPU (013): a pool and remote mode exclude each
    /// other.
    fn relaying(&self) -> bool {
        self.args
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .is_some_and(|a| a.relay.is_some())
    }

    fn server(&self) -> Option<(u16, Arc<crate::api::status::Tracker>)> {
        self.server
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Where this member's agents' requests go now: the host, while this Mac lends its memory
    /// to the pool's model.
    pub fn relay_target(&self) -> Option<(reqwest::Client, String)> {
        let address = self
            .inner()
            .joined
            .as_ref()
            .map(|j| (j.host.mac_id.clone(), j.address.clone()))?;
        #[cfg(feature = "engine")]
        if self.stage.get().state != "serving" {
            return None;
        }
        let client = self.client_for(&address.0).ok()?;
        Some((client, address.1))
    }

    /// The pool port's address while it listens.
    pub fn pool_addr(&self) -> Option<SocketAddr> {
        self.inner().listening.as_ref().map(|l| l.addr)
    }

    /// Opens the pool port (TLS, client certificates asked for) unless it is open.
    async fn listen(self: &Arc<Self>) -> Result<SocketAddr> {
        if let Some(addr) = self.pool_addr() {
            return Ok(addr);
        }
        let identity = self.identity()?;
        let config = tls::server_config(&identity).map_err(PoolError::internal)?;
        let tcp = tokio::net::TcpListener::bind((self.config.bind, self.config.port))
            .await
            .map_err(|e| {
                PoolError::internal(format!(
                    "cannot listen on the pool port {}: {e}",
                    self.config.port
                ))
            })?;
        let addr = tcp
            .local_addr()
            .map_err(|e| PoolError::internal(e.to_string()))?;
        let app = self
            .clone()
            .router()
            .into_make_service_with_connect_info::<tls::Peer>();
        let listener = tls::Listener::new(tcp, config);
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        self.inner().listening = Some(Listening { addr, server });
        Ok(addr)
    }

    /// Closes the pool port (FR-001: nothing beyond loopback without a pool).
    fn close(inner: &mut Inner) {
        if let Some(listening) = inner.listening.take() {
            listening.server.abort();
        }
    }

    /// The members answering now (hosting), in stage order after the host.
    pub fn live_members(&self) -> Vec<Member> {
        self.inner()
            .hosting
            .as_ref()
            .map(|h| {
                h.members
                    .iter()
                    .filter(|m| m.state != "lost")
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `GET /mrun/pool`.
    pub fn status(&self) -> Value {
        let me = self.identity().ok().map(|i| i.mac_id());
        let mut inner = self.inner();
        let now = Instant::now();
        if let Some(h) = inner.hosting.as_mut() {
            h.waiting.retain(|w| !w.expired(now));
        }
        let address = inner.listening.as_ref().map(|l| l.addr.to_string());
        let pool = if let Some(h) = &inner.hosting {
            json!({
                "role": "host",
                "name": h.name,
                "model": h.model,
                "degraded": h.degraded,
                "recovery": h.recovery,
                "replanning": h.replanning,
                "unencrypted": h.unencrypted,
                "plan": h.plan,
                "address": address,
                "members": h.members,
                "waiting": h.waiting.iter().map(Waiting::json).collect::<Vec<_>>(),
            })
        } else if let Some(j) = &inner.joined {
            json!({
                "role": "member",
                "name": j.pool,
                "host": {"mac_id": j.host.mac_id, "name": j.host.name},
                "address": j.address,
                "unencrypted": j.unencrypted,
            })
        } else {
            Value::Null
        };
        json!({
            "this": {"mac_id": me, "name": self.config.name, "chip": self.config.chip,
                "memory_bytes": self.config.memory_bytes},
            "pool": pool,
            "joining": inner.joining.as_ref().map(|j| json!({
                "pool": j.welcome.pool, "host": j.welcome.host, "code": j.code,
                "unencrypted": j.welcome.unencrypted})),
            // A member out of its pool (lost, or the host restarted) rejoins it without a code.
            "rejoin": (inner.hosting.is_none() && inner.joined.is_none())
                .then(|| self.last_host().map(|h| json!({"host": h.name, "mac_id": h.mac_id})))
                .flatten(),
            "memberships": self.members.list(),
        })
    }

    /// Starts hosting: the pool port listens and the pool is announced.
    pub async fn start(self: &Arc<Self>, name: Option<String>, unencrypted: bool) -> Result<Value> {
        {
            let inner = self.inner();
            if inner.hosting.is_some() {
                return Err(PoolError::usage("this Mac already hosts a pool"));
            }
            if inner.joined.is_some() {
                return Err(PoolError::usage(
                    "this Mac is in another pool; leave it first (mrun pool leave)",
                ));
            }
        }
        self.refuse_relaying()?;
        let identity = self.identity()?;
        let addr = self.listen().await?;
        let name = name.unwrap_or_else(|| format!("{}'s pool", short_name(&self.config.name)));
        let announced = discovery::Announced {
            name: name.clone(),
            host: self.config.name.clone(),
            host_id: identity.mac_id(),
            members: 1,
            model: None,
            protocol: discovery::PROTOCOL,
            addresses: Vec::new(),
        };
        let announcement = if self.config.announce {
            Some(
                discovery::Announcement::start(&announced, addr.port())
                    .map_err(PoolError::internal)?,
            )
        } else {
            None
        };
        tokio::spawn(health(self.clone()));
        self.inner().hosting = Some(Hosting {
            name,
            model: None,
            degraded: None,
            recovery: None,
            waiting_for: None,
            replanning: false,
            unencrypted,
            server: None,
            plan: None,
            members: Vec::new(),
            waiting: Vec::new(),
            announcement,
        });
        Ok(self.status())
    }

    /// Asks the pool at `address` (`host:port`) to join; returns the code to compare.
    pub async fn join(self: &Arc<Self>, address: &str) -> Result<Value> {
        if self.inner().hosting.is_some() {
            return Err(PoolError::usage(
                "this Mac hosts a pool; end it first (mrun pool leave)",
            ));
        }
        self.refuse_relaying()?;
        let identity = self.identity()?;
        let port = self.listen().await?.port();
        let client = tls::client(&identity, None).map_err(PoolError::internal)?;
        let nonce = join::nonce().map_err(|e| PoolError::internal(e.to_string()))?;
        let hello = Hello {
            name: self.config.name.clone(),
            chip: self.config.chip.clone(),
            memory_bytes: self.config.memory_bytes,
            power: crate::runtime::power::power_source().into(),
            protocol: discovery::PROTOCOL,
            nonce: nonce.clone(),
            port,
        };
        let unreachable = |e: reqwest::Error| {
            PoolError::new(
                StatusCode::BAD_GATEWAY,
                "E060",
                format!("cannot reach the pool at {address}: {e}"),
            )
        };
        let response = client
            .post(format!("https://{address}/pool/v1/hello"))
            .json(&hello)
            .send()
            .await
            .map_err(unreachable)?;
        let certificate = tls::server_certificate(&response)
            .ok_or_else(|| PoolError::internal("the pool sent no certificate"))?;
        if response.status() == StatusCode::CONFLICT {
            return Err(refusal(response).await);
        }
        if !response.status().is_success() {
            return Err(refusal(response).await);
        }
        let welcome: Welcome = response.json().await.map_err(unreachable)?;
        let code = join::code(
            &certificate,
            &identity.certificate,
            welcome.nonce.as_bytes(),
            nonce.as_bytes(),
        );
        let answer = json!({"pool": welcome.pool, "host": welcome.host, "code": code,
            "unencrypted": welcome.unencrypted});
        self.inner().joining = Some(Joining {
            welcome,
            address: address.to_string(),
            certificate,
            client,
            code,
        });
        Ok(answer)
    }

    /// The member's answer to its join: `yes` waits until the host decided. With `code`, the
    /// answer counts only when it is the code this Mac shows (the person read the host's).
    pub async fn confirm(&self, yes: bool, code: Option<&str>) -> Result<Value> {
        let Some(joining) = self.inner().joining.take() else {
            return Err(PoolError::usage(
                "no join is waiting for an answer (mrun pool join)",
            ));
        };
        if let Some(code) = code
            && code.replace(' ', "") != joining.code
        {
            let mut inner = self.inner();
            if inner.joined.is_none() && inner.hosting.is_none() {
                Self::close(&mut inner);
            }
            return Err(PoolError::usage(
                "the codes differ: this is not the Mac you meant to join, or someone is in between",
            ));
        }
        let close = |pool: &Self| {
            let mut inner = pool.inner();
            if inner.joined.is_none() && inner.hosting.is_none() {
                Self::close(&mut inner);
            }
        };
        if !yes {
            close(self);
            let _ = joining
                .client
                .post(format!("https://{}/pool/v1/confirm", joining.address))
                .json(&json!({"yes": false}))
                .send()
                .await;
            return Ok(json!({"joined": false, "reason": "declined"}));
        }
        let response = joining
            .client
            .post(format!("https://{}/pool/v1/confirm", joining.address))
            .json(&json!({"yes": true}))
            .timeout(join::JOIN_WINDOW + Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| PoolError::new(StatusCode::BAD_GATEWAY, "E060", e.to_string()))?;
        if !response.status().is_success() {
            close(self);
            return Err(refusal(response).await);
        }
        let answer: Value = response
            .json()
            .await
            .map_err(|e| PoolError::internal(e.to_string()))?;
        let secret = answer["secret"]
            .as_str()
            .and_then(crate::remote::from_hex)
            .and_then(|b| <[u8; 32]>::try_from(b).ok())
            .ok_or_else(|| PoolError::internal("the host sent no membership secret"))?;
        let host = Membership {
            mac_id: joining.welcome.host_id.clone(),
            name: joining.welcome.host.clone(),
            chip: String::new(),
            memory_bytes: 0,
            role: Role::Member,
            joined_at: crate::api::record::rfc3339_now(),
            address: Some(joining.address.clone()),
        };
        if identity::mac_id(&joining.certificate) != host.mac_id {
            return Err(PoolError::usage(
                "the host's certificate does not match its id",
            ));
        }
        self.members
            .add(host.clone(), &joining.certificate, &secret)
            .map_err(PoolError::internal)?;
        self.inner().joined = Some(Joined {
            unencrypted: joining.welcome.unencrypted,
            pool: joining.welcome.pool.clone(),
            host,
            address: joining.address,
        });
        Ok(json!({"joined": true, "pool": joining.welcome.pool}))
    }

    /// The host's decision on a waiting join.
    pub fn admit(&self, mac_id: &str, yes: bool) -> Result<Value> {
        let inner = self.inner();
        let waiting = inner
            .hosting
            .as_ref()
            .and_then(|h| h.waiting.iter().find(|w| w.mac_id == mac_id))
            .ok_or_else(|| PoolError::usage("no Mac with that id is waiting to join"))?;
        waiting.decision.send_replace(Some(yes));
        Ok(json!({"admitted": yes, "mac_id": mac_id}))
    }

    /// A pinned client towards `mac_id`.
    fn client_for(&self, mac_id: &str) -> Result<reqwest::Client> {
        let pin = self
            .members
            .pin(mac_id)
            .ok_or_else(|| PoolError::usage("this Mac is not paired with that Mac"))?;
        tls::client(&self.identity()?, Some(pin)).map_err(PoolError::internal)
    }

    /// Leaves: a member leaves its pool and forgets it; the host ends its pool for everyone
    /// (memberships stay, so members can rejoin a restarted pool).
    pub async fn leave(&self) -> Result<Value> {
        let (hosting, joined) = {
            let mut inner = self.inner();
            inner.joining = None;
            Self::close(&mut inner);
            (inner.hosting.take(), inner.joined.take())
        };
        if let Some(joined) = joined {
            if let Ok(client) = self.client_for(&joined.host.mac_id) {
                let _ = client
                    .post(format!("https://{}/pool/v1/bye", joined.address))
                    .timeout(Duration::from_secs(3))
                    .send()
                    .await;
            }
            self.members
                .remove(&joined.host.mac_id)
                .map_err(PoolError::internal)?;
            self.restore_own_model();
            return Ok(json!({"left": joined.pool}));
        }
        if let Some(hosting) = hosting {
            for member in &hosting.members {
                if let Ok(client) = self.client_for(&member.mac_id) {
                    let _ = client
                        .post(format!("https://{}/pool/v1/ended", member.address))
                        .timeout(Duration::from_secs(3))
                        .send()
                        .await;
                }
            }
            drop(hosting.announcement);
            #[cfg(feature = "engine")]
            self.close_ring("the pool ended");
            if hosting.model.is_some() {
                self.restore_own_model();
            }
            return Ok(json!({"ended": hosting.name}));
        }
        Err(PoolError::usage("this Mac is in no pool"))
    }

    /// The host removes a member: both Macs forget the membership.
    pub async fn remove(&self, mac_id: &str) -> Result<Value> {
        let member = {
            let mut inner = self.inner();
            let hosting = inner
                .hosting
                .as_mut()
                .ok_or_else(|| PoolError::usage("this Mac hosts no pool"))?;
            let at = hosting.members.iter().position(|m| m.mac_id == mac_id);
            at.map(|i| hosting.members.remove(i))
        };
        if let Some(member) = &member
            && let Ok(client) = self.client_for(mac_id)
        {
            let _ = client
                .post(format!("https://{}/pool/v1/bye", member.address))
                .timeout(Duration::from_secs(3))
                .send()
                .await;
        }
        let known = self.members.remove(mac_id).map_err(PoolError::internal)?;
        if member.is_none() && !known {
            return Err(PoolError::usage("no member with that id"));
        }
        if member.is_some() {
            self.serve_again();
        }
        Ok(json!({"removed": mac_id}))
    }

    /// Whether this Mac hosts an unencrypted pool.
    #[cfg_attr(not(feature = "engine"), allow(dead_code))]
    fn unencrypted(&self) -> bool {
        self.inner().hosting.as_ref().is_some_and(|h| h.unencrypted)
    }

    /// A member's place in its pool's queue, asked of the host (counts only).
    pub async fn place(&self) -> Option<Value> {
        let (host, address) = {
            let inner = self.inner();
            let joined = inner.joined.as_ref()?;
            (joined.host.mac_id.clone(), joined.address.clone())
        };
        let client = self.client_for(&host).ok()?;
        client
            .get(format!("https://{address}/pool/v1/queue"))
            .timeout(Duration::from_secs(1))
            .send()
            .await
            .ok()?
            .json()
            .await
            .ok()
    }

    /// Refuses a pool while this server relays to a remote GPU.
    fn refuse_relaying(&self) -> Result<()> {
        if self.relaying() {
            return Err(PoolError::usage(
                "this Mac relays to a remote GPU (mrun cloud); stop that first (mrun cloud down)",
            ));
        }
        Ok(())
    }

    /// The host this Mac joined last (a membership it keeps to rejoin).
    fn last_host(&self) -> Option<Membership> {
        self.members
            .list()
            .into_iter()
            .filter(|m| m.role == Role::Member && m.address.is_some())
            .max_by(|a, b| a.joined_at.cmp(&b.joined_at))
    }

    /// Rejoins the pool this Mac was in (after a loss, or its host's restart): the membership
    /// is the credential, no new code.
    pub async fn rejoin(self: &Arc<Self>) -> Result<Value> {
        {
            let inner = self.inner();
            if inner.hosting.is_some() || inner.joined.is_some() {
                return Err(PoolError::usage("this Mac is in a pool already"));
            }
        }
        let host = self.last_host().ok_or_else(|| {
            PoolError::usage("this Mac was in no pool to rejoin (mrun pool join <pool>)")
        })?;
        let address = host.address.clone().unwrap_or_default();
        let client = self.client_for(&host.mac_id)?;
        let port = self.listen().await?.port();
        let hello = Hello {
            name: self.config.name.clone(),
            chip: self.config.chip.clone(),
            memory_bytes: self.config.memory_bytes,
            power: crate::runtime::power::power_source().into(),
            protocol: discovery::PROTOCOL,
            nonce: String::new(),
            port,
        };
        let answer = client
            .post(format!("https://{address}/pool/v1/rejoin"))
            .json(&hello)
            .timeout(Duration::from_secs(10))
            .send()
            .await;
        let fail = |pool: &Self, error: PoolError| {
            let mut inner = pool.inner();
            if inner.joined.is_none() && inner.hosting.is_none() {
                Self::close(&mut inner);
            }
            Err(error)
        };
        let response = match answer {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => return fail(self, refusal(r).await),
            Err(e) => {
                return fail(
                    self,
                    PoolError::new(
                        StatusCode::BAD_GATEWAY,
                        "E060",
                        format!("{} does not answer: {e}", host.name),
                    ),
                );
            }
        };
        let welcome: Value = response.json().await.unwrap_or_default();
        let pool = welcome["pool"].as_str().unwrap_or_default().to_string();
        self.inner().joined = Some(Joined {
            unencrypted: welcome["unencrypted"] == true,
            pool: pool.clone(),
            host,
            address,
        });
        Ok(json!({"joined": true, "pool": pool}))
    }

    /// Serves the pool's model again on the Macs answering now, in the background (a Mac left,
    /// was removed or was lost while the pool served).
    fn serve_again(&self) {
        #[cfg(feature = "engine")]
        if let Some(pool) = self.me() {
            tokio::spawn(async move {
                let _ = pool.replan().await;
            });
        }
    }

    /// The pool port's routes.
    fn router(self: Arc<Self>) -> axum::Router {
        axum::Router::new()
            .route("/pool/v1/hello", post(hello))
            .route("/pool/v1/confirm", post(confirm))
            .route("/pool/v1/bye", post(bye))
            .route("/pool/v1/rejoin", post(rejoin))
            .route("/pool/v1/ended", post(ended))
            .route("/pool/v1/heartbeat", get(heartbeat))
            .route("/pool/v1/echo", post(echo))
            .merge(Self::stage_routes())
            .route("/v1/{*rest}", axum::routing::any(relay_in))
            .route("/pool/v1/sessions", get(member_sessions))
            .route("/pool/v1/queue", get(member_queue))
            .route("/pool/v1/excerpt/{id}", get(member_excerpt))
            .layer(axum::extract::DefaultBodyLimit::max(8 << 20))
            .layer(middleware::from_fn_with_state(self.clone(), members_only))
            .with_state(self)
    }

    #[cfg(feature = "engine")]
    fn stage_routes() -> axum::Router<Arc<Self>> {
        axum::Router::new()
            .route("/pool/v1/spec", get(serve::spec))
            .route("/pool/v1/load", post(serve::load))
            .route("/pool/v1/stage", get(serve::stage))
    }

    #[cfg(not(feature = "engine"))]
    fn stage_routes() -> axum::Router<Arc<Self>> {
        axum::Router::new()
    }

    /// Whether `peer` is a member this Mac pinned.
    fn pinned(&self, peer: &tls::Peer) -> bool {
        self.members
            .pin(&peer.mac_id)
            .is_some_and(|pin| pin == peer.certificate)
    }

    /// The pool port address another Mac reaches this one at.
    fn member_address(peer: &tls::Peer, port: u16) -> String {
        SocketAddr::new(peer.addr.ip(), port).to_string()
    }
}

/// "Ben" from "Ben's MacBook Pro"; the whole name otherwise.
fn short_name(name: &str) -> &str {
    name.split_once("'s ")
        .or_else(|| name.split_once("’s "))
        .map_or(name, |(owner, _)| owner)
}

async fn refusal(response: reqwest::Response) -> PoolError {
    let status = response.status();
    let body: Value = response.json().await.unwrap_or_default();
    PoolError::new(
        status,
        match body["error"]["code"].as_str() {
            Some("E063") => "E063",
            Some("E060") => "E060",
            _ => "E061",
        },
        body["error"]["message"]
            .as_str()
            .unwrap_or("the pool refused the join")
            .to_string(),
    )
}

/// Every route but joining needs a pinned member, checked before any body is read.
async fn members_only(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
    request: Request,
    next: Next,
) -> Response {
    let joining = matches!(request.uri().path(), "/pool/v1/hello" | "/pool/v1/confirm");
    if !joining && !pool.pinned(&peer) {
        return PoolError::new(StatusCode::FORBIDDEN, "E061", "this Mac is not a member")
            .into_response();
    }
    next.run(request).await
}

/// A Mac asks to join.
async fn hello(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
    Json(hello): Json<Hello>,
) -> Result<Json<Welcome>> {
    if hello.protocol != discovery::PROTOCOL {
        return Err(PoolError::new(
            StatusCode::CONFLICT,
            "E063",
            format!(
                "{} runs another mrun version; update mrun on both Macs",
                hello.name
            ),
        ));
    }
    let identity = pool.identity()?;
    let nonce = join::nonce().map_err(|e| PoolError::internal(e.to_string()))?;
    let code = join::code(
        &identity.certificate,
        &peer.certificate,
        nonce.as_bytes(),
        hello.nonce.as_bytes(),
    );
    let mut inner = pool.inner();
    let hosting = inner
        .hosting
        .as_mut()
        .ok_or_else(|| PoolError::new(StatusCode::NOT_FOUND, "E061", "this Mac hosts no pool"))?;
    hosting.waiting.retain(|w| w.mac_id != peer.mac_id);
    hosting.waiting.push(Waiting {
        mac_id: peer.mac_id.clone(),
        certificate: peer.certificate.clone(),
        hello,
        code,
        since: Instant::now(),
        decision: tokio::sync::watch::channel(None).0,
    });
    Ok(Json(Welcome {
        pool: hosting.name.clone(),
        host: pool.config.name.clone(),
        host_id: identity.mac_id(),
        protocol: discovery::PROTOCOL,
        nonce,
        unencrypted: hosting.unencrypted,
    }))
}

/// The member's answer; with yes, waits for the host's and hands over the secret.
async fn confirm(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
    Json(answer): Json<Value>,
) -> Result<Json<Value>> {
    let (decision, since) = {
        let mut inner = pool.inner();
        let hosting = inner.hosting.as_mut().ok_or_else(|| {
            PoolError::new(StatusCode::NOT_FOUND, "E061", "this Mac hosts no pool")
        })?;
        let at = hosting
            .waiting
            .iter()
            .position(|w| w.mac_id == peer.mac_id)
            .ok_or_else(|| PoolError::usage("no join from this Mac is waiting"))?;
        if answer["yes"] != true {
            hosting.waiting.remove(at);
            return Ok(Json(json!({"joined": false})));
        }
        let w = &hosting.waiting[at];
        (w.decision.subscribe(), w.since)
    };
    let decided = join::decided(decision, since).await;
    let waiting = {
        let mut inner = pool.inner();
        let hosting = inner.hosting.as_mut();
        hosting.and_then(|h| {
            let at = h.waiting.iter().position(|w| w.mac_id == peer.mac_id)?;
            Some(h.waiting.remove(at))
        })
    };
    let Some(waiting) = waiting else {
        return Err(PoolError::usage("the pool ended"));
    };
    if let Err(refused) = decided {
        let message = match refused {
            Refused::Declined => "the host declined the join",
            Refused::Expired => "nobody answered within 2 minutes",
        };
        return Err(PoolError::new(StatusCode::FORBIDDEN, "E061", message));
    }
    let secret = identity::random_secret().map_err(|e| PoolError::internal(e.to_string()))?;
    let address = Pool::member_address(&peer, waiting.hello.port);
    let membership = Membership {
        mac_id: peer.mac_id.clone(),
        name: waiting.hello.name.clone(),
        chip: waiting.hello.chip.clone(),
        memory_bytes: waiting.hello.memory_bytes,
        role: Role::Host,
        joined_at: crate::api::record::rfc3339_now(),
        address: None,
    };
    pool.members
        .add(membership, &peer.certificate, &secret)
        .map_err(PoolError::internal)?;
    if let Some(hosting) = pool.inner().hosting.as_mut() {
        hosting.members.retain(|m| m.mac_id != peer.mac_id);
        hosting.members.push(Member {
            mac_id: peer.mac_id.clone(),
            name: waiting.hello.name,
            chip: waiting.hello.chip,
            memory_bytes: waiting.hello.memory_bytes,
            power: waiting.hello.power,
            address,
            state: "joined".into(),
            missed: 0,
        });
    }
    let hex: String = secret.iter().map(|b| format!("{b:02x}")).collect();
    Ok(Json(json!({"joined": true, "secret": hex})))
}

/// The other Mac leaves (a member) or removes this one (the host): forget the membership.
async fn bye(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
) -> Result<Json<Value>> {
    let (left, removed) = {
        let mut inner = pool.inner();
        let left = inner.hosting.as_mut().is_some_and(|h| {
            let before = h.members.len();
            h.members.retain(|m| m.mac_id != peer.mac_id);
            h.members.len() < before && h.model.is_some()
        });
        let removed = inner
            .joined
            .as_ref()
            .is_some_and(|j| j.host.mac_id == peer.mac_id);
        if removed {
            inner.joined = None;
            Pool::close(&mut inner);
        }
        (left, removed)
    };
    pool.members
        .remove(&peer.mac_id)
        .map_err(PoolError::internal)?;
    if left {
        pool.serve_again();
    }
    if removed {
        pool.restore_own_model();
    }
    Ok(Json(json!({"bye": true})))
}

/// The host ended its pool: this member is out of it, and keeps the membership to rejoin.
async fn ended(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
) -> Json<Value> {
    let ended = {
        let mut inner = pool.inner();
        let ended = inner
            .joined
            .as_ref()
            .is_some_and(|j| j.host.mac_id == peer.mac_id);
        if ended {
            inner.joined = None;
            Pool::close(&mut inner);
        }
        ended
    };
    if ended {
        pool.restore_own_model();
    }
    Json(json!({"ended": true}))
}

/// A member is back (after a loss, or this host's restart): pinned, so no code.
async fn rejoin(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
    Json(hello): Json<Hello>,
) -> Result<Json<Value>> {
    if hello.protocol != discovery::PROTOCOL {
        return Err(PoolError::new(
            StatusCode::CONFLICT,
            "E063",
            format!(
                "{} runs another mrun version; update mrun on both Macs",
                hello.name
            ),
        ));
    }
    if pool
        .members
        .get(&peer.mac_id)
        .is_none_or(|m| m.role != Role::Host)
    {
        return Err(PoolError::new(
            StatusCode::FORBIDDEN,
            "E061",
            "this Mac is not a member of the pool",
        ));
    }
    let address = Pool::member_address(&peer, hello.port);
    let (name, waited, unencrypted) = {
        let mut inner = pool.inner();
        let hosting = inner.hosting.as_mut().ok_or_else(|| {
            PoolError::new(StatusCode::NOT_FOUND, "E061", "this Mac hosts no pool now")
        })?;
        hosting.members.retain(|m| m.mac_id != peer.mac_id);
        hosting.members.push(Member {
            mac_id: peer.mac_id.clone(),
            name: hello.name,
            chip: hello.chip,
            memory_bytes: hello.memory_bytes,
            power: hello.power,
            address,
            state: "joined".into(),
            missed: 0,
        });
        let waited = hosting.waiting_for.as_deref() == Some(peer.mac_id.as_str());
        (hosting.name.clone(), waited, hosting.unencrypted)
    };
    if waited {
        pool.serve_again();
    }
    let identity = pool.identity()?;
    Ok(Json(
        json!({"pool": name, "host": pool.config.name, "host_id": identity.mac_id(),
        "unencrypted": unencrypted}),
    ))
}

/// A member's heartbeat: its stage and power.
async fn heartbeat(State(pool): State<Arc<Pool>>) -> Json<Value> {
    #[cfg(feature = "engine")]
    let stage = json!(pool.stage.get());
    #[cfg(not(feature = "engine"))]
    let stage = json!({"state": "idle"});
    let _ = &pool;
    Json(json!({"ok": true, "stage": stage, "power": crate::runtime::power::power_source()}))
}

/// Heartbeats missed in a row before a member counts as lost (026 FR-009: within 5 s).
const MISSES: u32 = 3;
const BEAT: Duration = Duration::from_secs(1);

/// The host's watch over its members while it hosts: a heartbeat each second; a member that
/// misses [`MISSES`] is lost, the ring closes (requests on it fail at once) and the pool is
/// degraded until it returns.
async fn health(pool: Arc<Pool>) {
    loop {
        tokio::time::sleep(BEAT).await;
        let members = {
            let inner = pool.inner();
            let Some(h) = inner.hosting.as_ref() else {
                return;
            };
            h.members.clone()
        };
        for member in members {
            let beat = match pool.client_for(&member.mac_id) {
                Ok(client) => client
                    .get(format!("https://{}/pool/v1/heartbeat", member.address))
                    .timeout(Duration::from_millis(900))
                    .send()
                    .await
                    .ok()
                    .filter(|r| r.status().is_success()),
                Err(_) => None,
            };
            let beat = match beat {
                Some(r) => r.json::<Value>().await.ok(),
                None => None,
            };
            let mut lost = None;
            let mut back = false;
            if let Some(h) = pool.inner().hosting.as_mut()
                && let Some(m) = h.members.iter_mut().find(|m| m.mac_id == member.mac_id)
            {
                match &beat {
                    Some(b) => {
                        m.missed = 0;
                        back = m.state == "lost"
                            && h.waiting_for.as_deref() == Some(m.mac_id.as_str());
                        if let Some(state) = b["stage"]["state"].as_str() {
                            m.state = if state == "idle" {
                                "joined".into()
                            } else {
                                state.into()
                            };
                        }
                        if let Some(power) = b["power"].as_str() {
                            m.power = power.into();
                        }
                        // Nothing served: a Mac back is all the pool waited for.
                        if h.model.is_none() && h.members.iter().all(|m| m.state != "lost") {
                            h.degraded = None;
                        }
                    }
                    None => {
                        m.missed += 1;
                        if m.missed == MISSES {
                            m.state = "lost".into();
                            h.degraded = Some(format!("{} stopped answering", m.name));
                            lost = Some(m.name.clone());
                        }
                    }
                }
            }
            if let Some(name) = &lost {
                #[cfg(feature = "engine")]
                pool.close_ring(&format!("{name} stopped answering"));
                #[cfg(not(feature = "engine"))]
                let _ = name;
                // The rest may still hold the model: serve it on them, else wait.
                if pool.serves_pool() {
                    pool.serve_again();
                }
            }
            if back {
                pool.serve_again();
            }
        }
    }
}

/// A member's API call: to this Mac's own server, naming the member.
async fn relay_in(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
    request: Request,
) -> Response {
    let Some((port, _)) = pool.server() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let (parts, body) = request.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map_or(parts.uri.path(), |p| p.as_str());
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_default();
    relay::forward(
        &client,
        parts.method,
        format!("http://127.0.0.1:{port}{path}"),
        &parts.headers,
        Some((MEMBER_HEADER, &peer.mac_id)),
        axum::body::Body::new(body),
    )
    .await
}

/// A member's own sessions on this host (counters; its excerpts below).
async fn member_sessions(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
) -> Response {
    let Some((_, tracker)) = pool.server() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let history = tracker
        .history
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    Json(history.json_for(Some(&peer.mac_id), tracker.uptime_ms())).into_response()
}

/// A member's place in the pool's queue: whether its request runs now, how many of its own
/// wait, and how many of others' wait.
async fn member_queue(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
) -> Response {
    let Some((_, tracker)) = pool.server() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    Json(own_place(&tracker.pool_queue(), &peer.mac_id)).into_response()
}

/// `queue` (the host's) as `member` sees it: only counts, no other member named.
fn own_place(queue: &Value, member: &str) -> Value {
    let mine = |v: &Value| v["member"] == member;
    let waiting = queue["waiting"].as_array().cloned().unwrap_or_default();
    let count = |own: bool| -> u64 {
        waiting
            .iter()
            .filter(|w| mine(w) == own)
            .map(|w| w["waiting"].as_u64().unwrap_or(0))
            .sum()
    };
    json!({
        "running": queue["running"].is_object(),
        "yours_running": mine(&queue["running"]),
        "yours_waiting": count(true),
        "others_waiting": count(false),
    })
}

/// The text around a divergence of one of the member's own requests (decoded on the host, where
/// the tokens are).
async fn member_excerpt(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<tls::Peer>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> Response {
    let Some((port, _)) = pool.server() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_default();
    let url = format!("http://127.0.0.1:{port}/mrun/requests/{id}/excerpt");
    match client
        .get(url)
        .header(MEMBER_HEADER, &peer.mac_id)
        .send()
        .await
    {
        Ok(r) => (
            StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            Json(r.json::<Value>().await.unwrap_or_default()),
        )
            .into_response(),
        Err(_) => StatusCode::BAD_GATEWAY.into_response(),
    }
}

/// The link measurement's transfer: the bytes received.
async fn echo(body: axum::body::Bytes) -> Json<Value> {
    Json(json!({"bytes": body.len()}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_member_sees_its_place_and_counts_only() {
        let queue = json!({
            "running": {"member": "anna"},
            "waiting": [{"member": null, "waiting": 2}, {"member": "anna", "waiting": 1},
                {"member": "chris", "waiting": 3}],
        });
        assert_eq!(
            own_place(&queue, "anna"),
            json!({"running": true, "yours_running": true, "yours_waiting": 1, "others_waiting": 5})
        );
        assert_eq!(own_place(&queue, "dana")["yours_running"], false);
        assert!(!own_place(&queue, "dana").to_string().contains("chris"));
    }

    #[test]
    fn short_names_take_the_owner() {
        assert_eq!(short_name("Ben's MacBook Pro"), "Ben");
        assert_eq!(short_name("Anna’s Mac mini"), "Anna");
        assert_eq!(short_name("Studio"), "Studio");
    }
}

/// The loopback routes `mrun pool` and the window call (contracts/server.md).
pub mod routes {
    use super::{Pool, PoolError};
    use crate::cli::serve::ServerState;
    use axum::Json;
    use axum::extract::{Query, State};
    use axum::response::{IntoResponse, Response};
    use serde_json::{Value, json};
    use std::sync::Arc;

    fn pool(state: &ServerState) -> Arc<Pool> {
        state.pool.clone()
    }

    fn reply(result: Result<Value, PoolError>) -> Response {
        match result {
            Ok(value) => Json(value).into_response(),
            Err(error) => error.into_response(),
        }
    }

    #[derive(Debug, serde::Deserialize)]
    pub struct Browse {
        #[serde(default)]
        browse: bool,
    }

    /// `GET /mrun/pool[?browse=true]`: this Mac's pool, waiting joins and (browsing 2 s) the
    /// pools nearby.
    pub async fn status(State(state): State<ServerState>, Query(q): Query<Browse>) -> Response {
        let pool = pool(&state);
        let mut body = pool.status();
        if body["pool"]["role"] == "member"
            && let Some(place) = pool.place().await
        {
            body["pool"]["place"] = place;
        }
        if q.browse {
            let found = tokio::task::spawn_blocking(|| {
                super::discovery::browse(std::time::Duration::from_secs(2))
            })
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default();
            let me = body["this"]["mac_id"].clone();
            body["nearby"] = found
                .into_iter()
                .filter(|p| json!(p.host_id) != me)
                .map(|p| json!(p))
                .collect();
        }
        Json(body).into_response()
    }

    #[derive(Debug, Default, serde::Deserialize)]
    pub struct Start {
        name: Option<String>,
        #[serde(default)]
        unencrypted: bool,
    }

    pub async fn start(State(state): State<ServerState>, body: Option<Json<Start>>) -> Response {
        let body = body.map(|Json(b)| b).unwrap_or_default();
        let name = body.name.filter(|n| !n.trim().is_empty());
        reply(pool(&state).start(name, body.unencrypted).await)
    }

    #[derive(Debug, serde::Deserialize)]
    pub struct Join {
        address: String,
    }

    pub async fn join(State(state): State<ServerState>, Json(body): Json<Join>) -> Response {
        reply(pool(&state).join(&body.address).await)
    }

    #[derive(Debug, serde::Deserialize)]
    pub struct Answer {
        yes: bool,
        #[serde(default)]
        mac_id: Option<String>,
        #[serde(default)]
        code: Option<String>,
    }

    pub async fn confirm(State(state): State<ServerState>, Json(body): Json<Answer>) -> Response {
        reply(pool(&state).confirm(body.yes, body.code.as_deref()).await)
    }

    pub async fn admit(State(state): State<ServerState>, Json(body): Json<Answer>) -> Response {
        let Some(mac_id) = body.mac_id else {
            return PoolError::usage("which Mac? (mac_id)").into_response();
        };
        reply(pool(&state).admit(&mac_id, body.yes))
    }

    pub async fn rejoin(State(state): State<ServerState>) -> Response {
        reply(pool(&state).rejoin().await)
    }

    pub async fn leave(State(state): State<ServerState>) -> Response {
        reply(pool(&state).leave().await)
    }

    #[derive(Debug, serde::Deserialize)]
    pub struct Remove {
        mac_id: String,
    }

    pub async fn remove(State(state): State<ServerState>, Json(body): Json<Remove>) -> Response {
        reply(pool(&state).remove(&body.mac_id).await)
    }
}
