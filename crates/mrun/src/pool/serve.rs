//! Serving a model across the pool (026 US2, T014): the host plans from every Mac's own spec,
//! sends each member its layer range and the ring's key over the pinned connection, and loads its
//! own part on the engine worker; when every member is ready the ring forms and the server's model
//! becomes the pool's. A member loads its part on a thread of its own and runs [`serve_stage`]
//! until the host stops.
//!
//! A member reads the model from its own store; one that lacks the model says so (pull it there).

use super::engine::{PoolEngine, connect, serve_stage};
use super::link;
use super::plan::{self, Holds, LayerCost, MacSpec, ModelCost, Plan};
use super::{Pool, PoolError};
use crate::cli::model::OpenModel;
use crate::cli::serve::{Loaded, Readiness, ServerState};
use crate::errors::{CliError, Code};
use crate::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use axum::Json;
use axum::extract::{ConnectInfo, State};
use axum::http::StatusCode;
use mrun_engine::{LayerKind, NuvecEngine, Part};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use crate::hub::download::layer_of;

fn expert(name: &str) -> bool {
    name.contains("switch_mlp") || name.contains(".experts.")
}

/// The model's shape (layers, heads) from its `config.json`, as the engine reads it; a pulled
/// model's own info holds only what the budget needs.
pub fn shape(model: &OpenModel) -> mrun_engine::ModelInfo {
    std::fs::read_to_string(model.dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|c| mrun_engine::models::FamilyConfig::from_config(&c).ok())
        .map_or_else(
            || model.info.clone(),
            |f| f.model_info(String::new(), Vec::new()),
        )
}

/// What each layer of `model` costs, from its files and config.
pub fn model_cost(model: &OpenModel) -> ModelCost {
    let dir = &model.dir;
    let info = &shape(model);
    let config: Value = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    let text = config.get("text_config").unwrap_or(&config);
    let experts = text["num_experts"].as_f64().unwrap_or(0.0);
    let active_share = match (experts, text["num_experts_per_tok"].as_f64()) {
        (e, Some(k)) if e > 0.0 => k / e,
        _ => 1.0,
    };
    let footprint = |keep: &dyn Fn(&str) -> bool| {
        mrun_engine::tensor::weights_footprint(dir, |n| {
            !mrun_engine::load::unused_tensor(n) && keep(n)
        })
        .unwrap_or(0)
    };
    let attention_kv = 2 * u64::from(info.kv_heads) * u64::from(info.head_dim) * 2;
    let layers = (0..info.layer_kinds.len())
        .map(|i| {
            let dense = footprint(&|n| layer_of(n) == Some(i) && !expert(n));
            let sparse = footprint(&|n| layer_of(n) == Some(i) && expert(n));
            LayerCost {
                weights: dense + sparse,
                active: dense + (sparse as f64 * active_share) as u64,
                kv_per_token: match info.layer_kinds[i] {
                    LayerKind::FullAttention => attention_kv,
                    LayerKind::Linear => 0,
                },
                recurrent: 0,
            }
        })
        .collect();
    let head = footprint(&|n| layer_of(n).is_none());
    ModelCost {
        name: model.name.clone(),
        layers,
        head,
        head_active: head / 2,
        max_position: info.max_position,
    }
}

/// This Mac as a plan sees it: its own budget (D062) for a model of `weights` bytes.
pub fn local_spec(pool: &Pool) -> Result<MacSpec, PoolError> {
    let device = crate::hub::device_memory()
        .ok_or_else(|| PoolError::internal("this Mac reports no GPU memory"))?;
    let budget = mrun_engine::budget::plan(device, mrun_engine::budget::Overrides::default())
        .map_err(|e| PoolError::internal(e.to_string()))?;
    Ok(MacSpec {
        mac_id: pool.identity()?.mac_id(),
        name: pool.config.name.clone(),
        chip: pool.config.chip.clone(),
        budget: budget.budget,
        gpu_family: mrun_engine::tensor::numerics().gpu_arch,
        power: crate::runtime::power::power_source().into(),
        link: None,
    })
}

/// What this Mac holds of `model` (026 T029): the whole model, a part, or nothing.
pub fn holds(model: &str) -> Holds {
    match crate::hub::resolve(&crate::hub::Ctx::from_env().paths, model) {
        Ok(local) => match local.manifest.and_then(|m| m.part) {
            Some(part) => Holds::Part(part),
            None => Holds::Whole,
        },
        Err(_) => Holds::Nothing,
    }
}

/// A free TCP port on this Mac for its end of the ring.
fn free_port() -> Result<u16, PoolError> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .map_err(|e| PoolError::internal(e.to_string()))
}

/// A member's stage, as the host polls it.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StageState {
    /// `idle`, `downloading`, `loading`, `connecting`, `serving` or `failed`.
    pub state: &'static str,
    /// While downloading: the share of the file in flight.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f64>,
    pub first: usize,
    pub end: usize,
    pub state_tokens: u64,
    pub error: Option<String>,
}

/// This Mac's stage when it lends its memory to a pool.
#[derive(Debug, Default)]
pub struct Stage {
    state: Mutex<StageState>,
    /// Its end of the ring while it serves (closed to stop it).
    pipe: Mutex<Option<mrun_engine::Pipe>>,
    /// The thread that loads and serves it.
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Counts loads: a stopped stage's thread no longer writes the state.
    generation: std::sync::atomic::AtomicU64,
}

impl Stage {
    fn set(&self, f: impl FnOnce(&mut StageState)) {
        f(&mut self.state.lock().unwrap_or_else(PoisonError::into_inner));
    }

    /// [`Stage::set`] from the thread of load `generation`, unless a later load replaced it.
    fn set_for(&self, generation: u64, f: impl FnOnce(&mut StageState)) {
        if self.generation.load(std::sync::atomic::Ordering::SeqCst) == generation {
            self.set(f);
        }
    }

    /// Stops the stage: its ring closes and its thread ends (the host serves again, or the
    /// pool is over for this Mac).
    pub fn stop(&self) {
        if let Some(pipe) = self
            .pipe
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            pipe.close();
        }
    }

    pub fn get(&self) -> StageState {
        let state = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        StageState {
            state: if state.state.is_empty() {
                "idle"
            } else {
                state.state
            },
            ..state
        }
    }
}

/// What the host tells a member to load.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct LoadRequest {
    pub model: String,
    pub first: usize,
    pub end: usize,
    pub rank: usize,
    pub nodes: Vec<Vec<String>>,
    /// Hex, 32 bytes: the ring's key for this serving.
    pub key: String,
    /// The pool's ring is not encrypted (its host's choice).
    #[serde(default)]
    pub unencrypted: bool,
}

fn key_of(hex: &str) -> Result<[u8; 32], PoolError> {
    crate::remote::from_hex(hex)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| PoolError::usage("the ring key is not 32 bytes of hex"))
}

fn load_part(model: &OpenModel, part: Part) -> Result<NuvecEngine, String> {
    let mut options = mrun_engine::LoadOptions {
        identity: model.identity.value.clone(),
        overrides: mrun_engine::budget::Overrides::default(),
        fallback_device: crate::hub::device_memory(),
        cpu: false,
        calibration: mrun_engine::calibration::default_path(),
        part: Some(part),
    };
    if std::env::var_os("MRUN_POOL_CPU").is_some() {
        options.cpu = true;
        options.calibration = None;
    }
    NuvecEngine::load(&model.dir, options).map_err(|e| e.to_string())
}

/// `GET /pool/v1/spec` (members only): this Mac's spec, a port for its end of the ring, and the
/// address the host is seen from.
pub async fn spec(
    State(pool): State<Arc<Pool>>,
    ConnectInfo(peer): ConnectInfo<super::tls::Peer>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Value>, PoolError> {
    let model = q.get("model").cloned();
    let (spec, holds) = tokio::task::spawn_blocking({
        let pool = pool.clone();
        move || Ok::<_, PoolError>((local_spec(&pool)?, model.as_deref().map(holds)))
    })
    .await
    .map_err(|e| PoolError::internal(e.to_string()))??;
    Ok(Json(json!({
        "spec": spec,
        "holds": holds,
        "ring_port": free_port()?,
        "host_ip": peer.addr.ip().to_string(),
    })))
}

/// Pull progress lines (JSON) into the stage's state.
struct Downloading {
    stage: Arc<Stage>,
    generation: u64,
    line: Vec<u8>,
}

impl std::io::Write for Downloading {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.line.extend_from_slice(bytes);
        while let Some(at) = self.line.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.line.drain(..=at).collect();
            if let Ok(event) = serde_json::from_slice::<Value>(&line)
                && event["event"] == "progress"
            {
                let share = event["bytes"].as_f64().unwrap_or(0.0)
                    / event["total"].as_f64().unwrap_or(1.0).max(1.0);
                self.stage
                    .set_for(self.generation, |s| s.progress = Some(share.min(1.0)));
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `POST /pool/v1/load` (members only): loads this Mac's part and joins the ring, in the
/// background.
pub async fn load(
    State(pool): State<Arc<Pool>>,
    Json(request): Json<LoadRequest>,
) -> Result<Json<Value>, PoolError> {
    let key = key_of(&request.key)?;
    // The memory goes to the pool: this Mac's own model makes room first.
    let _before = pool.drop_own_model();
    let stage = pool.stage.clone();
    // A part already serving (the host serves again) stops first.
    stage.stop();
    let previous = stage
        .thread
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    let generation = stage
        .generation
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        + 1;
    stage.set(|s| {
        *s = StageState {
            state: "loading",
            first: request.first,
            end: request.end,
            ..StageState::default()
        }
    });
    let thread_stage = stage.clone();
    let runtime = tokio::runtime::Handle::current();
    let thread = std::thread::spawn(move || {
        let stage = thread_stage;
        if let Some(previous) = previous {
            let _ = previous.join();
        }
        let failed =
            |e: String| stage.set_for(generation, |s| (s.state, s.error) = ("failed", Some(e)));
        // A Mac that lacks the layers downloads only their files (a part in its store).
        if !holds(&request.model).covers(request.first, request.end) {
            stage.set_for(generation, |s| s.state = "downloading");
            let report = crate::hub::progress::Report::Json(crate::hub::progress::JsonLines::to(
                Downloading {
                    stage: stage.clone(),
                    generation,
                    line: Vec::new(),
                },
            ));
            let ctx = crate::hub::Ctx::from_env();
            if let Err(e) = runtime.block_on(crate::hub::download::pull_part(
                &ctx,
                &request.model,
                request.first,
                request.end,
                &report,
            )) {
                return failed(format!(
                    "cannot download its layers of {}: {}",
                    request.model, e.message
                ));
            }
            stage.set_for(generation, |s| (s.state, s.progress) = ("loading", None));
        }
        let model = match crate::cli::model::open_stage(&request.model) {
            Ok(model) => model,
            Err(e) => {
                return failed(format!(
                    "{}; pull it on this Mac first (mrun pull {})",
                    e.message, request.model
                ));
            }
        };
        let part = Part {
            first: request.first,
            end: request.end,
            head: false,
        };
        let mut engine = match load_part(&model, part) {
            Ok(engine) => engine,
            Err(e) => return failed(e),
        };
        let tokens = engine.state_tokens();
        stage.set_for(generation, |s| {
            (s.state, s.state_tokens) = ("connecting", tokens)
        });
        match connect(&request.nodes, request.rank, &key, request.unencrypted) {
            Ok(pipe) => {
                *stage.pipe.lock().unwrap_or_else(PoisonError::into_inner) = Some(pipe.clone());
                engine.set_pipe(Some(pipe));
            }
            Err(e) => return failed(e),
        }
        stage.set_for(generation, |s| s.state = "serving");
        match serve_stage(&mut engine) {
            Ok(()) => stage.set_for(generation, |s| *s = StageState::default()),
            Err(e) => failed(e),
        }
        if let Some(pipe) = engine.pipe() {
            pipe.close();
        }
    });
    *stage.thread.lock().unwrap_or_else(PoisonError::into_inner) = Some(thread);
    Ok(Json(json!({"loading": true})))
}

/// `GET /pool/v1/stage` (members only).
pub async fn stage(State(pool): State<Arc<Pool>>) -> Json<StageState> {
    Json(pool.stage.get())
}

/// A member's answer to the host's spec request.
struct MemberSpec {
    member: super::Member,
    holds: Holds,
    spec: MacSpec,
    ring_port: u16,
    host_ip: String,
    client: reqwest::Client,
}

impl Pool {
    /// Every Mac's spec, links measured, in stage order (the host first).
    async fn specs(&self, model: &str) -> Result<(MacSpec, Vec<MemberSpec>), PoolError> {
        let host = tokio::task::block_in_place(|| local_spec(self))?;
        let mut members = Vec::new();
        for member in self.live_members() {
            let client = self.client_for(&member.mac_id)?;
            let unreachable = |e: reqwest::Error| {
                PoolError::new(
                    StatusCode::BAD_GATEWAY,
                    "E060",
                    format!("{} stopped answering: {e}", member.name),
                )
            };
            let answer: Value = client
                .get(
                    reqwest::Url::parse_with_params(
                        &format!("https://{}/pool/v1/spec", member.address),
                        [("model", model)],
                    )
                    .map_err(|e| PoolError::internal(e.to_string()))?,
                )
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .map_err(unreachable)?
                .json()
                .await
                .map_err(unreachable)?;
            let mut spec: MacSpec = serde_json::from_value(answer["spec"].clone())
                .map_err(|e| PoolError::internal(e.to_string()))?;
            let ip: std::net::IpAddr = member
                .address
                .parse::<std::net::SocketAddr>()
                .map(|a| a.ip())
                .map_err(|e| PoolError::internal(e.to_string()))?;
            spec.link = Some(
                link::measure(&client, &member.address, link::kind(ip))
                    .await
                    .map_err(|e| PoolError::new(StatusCode::BAD_GATEWAY, "E060", e))?,
            );
            members.push(MemberSpec {
                holds: serde_json::from_value(answer["holds"].clone()).unwrap_or_default(),
                ring_port: answer["ring_port"].as_u64().unwrap_or(0) as u16,
                host_ip: answer["host_ip"]
                    .as_str()
                    .unwrap_or("127.0.0.1")
                    .to_string(),
                member,
                spec,
                client,
            });
        }
        Ok((host, members))
    }

    /// The plan for `model` on the pool's Macs now.
    pub async fn plan(&self, model: &str) -> Result<Plan, PoolError> {
        let opened = open(model).await?;
        let cost = tokio::task::block_in_place(|| model_cost(&opened));
        let (host, members) = self.specs(model).await?;
        let mut holds = vec![Holds::Whole];
        holds.extend(members.iter().map(|m| m.holds));
        let mut macs = vec![host];
        macs.extend(members.into_iter().map(|m| m.spec));
        let mut plan = plan::plan(&cost, &macs);
        plan::with_downloads(&mut plan, &cost, &holds);
        Ok(plan)
    }

    /// Serves `model` across the pool: plan, every member loads its part, the host its own, the
    /// ring forms, and the server's model becomes the pool's.
    pub async fn serve(&self, state: &ServerState, model: &str) -> Result<Value, PoolError> {
        if self.inner().hosting.is_none() {
            return Err(PoolError::usage("this Mac hosts no pool (mrun pool start)"));
        }
        let opened = Arc::new(open(model).await?);
        let cost = tokio::task::block_in_place(|| model_cost(&opened));
        let (host, members) = self.specs(model).await?;
        let mut macs = vec![host];
        macs.extend(members.iter().map(|m| m.spec.clone()));
        let mut plan = plan::plan(&cost, &macs);
        let mut holds = vec![Holds::Whole];
        holds.extend(members.iter().map(|m| m.holds));
        plan::with_downloads(&mut plan, &cost, &holds);
        if !plan.fits {
            return Err(PoolError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "E062",
                plan.refusal.clone().unwrap_or_default(),
            ));
        }
        // The host's memory goes to the pool's model (its own model comes back after).
        self.drop_own_model();
        if members.is_empty() {
            return self.serve_alone(state, opened, model, plan).await;
        }
        let host_ip = members
            .first()
            .map_or("127.0.0.1".to_string(), |m| m.host_ip.clone());
        let mut nodes = vec![vec![format!("{host_ip}:{}", free_port()?)]];
        for m in &members {
            let ip = m
                .member
                .address
                .rsplit_once(':')
                .map_or("", |(ip, _)| ip)
                .to_string();
            nodes.push(vec![format!("{ip}:{}", m.ring_port)]);
        }
        let unencrypted = self.unencrypted();
        let key =
            super::identity::random_secret().map_err(|e| PoolError::internal(e.to_string()))?;
        let key_hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
        for (i, m) in members.iter().enumerate() {
            let stage = &plan.stages[i + 1];
            let request = LoadRequest {
                model: model.to_string(),
                first: stage.first,
                end: stage.end,
                rank: i + 1,
                nodes: nodes.clone(),
                key: key_hex.clone(),
                unencrypted,
            };
            m.client
                .post(format!("https://{}/pool/v1/load", m.member.address))
                .json(&request)
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .map_err(|e| PoolError::new(StatusCode::BAD_GATEWAY, "E060", e.to_string()))?;
        }
        let watch: Vec<(String, String, reqwest::Client)> = members
            .iter()
            .map(|m| {
                (
                    m.member.name.clone(),
                    m.member.address.clone(),
                    m.client.clone(),
                )
            })
            .collect();
        let runtime = tokio::runtime::Handle::current();
        let ring = self.ring_slot();
        let reason = self.ring_reason.clone();
        *reason.lock().unwrap_or_else(PoisonError::into_inner) = None;
        let end = plan.stages[0].end;
        let engine_model = opened.clone();
        let factory = move || -> Result<PoolEngine, String> {
            let host = load_part(
                &engine_model,
                Part {
                    first: 0,
                    end,
                    head: true,
                },
            )?;
            // Every member loaded and connecting: the ring forms when the host connects.
            let tokens = runtime.block_on(ready(&watch))?;
            let pipe = connect(&nodes, 0, &key, unencrypted)?;
            *ring.lock().unwrap_or_else(PoisonError::into_inner) = Some(pipe.clone());
            Ok(PoolEngine::new(
                host,
                pipe,
                tokens.saturating_mul(9) / 10,
                reason,
            ))
        };
        let handle = EngineHandle::spawn_with(
            factory,
            opened.tokenizer.clone(),
            DEFAULT_STALL_TIMEOUT,
            crate::cli::serve::cache_policy(&state.args),
        )
        .map_err(|e| PoolError::new(StatusCode::BAD_GATEWAY, "E060", e))?;
        self.ready_with(state, &opened, model, handle);
        self.serving(state, model, &plan);
        Ok(json!({"serving": model, "plan": plan}))
    }

    /// Records what the pool serves, and the server it serves on (to serve again).
    fn serving(&self, state: &ServerState, model: &str, plan: &Plan) {
        if let Some(hosting) = self.inner().hosting.as_mut() {
            hosting.model = Some(model.to_string());
            hosting.plan = Some(json!(plan));
            hosting.server = Some(state.clone());
        }
    }

    /// The pool is the host alone (every member left): the whole model, no ring.
    async fn serve_alone(
        &self,
        state: &ServerState,
        opened: Arc<OpenModel>,
        model: &str,
        plan: Plan,
    ) -> Result<Value, PoolError> {
        let layers = plan.stages[0].end;
        let engine_model = opened.clone();
        let handle = EngineHandle::spawn_with(
            move || load_part(&engine_model, Part::whole(layers)),
            opened.tokenizer.clone(),
            DEFAULT_STALL_TIMEOUT,
            crate::cli::serve::cache_policy(&state.args),
        )
        .map_err(|e| PoolError::internal(e.to_string()))?;
        self.ready_with(state, &opened, model, handle);
        self.serving(state, model, &plan);
        Ok(json!({"serving": model, "plan": plan}))
    }

    /// The server's model becomes `handle`'s.
    fn ready_with(
        &self,
        state: &ServerState,
        opened: &OpenModel,
        model: &str,
        handle: EngineHandle,
    ) {
        let grammar =
            crate::chat::constrain::GrammarEnv::new(&opened.tokenizer, &handle.info().eos_tokens)
                .ok()
                .map(Arc::new);
        let loaded = Loaded {
            model_id: model.to_string(),
            syntax: opened.syntax(),
            supports_thinking: opened.supports_thinking(),
            defaults: opened.defaults,
            grammar,
            handle,
            tokenizer: opened.tokenizer.clone(),
            template: opened.template.clone(),
        };
        loaded.handle.queue().set_capacity(state.args.queue);
        state.ready.send_replace(Readiness::Ready(Arc::new(loaded)));
    }

    /// Serves the pool's model again on the Macs answering now (a Mac left, was removed or
    /// lost). When the rest cannot hold it, the pool waits for the lost Mac, offering the
    /// largest pulled model that fits the rest.
    pub async fn replan(self: Arc<Self>) -> Result<Value, PoolError> {
        let (state, model) = {
            let mut inner = self.inner();
            let hosting = inner
                .hosting
                .as_mut()
                .ok_or_else(|| PoolError::usage("this Mac hosts no pool"))?;
            let (Some(state), Some(model)) = (hosting.server.clone(), hosting.model.clone()) else {
                return Err(PoolError::usage(
                    "the pool serves no model yet (mrun pool serve <model>)",
                ));
            };
            if hosting.replanning {
                return Err(PoolError::usage("the pool is serving again already"));
            }
            hosting.replanning = true;
            (state, model)
        };
        self.close_ring("the pool is serving again on the Macs answering now");
        let result = self.serve(&state, &model).await;
        let lost = self.inner().hosting.as_ref().and_then(|h| {
            h.members
                .iter()
                .find(|m| m.state == "lost")
                .map(|m| (m.mac_id.clone(), m.name.clone()))
        });
        let smaller = match &result {
            Err(e) if e.code == "E062" => self.smaller(&model).await,
            _ => None,
        };
        if let Err(error) = &result {
            let code = if error.code == "E062" {
                Code::PoolDoesNotFit
            } else {
                Code::PoolMemberLost
            };
            state.ready.send_replace(Readiness::Failed(CliError::new(
                code,
                format!("the pool cannot serve {model} now: {}", error.message),
            )));
        }
        if let Some(hosting) = self.inner().hosting.as_mut() {
            hosting.replanning = false;
            match &result {
                Ok(_) => {
                    hosting.degraded = None;
                    hosting.recovery = None;
                    hosting.waiting_for = None;
                }
                Err(error) => {
                    hosting.degraded = Some(match &lost {
                        Some((_, name)) => format!("{name} stopped answering"),
                        None => error.message.clone(),
                    });
                    hosting.waiting_for = lost.as_ref().map(|(id, _)| id.clone());
                    hosting.recovery = Some(json!({
                        "wait": lost.as_ref().map(|(_, name)| name.clone()),
                        "smaller": smaller,
                    }));
                }
            }
        }
        result
    }

    /// The largest whole pulled model (by weight bytes) that fits the Macs answering now.
    async fn smaller(&self, current: &str) -> Option<String> {
        let (host, members) = self.specs(current).await.ok()?;
        let mut macs = vec![host];
        macs.extend(members.into_iter().map(|m| m.spec));
        let paths = crate::hub::Ctx::from_env().paths;
        let mut pulled: Vec<(u64, String)> = crate::hub::manifests(&paths)
            .into_iter()
            .filter(|m| m.name() != current)
            .map(|m| (m.loaded_weights(), m.name().to_string()))
            .collect();
        pulled.sort_by_key(|p| std::cmp::Reverse(p.0));
        for (_, name) in pulled {
            let Ok(opened) = open(&name).await else {
                continue;
            };
            let cost = tokio::task::block_in_place(|| model_cost(&opened));
            if plan::plan(&cost, &macs).fits {
                return Some(name);
            }
        }
        None
    }
}

impl Pool {
    fn ring_slot(&self) -> Arc<Mutex<Option<mrun_engine::Pipe>>> {
        self.ring.clone()
    }
}

/// Waits until every member's stage is connecting; their smallest state capacity.
async fn ready(members: &[(String, String, reqwest::Client)]) -> Result<u64, String> {
    let mut tokens = u64::MAX;
    for (name, address, client) in members {
        loop {
            let stage: StageState = client
                .get(format!("https://{address}/pool/v1/stage"))
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .map_err(|e| format!("{name} stopped answering: {e}"))?
                .json::<Value>()
                .await
                .map(|v| StageState {
                    state: match v["state"].as_str() {
                        Some("connecting") => "connecting",
                        Some("serving") => "serving",
                        Some("failed") => "failed",
                        _ => "loading",
                    },
                    state_tokens: v["state_tokens"].as_u64().unwrap_or(0),
                    error: v["error"].as_str().map(str::to_string),
                    ..StageState::default()
                })
                .map_err(|e| e.to_string())?;
            match stage.state {
                "connecting" | "serving" => {
                    tokens = tokens.min(stage.state_tokens);
                    break;
                }
                "failed" => return Err(format!("{name}: {}", stage.error.unwrap_or_default())),
                _ => tokio::time::sleep(Duration::from_millis(200)).await,
            }
        }
    }
    Ok(tokens)
}

async fn open(model: &str) -> Result<OpenModel, PoolError> {
    let name = model.to_string();
    tokio::task::spawn_blocking(move || crate::cli::model::open(&name))
        .await
        .map_err(|e| PoolError::internal(e.to_string()))?
        .map_err(|e| PoolError::new(StatusCode::NOT_FOUND, "E031", e.message))
}

/// The loopback routes for plans and serving.
pub mod routes {
    use super::super::PoolError;
    use crate::cli::serve::ServerState;
    use axum::Json;
    use axum::extract::{Query, State};
    use axum::response::{IntoResponse, Response};

    #[derive(Debug, serde::Deserialize)]
    pub struct Model {
        model: String,
    }

    fn reply(result: Result<serde_json::Value, PoolError>) -> Response {
        match result {
            Ok(value) => Json(value).into_response(),
            Err(error) => error.into_response(),
        }
    }

    /// `GET /mrun/pool/plan?model=`.
    pub async fn plan(State(state): State<ServerState>, Query(q): Query<Model>) -> Response {
        reply(
            state
                .pool
                .plan(&q.model)
                .await
                .map(|p| serde_json::json!(p)),
        )
    }

    /// `POST /mrun/pool/replan`: the pool's model again on the Macs answering now.
    pub async fn replan(State(state): State<ServerState>) -> Response {
        reply(state.pool.clone().replan().await)
    }

    /// `POST /mrun/pool/serve {model}`.
    pub async fn serve(State(state): State<ServerState>, Json(body): Json<Model>) -> Response {
        let pool = state.pool.clone();
        reply(pool.serve(&state, &body.model).await)
    }
}
