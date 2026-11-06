//! Session replay (025): finished requests grouped into sessions and rows from the prompts they
//! share, kept in memory (the last [`HISTORY`]), served at `GET /mrun/sessions` (counters, no
//! text) and `GET /mrun/requests/{id}/excerpt` (the text around a divergence, decoded on request).
//!
//! Clients send no session id. Requests of one client class, each within [`SESSION_IDLE_MS`] of
//! the previous, form a session. A request continues the row whose last request shares the most
//! whole conversation messages with it, else it opens a row: the first with tools is `main`,
//! later ones with tools are subagents. Requests without tools from a coding agent (titles,
//! compaction) share one `side` row.

use crate::api::handler;
use crate::cache::miss::{Cause, Excerpt, MessageRef};
use crate::chat::prompt::{RenderedPrompt, Role};
use crate::cli::serve::ServerState;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, PoisonError};

/// Finished requests session replay keeps.
pub const HISTORY: usize = 2000;
/// A session ends after this long without a request.
pub const SESSION_IDLE_MS: u64 = 30 * 60 * 1000;
/// Unreused tokens a re-rendered assistant message may cost before it is worth fixing (the
/// server's gap, `GAP_CAUSE`).
const ASSISTANT_SLACK: u32 = 256;
/// Characters of decoded text an excerpt shows per part.
const EXCERPT_CHARS: usize = 200;

/// What a request's prompt shares with others: a running hash at the end of each conversation
/// message, seeded with a hash of the system prompt and tools.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Fingerprint {
    bounds: Vec<u64>,
    tools: bool,
}

impl Fingerprint {
    pub fn of(prompt: &RenderedPrompt, tools: bool) -> Self {
        let mut hasher = DefaultHasher::new();
        let start = (prompt.system_tools_end as usize).min(prompt.tokens.len());
        prompt.tokens[..start].hash(&mut hasher);
        let marks: Vec<u32> = prompt
            .messages
            .iter()
            .map(|m| m.start)
            .filter(|&s| s >= prompt.system_tools_end)
            .collect();
        let mut bounds = Vec::with_capacity(marks.len());
        let mut at = start;
        for (i, _) in marks.iter().enumerate() {
            let end = marks
                .get(i + 1)
                .copied()
                .unwrap_or(prompt.generation_start)
                .min(prompt.len()) as usize;
            if end > at {
                prompt.tokens[at..end].hash(&mut hasher);
                at = end;
            }
            bounds.push(hasher.finish());
        }
        Self { bounds, tools }
    }

    fn shared(&self, other: &Self) -> usize {
        self.bounds
            .iter()
            .zip(&other.bounds)
            .take_while(|(a, b)| a == b)
            .count()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowKind {
    Main,
    Subagent,
    Side,
    /// Another conversation in a session without tools (the window's chat).
    Conversation,
}

impl RowKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Subagent => "subagent",
            Self::Side => "side",
            Self::Conversation => "conversation",
        }
    }
}

#[derive(Debug)]
struct Row {
    id: String,
    kind: RowKind,
    last: Fingerprint,
}

#[derive(Debug)]
struct Session {
    id: String,
    client: String,
    last_ms: u64,
    rows: Vec<Row>,
}

/// The display name of a client class.
pub fn label(client: &str) -> String {
    // A pool member's sessions are keyed `class@member`.
    match client.split('@').next().unwrap_or(client) {
        "claude-cli" => "Claude Code".into(),
        "opencode" => "OpenCode".into(),
        "mrun-ui" => "Chat".into(),
        c if c.starts_with("codex") => "Codex".into(),
        c => c.into(),
    }
}

/// Coding agents, whose requests without tools are side requests even before the main one.
fn agent(client: &str) -> bool {
    let client = client.split('@').next().unwrap_or(client);
    matches!(client, "claude-cli" | "opencode") || client.starts_with("codex")
}

/// Whether a miss is worth fixing: the page's `tone() == "attention"` rule (019).
pub fn worth_fixing(cause: Option<Cause>, message: Option<MessageRef>, unreused: u32) -> bool {
    match cause {
        Some(Cause::Evicted | Cause::Disabled) => true,
        Some(Cause::Diverged) => match message.map(|m| m.role) {
            None | Some(Role::User) => false,
            Some(Role::Assistant) => unreused > ASSISTANT_SLACK,
            Some(Role::System | Role::Tool) => true,
        },
        _ => false,
    }
}

/// What the run path hands over for a finished request.
#[derive(Debug)]
pub struct Finished {
    /// The pool member that sent the request (026), `None` from this Mac.
    pub member: Option<String>,
    pub client: String,
    pub fingerprint: Fingerprint,
    pub excerpt: Option<Arc<Excerpt>>,
}

#[derive(Debug)]
struct Kept {
    id: u64,
    session: String,
    record: Value,
    excerpt: Option<Arc<Excerpt>>,
}

/// Sessions and the ring of kept records.
#[derive(Debug, Default)]
pub struct History {
    sessions: Vec<Session>,
    kept: VecDeque<Kept>,
    dropped: u64,
    next_session: u32,
}

impl History {
    /// The session and row of a request from `client` that ended at `now_ms`.
    pub fn assign(&mut self, client: &str, fp: &Fingerprint, now_ms: u64) -> (String, String) {
        let open = self.sessions.iter().rposition(|s| {
            s.client == client && now_ms.saturating_sub(s.last_ms) <= SESSION_IDLE_MS
        });
        let index = open.unwrap_or_else(|| {
            self.next_session += 1;
            self.sessions.push(Session {
                id: format!("s{}", self.next_session),
                client: client.into(),
                last_ms: now_ms,
                rows: Vec::new(),
            });
            self.sessions.len() - 1
        });
        let session = &mut self.sessions[index];
        session.last_ms = now_ms;
        let main_tools = session
            .rows
            .iter()
            .any(|r| r.kind == RowKind::Main && r.last.tools);
        let row = if !fp.tools && (main_tools || agent(client)) {
            let at = session.rows.iter().position(|r| r.kind == RowKind::Side);
            let at = at.unwrap_or_else(|| {
                session.rows.push(Row {
                    id: "side".into(),
                    kind: RowKind::Side,
                    last: Fingerprint::default(),
                });
                session.rows.len() - 1
            });
            &mut session.rows[at]
        } else {
            let best = session
                .rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r.kind != RowKind::Side)
                .map(|(i, r)| (r.last.shared(fp), i))
                .filter(|&(shared, _)| shared > 0)
                .max_by_key(|&(shared, i)| (shared, i));
            let at = best.map(|(_, i)| i).unwrap_or_else(|| {
                let has_main = session.rows.iter().any(|r| r.kind == RowKind::Main);
                let count = |k| session.rows.iter().filter(|r| r.kind == k).count() + 1;
                let (id, kind) = if !has_main {
                    ("main".to_string(), RowKind::Main)
                } else if fp.tools {
                    (
                        format!("sub{}", count(RowKind::Subagent)),
                        RowKind::Subagent,
                    )
                } else {
                    (
                        format!("c{}", count(RowKind::Conversation) + 1),
                        RowKind::Conversation,
                    )
                };
                session.rows.push(Row {
                    id,
                    kind,
                    last: Fingerprint::default(),
                });
                session.rows.len() - 1
            });
            let row = &mut session.rows[at];
            row.last = fp.clone();
            row
        };
        (session.id.clone(), row.id.clone())
    }

    /// Keeps a finished record (with `session`, `row` and the rest already set).
    pub fn push(&mut self, id: u64, session: String, record: Value, excerpt: Option<Arc<Excerpt>>) {
        self.kept.push_back(Kept {
            id,
            session,
            record,
            excerpt,
        });
        while self.kept.len() > HISTORY {
            self.kept.pop_front();
            self.dropped += 1;
        }
        // Sessions whose records all left the ring and that ended are forgotten.
        let live: std::collections::HashSet<&str> =
            self.kept.iter().map(|k| k.session.as_str()).collect();
        let last = self.sessions.last().map(|s| s.id.clone());
        self.sessions
            .retain(|s| live.contains(s.id.as_str()) || Some(&s.id) == last.as_ref());
    }

    /// The excerpt of request `id` for `member` (`None`: this Mac): a request's text is shown
    /// only to the Mac that sent it (026 FR-010).
    pub fn excerpt(&self, id: u64, member: Option<&str>) -> Option<(Value, Arc<Excerpt>)> {
        let kept = self.kept.iter().find(|k| k.id == id)?;
        if kept.record["member"].as_str() != member {
            return None;
        }
        Some((kept.record["message"].clone(), kept.excerpt.clone()?))
    }

    /// The `/mrun/sessions` body at `now_ms`: every kept session, newest first, and the records
    /// after `since`, oldest first.
    pub fn json(&self, since: Option<u64>, now_ms: u64) -> Value {
        self.json_filtered(since, now_ms, None)
    }

    /// The sessions and records of one pool member only (026: what its Mac shows).
    pub fn json_for(&self, member: Option<&str>, now_ms: u64) -> Value {
        self.json_filtered(None, now_ms, Some(member))
    }

    fn json_filtered(
        &self,
        since: Option<u64>,
        now_ms: u64,
        member: Option<Option<&str>>,
    ) -> Value {
        let mine = |k: &&Kept| member.is_none_or(|m| k.record["member"].as_str() == m);
        let sessions: Vec<Value> = self
            .sessions
            .iter()
            .rev()
            .filter_map(|s| {
                let records: Vec<&Value> = self
                    .kept
                    .iter()
                    .filter(|k| k.session == s.id)
                    .filter(mine)
                    .map(|k| &k.record)
                    .collect();
                let first = records.first()?;
                let sum = |key: &str| {
                    records
                        .iter()
                        .map(|r| r[key].as_u64().unwrap_or(0))
                        .sum::<u64>()
                };
                let rows: Vec<Value> = s
                    .rows
                    .iter()
                    .map(|row| {
                        let n = records
                            .iter()
                            .filter(|r| r["row"] == row.id.as_str())
                            .count();
                        json!({"id": row.id, "kind": row.kind.as_str(), "requests": n})
                    })
                    .filter(|row| row["requests"] != 0)
                    .collect();
                Some(json!({
                    "id": s.id,
                    "client": s.client.split('@').next().unwrap_or(&s.client),
                    "member": s.client.split_once('@').map(|(_, m)| m),
                    "label": label(&s.client),
                    "first_ms": first["ended_ms"],
                    "last_ms": records.last().map_or(Value::Null, |r| r["ended_ms"].clone()),
                    "requests": records.len(),
                    "prompt_tokens": sum("prompt_tokens"),
                    "cached_tokens": sum("cached"),
                    "attention": records.iter().filter(|r| r["attention"] == true).count(),
                    "rows": rows,
                }))
            })
            .collect();
        let requests: Vec<&Value> = self
            .kept
            .iter()
            .filter(mine)
            .filter(|k| since.is_none_or(|since| k.id > since))
            .map(|k| &k.record)
            .collect();
        json!({
            "schema_version": 1,
            "kept": HISTORY,
            "dropped": self.dropped,
            "remote": false,
            "now_ms": now_ms,
            "sessions": sessions,
            "requests": requests,
        })
    }
}

/// The last `n` characters of `text`, on character boundaries, marked as cut when shortened.
fn tail(text: &str, n: usize) -> String {
    let count = text.chars().count();
    if count <= n {
        return text.into();
    }
    format!("…{}", text.chars().skip(count - n).collect::<String>())
}

/// The first `n` characters of `text`, on character boundaries.
fn head(text: &str, n: usize) -> String {
    if text.chars().count() <= n {
        return text.into();
    }
    format!("{}…", text.chars().take(n).collect::<String>())
}

/// Decoded tokens without the replacement characters a cut through a multi-byte character
/// leaves at either end.
fn decode(tokenizer: &tokenizers::Tokenizer, ids: &[u32]) -> String {
    tokenizer
        .decode(ids, false)
        .unwrap_or_default()
        .trim_matches('\u{FFFD}')
        .into()
}

#[derive(Debug, serde::Deserialize)]
pub struct Since {
    since: Option<u64>,
}

/// `GET /mrun/sessions[?since=<id>]`.
pub async fn sessions(State(state): State<ServerState>, Query(q): Query<Since>) -> Response {
    let now_ms = state.tracker.uptime_ms();
    let history = state
        .tracker
        .history
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    axum::Json(history.json(q.since, now_ms)).into_response()
}

/// `GET /mrun/requests/{id}/excerpt`: the text around the divergence, decoded now; a pool
/// member's request only for that member (asked through the pool port).
pub async fn excerpt(
    State(state): State<ServerState>,
    Path(id): Path<u64>,
    headers: axum::http::HeaderMap,
) -> Response {
    let member = headers
        .get(crate::pool::MEMBER_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let not_found = || {
        (
            StatusCode::NOT_FOUND,
            axum::Json(json!({"error": "no excerpt for this request"})),
        )
            .into_response()
    };
    let Ok(loaded) = handler::ready(&state) else {
        return not_found();
    };
    let found = state
        .tracker
        .history
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .excerpt(id, member.as_deref());
    let Some((message, excerpt)) = found else {
        return not_found();
    };
    let inner = loaded.tokenizer.inner();
    axum::Json(json!({
        "id": id,
        "message": message,
        "context": tail(&decode(inner, &excerpt.context), EXCERPT_CHARS),
        "before": head(&decode(inner, &excerpt.before), EXCERPT_CHARS),
        "after": head(&decode(inner, &excerpt.after), EXCERPT_CHARS),
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::prompt::MessageMark;

    /// A prompt with system prompt `system` and one message per entry of `messages` (each a
    /// token run), header after the last.
    fn prompt(system: &[u32], messages: &[&[u32]]) -> RenderedPrompt {
        let mut tokens = system.to_vec();
        let mut marks = vec![MessageMark {
            start: 0,
            role: Role::System,
            message: Some(0),
        }];
        for (i, m) in messages.iter().enumerate() {
            marks.push(MessageMark {
                start: tokens.len() as u32,
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                message: Some(i as u32 + 1),
            });
            tokens.extend_from_slice(m);
        }
        let generation_start = tokens.len() as u32;
        tokens.extend([7, 7]);
        RenderedPrompt {
            tokens,
            messages: marks,
            system_tools_end: system.len() as u32,
            generation_start,
        }
    }

    const MAIN: &[u32] = &[1, 2, 3];
    const SUB: &[u32] = &[4, 5];

    #[test]
    fn a_tool_loop_stays_on_main_and_a_subagent_opens_a_row() {
        let mut h = History::default();
        let fp =
            |system, messages: &[&[u32]], tools| Fingerprint::of(&prompt(system, messages), tools);
        assert_eq!(
            h.assign("claude-cli", &fp(MAIN, &[&[10]], true), 0),
            ("s1".into(), "main".into())
        );
        assert_eq!(
            h.assign("claude-cli", &fp(MAIN, &[&[10], &[11], &[12]], true), 1)
                .1,
            "main"
        );
        // Another system prompt, same client: a subagent in the same session.
        assert_eq!(
            h.assign("claude-cli", &fp(SUB, &[&[20]], true), 2),
            ("s1".into(), "sub1".into())
        );
        assert_eq!(
            h.assign("claude-cli", &fp(SUB, &[&[20], &[21], &[22]], true), 3)
                .1,
            "sub1"
        );
        // Back to the main conversation, and an edit of an earlier message stays on it.
        assert_eq!(
            h.assign(
                "claude-cli",
                &fp(MAIN, &[&[10], &[11], &[12], &[13], &[14]], true),
                4
            )
            .1,
            "main"
        );
        assert_eq!(
            h.assign("claude-cli", &fp(MAIN, &[&[10], &[99], &[12]], true), 5)
                .1,
            "main"
        );
        // A second subagent.
        assert_eq!(
            h.assign("claude-cli", &fp(SUB, &[&[30]], true), 6).1,
            "sub2"
        );
    }

    #[test]
    fn titles_go_to_side_and_clients_make_sessions() {
        let mut h = History::default();
        let fp =
            |system, messages: &[&[u32]], tools| Fingerprint::of(&prompt(system, messages), tools);
        // A title request before the main one: still side for a coding agent.
        assert_eq!(
            h.assign("claude-cli", &fp(SUB, &[&[1]], false), 0).1,
            "side"
        );
        assert_eq!(
            h.assign("claude-cli", &fp(MAIN, &[&[10]], true), 1).1,
            "main"
        );
        assert_eq!(
            h.assign("claude-cli", &fp(SUB, &[&[2]], false), 2).1,
            "side"
        );
        // Another client: its own session; the window's chat has no tools and gets main.
        assert_eq!(
            h.assign("mrun-ui", &fp(MAIN, &[&[10]], false), 3),
            ("s2".into(), "main".into())
        );
        assert_eq!(h.assign("mrun-ui", &fp(MAIN, &[&[40]], false), 4).1, "c2");
        // A pool member's agent: a session of its own, labelled like this Mac's.
        assert_eq!(
            h.assign("claude-cli@0123456789abcdef", &fp(SUB, &[&[1]], false), 5),
            ("s3".into(), "side".into())
        );
        assert_eq!(label("claude-cli@0123456789abcdef"), "Claude Code");
        // 31 minutes of silence: a new session.
        let later = 2 + 31 * 60 * 1000;
        assert_eq!(
            h.assign("claude-cli", &fp(MAIN, &[&[10], &[11]], true), later)
                .0,
            "s4"
        );
    }

    #[test]
    fn attention_follows_the_tone_rule() {
        let at = |role| Some(MessageRef { index: 3, role });
        assert!(worth_fixing(Some(Cause::Evicted), None, 0));
        assert!(worth_fixing(Some(Cause::Disabled), None, 0));
        assert!(worth_fixing(Some(Cause::Diverged), at(Role::Tool), 0));
        assert!(worth_fixing(Some(Cause::Diverged), at(Role::System), 0));
        assert!(!worth_fixing(Some(Cause::Diverged), at(Role::User), 9000));
        assert!(!worth_fixing(Some(Cause::Diverged), None, 9000));
        assert!(!worth_fixing(
            Some(Cause::Diverged),
            at(Role::Assistant),
            256
        ));
        assert!(worth_fixing(
            Some(Cause::Diverged),
            at(Role::Assistant),
            257
        ));
        for cause in [None, Some(Cause::New), Some(Cause::Gap)] {
            assert!(!worth_fixing(cause, None, 9000));
        }
    }

    #[test]
    fn the_ring_is_bounded_and_since_returns_newer_records() {
        let mut h = History::default();
        let fp = Fingerprint::of(&prompt(MAIN, &[&[10]]), true);
        for id in 1..=HISTORY as u64 + 5 {
            let (session, row) = h.assign("claude-cli", &fp, id);
            let record = json!({"id": id, "session": session, "row": row, "ended_ms": id,
                "prompt_tokens": 10, "cached": 9, "attention": id % 1000 == 0});
            h.push(id, session, record, None);
        }
        let body = h.json(None, 0);
        assert_eq!(body["requests"].as_array().unwrap().len(), HISTORY);
        assert_eq!(body["dropped"], 5);
        assert_eq!(body["sessions"][0]["requests"], HISTORY);
        assert_eq!(body["sessions"][0]["attention"], 2);
        assert_eq!(body["sessions"][0]["first_ms"], 6);
        let newer = h.json(Some(HISTORY as u64 + 3), 0);
        let ids: Vec<u64> = newer["requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_u64().unwrap())
            .collect();
        assert_eq!(ids, [HISTORY as u64 + 4, HISTORY as u64 + 5]);
        assert!(h.excerpt(HISTORY as u64, None).is_none());
    }

    #[test]
    fn excerpt_cuts_never_split_a_character() {
        let text = "ä".repeat(300);
        let cut = head(&text, EXCERPT_CHARS);
        assert_eq!(cut.chars().count(), EXCERPT_CHARS + 1);
        assert!(cut.ends_with('…'));
        let cut = tail(&text, EXCERPT_CHARS);
        assert!(cut.starts_with('…') && cut.chars().count() == EXCERPT_CHARS + 1);
        assert_eq!(head("short", EXCERPT_CHARS), "short");
    }
}
