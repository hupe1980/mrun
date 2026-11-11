// The shapes the page reads: `/mrun/status` (012, through the host's proxy) and the UI host's
// `/app/*` routes (spec 018: the UI host contract and data model).

export type Cause = "disabled" | "evicted" | "diverged" | "new" | "gap";
export type Phase = "queued" | "prefill" | "generating";

/** Speculative decoding's counts (006): draft tokens verified and kept, and verify passes. */
export interface Speculation {
  drafted: number;
  accepted: number;
  passes: number;
}

export interface Request {
  id: number;
  api: string;
  lane: number;
  phase: Phase;
  prompt_tokens: number;
  cached: number | null;
  progress: number;
  cause: Cause | null;
  unreused: number | null;
  generated: number;
  decode_tps: number | null;
  speculation?: Speculation | null;
  age_ms: number;
}

export interface Miss {
  cached: number;
  processed: number;
  cause: Cause | null;
  divergence: number | null;
  message: { index: number; role: string } | null;
  gap?: number;
  unreused: number;
}

export interface Finished extends Miss {
  id: number;
  api: string;
  prompt_tokens: number;
  generated: number;
  ttft_ms: number | null;
  decode_tps: number | null;
  stop: string | null;
  speculation?: Speculation | null;
}

/** A finished request as session replay keeps it (025, `/mrun/sessions`). */
export interface ReplayRecord extends Finished {
  session: string;
  row: string;
  client: string;
  /** Milliseconds since the server started, when the request ended. */
  ended_ms: number;
  attention: boolean;
  excerpt: boolean;
}

export type RowKind = "main" | "subagent" | "side" | "conversation";

export interface Session {
  id: string;
  client: string;
  label: string;
  first_ms: number;
  last_ms: number;
  requests: number;
  prompt_tokens: number;
  cached_tokens: number;
  attention: number;
  rows: { id: string; kind: RowKind; requests: number }[];
}

export interface Sessions {
  kept: number;
  dropped: number;
  remote: boolean;
  sessions: Session[];
  requests: ReplayRecord[];
}

/** The text around a divergence, decoded on request (`/mrun/requests/{id}/excerpt`). */
export interface Excerpt {
  id: number;
  message: { index: number; role: string } | null;
  context: string;
  before: string;
  after: string;
}

export interface Status {
  version: string;
  model: { id: string; identity: string; family: string; syntax: string } | null;
  numerics: { backend: string; nax: string; gpu_family: string } | null;
  budget: {
    budget: number;
    weights: number;
    scratch: number;
    kv_capacity_tokens: number;
    kv_bytes_per_token: number;
    /** What bounds it: `working set`, `physical`, `available memory`, `--budget-gib` or `context`. */
    binding?: string;
  } | null;
  queued: number;
  cache: {
    live_tokens: number;
    buffer_bytes: number;
    checkpoint_bytes: number;
    prompt_tokens_total: number;
    cached_tokens_total: number;
    enabled: boolean;
  } | null;
  last: Miss | null;
  requests: Request[];
  finished?: Finished[];
  lanes: { configured: number; busy: number };
  /** Remote mode (013): the pod the relay serves from. */
  pod?: Pod | null;
  /** This Mac's pool (026). */
  pool?: PoolView["pool"];
  /** Speculative decoding (006): the mode and totals since the server started. */
  speculation?: { mode: "auto" | "off" } & Speculation;
}

/** What the server is doing, as one word the whole UI agrees on. */
export type ServerState = "ready" | "reading" | "generating" | "loading" | "stopped" | "failed";

/** A Mac in a pool, as the host sees it (026). */
export interface PoolMember {
  mac_id: string;
  name: string;
  chip: string;
  memory_bytes: number;
  power: "adapter" | "battery" | string;
  address: string;
  /** `joined`, `downloading`, `loading`, `connecting`, `serving`, `failed` or `lost`. */
  state: string;
}

export interface PoolWaiting {
  mac_id: string;
  name: string;
  chip: string;
  memory_bytes: number;
  power: string;
  code: string;
  waiting_s: number;
}

/** `GET /mrun/pool` (026). */
export interface PoolView {
  this: { mac_id: string | null; name: string; chip: string; memory_bytes: number };
  pool:
    | {
        role: "host";
        name: string;
        model: string | null;
        degraded: string | null;
        /** While the pool waits for a lost Mac: whom, and the largest pulled model that fits the rest. */
        recovery?: { wait: string | null; smaller: string | null } | null;
        /** Serving again on the Macs answering now. */
        replanning?: boolean;
        /** The model's traffic between the Macs is not encrypted (the host's choice). */
        unencrypted?: boolean;
        address: string | null;
        members: PoolMember[];
        waiting: PoolWaiting[];
        plan?: PoolPlan | null;
        /** Who runs now, and how many requests wait per sender (`null`: this Mac). */
        queue?: { running: { member: string | null } | null; waiting: { member: string | null; waiting: number }[] };
      }
    | {
        role: "member";
        name: string;
        host: { mac_id: string; name: string };
        address: string;
        unencrypted?: boolean;
        /** This Mac's place in the pool's queue (counts only). */
        place?: { running: boolean; yours_running: boolean; yours_waiting: number; others_waiting: number } | null;
      }
    | null;
  joining: { pool: string; host: string; code: string; unencrypted?: boolean } | null;
  /** The pool this Mac was in and may rejoin without a code. */
  rejoin?: { host: string; mac_id: string } | null;
  nearby?: { name: string; host: string; host_id: string; members: number; model: string | null; addresses: string[] }[];
}

export interface PoolStage {
  mac_id: string;
  name: string;
  first: number;
  end: number;
  weights_bytes: number;
  state_bytes: number;
  scratch_bytes: number;
  budget: number;
  fits: boolean;
  short_bytes: number;
  ms_per_token: number;
  /** What this Mac downloads first (its layers' files). */
  download_bytes?: number;
  link?: { kind: string; rtt_ms_p50: number; rtt_ms_p90: number; mbps: number } | null;
}

/** `GET /mrun/pool/plan` (026 data model). */
export interface PoolPlan {
  model: string;
  fits: boolean;
  stages: PoolStage[];
  context_tokens: number;
  speed: { low: number; high: number };
  pace: { kind: "stage" | "link"; mac_id: string; name: string; share: number } | null;
  numerics_match: boolean;
  refusal: string | null;
}

/** The relay's pod (013, `/mrun/status` `pod`). */
export interface Pod {
  id: string | null;
  gpu: string | null;
  cost_per_hr: number | null;
  reachable: boolean;
  phase?: string;
  uptime_s?: number;
  spend_usd?: number;
  idle_s?: number | null;
  idle_stop_s?: number;
}

/** A step of `cloud up --json` (024): `pod`, `agent`, `loading`, `ready`. */
export type CloudStep = "pod" | "agent" | "loading" | "ready";

export interface Job {
  id: string;
  kind: ActionKind;
  target: string;
  state: "running" | "done" | "failed" | "cancelled";
  progress: {
    file: string;
    bytes: number;
    total: number;
    rate: number;
    eta_s: number | null;
    step?: CloudStep;
    text?: string;
  } | null;
  error: string | null;
  /** The last JSON line the command printed (`cloud up`'s `done`: model, gpu, cost_per_hr). */
  output?: Record<string, any> | null;
}

export type ActionKind =
  | "pull"
  | "use"
  | "remove"
  | "start"
  | "stop"
  | "connect"
  | "disconnect"
  | "cloud-up"
  | "cloud-down"
  | "cloud-login"
  | "cloud-logout";

/** A curated model for a rented GPU (`mrun cloud models --json`). */
export interface Offer {
  repo: string;
  weights_bytes: number;
  context: number;
  min_vram_gb: number;
  note: string;
  /** The cheapest fitting GPU and its $/h per cloud; `null` without a key or stock. */
  secure: [string, number] | null;
  community: [string, number] | null;
  /** GPU type ids it fits (empty without a key). */
  fits: string[];
}

/** A RunPod GPU type (`mrun cloud gpus --json`). */
export interface Gpu {
  id: string;
  name: string;
  memory_gb: number;
  secure_per_hr: number | null;
  community_per_hr: number | null;
  /** `High`, `Medium`, `Low`; `null`: none available now. */
  secure_stock: string | null;
  community_stock: string | null;
}

export type KeySource = "env" | "dotenv" | "keychain" | null;

/** The section `GET /app/cloud` (024 data model: CloudListing). */
export interface CloudListing {
  offers: Offer[];
  gpus: Gpu[];
  status: { pods: { id: string; model: string | null; gpu: string | null; cost_per_hr: number | null }[]; key: KeySource; error?: string };
  added: CloudModel[];
}

/** A model the user added to the cloud section (window prefs `cloud_models`). */
export interface CloudModel {
  repo: string;
  weights_bytes: number;
  family?: string | null;
}

export interface Prefs {
  keep_chats: boolean | null;
  menu_bar: boolean;
  open_at_login: boolean;
  server_port: number;
  /** The model the window last served (written by the host). */
  last_model?: string | null;
  cloud_models?: CloudModel[];
}

export interface AppState {
  ui_version: string;
  server: { port: number; running: boolean; started_by: "ui" | "other" | null };
  prefs: Prefs;
  jobs: Job[];
}

export interface ModelEntry {
  name: string;
  alias: string | null;
  repo: string;
  revision: string;
  size: number;
  pulled: boolean;
  /** `null`: this Mac's memory is unknown. */
  fit: "fits" | "tight" | "wont_fit" | null;
  verdict: string;
  max_context: number | null;
  recommended: boolean;
  preview: boolean;
  serving: boolean;
  family?: string;
  /** A part (026): only layers `[first, end)`, for pool stages. */
  part?: [number, number];
  /** What a context costs (027): `fixed + tokens × per_token` bytes. */
  memory?: { fixed: number; per_token: number } | null;
  /** The budget that counts against. */
  budget?: number | null;
}

export interface AgentEntry {
  client: "claude-code" | "codex" | "opencode";
  installed: boolean;
  version: string | null;
  connected: boolean;
  files: { path: string; change: string }[];
  refusal: string | null;
  /** The settings blocks `mrun connect <client>` prints (020 FR-008). */
  blocks?: { path: string; note: string; text: string }[];
}

/** Another app's settings (`mrun connect zed|crush --json`). */
export interface AppEntry {
  client: string;
  blocks?: { path: string; note: string; text: string }[];
}

/** `mrun pull <repo> --dry-run --json`'s plan (020 data model). */
export interface Plan {
  event: "plan";
  model: string;
  repo: string;
  revision: string;
  files: number;
  bytes: number;
  fit: "fits" | "tight" | "wont_fit" | null;
  verdict: string;
  family: string;
}

export interface Sampling {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  max_tokens?: number;
  seed?: number;
}

export interface Stats {
  prompt_tokens: number;
  cached_tokens: number;
  ttft_ms: number | null;
  decode_tps: number | null;
  stop: "end" | "length" | "stopped" | "error" | "tools";
  miss?: Miss;
  speculation?: Speculation;
}

/** A file in a user message (020 data model): sent whole, as a fenced block. */
export interface Attachment {
  name: string;
  text: string;
  bytes: number;
  tokens: number;
  /** Counted without a server (`bytes / 3.5`), shown as "about". */
  estimated: boolean;
}

export interface Message {
  role: "user" | "assistant";
  text: string;
  attachments?: Attachment[];
  reasoning?: string;
  reasoning_ms?: number;
  /** While reasoning streams (019 FR-001): when it began (epoch ms) and its tokens so far. */
  reasoning_started?: number;
  reasoning_tokens?: number;
  stats?: Stats;
  error?: string;
  /** The tools this reply called (028), in order; their results go back with the next request. */
  tools?: ToolCall[];
}

/** A denial a sandbox reported, with what "Allow …" adds to the server's block (028 FR-009). */
export interface Denial {
  text: string;
  fix?: { access: string; path: string } | { host: string };
  label?: string;
}

/** One tool call of a reply (028). */
export interface ToolCall {
  id: string;
  /** `<server>__<tool>`, as the model named it. */
  name: string;
  input: Record<string, unknown>;
  /** The streamed JSON of the input, until the reply ends. */
  raw?: string;
  state: "pending" | "asking" | "running" | "blocked" | "done" | "failed" | "denied";
  result?: string;
  denials?: Denial[];
  /** Ran without asking: this tool is always allowed in this chat. */
  auto?: boolean;
}

/** A server of `~/.mrun/mcp.json`, as `GET /app/mcp` shows it: never env values or headers. */
export interface McpServer {
  name: string;
  kind: "stdio" | "http";
  command?: string;
  host?: string;
  env?: string[];
  headers?: string[];
  sandbox: false | { workspace: string | null; write: boolean; hosts: string[]; allow: { access: string; path: string }[] };
  tools?: { name: string; qualified: string | null; description: string; input_schema: Record<string, unknown> }[];
  error?: string;
}

export interface McpList {
  path: string;
  servers: McpServer[];
  errors: string[];
}

export interface Chat {
  id: string;
  title: string;
  created: string;
  updated: string;
  model: string | null;
  system: string | null;
  sampling: Sampling;
  thinking: boolean | null;
  messages: Message[];
  /** The tool servers on in this chat (028), and the tools it always allows. */
  tools?: string[];
  always?: string[];
}

/** `mrun config show --json` (017): every setting with its value and where it comes from. */
export interface ConfigView {
  path: string;
  exists: boolean;
  /** `source`: `default`, `file:<line>` or `env:<VAR>`. */
  settings: { key: string; value: unknown; source: string }[];
  warnings?: string[];
}
