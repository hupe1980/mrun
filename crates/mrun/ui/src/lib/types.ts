// The shapes the page reads: `/mrun/status` (012, through the host's proxy) and the UI host's
// `/app/*` routes (spec 018: the UI host contract and data model).

export type Cause = "disabled" | "evicted" | "diverged" | "new" | "gap";
export type Phase = "queued" | "prefill" | "generating";

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
}

/** What the server is doing, as one word the whole UI agrees on. */
export type ServerState = "ready" | "reading" | "generating" | "loading" | "stopped" | "failed";

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
  stop: "end" | "length" | "stopped" | "error";
  miss?: Miss;
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
}
