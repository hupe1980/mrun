// The page's only way out: its UI host, on the same origin. Every request carries the host's
// per-start token, which arrives in the URL fragment (never sent to a server) and lives in this
// tab's sessionStorage (contracts/ui-host.md §Security).

import type { AgentEntry, AppEntry, AppState, Chat, CloudListing, ConfigView, Denial, Excerpt, Job, McpList, Miss, ModelEntry, Offer, Plan, PoolPlan, PoolView, Prefs, Sessions, Speculation, Status } from "./types";

const KEY = "mrun.token";

/** Moves `#t=<token>` into sessionStorage and clears it from the address. */
export function adoptToken(): string | null {
  const match = /(?:^#|&)t=([0-9a-f]{64})/.exec(location.hash);
  if (match?.[1]) {
    sessionStorage.setItem(KEY, match[1]);
    history.replaceState(null, "", location.pathname + location.search);
  }
  return sessionStorage.getItem(KEY);
}

export class HostError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function headers(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  const token = sessionStorage.getItem(KEY);
  if (token) h.set("Authorization", `Bearer ${token}`);
  return h;
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...init, headers: headers(init.headers) });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      // A CLI refusal carries `{error: <code>, message, hint}`: the sentence, not the code.
      message = body?.error?.message ?? body?.message ?? body?.error ?? message;
    } catch {
      /* not JSON */
    }
    throw new HostError(response.status, String(message));
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/**
 * A listing the host caches (019 R9): the cached value at once, marked stale while the host
 * refreshes it; `fresh` waits for the refresh.
 */
async function listing<T>(path: string, fresh: boolean): Promise<{ value: T; stale: boolean }> {
  const response = await fetch(fresh ? `${path}?fresh=1` : path, { headers: headers() });
  if (!response.ok) throw new HostError(response.status, response.statusText);
  return { value: (await response.json()) as T, stale: response.headers.get("x-mrun-stale") === "1" };
}

/** The server's own token count for a Messages body (`count_tokens`: no queue, 002). */
export async function countTokens(body: Record<string, unknown>): Promise<number> {
  const r = await call<{ input_tokens: number }>("/v1/messages/count_tokens", json("POST", body));
  return r.input_tokens;
}

export const host = {
  status: () => call<Status>("/mrun/status"),
  /** Session replay (025): every kept session, and the records after `since`. */
  sessions: (since?: number) => call<Sessions>(since === undefined ? "/mrun/sessions" : `/mrun/sessions?since=${since}`),
  excerpt: (id: number) => call<Excerpt>(`/mrun/requests/${id}/excerpt`),
  /** Mac pool (026): this Mac's pool, waiting joins and (browsing) the pools nearby. */
  pool: (browse = false) => call<PoolView>(browse ? "/mrun/pool?browse=true" : "/mrun/pool"),
  poolPlan: (model: string) => call<PoolPlan>(`/mrun/pool/plan?model=${encodeURIComponent(model)}`),
  state: () => call<AppState>("/app/state"),
  models: (fresh = false) => listing<ModelEntry[]>("/app/models", fresh),
  agents: (fresh = false) => listing<AgentEntry[]>("/app/agents", fresh),
  cloud: (fresh = false) => listing<CloudListing>("/app/cloud", fresh),
  /** A repo's weights and fitting GPUs (`mrun cloud models <repo> --json`). */
  cloudModel: (repo: string) => call<Offer>(`/app/cloud/model?repo=${encodeURIComponent(repo)}`),
  config: () => call<ConfigView>("/app/config"),
  modelInfo: (id: string) => call<Record<string, unknown>>(`/v1/models/${encodeURIComponent(id)}`),
  /** `secret` is `cloud-login`'s key: the host writes it to the command's stdin. */
  act: (kind: string, target: string, confirm = false, secret?: string) =>
    call<{ job: Job }>("/app/actions", json("POST", { kind, target, confirm, secret })),
  job: (id: string) => call<Job>(`/app/jobs/${id}`),
  cancel: (id: string) => call<Job>(`/app/jobs/${id}`, { method: "DELETE" }),
  command: (kind: string, target: string) =>
    call<{ argv: string[]; text: string }>(
      `/app/command?kind=${encodeURIComponent(kind)}&target=${encodeURIComponent(target)}`,
    ),
  chats: () => call<{ id: string; title: string; updated: string }[]>("/app/chats"),
  chat: (id: string) => call<Chat>(`/app/chats/${id}`),
  saveChat: (chat: Chat) => call<void>(`/app/chats/${chat.id}`, json("PUT", chat)),
  deleteChat: (id: string) => call<void>(`/app/chats/${id}`, { method: "DELETE" }),
  deleteAllChats: () => call<{ deleted: number }>("/app/chats", { method: "DELETE" }),
  prefs: (patch: Partial<Prefs>) => call<Prefs>("/app/prefs", json("PUT", patch)),
  quit: () => call<void>("/app/quit", { method: "POST" }),
  plan: (repo: string) => call<Plan>(`/app/plan?model=${encodeURIComponent(repo)}`),
  apps: () => call<AppEntry[]>("/app/apps"),
  exportChat: (name: string, text: string) => call<{ path: string }>("/app/export", json("POST", { name, text })),
  open: (what: "config" | "log" | "chats" | "mcp" | "url", url?: string) =>
    call<void>("/app/open", json("POST", { what, url })),
  /** Tools (028): the servers of mcp.json; `start` lists their tools, `refresh` rereads the file. */
  mcp: (mode: "" | "start" | "refresh" = "") => call<McpList>(mode ? `/app/mcp?${mode}=true` : "/app/mcp"),
  mcpCall: (server: string, tool: string, args: unknown) =>
    call<{ text: string; is_error: boolean; denials?: Denial[] }>("/app/mcp/call", json("POST", { server, tool, arguments: args })),
  mcpAllow: (server: string, fix: NonNullable<Denial["fix"]>) => call<{ command: string }>("/app/mcp/allow", json("POST", { server, fix })),
  mcpImport: (client: "claude-code" | "claude-desktop") => call<{ added: string[] }>("/app/mcp/import", json("POST", { client })),
  mcpAdd: (kit: "workspace" | "fetch", dir?: string) => call<{ added: string[] }>("/app/mcp/add", json("POST", { kit, dir })),
  mcpChoose: () => call<{ dir: string | null }>("/app/mcp/choose", { method: "POST" }),
};

/** Events of one streamed Messages reply. */
export type StreamEvent =
  | { type: "meta"; miss: Miss; max_tokens: number }
  | { type: "text"; text: string }
  | { type: "thinking"; text: string }
  | { type: "usage"; input: number; cached: number; output: number }
  | { type: "output"; tokens: number }
  | { type: "stop"; reason: string }
  | { type: "speculation"; speculation: Speculation }
  | { type: "error"; message: string }
  | { type: "tool_start"; index: number; id: string; name: string }
  | { type: "tool_args"; index: number; json: string };

/**
 * Streams `POST /v1/messages` (the public API every agent uses, FR-007). Aborting `signal` drops
 * the connection, which is how the server learns to stop (D052).
 */
export async function* streamMessages(
  body: Record<string, unknown>,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch("/v1/messages", {
    method: "POST",
    headers: headers({ "Content-Type": "application/json", "anthropic-version": "2023-06-01" }),
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!response.ok || !response.body) {
    let message = `The server answered ${response.status}.`;
    try {
      const err = await response.json();
      message = err?.error?.message ?? message;
    } catch {
      /* not JSON */
    }
    yield { type: "error", message };
    return;
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      yield* parseFrame(frame);
    }
  }
}

/** One SSE frame: mrun's `: mrun {…}` comment, or an Anthropic Messages event. */
export function* parseFrame(frame: string): Generator<StreamEvent> {
  for (const line of frame.split("\n")) {
    if (line.startsWith(": mrun ")) {
      try {
        const meta = JSON.parse(line.slice(7));
        if ("cached" in meta) {
          yield {
            type: "meta",
            max_tokens: meta.max_tokens ?? 0,
            miss: {
              cached: meta.cached,
              processed: meta.processed,
              cause: meta.cause ?? null,
              divergence: meta.divergence ?? null,
              message: meta.message ?? null,
              unreused: meta.unreused ?? 0,
            },
          };
        }
        if (meta.speculation) yield { type: "speculation", speculation: meta.speculation };
      } catch {
        /* a comment that is not mrun's JSON */
      }
    }
  }
  const data = frame
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => l.slice(6))
    .join("\n");
  if (!data) return;
  let event: Record<string, any>;
  try {
    event = JSON.parse(data);
  } catch {
    return;
  }
  switch (event.type) {
    case "message_start": {
      const u = event.message?.usage ?? {};
      yield {
        type: "usage",
        input: (u.input_tokens ?? 0) + (u.cache_read_input_tokens ?? 0),
        cached: u.cache_read_input_tokens ?? 0,
        output: u.output_tokens ?? 0,
      };
      break;
    }
    case "content_block_start":
      if (event.content_block?.type === "tool_use")
        yield { type: "tool_start", index: event.index ?? 0, id: event.content_block.id ?? "", name: event.content_block.name ?? "" };
      break;
    case "content_block_delta": {
      const d = event.delta ?? {};
      if (d.type === "input_json_delta") yield { type: "tool_args", index: event.index ?? 0, json: d.partial_json ?? "" };
      else if (d.type === "text_delta") yield { type: "text", text: d.text ?? "" };
      else if (d.type === "thinking_delta") yield { type: "thinking", text: d.thinking ?? "" };
      break;
    }
    case "message_delta":
      if (typeof event.usage?.output_tokens === "number")
        yield { type: "output", tokens: event.usage.output_tokens };
      if (event.delta?.stop_reason) yield { type: "stop", reason: event.delta.stop_reason };
      break;
    case "error":
      yield { type: "error", message: event.error?.message ?? "The server reported an error." };
      break;
  }
}
