// Numbers and causes in words: the page says what the CLI says, without jargon.

import type { Cause, Chat, Miss, Request, ServerState, Status } from "./types";

const GB = 1024 ** 3;
const MB = 1024 ** 2;

/** `19.5 GB`, `812 MB` (binary units, as `mrun ls`). */
export function bytes(n: number): string {
  if (n >= GB) return `${(n / GB).toFixed(n >= 10 * GB ? 1 : 2).replace(/\.?0+$/, "")} GB`;
  if (n >= MB) return `${Math.round(n / MB)} MB`;
  return `${Math.round(n / 1024)} KB`;
}

export function tokens(n: number): string {
  return n.toLocaleString("en-US");
}

/** `128K`, `4K`, `900` for context sizes. */
export function context(n: number): string {
  return n >= 1024 ? `${Math.round(n / 1024)}K` : String(n);
}

export function seconds(ms: number): string {
  if (ms < 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m} min ${s} s`;
}

export function percent(part: number, whole: number): string {
  if (whole <= 0) return "0 %";
  const p = (100 * part) / whole;
  return `${p >= 99.5 && part < whole ? 99 : Math.round(p)} %`;
}

/** The D053 cause of a miss, as a sentence fragment. */
export function cause(miss: Pick<Miss, "cause" | "message" | "divergence">): string {
  const at = miss.message ? `message ${miss.message.index + 1} (${miss.message.role})` : null;
  switch (miss.cause as Cause | null) {
    case "diverged":
      if (!at) return "the conversation changed";
      if (miss.message!.role === "user") return `edited at ${at}`;
      // Templates re-render earlier replies; the server resumes at their header (021).
      if (miss.message!.role === "assistant") return `reply re-rendered at ${at}`;
      return `client changed ${at}`;
    case "evicted":
      return "dropped from memory earlier";
    case "new":
      return "a new conversation";
    case "gap":
      return "resumed from an earlier checkpoint";
    case "disabled":
      return "the cache is off";
    default:
      return "fully reused";
  }
}

export type Tone = "expected" | "attention";

/** Unreused tokens the server itself calls a gap (`GAP_CAUSE`, 003). */
const GAP_TOKENS = 256;

/**
 * Whether a recompute deserves attention (019 data model): only what nobody intended, a cache
 * eviction, a disabled cache, or a client re-sending an earlier tool or system message
 * differently. New conversations, appended tokens, a person's own edit, checkpoint resumes and
 * a re-rendered reply that costs at most a gap (021) are expected.
 */
export function tone(
  miss: (Pick<Miss, "cause" | "message"> & { unreused?: number | null }) | null | undefined,
): Tone {
  switch (miss?.cause) {
    case "evicted":
    case "disabled":
      return "attention";
    case "diverged":
      if (!miss.message || miss.message.role === "user") return "expected";
      if (miss.message.role === "assistant" && (miss.unreused ?? 0) <= GAP_TOKENS) return "expected";
      return "attention";
    default:
      return "expected";
  }
}

/** One sentence for a reuse ribbon, read by VoiceOver. */
export function reuseSentence(cached: number, total: number, miss?: Miss | null): string {
  const base = `${tokens(cached)} of ${tokens(total)} prompt tokens reused`;
  if (!miss?.cause || cached >= total) return `${base}.`;
  return `${base}; ${tokens(total - cached)} computed because of ${cause(miss)}.`;
}

/** The server's state in one word, from `/health` reachability and `/mrun/status`. */
export function serverState(status: Status | null, reachable: boolean, loading: boolean): ServerState {
  if (!reachable) return "stopped";
  if (loading || !status?.model) return "loading";
  const active = status.requests.filter((r) => r.phase !== "queued");
  if (active.some((r) => r.phase === "prefill")) return "reading";
  if (active.length) return "generating";
  return "ready";
}

export const stateWord: Record<ServerState, string> = {
  ready: "Ready",
  reading: "Reading",
  generating: "Generating",
  loading: "Loading",
  stopped: "Stopped",
  failed: "Failed",
};

/** "Reading prompt · 12,400 of 30,000 tokens" or "Generating · 42 tok/s". */
export function phaseLine(r: Request): string {
  switch (r.phase) {
    case "queued":
      return `Waiting · ${tokens(r.prompt_tokens)} tokens`;
    case "prefill":
      return `Reading prompt · ${tokens(r.progress)} of ${tokens(r.prompt_tokens)} tokens`;
    case "generating":
      return r.decode_tps
        ? `Generating · ${r.decode_tps.toFixed(1)} tok/s · ${tokens(r.generated)} tokens`
        : `Generating · ${tokens(r.generated)} tokens`;
  }
}

export function stopWord(stop: string): string {
  switch (stop) {
    case "end":
    case "end_turn":
    case "stop_sequence":
      return "finished";
    case "length":
    case "max_tokens":
      return "reached the length limit";
    case "stopped":
      return "stopped by you";
    case "error":
      return "failed";
    default:
      return stop;
  }
}

/**
 * A model to add (020 FR-006): `org/name[@rev]` or a Hugging Face model URL, as `org/name[@rev]`;
 * `null` for anything else, which never reaches the network.
 */
export function repoFromInput(input: string): string | null {
  const s = input.trim();
  const part = "[A-Za-z0-9][A-Za-z0-9._-]{0,95}";
  const url = new RegExp(`^https?://(?:www\\.)?huggingface\\.co/(${part})/(${part})(?:/tree/([A-Za-z0-9._-]+))?/?$`).exec(s);
  if (url) return `${url[1]}/${url[2]}${url[3] ? `@${url[3]}` : ""}`;
  return new RegExp(`^${part}/${part}(?:@[A-Za-z0-9._-]+)?$`).test(s) ? s : null;
}

/** A chat as Markdown (020 FR-009): what was said, the files, the thoughts collapsed, and each
 * reply's stats line. */
export function exportMarkdown(chat: Chat, fence: (name: string, text: string) => string): string {
  const out = [`# ${chat.title}`, "", [chat.model, chat.updated.slice(0, 10)].filter(Boolean).join(" · ")];
  if (chat.system) out.push("", ...chat.system.split("\n").map((l) => `> ${l}`));
  for (const m of chat.messages) {
    out.push("", m.role === "user" ? "**You**" : `**${chat.model ?? "Model"}**`, "");
    for (const a of m.attachments ?? []) out.push(fence(a.name, a.text), "");
    if (m.reasoning) {
      out.push(`<details><summary>Thought for ${seconds(m.reasoning_ms ?? 0)}</summary>`, "", m.reasoning.trim(), "", "</details>", "");
    }
    if (m.text) out.push(m.text.trim());
    const st = m.stats;
    if (st) {
      const parts = [`${tokens(st.cached_tokens)} of ${tokens(st.prompt_tokens)} reused`];
      if (st.ttft_ms !== null) parts.push(`${seconds(st.ttft_ms)} to first token`);
      if (st.decode_tps) parts.push(`${st.decode_tps.toFixed(1)} tok/s`);
      parts.push(stopWord(st.stop));
      out.push("", `*${parts.join(" · ")}*`);
    }
  }
  return `${out.join("\n")}\n`;
}

/** Where a chat matches a search (020 FR-010): `null`, the title, or ~60 characters of text. */
export function searchHit(chat: Chat, query: string): { excerpt: string | null } | null {
  const q = query.trim().toLowerCase();
  if (!q) return { excerpt: null };
  if (chat.title.toLowerCase().includes(q)) return { excerpt: null };
  for (const m of chat.messages) {
    for (const text of [m.text, ...(m.attachments ?? []).map((a) => a.text)]) {
      const at = text.toLowerCase().indexOf(q);
      if (at >= 0) {
        const start = Math.max(0, at - 24);
        const piece = text.slice(start, at + q.length + 36).replace(/\s+/g, " ").trim();
        return { excerpt: `${start > 0 ? "…" : ""}${piece}…` };
      }
    }
  }
  return null;
}
