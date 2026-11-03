// Files in the chat (020 US1): whole text, never RAG, so the exact cache reuses them on every
// follow-up. A file becomes a fenced block headed by its name, ahead of the typed text
// (research R1); refusals say why before anything is sent (research R3).

import type { Attachment, Message } from "./types";

export const MAX_BYTES = 2 * 1024 * 1024;

export type Read = { ok: true; attachment: Attachment } | { ok: false; name: string; reason: string };

const languages: Record<string, string> = {
  rs: "rust", ts: "ts", tsx: "tsx", js: "js", jsx: "jsx", mjs: "js", py: "python", go: "go",
  java: "java", kt: "kotlin", swift: "swift", c: "c", h: "c", cpp: "cpp", cc: "cpp", hpp: "cpp",
  cs: "csharp", rb: "ruby", php: "php", sh: "sh", zsh: "sh", bash: "sh", sql: "sql",
  json: "json", toml: "toml", yaml: "yaml", yml: "yaml", md: "markdown", html: "html",
  css: "css", scss: "scss", svelte: "svelte", vue: "vue", xml: "xml", diff: "diff", patch: "diff",
};

/** Reads a dropped, pasted or chosen file, or says why it can't go in. */
export async function readAttachment(file: File): Promise<Read> {
  const name = file.name.split(/[\\/]/).pop()!.slice(0, 255) || "file";
  const refuse = (reason: string): Read => ({ ok: false, name, reason });
  if (file.type.startsWith("image/")) return refuse("Images need a vision model; mrun's models read text only for now.");
  if (file.type === "application/pdf" || /\.pdf$/i.test(name)) return refuse("PDFs aren't read; paste the text instead.");
  if (file.size > MAX_BYTES) return refuse(`Larger than ${MAX_BYTES / 1024 / 1024} MB.`);
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (bytes.subarray(0, 8192).includes(0)) return refuse("Not a text file.");
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return refuse("Not a text file (not UTF-8).");
  }
  return { ok: true, attachment: { name, text, bytes: file.size, tokens: estimate(file.size), estimated: true } };
}

/** Tokens without a server: a deliberately rough guess, always shown as "about". */
export function estimate(bytes: number): number {
  return Math.ceil(bytes / 3.5);
}

/** A file as the model reads it: its name, then a fence longer than any backtick run inside. */
export function fence(name: string, text: string): string {
  const longest = Math.max(0, ...[...text.matchAll(/`+/g)].map((m) => m[0].length));
  const ticks = "`".repeat(Math.max(3, longest + 1));
  const lang = languages[name.split(".").pop()?.toLowerCase() ?? ""] ?? "";
  return `\`${name}\`:\n${ticks}${lang}\n${text}${text.endsWith("\n") ? "" : "\n"}${ticks}`;
}

/** The text a user message sends: its files, in order, then what was typed. */
export function content(message: Pick<Message, "text" | "attachments">): string {
  const files = (message.attachments ?? []).map((a) => fence(a.name, a.text));
  return [...files, message.text].filter(Boolean).join("\n\n");
}
