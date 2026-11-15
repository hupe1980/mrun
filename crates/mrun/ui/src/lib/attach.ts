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

/** Image types the server reads (009 FR-001) and the most bytes one may have. */
export const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const TOKEN_PX = 32;
const MIN_TOKENS = 64;
const MAX_TOKENS = 2560;

/** The server's `resize_v1` grid (009 FR-002): tokens of a `width` × `height` image. */
export function imageTokens(width: number, height: number): number {
  const f = TOKEN_PX;
  const round = (v: number) => Math.max(f, Math.round(v / f) * f);
  let [hb, wb] = [round(height), round(width)];
  if (hb * wb > MAX_TOKENS * f * f) {
    const beta = Math.sqrt((height * width) / (MAX_TOKENS * f * f));
    hb = Math.max(f, Math.floor(height / beta / f) * f);
    wb = Math.max(f, Math.floor(width / beta / f) * f);
  } else if (hb * wb < MIN_TOKENS * f * f) {
    const beta = Math.sqrt((MIN_TOKENS * f * f) / (height * width));
    hb = Math.ceil((height * beta) / f) * f;
    wb = Math.ceil((width * beta) / f) * f;
  }
  return (hb / f) * (wb / f);
}

async function readImage(file: File, name: string): Promise<Read> {
  if (file.size > MAX_IMAGE_BYTES) return { ok: false, name, reason: "Larger than 20 MB." };
  const bytes = new Uint8Array(await file.arrayBuffer());
  let size: { width: number; height: number } | null = null;
  try {
    const bitmap = await createImageBitmap(new Blob([bytes], { type: file.type }));
    size = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
  } catch {
    size = null;
  }
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return {
    ok: true,
    attachment: {
      name,
      text: "",
      bytes: file.size,
      tokens: size ? imageTokens(size.width, size.height) : MAX_TOKENS,
      estimated: !size,
      image: { media: file.type, data: btoa(binary) },
    },
  };
}

/** Reads a dropped, pasted or chosen file, or says why it can't go in; images only when the
 *  served model reads them. */
export async function readAttachment(file: File, vision = false): Promise<Read> {
  const name = file.name.split(/[\\/]/).pop()!.slice(0, 255) || "file";
  const refuse = (reason: string): Read => ({ ok: false, name, reason });
  if (file.type.startsWith("image/")) {
    if (!vision) return refuse("The served model reads no images.");
    if (!IMAGE_TYPES.includes(file.type)) return refuse("Only PNG, JPEG, WebP and GIF images are read.");
    return readImage(file, name || "image");
  }
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
  const files = (message.attachments ?? []).filter((a) => !a.image).map((a) => fence(a.name, a.text));
  return [...files, message.text].filter(Boolean).join("\n\n");
}

/** A user message's content for the Messages API: its images first, then its text (009). */
export function blocks(message: Pick<Message, "text" | "attachments">): string | Record<string, unknown>[] {
  const images = (message.attachments ?? []).filter((a) => a.image);
  if (!images.length) return content(message);
  const text = content(message);
  return [
    ...images.map((a) => ({ type: "image", source: { type: "base64", media_type: a.image!.media, data: a.image!.data } })),
    ...(text ? [{ type: "text", text }] : []),
  ];
}
