// 020: files in the chat, adding a model by name, snippets, export and search.
import { describe, expect, it } from "vitest";
import { content, estimate, fence, MAX_BYTES, readAttachment } from "../src/lib/attach";
import { exportMarkdown, repoFromInput, searchHit } from "../src/lib/format";
import { apis, langs, snippet } from "../src/lib/snippets";
import type { Chat } from "../src/lib/types";
import golden from "./fixtures/chat.md?raw";

const file = (name: string, body: BlobPart, type = "") => new File([body], name, { type });

describe("files in the chat (US1, FR-003, FR-004)", () => {
  it("reads text and refuses what it can't send, each with its reason (SC-003)", async () => {
    const ok = await readAttachment(file("parser.rs", "fn main() {}\n"));
    expect(ok).toMatchObject({ ok: true, attachment: { name: "parser.rs", text: "fn main() {}\n", estimated: true } });
    const reasons = await Promise.all([
      readAttachment(file("shot.png", new Uint8Array([137, 80]), "image/png")),
      readAttachment(file("paper.pdf", "%PDF-1.7")),
      readAttachment(file("a.bin", new Uint8Array([1, 0, 2]))),
      readAttachment(file("latin1.txt", new Uint8Array([0xe9, 0x41]))),
      readAttachment(file("huge.log", new Uint8Array(MAX_BYTES + 1).fill(65))),
    ]);
    expect(reasons.map((r) => (r.ok ? "ok" : r.reason))).toEqual([
      "Images need a vision model; mrun's models read text only for now.",
      "PDFs aren't read; paste the text instead.",
      "Not a text file.",
      "Not a text file (not UTF-8).",
      "Larger than 2 MB.",
    ]);
    expect(estimate(35)).toBe(10);
  });

  it("fences a file longer than any backtick run inside, named, before the typed text", () => {
    expect(fence("a.md", "use ```js\nx\n```")).toBe("`a.md`:\n````markdown\nuse ```js\nx\n```\n````");
    expect(fence("Makefile", "all:\n")).toBe("`Makefile`:\n```\nall:\n```");
    const msg = { text: "Why?", attachments: [{ name: "x.py", text: "print(1)", bytes: 8, tokens: 3, estimated: false }] };
    expect(content(msg)).toBe("`x.py`:\n```python\nprint(1)\n```\n\nWhy?");
    expect(content({ text: "plain" })).toBe("plain");
  });
});

describe("adding a model by name (US2, FR-006)", () => {
  it("takes repo ids and Hugging Face links, and nothing else", () => {
    expect(repoFromInput(" mlx-community/Qwen3-8B-4bit ")).toBe("mlx-community/Qwen3-8B-4bit");
    expect(repoFromInput("org/name@abc123")).toBe("org/name@abc123");
    expect(repoFromInput("https://huggingface.co/mlx-community/Qwen3-8B-4bit")).toBe("mlx-community/Qwen3-8B-4bit");
    expect(repoFromInput("https://huggingface.co/org/name/tree/main/")).toBe("org/name@main");
    for (const bad of ["qwen3.6", "--help", "org/name; rm -rf /", "https://evil.example/org/name", "org/name/extra", ""]) {
      expect(repoFromInput(bad), bad).toBeNull();
    }
  });
});

describe("your code (US3, FR-007)", () => {
  it("fills every snippet with the live port and model", () => {
    for (const a of apis) {
      for (const l of langs) {
        const s = snippet(a.id, l.id, 9090, "qwen3.6");
        expect(s, `${a.id}/${l.id}`).toContain("http://127.0.0.1:9090");
        expect(s, `${a.id}/${l.id}`).toContain('"qwen3.6"');
      }
      expect(snippet(a.id, "curl", 9090, "m")).toContain(a.path);
    }
  });

  it("gives curl bodies that a shell passes through as valid JSON", () => {
    for (const a of apis) {
      const body = /-d '([^']*)'$/s.exec(snippet(a.id, "curl", 8484, "qwen3.6"));
      expect(body, a.id).not.toBeNull();
      expect(() => JSON.parse(body![1]!), a.id).not.toThrow();
    }
  });
});

const chat: Chat = {
  id: "0123456789abcdef",
  title: "Why the parser fails",
  created: "2026-11-01T10:00:00Z",
  updated: "2026-11-01T10:05:00Z",
  model: "qwen3.6",
  system: "Answer briefly.",
  sampling: {},
  thinking: null,
  messages: [
    { role: "user", text: "Why does this fail on empty input?", attachments: [{ name: "parser.rs", text: "fn parse(s: &str) -> char { s.chars().next().unwrap() }\n", bytes: 58, tokens: 19, estimated: false }] },
    {
      role: "assistant",
      text: "`next()` returns `None` for an empty string, and `unwrap` panics.",
      reasoning: "Empty string, so the iterator is empty.",
      reasoning_ms: 1200,
      stats: { prompt_tokens: 120, cached_tokens: 0, ttft_ms: 310, decode_tps: 41.2, stop: "end" },
    },
    { role: "user", text: "Fix it." },
  ],
};

describe("export and search (US4, FR-009, FR-010)", () => {
  it("exports a chat as the golden Markdown (SC-006)", () => {
    expect(exportMarkdown(chat, fence)).toBe(golden);
  });

  it("finds text inside messages and files, with an excerpt", () => {
    expect(searchHit(chat, "")).toEqual({ excerpt: null });
    expect(searchHit(chat, "PARSER")).toEqual({ excerpt: null });
    expect(searchHit(chat, "unwrap")?.excerpt).toContain("unwrap");
    expect(searchHit(chat, "chars()")?.excerpt).toContain("chars()");
    expect(searchHit(chat, "tokenizer")).toBeNull();
  });
});
