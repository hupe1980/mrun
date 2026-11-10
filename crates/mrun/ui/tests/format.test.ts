import { describe, expect, it } from "vitest";
import { parseFrame } from "../src/lib/api";
import { bytes, cause, fittingGpus, keptWords, needGb, percent, perHour, perPass, phaseLine, reuseSentence, spend, stopWord, stopsIn, tokens, tone, uptime } from "../src/lib/format";

describe("tones (019 FR-005, SC-003)", () => {
  it("amber only for what nobody intended", () => {
    const at = (role: string) => ({ index: 3, role });
    const expected = [
      { cause: null, message: null },
      { cause: "new", message: null },
      { cause: "gap", message: null },
      { cause: "diverged", message: at("user") },
      // A running lane carries no role: no evidence, no amber.
      { cause: "diverged", message: null },
      // A follow-up's re-rendered reply, resumed at its header (021).
      { cause: "diverged", message: at("assistant"), unreused: 3 },
    ] as const;
    const attention = [
      { cause: "evicted", message: null },
      { cause: "disabled", message: null },
      // The same, recomputed from the user message's start (020 R9).
      { cause: "diverged", message: at("assistant"), unreused: 7981 },
      { cause: "diverged", message: at("tool") },
      { cause: "diverged", message: at("system") },
    ] as const;
    for (const m of expected) expect(tone(m as any), JSON.stringify(m)).toBe("expected");
    for (const m of attention) expect(tone(m as any), JSON.stringify(m)).toBe("attention");
    expect(cause({ cause: "diverged", message: at("tool"), divergence: 1 })).toBe("client changed message 4 (tool)");
    expect(cause({ cause: "diverged", message: at("assistant"), divergence: 1 })).toBe("reply re-rendered at message 4 (assistant)");
  });

  it("groups every count of a thousand or more (SC-004)", () => {
    expect(tokens(221819)).toBe("221,819");
    expect(tokens(999)).toBe("999");
  });
});

describe("words for numbers", () => {
  it("bytes and percentages read as mrun ls prints them", () => {
    expect(bytes(19.5 * 1024 ** 3)).toBe("19.5 GB");
    expect(bytes(812 * 1024 ** 2)).toBe("812 MB");
    expect(percent(14212, 14380)).toBe("99 %");
    expect(percent(14380, 14380)).toBe("100 %");
  });

  it("causes are sentences, with the message where the prompt diverged", () => {
    expect(cause({ cause: "diverged", message: { index: 11, role: "user" }, divergence: 4000 })).toBe("edited at message 12 (user)");
    expect(cause({ cause: "evicted", message: null, divergence: null })).toBe("dropped from memory earlier");
    expect(cause({ cause: null, message: null, divergence: null })).toBe("fully reused");
    expect(
      reuseSentence(100, 150, { cached: 100, processed: 50, cause: "new", divergence: null, message: null, unreused: 0 }),
    ).toBe("100 of 150 prompt tokens reused; 50 computed because of a new conversation.");
  });

  it("progress is real, never a spinner", () => {
    const r = { id: 1, api: "messages", lane: 0, phase: "prefill" as const, prompt_tokens: 30000, cached: 0, progress: 12400, cause: null, unreused: null, generated: 0, decode_tps: null, age_ms: 1 };
    expect(phaseLine(r)).toBe("Reading prompt · 12,400 of 30,000 tokens");
    expect(stopWord("max_tokens")).toBe("reached the length limit");
  });
});

describe("the chat stream", () => {
  it("reads mrun's reuse comment and the Messages events", () => {
    const meta = [...parseFrame(': mrun {"cached":14212,"processed":168,"max_tokens":8192,"cause":"diverged","divergence":14000,"message":{"index":3,"role":"user"},"unreused":40}')];
    expect(meta).toEqual([
      { type: "meta", max_tokens: 8192, miss: { cached: 14212, processed: 168, cause: "diverged", divergence: 14000, message: { index: 3, role: "user" }, unreused: 40 } },
    ]);
    const text = [...parseFrame('event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}')];
    expect(text).toEqual([{ type: "text", text: "Hi" }]);
    const stop = [...parseFrame('event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}')];
    expect(stop).toEqual([{ type: "output", tokens: 12 }, { type: "stop", reason: "end_turn" }]);
    expect([...parseFrame(": keepalive")]).toEqual([]);
  });

  it("reads speculation from the closing comment (006)", () => {
    const end = [...parseFrame(': mrun {"warnings":["tool-args-invalid"],"speculation":{"drafted":120,"accepted":96,"passes":40}}')];
    expect(end).toEqual([{ type: "speculation", speculation: { drafted: 120, accepted: 96, passes: 40 } }]);
  });
});

describe("speculative decoding (006)", () => {
  const s = { drafted: 530, accepted: 412, passes: 200 };

  it("says tokens per pass and what was kept", () => {
    expect(perPass(s)).toBe("3.1 tokens per pass");
    expect(keptWords(s)).toBe("412 of 530 drafted tokens kept");
  });

  it("adds tokens per pass to a generating request", () => {
    const r = { id: 1, api: "messages", lane: 0, phase: "generating" as const, prompt_tokens: 900, cached: 0, progress: 900, cause: null, unreused: null, generated: 120, decode_tps: 61.2, age_ms: 1 };
    expect(phaseLine(r)).toBe("Generating · 61.2 tok/s · 120 tokens");
    expect(phaseLine({ ...r, speculation: s })).toBe("Generating · 61.2 tok/s · 120 tokens · 3.1 tokens per pass");
  });
});

describe("cloud (024)", () => {
  const gpu = (id: string, memory_gb: number, secure: number | null, stock: string | null, community: number | null = null) => ({
    id,
    name: id,
    memory_gb,
    secure_per_hr: secure,
    community_per_hr: community,
    secure_stock: stock,
    community_stock: community ? "High" : null,
  });

  it("prices, spend and uptime read as sentences", () => {
    expect(perHour(0.857)).toBe("$0.86/h");
    expect(spend(0.6)).toBe("$0.60");
    expect(spend(0.004)).toBe("under $0.01");
    expect(uptime(30)).toBe("under a minute");
    expect(uptime(42 * 60)).toBe("42 min");
    expect(uptime(125 * 60)).toBe("2 h 5 min");
    expect(uptime(120 * 60)).toBe("2 h");
    expect(stopsIn(12 * 60)).toBe(18);
    expect(stopsIn(4000)).toBe(0);
    expect(stopsIn(null)).toBeNull();
  });

  it("an added model needs its weights, a fifth more and 4 GB", () => {
    expect(needGb(30e9)).toBe(40);
  });

  it("fitting GPUs: in stock first, then cheapest; the catalog's ids win over memory", () => {
    const gpus = [gpu("A100", 80, 1.19, "Low"), gpu("L40S", 48, 0.86, "High"), gpu("H100", 80, 0.5, null), gpu("A5000", 24, 0.16, "High", 0.11)];
    expect(fittingGpus(gpus, "secure", { need: 40 }).map((c) => c.gpu.id)).toEqual(["L40S", "A100", "H100"]);
    expect(fittingGpus(gpus, "secure", { fits: ["A100", "A5000"] }).map((c) => c.gpu.id)).toEqual(["A5000", "A100"]);
    expect(fittingGpus(gpus, "community", { need: 10 }).map((c) => [c.gpu.id, c.price])).toEqual([["A5000", 0.11]]);
  });
});

describe("context length (027)", () => {
  it("offers powers of two up to what fits and costs fixed + tokens × per_token", async () => {
    const { contextChoices, contextTarget, memoryAt, targetModel } = await import("../src/lib/context");
    expect(contextChoices(100_000)).toEqual([16384, 32768, 65536]);
    expect(contextChoices(null)).toEqual([]);
    expect(memoryAt({ memory: { fixed: 100, per_token: 2 } }, 10)).toBe(120);
    expect(memoryAt({ memory: null }, 10)).toBeNull();
    expect(contextTarget("qwen3.6", 65536)).toBe("qwen3.6|65536");
    expect(contextTarget("qwen3.6", null)).toBe("qwen3.6");
    expect(targetModel("qwen3.6|65536")).toBe("qwen3.6");
  });
});
