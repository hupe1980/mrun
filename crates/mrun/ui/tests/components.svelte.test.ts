import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import BudgetBar from "../src/components/BudgetBar.svelte";
import Markdown from "../src/components/Markdown.svelte";
import Message from "../src/components/Message.svelte";
import ReuseRibbon from "../src/components/ReuseRibbon.svelte";
import { render, violations } from "./dom";

describe("reuse ribbon (SC-002)", () => {
  it("shows the server's numbers and cause exactly", async () => {
    const { target } = render(ReuseRibbon, {
      cached: 14212,
      total: 14380,
      miss: { cached: 14212, processed: 168, cause: "diverged", divergence: 14000, message: { index: 3, role: "user" }, unreused: 0 },
    });
    expect(target.textContent).toContain("14,212 of 14,380");
    expect(target.textContent).toContain("168 computed · edited at message 4 (user)");
    expect(target.querySelector("[role=img]")?.getAttribute("aria-label")).toBe(
      "14,212 of 14,380 prompt tokens reused; 168 computed because of edited at message 4 (user).",
    );
    expect(await violations()).toEqual([]);
  });
});

describe("budget bar", () => {
  it("reads as a sentence and names every part", async () => {
    const gib = 1024 ** 3;
    const { target } = render(BudgetBar, { budget: 32 * gib, weights: 19 * gib, scratch: gib, context: gib / 2, checkpoints: 0 });
    const label = target.querySelector("[role=img]")?.getAttribute("aria-label") ?? "";
    expect(label).toContain("20.5 GB of 32 GB in use");
    expect(label).toContain("Model 19 GB");
    expect(target.textContent).toContain("Free");
    expect(await violations()).toEqual([]);
  });
});

describe("markdown from the model", () => {
  it("never injects markup", () => {
    const { target } = render(Markdown, { text: 'Hello <img src=x onerror="alert(1)"> **bold** `code`\n\n```js\nlet a = 1;\n```' });
    expect(target.querySelector("img")).toBeNull();
    expect(target.textContent).toContain('<img src=x onerror="alert(1)">');
    expect(target.querySelector("strong")?.textContent).toBe("bold");
    expect(target.querySelector("pre code")?.textContent).toBe("let a = 1;");
  });
});

describe("a reply that shows its work (019 US1)", () => {
  afterEach(() => vi.useRealTimers());

  it("ticks seconds and tokens while thinking, with a preview line (FR-001, SC-001)", () => {
    vi.useFakeTimers();
    vi.setSystemTime(100_000);
    const message = $state({ role: "assistant" as const, text: "", reasoning: "First idea.\nChecking the cache rule", reasoning_started: 100_000, reasoning_tokens: 1104 });
    const { target } = render(Message, { message, streaming: true });
    expect(target.querySelector(".thought")?.textContent).toContain("Thinking · 0 s · 1,104 tokens");
    expect(target.querySelector(".preview")?.textContent).toBe("Checking the cache rule");
    vi.advanceTimersByTime(23_000);
    message.reasoning_tokens = 1500;
    flushSync();
    expect(target.querySelector(".thought")?.textContent).toContain("Thinking · 23 s · 1,500 tokens");
  });

  it("offers Copy and Regenerate on the last reply, Edit on a question (FR-002, FR-003, SC-002)", async () => {
    let regenerated = 0;
    const reply = render(Message, { message: { role: "assistant", text: "Done." }, last: true, onregenerate: () => regenerated++ });
    const labels = [...reply.target.querySelectorAll("button")].map((b) => b.getAttribute("aria-label"));
    expect(labels).toEqual(["Copy reply", "Regenerate reply"]);
    (reply.target.querySelector("[aria-label='Regenerate reply']") as HTMLButtonElement).click();
    expect(regenerated).toBe(1);
    expect(await violations()).toEqual([]);
    reply.destroy();

    let sent = "";
    const question = $state({ editing: false });
    const user = render(Message, {
      message: { role: "user", text: "Wrte a haiku" },
      get editing() { return question.editing; },
      onedit: () => (question.editing = true),
      oncommit: (t: string) => (sent = t),
      oncancel: () => (question.editing = false),
    } as any);
    (user.target.querySelector("[aria-label='Edit question']") as HTMLButtonElement).click();
    flushSync();
    const field = user.target.querySelector("textarea")!;
    expect(field.value).toBe("Wrte a haiku");
    field.value = "Write a haiku";
    field.dispatchEvent(new Event("input"));
    field.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    expect(sent).toBe("Write a haiku");
    expect(await violations()).toEqual([]);
  });
});

describe("chat message stats (FR-008)", () => {
  it("leads with reuse, then timing, then why it stopped", async () => {
    const { target } = render(Message, {
      message: {
        role: "assistant",
        text: "Done.",
        reasoning: "Thinking about it",
        reasoning_ms: 2400,
        stats: { prompt_tokens: 14380, cached_tokens: 14212, ttft_ms: 310, decode_tps: 42.13, stop: "end" },
      },
    });
    const stats = target.querySelector(".stats")?.textContent?.replace(/\s+/g, " ").trim();
    expect(stats).toBe("14,212 of 14,380 reused · 0.31 s to first token · 42.1 tok/s · finished");
    expect(target.textContent).toContain("Thought for 2.4 s");
    expect(target.querySelector(".reasoning")).toBeNull();
    expect(await violations()).toEqual([]);
  });

  it("adds tokens per pass when the reply was drafted (006)", async () => {
    const { target } = render(Message, {
      message: {
        role: "assistant",
        text: "Done.",
        stats: { prompt_tokens: 900, cached_tokens: 0, ttft_ms: 310, decode_tps: 61.2, stop: "end", speculation: { drafted: 120, accepted: 96, passes: 40 } },
      },
    });
    const stats = target.querySelector(".stats")?.textContent?.replace(/\s+/g, " ").trim();
    expect(stats).toBe("0 of 900 reused · 0.31 s to first token · 61.2 tok/s · 3.4 tokens per pass · finished");
    expect(target.querySelector(".stats [title]")?.getAttribute("title")).toBe("96 of 120 drafted tokens kept");
    expect(await violations()).toEqual([]);
  });
});
