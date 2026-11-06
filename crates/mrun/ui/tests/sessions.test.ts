import { describe, expect, it } from "vitest";
import { hint } from "../src/lib/format";
import { changed, fitWidth, layout, merge, move, rowLabel, summary } from "../src/lib/sessions";
import type { ReplayRecord, Session } from "../src/lib/types";

const record = (id: number, row: string, over: Partial<ReplayRecord> = {}): ReplayRecord => ({
  id, session: "s1", row, client: "claude-cli", ended_ms: id * 1000, attention: false, excerpt: false,
  api: "messages", prompt_tokens: 1000 * id, cached: 900 * id, processed: 100 * id, generated: 10,
  ttft_ms: 100 * id, decode_tps: 40, stop: "end", cause: null, divergence: null, message: null, unreused: 0,
  ...over,
});

const session: Session = {
  id: "s1", client: "claude-cli", label: "Claude Code", first_ms: 1000, last_ms: 9000, requests: 6,
  prompt_tokens: 21000, cached_tokens: 18900, attention: 1,
  rows: [{ id: "main", kind: "main", requests: 3 }, { id: "sub1", kind: "subagent", requests: 2 }, { id: "side", kind: "side", requests: 1 }],
};

const records = [
  record(1, "main"),
  record(2, "side"),
  record(3, "sub1"),
  record(4, "sub1", { attention: true }),
  // Five minutes later.
  record(5, "main", { ended_ms: 4000 + 5 * 60_000 }),
  record(6, "main", { ended_ms: 4000 + 5 * 60_000 + 500, attention: true }),
];

describe("session replay layout (025)", () => {
  it("merges polls by id, oldest first, bounded like the server", () => {
    const kept = merge([record(1, "main"), record(2, "main")], [record(2, "main", { cached: 1 }), record(3, "main")]);
    expect(kept.map((r) => r.id)).toEqual([1, 2, 3]);
    expect(kept[1]!.cached).toBe(1);
    expect(merge(kept, [record(4, "main")], 2).map((r) => r.id)).toEqual([3, 4]);
  });

  it("puts each row in its band, columns in time order, and idle gaps as breaks", () => {
    const l = layout(session, records, 6);
    expect(l.bands.map((b) => b.label)).toEqual(["Main", "Subagent 1", "Side requests"]);
    expect(l.columns.map((c) => c.row)).toEqual([0, 2, 1, 1, 0, 0]);
    expect(l.columns[1]!.x).toBe(8);
    expect(l.breaks).toEqual([{ x: 32, minutes: 5 }]);
    expect(l.columns[4]!.x).toBe(32 + 64);
    // One scale per session: the tallest request is 64 px, reused below computed.
    expect(l.columns[5]!.height).toBe(64);
    expect(l.columns[5]!.reused).toBeCloseTo(57.6);
    // Bands stack: the main band holds its tallest column.
    expect(l.bands[1]!.y).toBe(l.bands[0]!.height);
    expect(l.bands[0]!.height).toBeGreaterThanOrEqual(64);
    // Slowest first token strongest.
    expect(l.columns[0]!.wait).toBe(0);
    expect(l.columns[5]!.wait).toBe(1);
  });

  it("fits a session into the visible width", () => {
    expect(fitWidth(session, records, 600)).toBe(24);
    expect(fitWidth(session, records, 100)).toBe(4);
  });

  it("moves along rows, across rows and to the misses worth fixing", () => {
    const { columns } = layout(session, records, 6);
    expect(move(columns, 0, "right")).toBe(4);
    expect(move(columns, 4, "left")).toBe(0);
    expect(move(columns, 0, "left")).toBe(0);
    expect(move(columns, 4, "down")).toBe(3);
    expect(move(columns, 3, "down")).toBe(1);
    expect(move(columns, 1, "down")).toBe(1);
    expect(move(columns, 0, "next")).toBe(3);
    expect(move(columns, 3, "next")).toBe(5);
    expect(move(columns, 5, "next")).toBe(5);
    expect(move(columns, 5, "previous")).toBe(3);
    expect(move(columns, null, "next")).toBe(3);
    expect(move([], null, "right")).toBeNull();
  });

  it("says a session in one line and names rows", () => {
    expect(summary(session)).toBe("6 requests · 90 % of prompt tokens reused · 1 worth fixing");
    expect(rowLabel("conversation", "c2")).toBe("Conversation 2");
  });

  it("marks only what changed between the cached and the new text", () => {
    const before = '12.4s"}, {"role": "assistant"';
    const after = '11.9s"}, {"role": "assistant"';
    const at = changed(`build ok in ${before}`, `build ok in ${after}`);
    expect(at).toEqual({ shared: 13, before: 16, after: 16 });
    expect(changed("abc", "xyz")).toEqual({ shared: 0, before: 3, after: 3 });
    expect(changed("same text here", "same text here")).toEqual({ shared: 14, before: 14, after: 14 });
    // An insertion: nothing removed, the new words marked.
    expect(changed("run the tests now please", "run the unit tests now please")).toEqual({ shared: 8, before: 8, after: 13 });
  });

  it("gives a hint for each cause", () => {
    expect(hint({ cause: "diverged", message: { index: 3, role: "tool" } })).toContain("tool result");
    expect(hint({ cause: "evicted", message: null })).toContain("Memory ran short");
  });
});
