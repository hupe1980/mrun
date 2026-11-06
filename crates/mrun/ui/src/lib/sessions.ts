// Session replay's layout and keyboard model (025): pure, so the skyline only draws what this
// computes and the tests check geometry and keys without a DOM.

import { percent } from "./format";
import type { ReplayRecord, RowKind, Session } from "./types";

/** Records the page keeps, like the server. */
export const KEPT = 2000;
/** A gap between two requests longer than this is drawn as an idle break. */
export const IDLE_MS = 30_000;
/** The tallest column, in pixels. */
const TALLEST = 64;
const BREAK_W = 64;
const GAP = 2;
const ROW_PAD = 10;

/** Merges polled records into the kept ones by id, oldest first, at most `cap`. */
export function merge(kept: ReplayRecord[], next: ReplayRecord[], cap = KEPT): ReplayRecord[] {
  if (!next.length) return kept;
  const byId = new Map(kept.map((r) => [r.id, r]));
  for (const r of next) byId.set(r.id, r);
  const all = [...byId.values()].sort((a, b) => a.id - b.id);
  return all.length > cap ? all.slice(all.length - cap) : all;
}

export function rowLabel(kind: RowKind, id: string): string {
  const n = id.replace(/^\D+/, "");
  switch (kind) {
    case "main":
      return "Main";
    case "subagent":
      return `Subagent ${n}`;
    case "side":
      return "Side requests";
    case "conversation":
      return `Conversation ${n}`;
  }
}

export interface Column {
  record: ReplayRecord;
  /** Index in time order across the session. */
  index: number;
  row: number;
  x: number;
  /** Total height (prompt tokens) and the reused part. */
  height: number;
  reused: number;
  /** Time to first token as a share of the session's spread, 0 (fastest) to 1 (slowest). */
  wait: number;
}

export interface Band {
  id: string;
  kind: RowKind;
  label: string;
  y: number;
  height: number;
}

export interface Break {
  x: number;
  minutes: number;
}

export interface Layout {
  width: number;
  height: number;
  columnWidth: number;
  bands: Band[];
  columns: Column[];
  breaks: Break[];
}

/** Rank of each value among `values`, 0..1 (ties share a rank). */
function quantiles(values: (number | null)[]): number[] {
  const known = values.filter((v): v is number => v !== null).sort((a, b) => a - b);
  if (known.length < 2) return values.map(() => 0);
  return values.map((v) => (v === null ? 0 : known.indexOf(v) / (known.length - 1)));
}

/**
 * The skyline of one session: a band per row (as tall as its tallest request on the session's
 * shared scale), a column per request in time order with `columnWidth`, and an idle break for
 * each gap over a 30 s.
 */
export function layout(session: Session, records: ReplayRecord[], columnWidth: number): Layout {
  const mine = records.filter((r) => r.session === session.id);
  // Main first, then the other conversations and subagents as they started, side requests last.
  const order: Record<RowKind, number> = { main: 0, conversation: 1, subagent: 1, side: 2 };
  const rows = session.rows
    .filter((row) => mine.some((r) => r.row === row.id))
    .sort((a, b) => order[a.kind] - order[b.kind]);
  const tallest = Math.max(1, ...mine.map((r) => r.prompt_tokens));
  const scale = TALLEST / tallest;
  let y = 0;
  const bands: Band[] = rows.map((row) => {
    const top = Math.max(...mine.filter((r) => r.row === row.id).map((r) => r.prompt_tokens));
    const band = { id: row.id, kind: row.kind, label: rowLabel(row.kind, row.id), y, height: Math.max(18, top * scale) + ROW_PAD };
    y += band.height;
    return band;
  });
  const waits = quantiles(mine.map((r) => r.ttft_ms));
  const columns: Column[] = [];
  const breaks: Break[] = [];
  let x = 0;
  mine.forEach((record, index) => {
    const prev = mine[index - 1];
    if (prev && record.ended_ms - prev.ended_ms > IDLE_MS) {
      breaks.push({ x, minutes: Math.max(1, Math.round((record.ended_ms - prev.ended_ms) / 60_000)) });
      x += BREAK_W;
    }
    columns.push({
      record,
      index,
      row: Math.max(0, rows.findIndex((row) => row.id === record.row)),
      x,
      height: Math.max(2, record.prompt_tokens * scale),
      reused: Math.min(record.cached, record.prompt_tokens) * scale,
      wait: waits[index] ?? 0,
    });
    x += columnWidth + GAP;
  });
  return { width: Math.max(0, x - GAP), height: y, columnWidth, bands, columns, breaks };
}

/** The column width that fits the whole session in `width` pixels (⌘0), within 2–24 px. */
export function fitWidth(session: Session, records: ReplayRecord[], width: number): number {
  const mine = records.filter((r) => r.session === session.id);
  const breaks = mine.filter((r, i) => i > 0 && r.ended_ms - mine[i - 1]!.ended_ms > IDLE_MS).length;
  const room = width - breaks * BREAK_W;
  return Math.min(24, Math.max(2, Math.floor(room / Math.max(1, mine.length)) - GAP));
}

export type Key = "left" | "right" | "up" | "down" | "next" | "previous" | "first" | "last";

/**
 * The column a key selects from `current` (an index into `columns`): ←/→ the previous or next
 * request in the same row, ↑/↓ the nearest request in time in the row above or below, ⌥→/⌥←
 * the next or previous request worth fixing in any row. Stays put at an edge.
 */
export function move(columns: Column[], current: number | null, key: Key): number | null {
  if (!columns.length) return null;
  if (current === null || !columns[current]) {
    return key === "previous" || key === "left" ? columns.length - 1 : key === "next" ? next(columns, -1, 1) : columns.length - 1;
  }
  const here = columns[current]!;
  switch (key) {
    case "first":
      return 0;
    case "last":
      return columns.length - 1;
    case "left":
    case "right": {
      const step = key === "right" ? 1 : -1;
      for (let i = current + step; i >= 0 && i < columns.length; i += step) if (columns[i]!.row === here.row) return i;
      return current;
    }
    case "up":
    case "down": {
      const row = here.row + (key === "down" ? 1 : -1);
      let best: number | null = null;
      columns.forEach((c, i) => {
        if (c.row === row && (best === null || Math.abs(c.x - here.x) < Math.abs(columns[best]!.x - here.x))) best = i;
      });
      return best ?? current;
    }
    case "next":
      return next(columns, current, 1) ?? current;
    case "previous":
      return next(columns, current, -1) ?? current;
  }
}

function next(columns: Column[], from: number, step: number): number | null {
  for (let i = from + step; i >= 0 && i < columns.length; i += step) if (columns[i]!.record.attention) return i;
  return null;
}

/** "142 requests · 96 % of prompt tokens reused · 2 worth fixing". */
export function summary(session: Session): string {
  const requests = `${session.requests.toLocaleString("en-US")} ${session.requests === 1 ? "request" : "requests"}`;
  const reused = `${percent(session.cached_tokens, session.prompt_tokens)} of prompt tokens reused`;
  const fixing = session.attention ? `${session.attention} worth fixing` : "nothing worth fixing";
  return `${requests} · ${reused} · ${fixing}`;
}

/**
 * Where `before` and `after` differ: the shared start, then the end of the changed part of each,
 * where they agree again for 6 characters with the least changed (else their ends).
 */
export function changed(before: string, after: string): { shared: number; before: number; after: number } {
  const AGREE = 6;
  let shared = 0;
  while (shared < before.length && shared < after.length && before[shared] === after[shared]) shared++;
  const most = before.length + after.length - 2 * shared;
  for (let total = 0; total <= most; total++) {
    for (let a = 0; a <= total; a++) {
      const i = shared + a;
      const j = shared + total - a;
      if (i + AGREE > before.length || j + AGREE > after.length) continue;
      if (before.slice(i, i + AGREE) === after.slice(j, j + AGREE)) return { shared, before: i, after: j };
    }
  }
  return { shared, before: before.length, after: after.length };
}
