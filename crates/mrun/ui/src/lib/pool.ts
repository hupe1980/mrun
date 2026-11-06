// The pool's ring (026 R7): Macs as tiles on a circle, the model's layers as the circle's arc
// (one segment per Mac, sized by its layers), link chips at the joints. Pure, so the component
// only draws and the tests check geometry.

/** Degrees of empty arc around each tile. */
const TILE_GAP = 26;

export interface RingStage {
  mac_id: string;
  name: string;
  layers: number;
}

export interface Arc {
  mac_id: string;
  /** Degrees, clockwise from 12 o'clock. */
  start: number;
  end: number;
  /** SVG path of the arc. */
  d: string;
}

export interface Tile {
  mac_id: string;
  angle: number;
  x: number;
  y: number;
}

export interface Joint {
  /** The link into this Mac (from the previous one). */
  mac_id: string;
  x: number;
  y: number;
}

export interface Ring {
  arcs: Arc[];
  tiles: Tile[];
  joints: Joint[];
}

function point(cx: number, cy: number, r: number, degrees: number): [number, number] {
  const rad = ((degrees - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

function arcPath(cx: number, cy: number, r: number, start: number, end: number): string {
  const [x1, y1] = point(cx, cy, r, start);
  const [x2, y2] = point(cx, cy, r, end);
  const large = end - start > 180 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

/**
 * The ring of `stages` (stage order, the host first at 12 o'clock) around (cx, cy) with radius
 * r: each Mac's tile sits at the start of its arc, the arc's length follows its layers.
 */
export function ring(stages: RingStage[], cx: number, cy: number, r: number): Ring {
  const n = stages.length;
  if (!n) return { arcs: [], tiles: [], joints: [] };
  const total = stages.reduce((a, s) => a + Math.max(1, s.layers), 0);
  const free = 360 - n * TILE_GAP;
  let at = 0;
  const arcs: Arc[] = [];
  const tiles: Tile[] = [];
  const joints: Joint[] = [];
  for (const s of stages) {
    const [tx, ty] = point(cx, cy, r, at);
    tiles.push({ mac_id: s.mac_id, angle: at, x: tx, y: ty });
    const start = at + TILE_GAP / 2;
    const end = start + (free * Math.max(1, s.layers)) / total;
    arcs.push({ mac_id: s.mac_id, start, end, d: arcPath(cx, cy, r, start, end) });
    at = end + TILE_GAP / 2;
  }
  // A Mac's link chip sits inside the ring, between its tile and the centre.
  if (n > 1) {
    for (let i = 1; i < n; i++) {
      const [x, y] = point(cx, cy, r * 0.62, tiles[i]!.angle);
      joints.push({ mac_id: stages[i]!.mac_id, x, y });
    }
  }
  return { arcs, tiles, joints };
}

/** A Mac's colour on the ring: a fixed sequence, the host first. */
export const COLORS = ["--pool-1", "--pool-2", "--pool-3", "--pool-4", "--pool-5", "--pool-6", "--pool-7", "--pool-8"];

/** "Thunderbolt · 0.1 ms" for a link chip. */
export function linkLabel(kind: string, ms: number): string {
  const word = ({ thunderbolt: "Thunderbolt", ethernet: "Ethernet", wifi: "Wi-Fi", loopback: "This Mac" } as Record<string, string>)[kind] ?? kind;
  return `${word} · ${ms < 1 ? ms.toFixed(1) : Math.round(ms)} ms`;
}

/** What a member's state means, in words. */
export function memberWord(state: string): string {
  switch (state) {
    case "downloading":
      return "Downloading its layers";
    case "loading":
      return "Loading its layers";
    case "connecting":
      return "Joining the ring";
    case "serving":
      return "Computing for the pool";
    case "failed":
      return "Could not load";
    case "lost":
      return "Stopped answering";
    default:
      return "Ready to lend";
  }
}

/** "Ben" from "Ben's MacBook Pro". */
export function owner(name: string): string {
  const m = /^(.+?)[’']s /.exec(name);
  return m ? m[1]! : name;
}

/** A Mac as the ring draws it. */
export interface RingMac {
  mac_id: string;
  name: string;
  chip: string;
  /** Layers this Mac runs (0 before a plan). */
  layers: number;
  first?: number;
  end?: number;
  /** `used / budget` bytes, when known. */
  used?: number;
  budget?: number;
  power?: string;
  state?: string;
  link?: { kind: string; ms: number } | null;
  host?: boolean;
}

/** The share of a Mac's memory a plan can use, as Apple's GPU working set allows (about 75 %). */
const USABLE = 0.75;

/**
 * Whether a model of `weights` bytes could fit the pool: the Macs' usable memory together holds
 * it with room to spare for context. A hint for Models; the plan gives the verdict.
 */
export function fitsPool(weights: number, memories: number[]): boolean {
  if (memories.length < 2 || weights <= 0) return false;
  const usable = memories.reduce((a, m) => a + m * USABLE, 0);
  return weights * 1.1 <= usable;
}

/** Macs back in the pool that the plan it serves does not include (Re-plan takes them in). */
export function backMacs(members: { mac_id: string; name: string; state: string }[], stages: { mac_id: string }[] | undefined): string[] {
  if (!stages?.length) return [];
  return members.filter((m) => m.state !== "lost" && !stages.some((s) => s.mac_id === m.mac_id)).map((m) => m.name);
}

let planned: string | null = null;

/** Models asks Pool to show the plan for `model` (Pool takes it when it opens). */
export function planFor(model: string): void {
  planned = model;
}

/** The model Models asked a plan for, once. */
export function takePlan(): string | null {
  const model = planned;
  planned = null;
  return model;
}

/** The host's ring: this Mac first, then its members, each with its layers in `stages`. */
export function hostRing(
  me: RingMac,
  members: { mac_id: string; name: string; chip: string; memory_bytes: number; power: string; state: string }[],
  stages: { mac_id: string; first: number; end: number; link?: { kind: string; rtt_ms_p50: number } | null }[] | undefined,
): RingMac[] {
  const at = (id: string | null) => (stages ?? []).find((s) => s.mac_id === id);
  const mine = at(me.mac_id) ?? stages?.[0];
  const macs: RingMac[] = [{ ...me, host: true, layers: mine ? mine.end - mine.first : 0, first: mine?.first, end: mine?.end }];
  for (const m of members) {
    const s = at(m.mac_id);
    const link = s?.link ? { kind: s.link.kind, ms: s.link.rtt_ms_p50 / 2 } : null;
    macs.push({ mac_id: m.mac_id, name: m.name, chip: m.chip, layers: s ? s.end - s.first : 0, first: s?.first, end: s?.end, power: m.power, state: m.state, budget: m.memory_bytes, link });
  }
  return macs;
}

/** A member's place in the pool's queue, in words; `null` when nothing of its waits or runs. */
export function placeWords(place: { yours_running: boolean; yours_waiting: number; others_waiting: number } | null | undefined): string | null {
  if (!place) return null;
  if (place.yours_running) return place.yours_waiting ? `Yours runs now · ${place.yours_waiting} more of yours waiting` : "Yours runs now";
  if (place.yours_waiting) return `${place.yours_waiting} of yours waiting · ${place.others_waiting} of others' in the queue`;
  return null;
}
