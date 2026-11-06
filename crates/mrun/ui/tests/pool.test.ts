import { describe, expect, it } from "vitest";
import { backMacs, fitsPool, hostRing, linkLabel, memberWord, owner, placeWords, planFor, ring, takePlan } from "../src/lib/pool";

describe("the pool's ring (026)", () => {
  const stages = [
    { mac_id: "ben", name: "Ben", layers: 24 },
    { mac_id: "anna", name: "Anna", layers: 24 },
    { mac_id: "chris", name: "Chris", layers: 34 },
    { mac_id: "dana", name: "Dana", layers: 12 },
  ];

  it("puts the host at 12 o'clock and sizes arcs by layers", () => {
    const r = ring(stages, 100, 100, 80);
    expect(r.tiles[0]!.angle).toBe(0);
    expect(r.tiles[0]!.x).toBeCloseTo(100);
    expect(r.tiles[0]!.y).toBeCloseTo(20);
    const len = (i: number) => r.arcs[i]!.end - r.arcs[i]!.start;
    expect(len(2)).toBeGreaterThan(len(0));
    expect(len(3)).toBeLessThan(len(0));
    expect(len(0)).toBeCloseTo(len(1));
    // The whole circle: arcs plus a gap around each tile.
    expect(r.arcs[3]!.end + 13).toBeCloseTo(360);
    // A chip for each member's link to the host, none for the host.
    expect(r.joints).toHaveLength(3);
  });

  it("scales from two to eight Macs", () => {
    expect(ring(stages.slice(0, 2), 0, 0, 10).arcs).toHaveLength(2);
    const eight = Array.from({ length: 8 }, (_, i) => ({ mac_id: `m${i}`, name: "", layers: 10 }));
    const r = ring(eight, 0, 0, 10);
    expect(r.tiles.map((t) => Math.round(t.angle))).toEqual([0, 45, 90, 135, 180, 225, 270, 315]);
    expect(ring([], 0, 0, 10).arcs).toEqual([]);
  });

  it("says links, states and owners in words", () => {
    expect(linkLabel("thunderbolt", 0.12)).toBe("Thunderbolt · 0.1 ms");
    expect(linkLabel("wifi", 3.8)).toBe("Wi-Fi · 4 ms");
    expect(memberWord("lost")).toBe("Stopped answering");
    expect(owner("Anna’s MacBook Pro")).toBe("Anna");
    expect(owner("Studio")).toBe("Studio");
  });
});

describe("the pool beyond the ring (026 US3, US4)", () => {
  const gb = 1e9;
  it("marks a model the Macs together hold, and never for one Mac", () => {
    expect(fitsPool(85 * gb, [64 * gb, 64 * gb])).toBe(true);
    expect(fitsPool(100 * gb, [64 * gb, 64 * gb])).toBe(false);
    expect(fitsPool(20 * gb, [64 * gb])).toBe(false);
  });

  it("names the Macs back that the plan leaves out", () => {
    const members = [
      { mac_id: "anna", name: "Anna's MacBook Pro", state: "joined" },
      { mac_id: "chris", name: "Chris's Mac mini", state: "serving" },
      { mac_id: "dana", name: "Dana's MacBook Air", state: "lost" },
    ];
    const stages = [{ mac_id: "ben" }, { mac_id: "chris" }];
    expect(backMacs(members, stages)).toEqual(["Anna's MacBook Pro"]);
    expect(backMacs(members, undefined)).toEqual([]);
  });

  it("builds the host's ring with each Mac's layers and link", () => {
    const macs = hostRing(
      { mac_id: "ben", name: "Ben", chip: "", layers: 0 },
      [{ mac_id: "anna", name: "Anna", chip: "Apple M4 Pro", memory_bytes: 48 * gb, power: "adapter", state: "serving" }],
      [
        { mac_id: "ben", first: 0, end: 30 },
        { mac_id: "anna", first: 30, end: 48, link: { kind: "thunderbolt", rtt_ms_p50: 0.2 } },
      ],
    );
    expect(macs.map((m) => m.layers)).toEqual([30, 18]);
    expect(macs[0]!.host).toBe(true);
    expect(macs[1]!.link).toEqual({ kind: "thunderbolt", ms: 0.1 });
  });

  it("says a member's place in counts only", () => {
    expect(placeWords({ yours_running: true, yours_waiting: 0, others_waiting: 3 })).toBe("Yours runs now");
    expect(placeWords({ yours_running: false, yours_waiting: 2, others_waiting: 1 })).toBe("2 of yours waiting · 1 of others' in the queue");
    expect(placeWords({ yours_running: false, yours_waiting: 0, others_waiting: 4 })).toBeNull();
    expect(memberWord("downloading")).toBe("Downloading its layers");
  });

  it("hands a plan from Models to Pool once", () => {
    planFor("qwen3-235b");
    expect(takePlan()).toBe("qwen3-235b");
    expect(takePlan()).toBeNull();
  });
});
