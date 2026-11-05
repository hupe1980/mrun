import { flushSync } from "svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { live } from "../src/lib/live.svelte";
import type { CloudListing, ModelEntry, Status } from "../src/lib/types";
import CloudSheet from "../src/components/CloudSheet.svelte";
import Agents from "../src/places/Agents.svelte";
import Models from "../src/places/Models.svelte";
import Now from "../src/places/Now.svelte";
import Settings from "../src/places/Settings.svelte";
import { render, settle, violations } from "./dom";

const gib = 1024 ** 3;
const status: Status = {
  version: "0.1.0",
  model: { id: "qwen3.6", identity: "x", family: "qwen3_5_moe", syntax: "qwen" },
  numerics: { backend: "nuvec", nax: "off", gpu_family: "applegpu_g16s" },
  budget: { budget: 32 * gib, weights: 19.5 * gib, scratch: gib, kv_capacity_tokens: 262144, kv_bytes_per_token: 20480 },
  queued: 0,
  cache: { live_tokens: 14380, buffer_bytes: gib / 4, checkpoint_bytes: gib / 8, prompt_tokens_total: 100000, cached_tokens_total: 92000, enabled: true },
  last: null,
  requests: [
    { id: 9, api: "messages", lane: 0, phase: "prefill", prompt_tokens: 30000, cached: 17600, progress: 24000, cause: null, unreused: 0, generated: 0, decode_tps: null, age_ms: 900 },
  ],
  finished: [
    { id: 8, api: "messages", prompt_tokens: 14380, generated: 210, ttft_ms: 310, decode_tps: 42.1, stop: "end", cached: 14212, processed: 168, cause: "diverged", divergence: 14000, message: { index: 3, role: "user" }, unreused: 0 },
  ],
  lanes: { configured: 1, busy: 1 },
};

// Places fetch on mount; answer with empty listings instead of a real host.
beforeEach(() => {
  vi.stubGlobal("fetch", async () => new Response("[]", { status: 200, headers: { "content-type": "application/json" } }));
});

describe("Now (US1)", () => {
  it("shows the state, the budget, real progress and every request's reuse", async () => {
    live.app = { ui_version: "0.1.0", server: { port: 8484, running: true, started_by: "ui" }, prefs: { keep_chats: null, menu_bar: true, open_at_login: false, server_port: 8484 }, jobs: [] };
    live.status = status;
    live.reachable = true;
    const { target } = render(Now, { onAction: () => {} });
    flushSync();
    const text = target.textContent ?? "";
    expect(text).toContain("qwen3.6 is working");
    expect(text).toContain("Reading prompt · 24,000 of 30,000 tokens");
    // One finished request is not yet a share worth a headline (019 FR-006).
    expect(text).not.toContain("92 %");
    expect(text).toContain("168 computed · edited at message 4 (user)");
    expect(target.querySelector("[role=progressbar]")?.getAttribute("aria-valuenow")).toBe("24000");
    expect(await violations()).toEqual([]);
  });

  it("shows the lifetime share once there is history, never as success green (019 FR-006)", () => {
    const one = status.finished![0]!;
    live.status = { ...status, finished: [one, { ...one, id: 6 }, { ...one, id: 7 }] };
    const { target } = render(Now, { onAction: () => {} });
    flushSync();
    expect(target.textContent).toContain("92 %");
  });

  it("paints only unintended recomputes amber (019 FR-005, SC-003)", () => {
    const one = status.finished![0]!;
    live.status = {
      ...status,
      finished: [
        { ...one, id: 1, cause: "new", message: null, cached: 0, processed: 900, prompt_tokens: 900 },
        { ...one, id: 2, cause: "diverged", message: { index: 6, role: "tool" }, cached: 400, processed: 1980, prompt_tokens: 2380 },
      ],
    };
    const { target } = render(Now, { onAction: () => {} });
    flushSync();
    const ribbons = [...target.querySelectorAll(".recent .ribbon")];
    // Newest first: the tool rewrite, then the new conversation.
    expect(ribbons[0]?.classList.contains("attention")).toBe(true);
    expect(ribbons[0]?.textContent).toContain("1,980 computed · client changed message 7 (tool)");
    expect(ribbons[1]?.classList.contains("attention")).toBe(false);
    expect(ribbons[1]?.textContent).toContain("900 computed · a new conversation");
  });

  it("offers Start when the server is not running", () => {
    live.app = { ui_version: "0.1.0", server: { port: 8484, running: false, started_by: null }, prefs: { keep_chats: null, menu_bar: true, open_at_login: false, server_port: 8484 }, jobs: [] };
    live.status = null;
    live.reachable = false;
    const { target } = render(Now, { onAction: () => {} });
    expect(target.textContent).toContain("mrun is not running");
    expect([...target.querySelectorAll("button")].some((b) => b.textContent?.includes("Start"))).toBe(true);
  });
});

describe("Models (US3)", () => {
  const model = (m: Partial<ModelEntry>): ModelEntry => ({
    name: "qwen3.6", alias: "qwen3.6", repo: "mlx-community/Qwen3.6-35B-A3B-4bit", revision: "abc", size: 20 * gib,
    pulled: false, fit: "fits", verdict: "Fits with 262144 tokens.", max_context: 262144, recommended: false,
    preview: false, serving: false, ...m,
  });

  it("shows the CLI's verdicts verbatim and never loads a model that won't fit (SC-005, US3 AS5)", async () => {
    live.app = { ui_version: "0.1.0", server: { port: 8484, running: true, started_by: "ui" }, prefs: { keep_chats: null, menu_bar: true, open_at_login: false, server_port: 8484 }, jobs: [] };
    live.models = [
      model({ recommended: true, pulled: true, serving: true }),
      model({ name: "qwen3.5", alias: "qwen3.5", pulled: true, fit: "tight", verdict: "Fits with 40960 tokens, less than the 65536 coding agents need." }),
      model({ name: "big", alias: null, repo: "org/big", fit: "wont_fit", verdict: "Does not fit the budget of 36.0 GB." }),
    ];
    const { target } = render(Models, { onError: () => {} });
    const text = target.textContent ?? "";
    expect(text).toContain("Fits with 262144 tokens.");
    expect(text).toContain("Fits with 40960 tokens, less than the 65536 coding agents need.");
    expect(text).toContain("Does not fit the budget of 36.0 GB.");
    expect(text).toContain("Best for this Mac");
    const rows = [...target.querySelectorAll("li")].filter((li) => li.textContent?.includes("Does not fit"));
    const download = [...(rows[0]?.querySelectorAll("button") ?? [])].find((b) => b.textContent?.includes("Download"));
    expect(download?.disabled).toBe(true);
    expect(await violations()).toEqual([]);
  });

  it("marks the served model In Use from the server, with one filled button (019 FR-008, FR-009)", () => {
    live.app = { ui_version: "0.1.0", server: { port: 8484, running: true, started_by: "ui" }, prefs: { keep_chats: null, menu_bar: true, open_at_login: false, server_port: 8484 }, jobs: [] };
    live.status = { ...status, model: { ...status.model!, id: "qwen3.5" } };
    live.reachable = true;
    live.models = [
      model({ recommended: true, pulled: true }),
      model({ name: "qwen3.5", alias: "qwen3.5", pulled: true }),
      model({ name: "other", alias: null, repo: "org/other", pulled: true }),
    ];
    const { target } = render(Models, { onError: () => {} });
    const row = [...target.querySelectorAll("li")].find((li) => li.textContent?.includes("qwen3.5"))!;
    expect(row.textContent).toContain("In Use");
    expect([...row.querySelectorAll("button")].some((b) => b.textContent?.trim() === "Use")).toBe(false);
    expect(target.querySelectorAll(".btn.primary").length).toBe(1);
    expect(target.querySelector("h1")).toBeNull();
  });

  it("shows skeleton rows before the first listing (019 FR-017)", async () => {
    live.models = null;
    const { target } = render(Models, { onError: () => {} });
    const list = target.querySelector("[aria-busy=true]");
    expect(list?.getAttribute("aria-label")).toBe("Loading models");
    expect(list?.querySelectorAll(".skeleton").length).toBeGreaterThan(0);
    expect(await violations()).toEqual([]);
  });

  it("opens the row's actions on right-click, keyboard-navigable (019 FR-012)", async () => {
    live.models = [model({ recommended: true, pulled: true })];
    live.status = null;
    const { target } = render(Models, { onError: () => {} });
    const { default: ContextMenu } = await import("../src/components/ContextMenu.svelte");
    const { mount } = await import("svelte");
    mount(ContextMenu, { target: document.body, props: {} });
    target.querySelector("li")!.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 40, clientY: 40 }));
    flushSync();
    const items = [...document.querySelectorAll("[role=menu] [role=menuitem]")].map((b) => b.textContent);
    expect(items).toEqual(["Use", "Copy as Command", "Remove…"]);
    expect(await violations()).toEqual([]);
  });
});

describe("Agents (019 US4)", () => {
  it("shows the last rows at once and skeletons only before the first listing", () => {
    live.agents = null;
    const first = render(Agents, { onError: () => {} });
    expect(first.target.querySelector("[aria-busy=true]")?.getAttribute("aria-label")).toBe("Loading agents");
    first.destroy();
    live.agents = [{ client: "claude-code", installed: true, connected: true, version: "2.1", refusal: null, files: [] } as any];
    const { target } = render(Agents, { onError: () => {} });
    expect(target.textContent).toContain("Connected to mrun");
    expect(target.querySelector("h1")).toBeNull();
  });
});

describe("Settings (019 FR-013)", () => {
  it("uses switches for every on/off preference", async () => {
    live.app = { ui_version: "0.1.0", server: { port: 8484, running: true, started_by: "ui" }, prefs: { keep_chats: true, menu_bar: true, open_at_login: false, server_port: 8484 }, jobs: [] };
    const { target } = render(Settings, { onclose: () => {} });
    const boxes = [...target.querySelectorAll<HTMLInputElement>("input[type=checkbox]")];
    expect(boxes.length).toBeGreaterThanOrEqual(3);
    expect(boxes.every((b) => b.getAttribute("role") === "switch")).toBe(true);
    expect(await violations()).toEqual([]);
  });
});

describe("Cloud in the window (024)", () => {
  const listing = (key: CloudListing["status"]["key"]): CloudListing => ({
    offers: [
      { repo: "Qwen/Qwen3.6-35B-A3B-FP8", weights_bytes: 35e9, context: 262144, min_vram_gb: 48, note: "the local qwen3.6 at full precision", secure: key ? ["NVIDIA L40S", 0.86] : null, community: key ? ["NVIDIA L40S", 0.69] : null, fits: key ? ["NVIDIA L40S", "NVIDIA A100 80GB PCIe"] : [] },
    ],
    gpus: key
      ? [
          { id: "NVIDIA A100 80GB PCIe", name: "A100 80GB", memory_gb: 80, secure_per_hr: 1.19, community_per_hr: null, secure_stock: "Low", community_stock: null },
          { id: "NVIDIA L40S", name: "L40S", memory_gb: 48, secure_per_hr: 0.86, community_per_hr: 0.69, secure_stock: "High", community_stock: "High" },
          { id: "NVIDIA RTX A5000", name: "RTX A5000", memory_gb: 24, secure_per_hr: 0.16, community_per_hr: null, secure_stock: "High", community_stock: null },
        ]
      : [],
    status: { pods: [], key },
    added: [{ repo: "org/mine", weights_bytes: 30e9 }],
  });
  const serve = (value: CloudListing) =>
    vi.stubGlobal("fetch", async (url: string) =>
      new Response(JSON.stringify(String(url).startsWith("/app/cloud") ? value : []), { status: 200, headers: { "content-type": "application/json" } }),
    );

  it("Models lists the catalog and added models, priced once a key is set", async () => {
    live.models = [];
    serve(listing(null));
    const first = render(Models, { onError: () => {} });
    await settle();
    expect(first.target.textContent).toContain("On a rented GPU · RunPod");
    expect(first.target.textContent).toContain("No RunPod key");
    expect(first.target.textContent).toContain("price after you add your key");
    expect(first.target.textContent).toContain("needs about 40 GB");
    first.destroy();
    serve(listing("keychain"));
    const { target } = render(Models, { onError: () => {} });
    await settle();
    expect(target.textContent).toContain("Key in Keychain");
    expect(target.textContent).toContain("NVIDIA L40S · $0.86/h");
    expect(await violations()).toEqual([]);
  });

  it("the sheet asks for the key first when none is set", async () => {
    live.cloud = listing(null);
    const { target } = render(CloudSheet, { pick: { repo: "org/mine", weights_bytes: 30e9, need: 40 }, onclose: () => {}, onstarted: () => {} });
    await settle();
    expect(target.querySelector("input[type=password]")).not.toBeNull();
    expect(target.textContent).toContain("kept in your Keychain");
    expect(await violations()).toEqual([]);
  });

  it("the sheet offers fitting GPUs, cheapest in stock first, Community only as a saving", async () => {
    live.cloud = listing("keychain");
    live.app = { ui_version: "0.1.0", server: { port: 8484, running: false, started_by: null }, prefs: { keep_chats: null, menu_bar: true, open_at_login: false, server_port: 8484 }, jobs: [] };
    const { target } = render(CloudSheet, {
      pick: { repo: "Qwen/Qwen3.6-35B-A3B-FP8", weights_bytes: 35e9, need: 48, fits: ["NVIDIA L40S", "NVIDIA A100 80GB PCIe"] },
      onclose: () => {},
      onstarted: () => {},
    });
    await settle();
    const gpus = [...target.querySelectorAll(".gpu .gname")].map((n) => n.textContent);
    expect(gpus).toEqual(["L40S", "A100 80GB"]);
    expect(target.querySelector<HTMLInputElement>(".gpu input")?.checked).toBe(true);
    expect(target.textContent).toContain("Cheaper on Community Cloud: $0.69/h");
    expect(target.textContent).toContain("Your prompts and code go to RunPod");
    expect(target.textContent).toContain("Stops itself after 30 idle minutes");
    expect(target.querySelector(".footer .primary")?.textContent?.trim()).toBe("Start · $0.86/h");
    expect(await violations()).toEqual([]);
  });

  it("Now shows the pod's cost and when it stops by itself", async () => {
    live.app = { ui_version: "0.1.0", server: { port: 8484, running: true, started_by: "ui" }, prefs: { keep_chats: null, menu_bar: true, open_at_login: false, server_port: 8484 }, jobs: [] };
    live.reachable = true;
    live.status = { ...status, budget: null, cache: null, requests: [], finished: [], pod: { id: "p1", gpu: "NVIDIA L40S", cost_per_hr: 0.86, reachable: true, uptime_s: 42 * 60, spend_usd: 0.6, idle_s: 12 * 60, idle_stop_s: 1800 } };
    const { target } = render(Now, { onAction: () => {} });
    flushSync();
    const text = target.textContent ?? "";
    expect(text).toContain("on RunPod · NVIDIA L40S");
    expect(text).toContain("$0.86/h");
    expect(text).toContain("up 42 min");
    expect(text).toContain("about $0.60 so far");
    expect(text).toContain("stops in 18 min unless used");
    expect(text).toContain("Stop Pod");
    expect(await violations()).toEqual([]);
    live.status = status;
  });

  it("Settings says where the key comes from", async () => {
    serve(listing("env"));
    const { target } = render(Settings, { onclose: () => {} });
    await settle();
    expect(target.textContent).toContain("From RUNPOD_API_KEY.");
    expect(target.textContent).not.toContain("Add…");
  });
});
