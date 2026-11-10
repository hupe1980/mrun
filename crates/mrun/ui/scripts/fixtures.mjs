// Fixed data for `npm run shots` (019 FR-018): one Mac, one model, every tone of recompute, so
// before/after screenshots differ only where the page changed.

const gib = 1024 ** 3;
const finished = (id, f) => ({
  id, api: "messages", generated: 210, ttft_ms: 310, decode_tps: 42.1, stop: "end", divergence: null, message: null, unreused: 0, ...f,
});

export const appState = {
  ui_version: "0.1.0",
  server: { port: 8484, running: true, started_by: "ui" },
  prefs: { keep_chats: true, menu_bar: true, open_at_login: false, last_model: "qwen3.6" },
  jobs: [],
};

export const status = {
  schema_version: 1,
  version: "0.1.0",
  model: { id: "qwen3.6", identity: "x", family: "qwen3_5_moe", syntax: "qwen" },
  numerics: { backend: "nuvec", nax: "off", gpu_family: "applegpu_g16s" },
  budget: { budget: 32 * gib, weights: 19.5 * gib, scratch: gib, kv_capacity_tokens: 221819, kv_bytes_per_token: 20480 },
  queued: 0,
  cache: { live_tokens: 14380, buffer_bytes: gib / 4, checkpoint_bytes: gib / 8, prompt_tokens_total: 186000, cached_tokens_total: 171500, enabled: true },
  last: null,
  requests: [
    { id: 12, api: "messages", lane: 0, phase: "prefill", prompt_tokens: 30000, cached: 17600, progress: 24000, cause: null, unreused: 0, generated: 0, decode_tps: null, age_ms: 900 },
  ],
  finished: [
    finished(7, { prompt_tokens: 900, cached: 0, processed: 900, cause: "new" }),
    finished(8, { prompt_tokens: 14380, cached: 14212, processed: 168, cause: "diverged", message: { index: 3, role: "user" } }),
    finished(9, { prompt_tokens: 22480, cached: 20500, processed: 1980, cause: "diverged", message: { index: 6, role: "tool" } }),
    finished(10, { prompt_tokens: 24100, cached: 24060, processed: 40, cause: null, speculation: { drafted: 168, accepted: 131, passes: 52 } }),
    finished(11, { prompt_tokens: 26000, cached: 0, processed: 26000, cause: "evicted" }),
  ],
  lanes: { configured: 1, busy: 1 },
  speculation: { mode: "auto", drafted: 4120, accepted: 3190, passes: 1450 },
};

const model = (m) => ({
  revision: "abc", pulled: true, fit: "fits", max_context: 262144, recommended: false, preview: false, serving: false,
  memory: { fixed: 20.6 * gib, per_token: 21 * 1024 }, budget: 32 * gib, ...m,
});

export const models = [
  model({ name: "qwen3.6", alias: "qwen3.6", repo: "mlx-community/Qwen3.6-35B-A3B-4bit", size: 19.4 * gib, recommended: true, serving: true, max_context: 221819, verdict: "Fits with 221,819 tokens." }),
  model({ name: "qwen3.5", alias: "qwen3.5", repo: "mlx-community/Qwen3.5-9B-4bit", size: 5.57 * gib, preview: true, memory: { fixed: 6.4 * gib, per_token: 33 * 1024 }, verdict: "Fits with 262,144 tokens." }),
  model({ name: "mlx-community/Qwen3-8B-4bit", alias: null, repo: "mlx-community/Qwen3-8B-4bit", size: 4.3 * gib, fit: "tight", max_context: 40960, memory: { fixed: 5.2 * gib, per_token: 144 * 1024 }, verdict: "Fits with 40,960 tokens, less than the 65,536 coding agents need." }),
  model({ name: "qwen3.8-big", alias: "qwen3.8-big", repo: "mlx-community/Qwen3.8-235B-A22B-4bit", size: 124 * gib, pulled: false, fit: "wont_fit", verdict: "Does not fit the budget of 32 GB." }),
];

export const agents = [
  { client: "claude-code", installed: true, version: "2.1.0", connected: true, files: [], refusal: null },
  { client: "codex", installed: true, version: "0.40.0", connected: false, files: [], refusal: null },
  { client: "opencode", installed: false, version: null, connected: false, files: [], refusal: null },
];

export const plan = {
  event: "plan", model: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", repo: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
  revision: "6e3f2a9c1b", files: 5, bytes: 17.2 * gib, fit: "fits", verdict: "Fits with 98,304 tokens.", family: "qwen3_moe",
};

export const apps = [
  {
    client: "zed",
    blocks: [{ path: "~/.config/zed/settings.json", note: "merge into Zed's settings", text: '{\n  "language_models": {\n    "openai_compatible": {\n      "mrun": { "api_url": "http://127.0.0.1:8484/v1" }\n    }\n  }\n}' }],
  },
];

/** The file the shots attach to a chat. */
// Cloud in the window (024): the catalog priced with a key, one added model, a start on its way.
const gpu = (id, name, memory_gb, secure_per_hr, secure_stock, community_per_hr = null) => ({
  id, name, memory_gb, secure_per_hr, community_per_hr, secure_stock, community_stock: community_per_hr ? "High" : null,
});

export const cloud = {
  offers: [
    { repo: "Qwen/Qwen3.6-35B-A3B-FP8", weights_bytes: 35.1e9, context: 262144, min_vram_gb: 48, note: "The local qwen3.6 at full precision and GPU speed.", secure: ["L40S", 0.86], community: ["L40S", 0.69], fits: ["NVIDIA L40S", "NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3"] },
    { repo: "Qwen/Qwen3-Coder-Next-FP8", weights_bytes: 81.4e9, context: 262144, min_vram_gb: 96, note: "An 80B coder; does not fit a 48 GB Mac.", secure: ["RTX PRO 6000", 1.89], community: null, fits: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] },
  ],
  gpus: [
    gpu("NVIDIA L40S", "L40S", 48, 0.86, "High", 0.69),
    gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.19, "Low"),
    gpu("NVIDIA H100 80GB HBM3", "H100 80GB", 80, 2.69, null),
    gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000", 96, 1.89, "Medium"),
  ],
  status: { pods: [], relay: null, key: "keychain" },
  added: [{ repo: "org/my-finetune", weights_bytes: 30.4e9 }],
};

export const cloudJob = {
  id: "j9", kind: "cloud-up", target: "Qwen/Qwen3.6-35B-A3B-FP8|NVIDIA L40S|secure", state: "running", error: null,
  progress: { file: "", bytes: 12.4e9, total: 35.1e9, rate: 0, eta_s: null, step: "loading", text: "loading: 12.4 GB of weights on disk" },
};

export const podStatus = {
  ...status,
  model: { id: "Qwen/Qwen3.6-35B-A3B-FP8", identity: "remote", family: "remote", syntax: "remote" },
  numerics: null, budget: null, cache: null, requests: [], finished: [], last: null,
  pod: { id: "p1", gpu: "NVIDIA L40S", cost_per_hr: 0.86, reachable: true, phase: "ready", uptime_s: 42 * 60, spend_usd: 0.6, idle_s: 12 * 60, idle_stop_s: 1800 },
};

export const sourceFile = `pub fn parse(input: &str) -> Result<Ast, Error> {
    let mut tokens = lex(input)?.peekable();
    let first = tokens.next().unwrap();
    expression(first, &mut tokens)
}
`;

export const modelInfo = { id: "qwen3.6", defaults: { temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0 }, thinking: true };

const frame = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

/** A finished reply: mrun's comment, thinking, a Markdown answer with code, usage. One frame per
 * token, as mrun streams, so the page's times and rates read as on a real Mac. */
export function reply() {
  const thought = "The user wants a haiku about caches . Keep it concrete , then explain prefix reuse in two sentences .".split(" ");
  const answer = (
    "*Lines held from before —*  \n*only the new words are read;*  \n*the rest, remembered.*\n\n" +
    "Every turn resends the whole conversation, but only its new end changed. mrun keeps the state of the shared start, " +
    "so it reads just the new tokens:\n\n```rust\nlet hit = cache.longest_prefix(&tokens);\n```\n"
  ).split(/(?<= )/);
  return [
    `: mrun ${JSON.stringify({ cached: 14212, processed: 168, cause: "diverged", divergence: 14000, message: { index: 2, role: "user" }, unreused: 0 })}\n\n`,
    frame("message_start", { type: "message_start", message: { usage: { input_tokens: 168, cache_read_input_tokens: 14212, output_tokens: 0 } } }),
    ...thought.map((w) => frame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: `${w} ` } })),
    ...answer.map((text) => frame("content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "text_delta", text } })),
    frame("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: thought.length + answer.length } }),
    frame("message_stop", { type: "message_stop" }),
  ];
}

/** A reply still thinking: the stream stays open after these frames. */
export function thinking() {
  return [
    `: mrun ${JSON.stringify({ cached: 14400, processed: 40, cause: null, divergence: null, message: null, unreused: 0 })}\n\n`,
    frame("message_start", { type: "message_start", message: { usage: { input_tokens: 40, cache_read_input_tokens: 14400, output_tokens: 0 } } }),
    ...Array.from({ length: 40 }, (_, i) =>
      frame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: i === 39 ? "\nComparing the prefix rule with checkpoint resume" : " step" } }),
    ),
  ];
}

// Session replay (025): a Claude Code session of 140 requests (a main tool loop, two subagents,
// side requests, a 6-minute break, two misses worth fixing) and an older chat.
export function sessions(n = 140) {
  const requests = [];
  let id = 100;
  let at = 60_000;
  let main = 18_000;
  const push = (session, row, client, prompt, cached, f = {}) => {
    at += 2500 + ((id * 7919) % 6500);
    requests.push({
      id: id++, session, row, client, ended_ms: at, api: "messages", prompt_tokens: prompt, cached, processed: prompt - cached,
      generated: 60 + ((id * 31) % 400), ttft_ms: 180 + ((id * 97) % 900) + (prompt - cached) / 40, decode_tps: 38 + ((id * 13) % 60) / 10,
      stop: "end", cause: cached < prompt - 600 ? "new" : null, divergence: null, message: null, unreused: 0, attention: false, excerpt: false, ...f,
    });
  };
  for (let i = 0; i < 6; i++) push("s1", "main", "mrun-ui", 1200 + i * 400, 1100 + i * 400 - 300, { session: "s2" });
  at += 25 * 60_000;
  for (let i = 0; i < n; i++) {
    if (i % 23 === 3) push("s1", "side", "claude-cli", 700 + (i % 5) * 90, 0, { cause: "new" });
    if (i >= 38 && i < 50) {
      const sub = 6000 + (i - 38) * 1400;
      push("s1", "sub1", "claude-cli", sub, i === 38 ? 0 : sub - 1300, i === 38 ? { cause: "new" } : {});
      continue;
    }
    if (i >= 104 && i < 112) {
      const sub = 5200 + (i - 104) * 1100;
      push("s1", "sub2", "claude-cli", sub, i === 104 ? 0 : sub - 1000, i === 104 ? { cause: "new" } : {});
      continue;
    }
    if (i === 70) at += 6 * 60_000;
    const grew = 280 + ((i * 131) % 900);
    if (i === 57) {
      push("s1", "main", "claude-cli", main + grew, main - 9200, { cause: "diverged", message: { index: 41, role: "tool" }, divergence: main - 9200, unreused: 9200, attention: true, excerpt: true });
    } else if (i === 118) {
      push("s1", "main", "claude-cli", main + grew, 0, { cause: "evicted", attention: true });
    } else {
      push("s1", "main", "claude-cli", main + grew, main, {});
    }
    main += grew;
  }
  const summary = (sid, label, client) => {
    const mine = requests.filter((r) => r.session === sid);
    const rows = [...new Set(mine.map((r) => r.row))].map((row) => ({
      id: row, kind: row === "main" ? "main" : row === "side" ? "side" : "subagent", requests: mine.filter((r) => r.row === row).length,
    }));
    return {
      id: sid, client, label, first_ms: mine[0].ended_ms, last_ms: mine.at(-1).ended_ms, requests: mine.length,
      prompt_tokens: mine.reduce((a, r) => a + r.prompt_tokens, 0), cached_tokens: mine.reduce((a, r) => a + r.cached, 0),
      attention: mine.filter((r) => r.attention).length, rows,
    };
  };
  return { kept: 2000, dropped: 0, remote: false, now_ms: at, sessions: [summary("s1", "Claude Code", "claude-cli"), summary("s2", "Chat", "mrun-ui")], requests };
}

export const excerpt = {
  id: 0, message: { index: 41, role: "tool" },
  context: "…{\"type\": \"tool_result\", \"content\": \"Compiling mrun v0.1.1\\n    Finished `dev` profile in ",
  before: "12.4s\\nrunning 448 tests\\ntest result: ok. 448 passed\"}]}, {\"role\": \"assistant\", …",
  after: "11.9s\\nrunning 448 tests\\ntest result: ok. 448 passed\"}]}, {\"role\": \"assistant\", …",
};

// Mac pool (026): four MacBooks in an office, one on Wi-Fi.
const pgib = 1024 ** 3;
const thisMac = { mac_id: "ben0000000000000", name: "Ben's MacBook Pro", chip: "Apple M4 Max", memory_bytes: 48 * pgib };
const poolMembers = [
  { mac_id: "anna000000000000", name: "Anna's MacBook Pro", chip: "Apple M4 Pro", memory_bytes: 48 * pgib, power: "adapter", address: "10.0.0.5:8485", state: "serving" },
  { mac_id: "chris00000000000", name: "Chris's MacBook Pro", chip: "Apple M3 Max", memory_bytes: 36 * pgib, power: "adapter", address: "10.0.0.7:8485", state: "serving" },
  { mac_id: "dana000000000000", name: "Dana's MacBook Air", chip: "Apple M3", memory_bytes: 24 * pgib, power: "battery", address: "10.0.0.9:8485", state: "serving" },
];
const stage = (mac, first, end, used, budget, ms, link = null) => ({
  mac_id: mac.mac_id, name: mac.name, first, end, weights_bytes: used * 0.86, state_bytes: used * 0.1, scratch_bytes: used * 0.04, budget, fits: true, short_bytes: 0, ms_per_token: ms,
  link: link && { kind: link[0], rtt_ms_p50: link[1], rtt_ms_p90: link[1] * 1.6, mbps: 900 },
});
export const poolPlan = {
  model: "qwen3.8-235b", fits: true, context_tokens: 98304, speed: { low: 11.4, high: 14.9 },
  pace: { kind: "link", mac_id: "dana000000000000", name: "Dana's MacBook Air", share: 0.31 }, numerics_match: false, refusal: null,
  stages: [
    stage(thisMac, 0, 30, 36 * pgib, 40 * pgib, 18),
    stage(poolMembers[0], 30, 60, 37 * pgib, 40 * pgib, 19, ["thunderbolt", 0.24]),
    stage(poolMembers[1], 60, 82, 27 * pgib, 30 * pgib, 14, ["ethernet", 0.6]),
    stage(poolMembers[2], 82, 94, 15 * pgib, 18 * pgib, 9, ["wifi", 7.6]),
  ],
};
export function poolView(mode) {
  const host = (extra) => ({ role: "host", name: "Ben's pool", model: null, degraded: null, address: "10.0.0.2:8485", plan: null, members: poolMembers.map((m) => ({ ...m, state: "joined" })), waiting: [], ...extra });
  switch (mode) {
    case "empty":
      return { this: thisMac, pool: null, joining: null, nearby: [
        { name: "Platform team pool", host: "Chris's MacBook Pro", host_id: "chris00000000000", members: 3, model: "qwen3.8-235b", addresses: ["10.0.0.7:8485"] },
        { name: "Dana's pool", host: "Dana's MacBook Air", host_id: "dana000000000000", members: 1, model: null, addresses: ["10.0.0.9:8485"] },
      ] };
    case "join":
      return { this: thisMac, pool: host({ members: poolMembers.slice(0, 2).map((m) => ({ ...m, state: "joined" })), waiting: [
        { mac_id: "dana000000000000", name: "Dana's MacBook Air", chip: "Apple M3", memory_bytes: 24 * pgib, power: "battery", code: "482913", waiting_s: 4 },
      ] }), joining: null };
    case "open":
      return { this: thisMac, pool: host({}), joining: null };
    case "serving":
      return { this: thisMac, pool: host({ model: "qwen3.8-235b", plan: poolPlan, members: poolMembers }), joining: null };
    case "lost":
      return { this: thisMac, pool: host({ model: "qwen3.8-235b", plan: poolPlan, degraded: "Dana's MacBook Air stopped answering", recovery: { wait: "Dana's MacBook Air", smaller: "qwen3.6" },
        members: poolMembers.map((m) => (m.mac_id.startsWith("dana") ? { ...m, state: "lost" } : m)) }), joining: null };
  }
}

/** `mrun config show --json`: a few settings from the file, the rest defaults. */
const setting = (key, value, source = "default") => ({ key, value, source, overridden: [], needs: null });
export const configView = {
  path: "/Users/you/.mrun/config.toml",
  exists: true,
  settings: [
    setting("port", 8484),
    setting("model", "qwen3.6", "file:2"),
    setting("budget-gib", null),
    setting("serve.max-output", null),
    setting("serve.queue", 4, "file:5"),
    setting("serve.queue-wait-s", 240),
    setting("serve.no-cache", null),
    setting("serve.speculation", "auto"),
    setting("run.max-tokens", 2048),
  ],
  warnings: [],
};
