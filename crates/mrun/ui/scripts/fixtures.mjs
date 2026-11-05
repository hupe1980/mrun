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
    finished(10, { prompt_tokens: 24100, cached: 24060, processed: 40, cause: null }),
    finished(11, { prompt_tokens: 26000, cached: 0, processed: 26000, cause: "evicted" }),
  ],
  lanes: { configured: 1, busy: 1 },
};

const model = (m) => ({
  revision: "abc", pulled: true, fit: "fits", max_context: 262144, recommended: false, preview: false, serving: false, ...m,
});

export const models = [
  model({ name: "qwen3.6", alias: "qwen3.6", repo: "mlx-community/Qwen3.6-35B-A3B-4bit", size: 19.4 * gib, recommended: true, serving: true, verdict: "Fits with 221,819 tokens." }),
  model({ name: "qwen3.5", alias: "qwen3.5", repo: "mlx-community/Qwen3.5-9B-4bit", size: 5.57 * gib, preview: true, verdict: "Fits with 262,144 tokens." }),
  model({ name: "mlx-community/Qwen3-8B-4bit", alias: null, repo: "mlx-community/Qwen3-8B-4bit", size: 4.3 * gib, fit: "tight", verdict: "Fits with 40,960 tokens, less than the 65,536 coding agents need." }),
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
