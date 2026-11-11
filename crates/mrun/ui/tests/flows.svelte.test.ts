import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import FirstRun from "../src/components/FirstRun.svelte";
import Inspector from "../src/components/Inspector.svelte";
import { live } from "../src/lib/live.svelte";
import type { Chat as ChatT, ModelEntry } from "../src/lib/types";
import Agents from "../src/places/Agents.svelte";
import Chat from "../src/places/Chat.svelte";
import { render, settle, violations } from "./dom";

const gib = 1024 ** 3;
const app = (keep: boolean | null) => ({
  ui_version: "0.1.0",
  server: { port: 8484, running: true, started_by: "ui" as const },
  prefs: { keep_chats: keep, menu_bar: true, open_at_login: false, server_port: 8484 },
  jobs: [],
});

/** Answers the page's host calls from a table, recording each request. */
function stubFetch(routes: Record<string, unknown>) {
  const calls: string[] = [];
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    calls.push(`${init?.method ?? "GET"} ${url}`);
    const body = Object.entries(routes).find(([k]) => url.startsWith(k))?.[1] ?? {};
    return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
  });
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe("chat (US2)", () => {
  it("asks once whether to keep chats, and the answer becomes the setting (FR-010)", async () => {
    const calls = stubFetch({ "/app/state": app(true), "/app/prefs": app(true).prefs, "/v1/models": { defaults: { temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0 }, thinking: true } });
    live.app = app(null);
    live.status = { model: { id: "qwen3.6" }, requests: [] } as any;
    live.reachable = true;
    const { target } = render(Chat, { active: true });
    expect(target.textContent).toContain("Keep chats on this Mac?");
    const keep = [...target.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Keep")!;
    keep.click();
    await settle();
    expect(calls.some((c) => c.startsWith("PUT /app/prefs"))).toBe(true);
    live.app = app(true);
    flushSync();
    expect(target.textContent).not.toContain("Keep chats on this Mac?");
    expect(await violations()).toEqual([]);
  });

  it("the inspector shows each model default and resets every change", async () => {
    const chat: ChatT = { id: "0123456789abcdef", title: "t", created: "", updated: "", model: null, system: null, sampling: { temperature: 1.2, top_k: 5 }, thinking: null, messages: [] };
    const state = $state({ chat });
    const { target } = render(Inspector, {
      get chat() { return state.chat; },
      set chat(v) { state.chat = v; },
      defaults: { temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0 },
      thinkingFamily: true,
      onclose: () => {},
    } as any);
    const placeholders = [...target.querySelectorAll("input[type=number]")].map((i) => i.getAttribute("placeholder"));
    expect(placeholders).toContain("Default (0.95)");
    expect(target.querySelectorAll("[aria-label=changed]").length).toBe(2);
    [...target.querySelectorAll("button")].find((b) => b.textContent?.includes("Reset to Model Defaults"))!.click();
    flushSync();
    expect(state.chat.sampling).toEqual({});
    expect(await violations()).toEqual([]);
  });
});

describe("first run (US3 AS1, FR-014)", () => {
  it("offers the best model already chosen, with the terminal's commands one click away", async () => {
    const entry: ModelEntry = { name: "qwen3.6", alias: "qwen3.6", repo: "r", revision: "x", size: 20 * gib, pulled: false, fit: "fits", verdict: "Fits with 262144 tokens.", max_context: 262144, recommended: true, preview: false, serving: false };
    live.models = [entry];
    const { target } = render(FirstRun, { ondone: () => {}, onchoose: () => {} });
    expect(target.ownerDocument.body.textContent).toContain("Set up mrun on this Mac");
    expect(target.ownerDocument.body.textContent).toContain("qwen3.6");
    const buttons = [...target.ownerDocument.querySelectorAll("button")].map((b) => b.textContent?.trim());
    expect(buttons).toContain("Continue");
    [...target.ownerDocument.querySelectorAll("button")].find((b) => b.textContent?.includes("Prefer the terminal?"))!.click();
    flushSync();
    expect(target.ownerDocument.querySelector(".commands")?.textContent).toContain("mrun pull qwen3.6");
    expect(await violations()).toEqual([]);
  });
});

describe("agents (US4)", () => {
  it("shows install and connection state and the refusal reason", async () => {
    stubFetch({
      "/app/agents": [
        { client: "claude-code", installed: true, version: "2.1.0", connected: true, files: [], refusal: null },
        { client: "codex", installed: true, version: null, connected: false, files: [], refusal: "qwen3.5 leaves 40960 tokens of context; codex needs at least 65536" },
        { client: "opencode", installed: false, version: null, connected: false, files: [], refusal: null },
      ],
    });
    const { target } = render(Agents, { onError: () => {} });
    await settle(12);
    const text = target.textContent ?? "";
    expect(text).toContain("Connected to mrun");
    expect(text).toContain("codex needs at least 65536");
    expect(text).toContain("Not installed.");
    const connect = [...target.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Connect");
    expect(connect?.disabled).toBe(true);
    expect(await violations()).toEqual([]);
  });
});

describe("edit and resend (019 US1)", () => {
  it("cuts the chat at the edited question and asks again (FR-003, SC-002)", async () => {
    const bodies: any[] = [];
    const sse = (text: string) =>
      [
        `: mrun {"cached":0,"processed":12,"cause":"new","divergence":null,"message":null,"unreused":0}`,
        `event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12,"cache_read_input_tokens":0,"output_tokens":0}}}`,
        `event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"${text}"}}`,
        `event: message_stop\ndata: {"type":"message_stop"}`,
        "",
      ].join("\n\n");
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      if (url.startsWith("/v1/messages")) {
        bodies.push(JSON.parse(String(init?.body)));
        return new Response(sse(`Answer ${bodies.length}`), { status: 200, headers: { "content-type": "text/event-stream" } });
      }
      return new Response(url.startsWith("/v1/models") ? '{"defaults":{}}' : "{}", { status: 200, headers: { "content-type": "application/json" } });
    });
    live.app = app(false);
    live.status = { model: { id: "qwen3.6" }, requests: [] } as any;
    live.reachable = true;
    const { target } = render(Chat, { active: true });
    const composer = target.querySelector<HTMLTextAreaElement>("textarea[aria-label=Message]")!;
    composer.value = "Wrte a haiku";
    composer.dispatchEvent(new Event("input"));
    composer.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    await settle(30);
    expect(target.textContent).toContain("Answer 1");

    // ↑ in the empty composer edits the last question.
    composer.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowUp", bubbles: true }));
    flushSync();
    const field = target.querySelector<HTMLTextAreaElement>("textarea[aria-label='Edit question']")!;
    expect(field.value).toBe("Wrte a haiku");
    field.value = "Write a haiku";
    field.dispatchEvent(new Event("input"));
    field.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    await settle(30);
    expect(bodies[1].messages).toEqual([{ role: "user", content: "Write a haiku" }]);
    expect(target.textContent).toContain("Answer 2");
    expect(target.textContent).not.toContain("Answer 1");
    expect(await violations()).toEqual([]);
  });
});

describe("files in the chat (020 US1)", () => {
  function server(capacity: number) {
    const bodies: any[] = [];
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      if (url.startsWith("/v1/messages/count_tokens")) {
        const body = JSON.parse(String(init?.body));
        const chars = JSON.stringify(body.messages).length;
        return new Response(JSON.stringify({ input_tokens: Math.ceil(chars / 4) }), { status: 200 });
      }
      if (url.startsWith("/v1/messages")) {
        bodies.push(JSON.parse(String(init?.body)));
        return new Response('event: message_stop\ndata: {"type":"message_stop"}\n\n', { status: 200, headers: { "content-type": "text/event-stream" } });
      }
      return new Response(url.startsWith("/v1/models") ? '{"defaults":{}}' : "{}", { status: 200 });
    });
    live.app = app(false);
    live.status = { model: { id: "qwen3.6" }, requests: [], budget: { kv_capacity_tokens: capacity } } as any;
    live.reachable = true;
    return bodies;
  }

  const drop = (target: Element, files: File[]) => {
    const event = new Event("drop", { bubbles: true, cancelable: true }) as any;
    event.dataTransfer = { files, types: ["Files"] };
    target.dispatchEvent(event);
  };

  it("counts a dropped file with the server and sends it fenced before the question (FR-001–FR-004)", async () => {
    const bodies = server(100_000);
    const { target } = render(Chat, { active: true });
    drop(target.querySelector("section.conversation")!, [new File(["fn main() {}\n"], "main.rs")]);
    await settle(40);
    const chip = target.querySelector(".attachments .chip");
    expect(chip?.textContent).toContain("main.rs");
    expect(chip?.textContent).toMatch(/\d+ tokens/);
    expect(chip?.textContent).not.toContain("about");
    const composer = target.querySelector<HTMLTextAreaElement>("textarea[aria-label=Message]")!;
    composer.value = "What does it do?";
    composer.dispatchEvent(new Event("input"));
    composer.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    await settle(30);
    expect(bodies[0].messages[0].content).toBe("`main.rs`:\n```rust\nfn main() {}\n```\n\nWhat does it do?");
    expect(target.querySelector(".user .file")?.textContent).toContain("main.rs");
    expect(await violations()).toEqual([]);
  });

  it("refuses a file over the remaining context and an image, each with its reason (FR-003, SC-003)", async () => {
    server(50);
    const { target } = render(Chat, { active: true });
    drop(target.querySelector("section.conversation")!, [new File(["x".repeat(4000)], "big.log"), new File([new Uint8Array([1])], "a.png", { type: "image/png" })]);
    await settle(40);
    const notes = [...target.querySelectorAll(".chip.refused")].map((n) => n.textContent?.replace(/\s+/g, " ").trim());
    expect(notes[0]).toMatch(/big\.log [\d,]+ tokens; [\d,]+ are left in this chat's context\./);
    expect(notes[1]).toContain("Images need a vision model");
    expect(target.querySelectorAll(".attachments .chip:not(.refused)").length).toBe(0);
  });
});

describe("add a model and your code (020 US2, US3)", () => {
  it("shows the CLI's plan before Download, and refuses other input without a request", async () => {
    const calls = stubFetch({
      "/app/plan": { event: "plan", model: "org/coder", repo: "org/coder", revision: "abcdef123", files: 4, bytes: 17 * gib, fit: "fits", verdict: "Fits with 98,304 tokens.", family: "qwen3_moe" },
    });
    live.models = [];
    const { default: Models } = await import("../src/places/Models.svelte");
    const { target } = render(Models, { onError: () => {} });
    const input = target.querySelector<HTMLInputElement>("input[aria-label='Model to add']")!;
    const submit = () => target.querySelector<HTMLFormElement>(".add-form")!.dispatchEvent(new Event("submit", { cancelable: true }));
    input.value = "not a repo";
    input.dispatchEvent(new Event("input"));
    submit();
    await settle();
    expect(target.querySelector(".add-error")?.textContent).toContain("org/name");
    expect(calls.some((c) => c.includes("/app/plan"))).toBe(false);
    input.value = "https://huggingface.co/org/coder";
    input.dispatchEvent(new Event("input"));
    submit();
    await settle(12);
    expect(calls).toContain("GET /app/plan?model=org%2Fcoder");
    const plan = target.querySelector(".plan")?.textContent?.replace(/\s+/g, " ");
    expect(plan).toContain("17 GB · 4 files · family qwen3_moe");
    expect(plan).toContain("Fits with 98,304 tokens.");
    expect(await violations()).toEqual([]);
  });

  it("gives code for all three APIs on the live port (FR-007)", async () => {
    stubFetch({ "/app/agents": [], "/app/apps": [{ client: "zed", blocks: [{ path: "~/.config/zed/settings.json", note: "merge", text: "{}" }] }] });
    live.app = { ...app(true), server: { port: 9191, running: true, started_by: "ui" } };
    live.status = { model: { id: "qwen3.6" }, requests: [] } as any;
    live.agents = [];
    const { target } = render(Agents, { onError: () => {} });
    await settle(12);
    const code = [...target.querySelectorAll(".snippet pre")].map((p) => p.textContent ?? "");
    expect(code.length).toBeGreaterThanOrEqual(4);
    expect(code[0]).toContain("http://127.0.0.1:9191/v1/messages");
    expect(target.textContent).toContain("Other apps");
    expect(await violations()).toEqual([]);
  });
});

describe("places keep their state", () => {
  it("a chat survives switching to Sessions and back, unsaved", async () => {
    const { default: App } = await import("../src/App.svelte");
    stubFetch({ "/app/state": app(false), "/app/models": [], "/app/agents": [], "/app/apps": [], "/app/chats": [], "/v1/models": { defaults: {}, thinking: true }, "/mrun/sessions": { kept: 0, dropped: 0, remote: false, sessions: [], requests: [] } });
    live.app = app(false);
    live.status = { model: { id: "qwen3.6" }, requests: [] } as any;
    live.reachable = true;
    live.models = [];
    sessionStorage.setItem("mrun.token", "0".repeat(64));
    const { target } = render(App, {});
    await settle();
    const go = (name: string) =>
      [...target.querySelectorAll<HTMLButtonElement>("li > button")].find((b) => b.textContent?.trim().startsWith(name))!.click();
    go("Chat");
    await settle();
    const field = target.querySelector<HTMLTextAreaElement>("textarea")!;
    field.value = "Explain the prefix cache";
    field.dispatchEvent(new Event("input", { bubbles: true }));
    await settle();
    go("Sessions");
    await settle();
    expect(target.querySelector<HTMLElement>(".keep")?.hidden).toBe(true);
    go("Chat");
    await settle();
    expect(target.querySelector<HTMLTextAreaElement>("textarea")!.value).toBe("Explain the prefix cache");
  });
});

describe("tools in the chat (028 US1)", () => {
  it("runs a call only after Allow, sends Deny back as an error, and Always stops asking (SC-001)", async () => {
    const bodies: any[] = [];
    const calls: any[] = [];
    let n = 0;
    const frame = (e: object) => `event: x\ndata: ${JSON.stringify(e)}`;
    const toolReply = (id: string) =>
      [
        frame({ type: "message_start", message: { usage: { input_tokens: 10, output_tokens: 0 } } }),
        frame({ type: "content_block_start", index: 0, content_block: { type: "tool_use", id, name: "workspace__read", input: {} } }),
        frame({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: '{"path":' } }),
        frame({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: '"src/a.rs"}' } }),
        frame({ type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 5 } }),
        "",
      ].join("\n\n");
    const textReply = (t: string) =>
      [frame({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: t } }), frame({ type: "message_delta", delta: { stop_reason: "end_turn" } }), ""].join("\n\n");
    const list = {
      path: "/Users/me/.mrun/mcp.json",
      errors: [],
      servers: [
        {
          name: "workspace",
          kind: "stdio",
          command: "mrun mcp serve workspace --root /Users/me/app",
          env: [],
          sandbox: { workspace: "/Users/me/app", write: true, hosts: [], allow: [] },
          tools: [{ name: "read", qualified: "workspace__read", description: "Read a file", input_schema: { type: "object" } }],
        },
      ],
    };
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      const ok = (body: unknown, type = "application/json") => new Response(typeof body === "string" ? body : JSON.stringify(body), { status: 200, headers: { "content-type": type } });
      if (url.startsWith("/app/state")) return ok(app(false));
      if (url.startsWith("/v1/messages/count_tokens")) return ok({ input_tokens: JSON.parse(String(init?.body)).tools ? 140 : 10 });
      if (url.startsWith("/v1/messages")) {
        bodies.push(JSON.parse(String(init?.body)));
        n += 1;
        return ok(n % 2 ? toolReply(`toolu_${n}`) : textReply(`Answer ${n}`), "text/event-stream");
      }
      if (url.startsWith("/app/mcp/call")) {
        calls.push(JSON.parse(String(init?.body)));
        return ok({ text: "fn a() {}", is_error: false });
      }
      if (url.startsWith("/app/mcp")) return ok(list);
      return ok(url.startsWith("/v1/models") ? { defaults: {} } : {});
    });
    live.app = app(false);
    live.status = { model: { id: "qwen3.6" }, requests: [] } as any;
    live.reachable = true;
    const { target } = render(Chat, { active: true });
    const doc = target.ownerDocument;
    const until = async (ok: () => boolean) => {
      for (let i = 0; i < 200 && !ok(); i++) await new Promise((r) => setTimeout(r, 0));
      flushSync();
    };
    const button = (label: string) => [...doc.querySelectorAll("button")].find((b) => b.textContent?.trim() === label);

    // Turn the server on in the picker: tool count and the tokens its definitions add.
    target.querySelector<HTMLButtonElement>("button[aria-label^=Tools]")!.click();
    await until(() => doc.body.textContent!.includes("130 tokens"));
    expect(doc.body.textContent).toContain("Sandboxed · writes ~/app · no network");
    expect(doc.body.textContent).toContain("1 tool · 130 tokens");
    doc.querySelector<HTMLInputElement>("[role=dialog][aria-label='Tools for this chat'] input[role=switch]")!.click();
    flushSync();
    doc.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    flushSync();

    const composer = target.querySelector<HTMLTextAreaElement>("textarea[aria-label=Message]")!;
    const say = async (text: string) => {
      composer.value = text;
      composer.dispatchEvent(new Event("input"));
      composer.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      const before = bodies.length;
      await until(() => bodies.length > before && (Boolean(button("Allow")) || target.textContent!.includes(`Answer ${before + 2}`)));
    };

    await say("What is in a.rs?");
    expect(bodies[0].tools).toEqual([{ name: "workspace__read", description: "Read a file", input_schema: { type: "object" } }]);
    expect(target.textContent).toContain("Read");
    expect(target.textContent).toContain("src/a.rs");
    expect(target.textContent).toContain("Waiting for you");
    expect(calls).toEqual([]);
    button("Deny")!.click();
    await until(() => target.textContent!.includes("Answer 2"));
    expect(calls).toEqual([]);
    const denied = bodies[1].messages.at(-1).content[0];
    expect(denied).toMatchObject({ type: "tool_result", tool_use_id: "toolu_1", is_error: true, content: "The user declined this call." });
    expect(target.textContent).toContain("Answer 2");

    await say("Again, please");
    button("Allow")!.click();
    await until(() => target.textContent!.includes("Answer 4"));
    expect(calls).toEqual([{ server: "workspace", tool: "read", arguments: { path: "src/a.rs" } }]);
    expect(bodies[3].messages.at(-1).content[0]).toMatchObject({ tool_use_id: "toolu_3", is_error: false, content: "fn a() {}" });

    await say("And once more");
    button("Always Allow")!.click();
    await until(() => target.textContent!.includes("Answer 6"));
    await say("Last time");
    expect(calls.length).toBe(3);
    expect(target.textContent).not.toContain("Waiting for you");
    expect(await violations()).toEqual([]);
  });
});

describe("Settings › Tools (028 US2, US4)", () => {
  it("lists each server with its sandbox and tools, never a secret, and adds mrun's own tools", async () => {
    const { default: Settings } = await import("../src/places/Settings.svelte");
    const { tools } = await import("../src/lib/tools.svelte");
    tools.list = null;
    const posts: string[] = [];
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      if (init?.method === "POST") posts.push(`${url} ${init.body ?? ""}`);
      const body = url.startsWith("/app/state")
        ? app(false)
        : url.startsWith("/app/mcp/choose")
        ? { dir: "/Users/me/app" }
        : url.startsWith("/app/mcp/add")
          ? { added: ["workspace"] }
          : url.startsWith("/app/mcp")
            ? {
                path: "/Users/me/.mrun/mcp.json",
                errors: [],
                servers: [
                  { name: "github", kind: "http", host: "api.example.com", headers: ["Authorization"], sandbox: false, error: "api.example.com: HTTP 401" },
                  { name: "notes", kind: "stdio", command: "npx -y notes-mcp", env: ["NOTES_TOKEN"], sandbox: { workspace: null, write: false, hosts: ["registry.npmjs.org", "pypi.org", "files.pythonhosted.org", "github.com"], allow: [] }, tools: [{ name: "search", qualified: "notes__search", description: "", input_schema: {} }] },
                ],
              }
            : {};
      return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
    });
    const { target } = render(Settings, { onclose: () => {}, pane: "tools" } as any);
    const doc = target.ownerDocument;
    for (let i = 0; i < 50 && !doc.body.textContent!.includes("notes"); i++) await new Promise((r) => setTimeout(r, 0));
    flushSync();
    const text = doc.body.textContent!;
    expect(text).toContain("Remote · api.example.com");
    expect(text).toContain("api.example.com: HTTP 401");
    expect(text).toContain("Sandboxed · reads ~ · package registries");
    expect(text).toContain("search");
    [...doc.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Add Workspace…")!.click();
    for (let i = 0; i < 50 && !doc.body.textContent!.includes("Added workspace"); i++) await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(posts.some((p) => p.includes("/app/mcp/add") && p.includes('"dir":"/Users/me/app"'))).toBe(true);
    expect(doc.body.textContent).toContain("Added workspace.");
    expect(await violations()).toEqual([]);
  });
});
