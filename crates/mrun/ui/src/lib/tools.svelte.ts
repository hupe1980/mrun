// Tools in mrun's chats (028): the servers of ~/.mrun/mcp.json, shared by Chat and Settings ›
// Tools, and how a call reads as a sentence on its card ("Run cargo test", "Edit src/a.rs").

import { countTokens, host } from "./api";
import type { McpList, McpServer, ToolCall } from "./types";

class Tools {
  list = $state<McpList | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);
  /** Prompt tokens of each server's tool definitions, for one model. */
  tokens = $state<Record<string, number>>({});
  private countedFor = "";
  /** Remote hosts the current chat sends tool calls to (the privacy line, FR-008). */
  hosts = $state<string[]>([]);

  /** Lists the servers; `start` also starts them for their tools (first open, Refresh). */
  async load(mode: "" | "start" | "refresh" = "start") {
    this.loading = true;
    try {
      this.list = await host.mcp(mode);
      this.error = null;
      this.countedFor = "";
    } catch (e) {
      this.error = (e as Error).message;
    } finally {
      this.loading = false;
    }
  }

  /** Adds mrun's own workspace tools for a folder the user chooses; the name added, if any. */
  async addWorkspace(): Promise<string | null> {
    const { dir } = await host.mcpChoose();
    if (!dir) return null;
    return this.added(await host.mcpAdd("workspace", dir));
  }

  async addFetch(): Promise<string | null> {
    return this.added(await host.mcpAdd("fetch"));
  }

  /** Copies another client's servers; the names added. */
  async importFrom(client: "claude-code" | "claude-desktop"): Promise<string[]> {
    const { added } = await host.mcpImport(client);
    await this.load("refresh");
    return added;
  }

  private async added(r: { added: string[] }): Promise<string | null> {
    await this.load("refresh");
    return r.added[0] ?? null;
  }

  server(name: string): McpServer | undefined {
    return this.list?.servers.find((s) => s.name === name);
  }

  /** The definitions the model gets for these servers. */
  definitions(servers: string[]) {
    return servers.flatMap((name) =>
      (this.server(name)?.tools ?? [])
        .filter((t) => t.qualified)
        .map((t) => ({ name: t.qualified!, description: t.description, input_schema: t.input_schema })),
    );
  }

  /** The server and tool of a name the model called, if that server is on in the chat. */
  resolve(name: string, on: string[]): { server: McpServer; tool: string } | null {
    for (const s of on) {
      const server = this.server(s);
      const t = server?.tools?.find((t) => t.qualified === name);
      if (server && t) return { server, tool: t.name };
    }
    return null;
  }

  /** Counts each server's definitions with the served model's tokenizer (FR-007). */
  async count(model: string) {
    if (!this.list || this.countedFor === model) return;
    this.countedFor = model;
    try {
      const base = await countTokens({ model, messages: [{ role: "user", content: "." }] });
      const counts: Record<string, number> = {};
      for (const s of this.list.servers) {
        const defs = this.definitions([s.name]);
        if (!defs.length) continue;
        counts[s.name] = Math.max(0, (await countTokens({ model, messages: [{ role: "user", content: "." }], tools: defs })) - base);
      }
      this.tokens = counts;
    } catch {
      this.countedFor = "";
    }
  }
}

export const tools = new Tools();

/** mrun's own tools (`mrun mcp serve …`): their calls read as sentences. */
export function builtin(server: McpServer | undefined): boolean {
  return Boolean(server?.command?.startsWith("mrun mcp serve"));
}

export interface Described {
  verb: string;
  /** The path, command, pattern or URL, shown in mono. */
  subject: string;
  /** What the card shows below while asking: a command, a diff, the new file. */
  body?: { kind: "code" | "diff" | "json"; text?: string; old?: string; new?: string };
  /** A host beyond this Mac the call reaches. */
  reaches?: string;
}

const str = (v: unknown) => (typeof v === "string" ? v : v === undefined ? "" : JSON.stringify(v));

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** A call as its card's first line, from its tool and input. */
export function describe(call: ToolCall, server: McpServer | undefined, tool: string): Described {
  const i = call.input;
  if (builtin(server)) {
    switch (tool) {
      case "read":
        return { verb: "Read", subject: str(i.path) };
      case "write": {
        const text = str(i.content);
        const lines = text ? text.split("\n").length : 0;
        return { verb: "Write", subject: `${str(i.path)} · ${lines} ${lines === 1 ? "line" : "lines"}`, body: { kind: "code", text: text.length > 4000 ? `${text.slice(0, 4000)}\n…` : text } };
      }
      case "edit":
        return { verb: "Edit", subject: str(i.path), body: { kind: "diff", old: str(i.old), new: str(i.new) } };
      case "glob":
        return { verb: "Find", subject: str(i.pattern) };
      case "grep":
        return { verb: "Search", subject: i.path ? `${str(i.pattern)} in ${str(i.path)}` : str(i.pattern) };
      case "bash":
        return { verb: "Run", subject: str(i.command).split("\n")[0]!, body: str(i.command).includes("\n") ? { kind: "code", text: str(i.command) } : undefined };
      case "fetch":
        return { verb: "Fetch", subject: str(i.url), reaches: hostOf(str(i.url)) };
    }
  }
  const words = tool.replace(/[_-]+/g, " ").trim();
  const verb = words.charAt(0).toUpperCase() + words.slice(1);
  const keys = Object.keys(i);
  const one = keys.length === 1 && typeof i[keys[0]!] === "string" ? str(i[keys[0]!]) : "";
  return {
    verb,
    subject: one,
    body: keys.length && !one ? { kind: "json", text: JSON.stringify(i, null, 2) } : undefined,
    reaches: server?.kind === "http" ? server.host : undefined,
  };
}

/** The sandbox line of a server, as Settings and the picker show it. */
export function cageWords(s: McpServer): string {
  if (s.kind === "http") return `Remote · ${s.host}`;
  if (s.sandbox === false) return s.command === "mrun mcp serve fetch" ? "Reads the web · each call names the site" : "Not sandboxed";
  const where = s.sandbox.workspace ? s.sandbox.workspace.replace(/^\/Users\/[^/]+/, "~") : "~";
  const h = s.sandbox.hosts;
  const net = !h.length ? "no network" : h.includes("registry.npmjs.org") && h.length === 4 ? "package registries" : `${h.length} ${h.length === 1 ? "host" : "hosts"}`;
  return `Sandboxed · ${s.sandbox.write ? "writes" : "reads"} ${where} · ${net}`;
}
