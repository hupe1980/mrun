<script lang="ts">
  // Settings, a sheet opened with ⌘, (FR-001, US6): few settings and strong defaults. Server
  // settings are shown with their source and edited in config.toml, which mrun never writes
  // (D066): one settings source for the CLI and the UI.
  import { ask } from "../lib/ask.svelte";
  import { onMount } from "svelte";
  import Symbol from "../components/Symbol.svelte";
  import { host } from "../lib/api";
  import { live } from "../lib/live.svelte";
  import { jobs } from "../lib/jobs.svelte";
  import { builtin, cageWords, tools } from "../lib/tools.svelte";
  import { contextMenu } from "../lib/native.svelte";
  import type { ConfigView } from "../lib/types";

  let { onclose, pane: initial }: { onclose: () => void; pane?: string } = $props();

  type Pane = "general" | "server" | "tools" | "about";
  // svelte-ignore state_referenced_locally
  let pane = $state<Pane>((initial as Pane) ?? "general");
  let toolNote = $state<string | null>(null);
  let toolBusy = $state(false);
  $effect(() => {
    if (pane === "tools" && !tools.list?.servers.some((s) => s.tools || s.error)) void tools.load("start");
  });

  /** Runs an add or import and says what changed. */
  async function toolAction(f: () => Promise<string | string[] | null>, what: string) {
    toolBusy = true;
    toolNote = null;
    try {
      const got = await f();
      const names = Array.isArray(got) ? got : got ? [got] : null;
      if (names) toolNote = names.length ? `Added ${names.join(", ")}. Turn ${names.length === 1 ? "it" : "them"} on in a chat with the tools button.` : `Nothing new from ${what}.`;
    } catch (e) {
      toolNote = (e as Error).message;
    } finally {
      toolBusy = false;
    }
  }
  let config = $state<ConfigView | null>(null);
  let configFailed = $state(false);
  let deleted = $state<number | null>(null);
  const prefs = $derived(live.app?.prefs);

  onMount(async () => {
    void live.refreshCloud();
    config = await host.config().catch(() => {
      configFailed = true;
      return null;
    });
  });

  // The RunPod key (024 US3): where it comes from; added and removed here, kept in the Keychain.
  const keySource = $derived(live.cloud?.status.key ?? null);
  let addingKey = $state(false);
  let secret = $state("");
  let keyError = $state<string | null>(null);
  const keyBusy = $derived(jobs.running("cloud-login", "") || jobs.running("cloud-logout", ""));

  async function saveKey(e: SubmitEvent) {
    e.preventDefault();
    const key = secret.trim();
    secret = "";
    keyError = null;
    const job = await jobs.run("cloud-login", "", false, key).catch((err) => ({ state: "failed", error: String(err.message) }));
    if (job.state === "done") addingKey = false;
    else keyError = job.error ?? "RunPod did not accept this key.";
  }

  async function removeKey() {
    if (!(await ask("Remove the RunPod key from your Keychain?", "Remove", { destructive: true }))) return;
    await jobs.run("cloud-logout", "").catch(() => {});
  }

  async function setPref(key: string, value: boolean) {
    await host.prefs({ [key]: value } as any).catch(() => {});
    await live.tick();
  }

  async function deleteAll() {
    if (!(await ask("Delete all chats kept on this Mac?", "Delete All", { detail: "This cannot be undone.", destructive: true }))) return;
    deleted = (await host.deleteAllChats().catch(() => ({ deleted: 0 }))).deleted;
  }

  // The settings of `mrun serve`, as `mrun config show` reports them (`serve.` keys are
  // `[serve]` in config.toml); `null` means mrun picks the value itself.
  const SERVER: [string, string][] = [
    ["port", "Port"],
    ["model", "Model"],
    ["budget-gib", "Memory budget (GiB)"],
    ["serve.max-output", "Output limit"],
    ["serve.queue", "Queue"],
    ["serve.queue-wait-s", "Queue wait (s)"],
    ["serve.no-cache", "Prefix cache"],
    ["serve.speculation", "Speculative decoding"],
  ];
  const serverRows = $derived(
    SERVER.flatMap(([key, label]) => {
      const s = config?.settings.find((s) => s.key === key);
      if (!s) return [];
      // `no-cache` unset or false: the cache is on.
      const value = key === "serve.no-cache" ? (s.value === true ? "off" : "on") : shown(s.value);
      return [{ key, label, value, source: sourceWords(s.source) }];
    }),
  );

  function shown(value: unknown): string {
    if (value === null || value === undefined) return "automatic";
    if (typeof value === "boolean") return value ? "yes" : "no";
    return String(value);
  }

  function sourceWords(source: string): string {
    if (source.startsWith("file:")) return `config.toml, line ${source.slice(5)}`;
    if (source.startsWith("env:")) return source.slice(4);
    return source;
  }
</script>

<div class="scrim" role="presentation" onclick={onclose}></div>
<div class="sheet glass" role="dialog" aria-modal="true" aria-label="Settings">
  <header>
    <div class="tabs" role="tablist">
      {#each [["general", "General", "settings"], ["server", "Server", "now"], ["tools", "Tools", "tool"], ["about", "About", "bolt"]] as [id, label, icon] (id)}
        <button role="tab" aria-selected={pane === id} class:sel={pane === id} onclick={() => (pane = id as Pane)}>
          <Symbol name={icon!} size={18} /><span>{label}</span>
        </button>
      {/each}
    </div>
    <button class="btn plain close" onclick={onclose} aria-label="Close settings"><Symbol name="xmark" size={14} /></button>
  </header>

  <div class="body">
    {#if pane === "general"}
      <section class="form">
        <label class="row"><span>Open mrun at login<small>Starts in the menu bar when you log in.</small></span>
          <input type="checkbox" role="switch" checked={prefs?.open_at_login ?? false} onchange={(e) => setPref("open_at_login", (e.currentTarget as HTMLInputElement).checked)} /></label>
        <label class="row"><span>Show in menu bar<small>The model’s state and memory at a glance while mrun runs. Closing the window quits mrun and ejects the model it started.</small></span>
          <input type="checkbox" role="switch" checked={prefs?.menu_bar ?? true} onchange={(e) => setPref("menu_bar", (e.currentTarget as HTMLInputElement).checked)} /></label>
        <label class="row"><span>Keep chats on this Mac<small>Saved in ~/.mrun/chats, never anywhere else.</small></span>
          <input type="checkbox" role="switch" checked={prefs?.keep_chats === true} onchange={(e) => setPref("keep_chats", (e.currentTarget as HTMLInputElement).checked)} /></label>
        <div class="row"><span>RunPod key<small>{keySource === "keychain" ? "In your Keychain. Used to rent GPUs from Models." : keySource === "env" ? "From RUNPOD_API_KEY." : keySource === "dotenv" ? "From ./.env." : "Not set. Needed to rent a GPU from Models."}</small></span>
          {#if keySource === "keychain"}
            <button class="btn" disabled={keyBusy} onclick={removeKey}>Remove</button>
          {:else if keySource === null && !addingKey}
            <button class="btn" onclick={() => (addingKey = true)}>Add…</button>
          {/if}</div>
        {#if addingKey && keySource === null}
          <form class="row keyform" onsubmit={saveKey}>
            <!-- svelte-ignore a11y_autofocus -->
            <input type="password" bind:value={secret} placeholder="rpa_…" aria-label="RunPod API key" autocomplete="off" spellcheck="false" autofocus />
            <button type="button" class="btn plain" onclick={() => ((addingKey = false), (secret = ""), (keyError = null))}>Cancel</button>
            <button type="submit" class="btn primary" disabled={!secret.trim() || keyBusy}>{keyBusy ? "Checking…" : "Save"}</button>
          </form>
          {#if keyError}<p class="key-error" role="alert">{keyError}</p>{/if}
        {/if}
        <div class="row"><span>Kept chats<small>{deleted !== null ? `${deleted} deleted.` : "Remove every chat saved on this Mac."}</small></span>
          <button class="btn danger" onclick={deleteAll}>Delete All Chats…</button></div>
      </section>
      <p class="privacy"><Symbol name={live.remote ? "cloud" : "lock"} size={13} />
        {#if live.remote}Remote mode is on: prompts go to your RunPod pod.{:else}Runs on this Mac. Nothing leaves it: no account, no telemetry, no update checks.{/if}
      </p>
    {:else if pane === "server"}
      <section class="form">
        {#each serverRows as r (r.key)}
          <div class="row"><span>{r.label}<small>{r.source}</small></span><code>{r.value}</code></div>
        {:else}
          <p class="muted">{configFailed ? "mrun could not read its settings (mrun config show)." : "Reading the settings…"}</p>
        {/each}
      </section>
      <div class="buttons">
        <button class="btn" onclick={() => host.open("config")}>Edit config.toml…</button>
        <button class="btn" onclick={() => host.open("log")}>Open Server Log</button>
        <button class="btn primary" disabled={!live.app?.server.running || jobs.running("use", live.status?.model?.id ?? "")} onclick={() => live.status?.model && jobs.run("use", live.status.model.id, true)}>Restart Server</button>
      </div>
      <p class="muted small">mrun reads settings from defaults, then {config?.path ?? "~/.mrun/config.toml"}{config && !config.exists ? " (not created yet)" : ""}, then MRUN_* variables, then flags. The window shows them; it never writes the file.</p>
    {:else if pane === "tools"}
      <section class="form">
        {#each tools.list?.servers ?? [] as s (s.name)}
          <div class="row tool">
            <span>
              <b>{s.name}</b>
              <small class="cage" class:remote={s.kind === "http"} class:open={s.kind === "stdio" && s.sandbox === false && !builtin(s)}>{cageWords(s)}</small>
              {#if s.error}
                <small class="err" role="alert">{s.error}</small>
              {:else if s.tools}
                <span class="chips">{#each s.tools as t (t.name)}<code title={t.description}>{t.name}</code>{/each}</span>
              {:else}
                <small>{tools.loading ? "Starting…" : "Not started yet."}</small>
              {/if}
            </span>
            <code class="cmd" title={s.command ?? s.host}>{s.kind === "http" ? s.host : s.command}</code>
          </div>
        {:else}
          <p class="muted pad">{tools.loading ? "Starting the servers…" : tools.error ?? "No tool servers yet. Add mrun’s own, or import the ones Claude Code has."}</p>
        {/each}
      </section>
      <div class="buttons wrap">
        <button class="btn" disabled={toolBusy} onclick={() => toolAction(() => tools.addWorkspace(), "")} title="mrun mcp add workspace <folder>">Add Workspace…</button>
        <button class="btn" disabled={toolBusy || Boolean(tools.server("fetch"))} onclick={() => toolAction(() => tools.addFetch(), "")} title="mrun mcp add fetch">Add Fetch</button>
        <button class="btn" disabled={toolBusy} aria-haspopup="menu" onclick={(e) => void contextMenu(e, [{ id: "claude-code", title: "From Claude Code" }, { id: "claude-desktop", title: "From Claude Desktop" }]).then((id) => {
          if (id === "claude-code" || id === "claude-desktop") void toolAction(() => tools.importFrom(id), id === "claude-code" ? "Claude Code" : "Claude Desktop");
        })} title="mrun mcp import claude-code | claude-desktop">Import…</button>
        <span class="grow"></span>
        <button class="btn plain" onclick={() => host.open("mcp").then(() => tools.load(""))}>Edit…</button>
        <button class="btn plain" disabled={tools.loading} onclick={() => tools.load("refresh")} title="mrun mcp ls"><Symbol name="reload" size={13} /> Refresh</button>
      </div>
      {#if toolNote}<p class="small note" role="status">{toolNote}</p>{/if}
      <p class="muted small">Your chats call these tools; mrun’s server never does. Each call asks first. Local servers run sandboxed to their folder unless their entry in {tools.list?.path ?? "~/.mrun/mcp.json"} says <code>"sandbox": false</code>.</p>
    {:else}
      <section class="about">
        <div class="mark" aria-hidden="true"><Symbol name="bolt" size={26} weight={1.8} /></div>
        <h2>mrun</h2>
        <p class="num">Version {live.status?.version ?? live.app?.ui_version ?? "—"}</p>
        {#if live.status?.numerics}
          <p class="muted num">{live.status.numerics.backend} · {live.status.numerics.gpu_family} · NAX {live.status.numerics.nax}</p>
        {/if}
        <p class="muted">A local inference server for coding agents on Apple Silicon.</p>
      </section>
    {/if}
  </div>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    z-index: 60;
    background: rgba(0, 0, 0, 0.18);
  }
  .sheet {
    position: fixed;
    z-index: 61;
    left: 50%;
    top: 64px;
    transform: translateX(-50%);
    width: min(560px, calc(100vw - 32px));
    max-height: calc(100vh - 96px);
    overflow-y: auto;
    border-radius: var(--r-xl);
    border: 0.5px solid var(--glass-edge);
    box-shadow: var(--shadow-sheet);
  }
  header {
    display: flex;
    justify-content: center;
    position: relative;
    padding: var(--s2);
    border-bottom: 0.5px solid var(--separator);
  }
  .tabs {
    display: flex;
    gap: 2px;
  }
  .tabs button {
    display: grid;
    justify-items: center;
    gap: 2px;
    min-width: 72px;
    padding: 6px 10px;
    border: none;
    border-radius: var(--r-sm);
    background: none;
    color: var(--muted);
    font-size: 11px;
  }
  .tabs button.sel {
    background: var(--fill);
    color: var(--accent-ink);
  }
  .close {
    position: absolute;
    right: 8px;
    top: 12px;
  }
  .body {
    padding: var(--s4) var(--s5) var(--s5);
  }
  .form {
    background: var(--surface);
    border-radius: var(--r);
    padding: 0 var(--s4);
  }
  .row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--s4);
    padding: 10px 0;
    border-top: 0.5px solid var(--separator);
  }
  .row:first-child {
    border-top: none;
  }
  .row small {
    display: block;
    color: var(--muted);
    font-size: 11px;
  }
  .keyform {
    justify-content: flex-end;
  }
  .keyform input {
    flex: 1;
    font-family: var(--mono);
  }
  .key-error {
    margin: 0 0 var(--s2);
    color: var(--failed);
    font-size: 12px;
  }
  code {
    font: 12px var(--mono);
    color: var(--text-2);
  }
  .buttons {
    display: flex;
    gap: var(--s2);
    justify-content: flex-end;
    margin-top: var(--s3);
  }
  .privacy {
    display: flex;
    gap: 6px;
    align-items: center;
    color: var(--muted);
    font-size: 12px;
    margin: var(--s3) 0 0;
  }
  .muted {
    color: var(--muted);
  }
  .small {
    font-size: 11px;
  }
  .tool {
    align-items: flex-start;
  }
  .tool > span {
    min-width: 0;
  }
  .cage {
    color: var(--muted);
  }
  .cage.remote {
    color: var(--remote) !important;
  }
  .cage.open {
    color: var(--warn) !important;
  }
  .err {
    color: var(--failed) !important;
    overflow-wrap: anywhere;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
  }
  .chips code {
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--fill);
  }
  .cmd {
    max-width: 45%;
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--muted);
  }
  .pad {
    padding: var(--s3) 0;
    margin: 0;
  }
  .buttons.wrap {
    flex-wrap: wrap;
  }
  .grow {
    flex: 1;
  }
  .note {
    margin: var(--s2) 0 0;
  }
  .about {
    text-align: center;
    display: grid;
    gap: 4px;
    justify-items: center;
  }
  .about .mark {
    width: 56px;
    height: 56px;
    border-radius: 15px;
    display: grid;
    place-items: center;
    color: #fff;
    background: linear-gradient(160deg, var(--accent), var(--generating-fill));
  }
  .about h2 {
    margin: 6px 0 0;
    font: 700 18px var(--display);
  }
  .about p {
    margin: 0;
  }
</style>
