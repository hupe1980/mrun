<script lang="ts">
  // The chat's tools (028 FR-007): which servers this chat may call, each with its tool count,
  // the prompt tokens its definitions add, and where it runs (sandboxed, not, or a remote host).
  // With none set up yet it offers mrun's own tools and an import, one click each.
  import { onMount } from "svelte";
  import { tokens as fmt } from "../lib/format";
  import { builtin, cageWords, tools } from "../lib/tools.svelte";
  import Symbol from "./Symbol.svelte";

  let {
    on = $bindable([]),
    model,
    onclose,
    onmanage,
  }: { on: string[]; model: string | null; onclose: () => void; onmanage: () => void } = $props();

  let busy = $state(false);
  let problem = $state<string | null>(null);

  onMount(() => {
    if (!tools.list?.servers.some((s) => s.tools || s.error)) void tools.load("start");
  });
  $effect(() => {
    if (model && tools.list) void tools.count(model);
  });

  function toggle(name: string, yes: boolean) {
    on = yes ? [...on.filter((n) => n !== name), name] : on.filter((n) => n !== name);
  }

  async function run(f: () => Promise<string | string[] | null>) {
    busy = true;
    problem = null;
    try {
      const got = await f();
      for (const name of Array.isArray(got) ? got : got ? [got] : []) toggle(name, true);
    } catch (e) {
      problem = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  const servers = $derived(tools.list?.servers ?? []);
</script>

<svelte:window onkeydown={(e) => e.key === "Escape" && (e.preventDefault(), onclose())} />
<div class="scrim" role="presentation" onclick={onclose}></div>
<div class="pop glass" role="dialog" aria-label="Tools for this chat">
  <header>
    <h3>Tools</h3>
    <span>Each call asks you first.</span>
  </header>

  {#if !servers.length && !tools.loading}
    <div class="empty">
      <button class="offer" disabled={busy} onclick={() => run(() => tools.addWorkspace())}>
        <Symbol name="tool" size={18} />
        <span><b>Workspace…</b><small>Read, edit and run in a folder you choose, sandboxed to it.</small></span>
      </button>
      <button class="offer" disabled={busy} onclick={() => run(() => tools.addFetch())}>
        <Symbol name="globe" size={18} />
        <span><b>Fetch</b><small>Read web pages; each call names the site.</small></span>
      </button>
      <button class="offer" disabled={busy} onclick={() => run(() => tools.importFrom("claude-code"))}>
        <Symbol name="download" size={18} />
        <span><b>Claude Code’s servers</b><small>Copied into ~/.mrun/mcp.json, sandboxed here.</small></span>
      </button>
    </div>
  {:else}
    <ul>
      {#each servers as s (s.name)}
        {@const n = s.tools?.length ?? 0}
        <li>
          <label class:off={!s.tools}>
            <input type="checkbox" role="switch" checked={on.includes(s.name)} disabled={!s.tools} onchange={(e) => toggle(s.name, e.currentTarget.checked)} />
            <span class="name">{s.name}</span>
            <span class="where" class:remote={s.kind === "http"} class:open={s.kind === "stdio" && s.sandbox === false && !builtin(s)}>
              <Symbol name={s.kind === "http" || builtin(s) ? "globe" : s.sandbox === false ? "warning" : "shield"} size={11} />{cageWords(s)}
            </span>
            {#if s.error}
              <span class="err" role="alert">{s.error}</span>
            {:else if s.tools}
              <span class="count num">{n} {n === 1 ? "tool" : "tools"}{tools.tokens[s.name] ? ` · ${fmt(tools.tokens[s.name]!)} tokens` : ""}</span>
            {:else}
              <span class="count">Starting…</span>
            {/if}
          </label>
        </li>
      {/each}
    </ul>
  {/if}
  {#if problem}<p class="err pad" role="alert">{problem}</p>{/if}

  <footer>
    <button class="btn plain" disabled={tools.loading} onclick={() => tools.load("refresh")}><Symbol name="reload" size={13} />{tools.loading ? "Starting…" : "Refresh"}</button>
    <button class="btn plain" onclick={onmanage}>Settings › Tools…</button>
  </footer>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    z-index: 40;
  }
  .pop {
    position: absolute;
    z-index: 41;
    bottom: calc(100% + 8px);
    right: 0;
    width: min(400px, calc(100vw - 32px));
    border-radius: var(--r-lg);
    border: 0.5px solid var(--glass-edge);
    box-shadow: var(--shadow-sheet);
    padding: var(--s2);
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 4px 8px 8px;
  }
  h3 {
    margin: 0;
    font: 650 13px var(--display);
  }
  header span {
    color: var(--muted);
    font-size: 12px;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    max-height: 320px;
    overflow-y: auto;
  }
  label {
    display: grid;
    grid-template-columns: auto 1fr auto;
    column-gap: 8px;
    align-items: center;
    padding: 7px 8px;
    border-radius: var(--r-sm);
  }
  label:hover {
    background: var(--fill);
  }
  label input {
    grid-row: span 2;
  }
  .name {
    font-weight: 600;
  }
  .where {
    grid-column: 2 / 4;
    display: inline-flex;
    gap: 4px;
    align-items: center;
    color: var(--muted);
    font-size: 11px;
  }
  .where.remote {
    color: var(--remote);
  }
  .where.open {
    color: var(--warn);
  }
  .count {
    grid-row: 1;
    grid-column: 3;
    color: var(--muted);
    font-size: 11px;
  }
  .err {
    color: var(--failed);
    font-size: 11px;
    grid-column: 2 / 4;
    overflow-wrap: anywhere;
  }
  .pad {
    padding: 0 8px;
  }
  .empty {
    display: grid;
    gap: 4px;
  }
  .offer {
    display: flex;
    gap: 10px;
    align-items: center;
    text-align: left;
    padding: 9px 10px;
    border: none;
    border-radius: var(--r);
    background: none;
    color: var(--accent-ink);
  }
  .offer:hover {
    background: var(--accent-soft);
  }
  .offer span {
    display: grid;
    color: var(--text);
  }
  .offer small {
    color: var(--muted);
    font-size: 11px;
  }
  footer {
    display: flex;
    justify-content: space-between;
    border-top: 0.5px solid var(--separator);
    margin-top: 6px;
    padding-top: 6px;
  }
</style>
