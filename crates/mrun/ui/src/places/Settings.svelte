<script lang="ts">
  // Settings, a sheet opened with ⌘, (FR-001, US6): few settings and strong defaults. Server
  // settings are shown with their source and edited in config.toml, which mrun never writes
  // (D066): one settings source for the CLI and the UI.
  import { onMount } from "svelte";
  import Symbol from "../components/Symbol.svelte";
  import { host } from "../lib/api";
  import { live } from "../lib/live.svelte";
  import { jobs } from "../lib/jobs.svelte";

  let { onclose }: { onclose: () => void } = $props();

  type Pane = "general" | "server" | "about";
  let pane = $state<Pane>("general");
  let config = $state<any>(null);
  let deleted = $state<number | null>(null);
  const prefs = $derived(live.app?.prefs);

  onMount(async () => {
    config = await host.config().catch(() => null);
  });

  async function setPref(key: string, value: boolean) {
    await host.prefs({ [key]: value } as any).catch(() => {});
    await live.tick();
  }

  async function deleteAll() {
    if (!window.confirm("Delete all chats kept on this Mac? This cannot be undone.")) return;
    deleted = (await host.deleteAllChats().catch(() => ({ deleted: 0 }))).deleted;
  }

  const serverRows = $derived.by(() => {
    const settings: any[] = config?.settings ?? config?.serve ?? [];
    const want = ["port", "model", "budget-gib", "max-output", "queue"];
    if (Array.isArray(settings)) return settings.filter((s: any) => want.includes(s.name ?? s.key));
    return Object.entries(settings)
      .filter(([k]) => want.includes(k))
      .map(([name, v]: [string, any]) => ({ name, value: v?.value ?? v, source: v?.source ?? "" }));
  });
</script>

<div class="scrim" role="presentation" onclick={onclose}></div>
<div class="sheet glass" role="dialog" aria-modal="true" aria-label="Settings">
  <header>
    <div class="tabs" role="tablist">
      {#each [["general", "General", "settings"], ["server", "Server", "now"], ["about", "About", "bolt"]] as [id, label, icon] (id)}
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
        <label class="row"><span>Show in menu bar<small>The model’s state and memory at a glance.</small></span>
          <input type="checkbox" role="switch" checked={prefs?.menu_bar ?? true} onchange={(e) => setPref("menu_bar", (e.currentTarget as HTMLInputElement).checked)} /></label>
        <label class="row"><span>Keep chats on this Mac<small>Saved in ~/.mrun/chats, never anywhere else.</small></span>
          <input type="checkbox" role="switch" checked={prefs?.keep_chats === true} onchange={(e) => setPref("keep_chats", (e.currentTarget as HTMLInputElement).checked)} /></label>
        <div class="row"><span>Kept chats<small>{deleted !== null ? `${deleted} deleted.` : "Remove every chat saved on this Mac."}</small></span>
          <button class="btn danger" onclick={deleteAll}>Delete All Chats…</button></div>
      </section>
      <p class="privacy"><Symbol name="lock" size={13} />
        {#if live.remote}Remote mode is on: prompts go to your RunPod pod.{:else}Runs on this Mac. Nothing leaves it: no account, no telemetry, no update checks.{/if}
      </p>
    {:else if pane === "server"}
      <section class="form">
        {#each serverRows as r (r.name)}
          <div class="row"><span>{r.name}<small>{r.source || "default"}</small></span><code>{String(r.value ?? "—")}</code></div>
        {:else}
          <p class="muted">Start the server to see its settings.</p>
        {/each}
      </section>
      <div class="buttons">
        <button class="btn" onclick={() => host.open("config")}>Edit config.toml…</button>
        <button class="btn" onclick={() => host.open("log")}>Show Server Log</button>
        <button class="btn primary" disabled={!live.app?.server.running || jobs.running("use", live.status?.model?.id ?? "")} onclick={() => live.status?.model && jobs.run("use", live.status.model.id, true)}>Restart Server</button>
      </div>
      <p class="muted small">mrun reads settings from defaults, then ~/.mrun/config.toml, then MRUN_* variables, then flags. The UI shows them; it never writes the file.</p>
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
