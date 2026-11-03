<script lang="ts">
  // Agents, like Internet Accounts: each client, whether it is connected to this mrun, and one
  // button. Connect does exactly what `mrun connect <client> --write` does, Disconnect what
  // `--restore` does, and both show the files they changed (FR-015).
  import { onMount } from "svelte";
  import ActionMenu from "../components/ActionMenu.svelte";
  import Symbol from "../components/Symbol.svelte";
  import { host } from "../lib/api";
  import { jobs } from "../lib/jobs.svelte";
  import { copyText } from "../lib/clipboard";
  import { live } from "../lib/live.svelte";
  import { apis, langs, snippet, type Lang } from "../lib/snippets";
  import type { AppEntry } from "../lib/types";
  import { contextMenu } from "../lib/native.svelte";
  import type { AgentEntry } from "../lib/types";

  let { onError }: { onError: (message: string) => void } = $props();

  // The last rows of this run show at once; the host refreshes them behind (019 US4).
  const agents = $derived(live.agents);
  let changed = $state<Record<string, { path: string; change: string }[]>>({});

  const names: Record<string, { title: string; site: string }> = {
    "claude-code": { title: "Claude Code", site: "https://docs.claude.com/en/docs/claude-code" },
    codex: { title: "Codex", site: "https://github.com/openai/codex" },
    opencode: { title: "OpenCode", site: "https://opencode.ai" },
  };

  async function load() {
    try {
      await live.refreshAgents();
    } catch (e) {
      live.agents ??= [];
      onError(String((e as Error).message));
    }
  }
  onMount(load);

  let lang = $state<Lang>("curl");
  let copied = $state<string | null>(null);
  let apps = $state<AppEntry[] | null>(null);
  const appNames: Record<string, string> = { zed: "Zed", crush: "Crush" };
  const port = $derived(live.app?.server?.port ?? 8484);
  const running = $derived(live.state !== "stopped");
  const modelId = $derived(live.status?.model?.id ?? live.app?.prefs?.last_model ?? "qwen3.6");
  const base = $derived(`http://127.0.0.1:${port}`);

  onMount(() => {
    host.apps().then((list) => (apps = Array.isArray(list) ? list : [])).catch(() => (apps = []));
  });

  async function copy(id: string, text: string) {
    if (await copyText(text)) {
      copied = id;
      setTimeout(() => copied === id && (copied = null), 1200);
    } else onError("Couldn't copy.");
  }

  async function act(kind: "connect" | "disconnect", a: AgentEntry) {
    const job = await jobs.run(kind, a.client);
    if (job.state === "failed") onError(job.error ?? `${names[a.client]?.title} could not be changed.`);
    const now = live.agents?.find((x) => x.client === a.client);
    if (now) changed[a.client] = now.files;
  }

  function menu(e: MouseEvent | KeyboardEvent, a: AgentEntry) {
    const title = names[a.client]?.title ?? a.client;
    const connect = a.connected
      ? { id: "disconnect", title: "Disconnect" }
      : { id: "connect", title: `Connect ${title}`, enabled: a.installed && !a.refusal };
    void contextMenu(e, [connect, "-", { id: "copy", title: "Copy as Command" }]).then(async (id) => {
      if (id === "connect" || id === "disconnect") await act(id, a);
      else if (id === "copy") {
        const shown = await host.command(a.connected ? "disconnect" : "connect", a.client);
        await navigator.clipboard?.writeText(shown.text).catch(() => {});
      }
    });
  }
</script>

<div class="agents">
  <p class="page-lede">Connect a coding agent to the model on this Mac. mrun writes the agent’s settings, keeps a backup, and puts them back when you disconnect.</p>

  {#if agents === null}
    <ul class="card list" aria-busy="true" aria-label="Loading agents">
      {#each Object.values(names) as n (n.title)}
        <li class="skeleton-row" aria-hidden="true">
          <div class="icon"></div>
          <div class="main"><span class="skeleton" style:width="30%"></span><span class="skeleton" style:width="45%" style:margin-top="8px"></span></div>
        </li>
      {/each}
    </ul>
  {:else}
    <ul class="card list">
      {#each agents as a (a.client)}
        {@const n = names[a.client] ?? { title: a.client, site: "" }}
        <!-- Right-click and Shift-F10 reach the row's actions; its buttons stay the primary way. -->
        <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
        <li oncontextmenu={(e) => menu(e, a)} onkeydown={(e) => (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) && menu(e, a)}>
          <div class="icon" class:on={a.connected} aria-hidden="true"><Symbol name="agents" size={18} /></div>
          <div class="main">
            <div class="name">{n.title}{#if a.version}<span class="ver">{a.version}</span>{/if}</div>
            {#if !a.installed}
              <div class="sub">Not installed. <a href={n.site} onclick={(e) => { e.preventDefault(); void host.open("url", n.site); }}>Get {n.title}</a></div>
            {:else if a.connected}
              <div class="sub ok"><Symbol name="check" size={12} weight={2} /> Connected to mrun</div>
            {:else if a.refusal}
              <div class="sub warn">{a.refusal}</div>
            {:else}
              <div class="sub">Not connected</div>
            {/if}
            {#if changed[a.client]?.length}
              <ul class="files">
                {#each changed[a.client] ?? [] as f (f.path)}<li><span class="chg">{f.change}</span> <code>{f.path}</code></li>{/each}
              </ul>
            {/if}
          </div>
          <div class="actions">
            {#if a.installed}
              {#if a.connected}
                <button class="btn" disabled={jobs.running("disconnect", a.client)} onclick={() => act("disconnect", a)}>Disconnect</button>
              {:else}
                <button class="btn primary" disabled={Boolean(a.refusal) || jobs.running("connect", a.client)} onclick={() => act("connect", a)}>Connect</button>
              {/if}
            {/if}
            <ActionMenu kind={a.connected ? "disconnect" : "connect"} target={a.client} label="More for {n.title}" />
          </div>
        </li>
      {/each}
    </ul>
  {/if}

  <!-- LM Studio's Developer tab, the mrun way: the live address and model in ready-to-paste
       code for all three APIs (020 FR-007), and the settings for print-only apps (FR-008). -->
  <section class="card code" aria-labelledby="code-title">
    <h2 id="code-title" class="section-title">Your code</h2>
    <p class="lede">
      Any program can use the model at <code>{base}</code> as <code>{modelId}</code>. No key is checked; SDKs need any string.
      {#if !running}<strong>Start the server to try them.</strong>{/if}
    </p>
    <div class="tabs" role="tablist" aria-label="Language">
      {#each langs as l (l.id)}
        <button role="tab" aria-selected={lang === l.id} class:on={lang === l.id} onclick={() => (lang = l.id)}>{l.title}</button>
      {/each}
    </div>
    {#each apis as a (a.id)}
      {@const text = snippet(a.id, lang, port, modelId)}
      <div class="snippet">
        <div class="snippet-bar">
          <span>{a.title}</span>
          <button class="btn plain" onclick={() => copy(a.id, text)} aria-label="Copy the {a.title} example">
            <Symbol name={copied === a.id ? "check" : "copy"} size={13} />{copied === a.id ? "Copied" : "Copy"}
          </button>
        </div>
        <pre><code>{text}</code></pre>
      </div>
    {/each}
  </section>

  {#if apps?.some((x) => x.blocks?.length)}
    <section class="card code" aria-labelledby="apps-title">
      <h2 id="apps-title" class="section-title">Other apps</h2>
      {#each apps as app (app.client)}
        {#each app.blocks ?? [] as b, i (b.path + i)}
          <div class="snippet">
            <div class="snippet-bar">
              <span>{appNames[app.client] ?? app.client} · <code>{b.path}</code>{b.note ? ` · ${b.note}` : ""}</span>
              <button class="btn plain" onclick={() => copy(`${app.client}${i}`, b.text)} aria-label="Copy the {appNames[app.client] ?? app.client} settings">
                <Symbol name={copied === `${app.client}${i}` ? "check" : "copy"} size={13} />{copied === `${app.client}${i}` ? "Copied" : "Copy"}
              </button>
            </div>
            <pre><code>{b.text}</code></pre>
          </div>
        {/each}
      {/each}
    </section>
  {/if}
</div>

<style>
  .agents {
    display: grid;
    gap: var(--s4);
  }
  .code .lede {
    margin: 0 0 var(--s3);
    color: var(--muted);
  }
  .code .lede code,
  .snippet-bar code {
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--text);
  }
  .tabs {
    display: inline-flex;
    padding: 2px;
    border-radius: 8px;
    background: var(--fill);
    margin-bottom: var(--s3);
  }
  .tabs button {
    border: none;
    background: none;
    padding: 3px 12px;
    border-radius: 6px;
    font-size: 12px;
  }
  .tabs button.on {
    background: var(--surface);
    box-shadow: 0 0.5px 2px rgba(0, 0, 0, 0.12);
    font-weight: 500;
  }
  .snippet {
    border: 0.5px solid var(--separator);
    border-radius: var(--r);
    overflow: hidden;
    margin-top: var(--s2);
  }
  .snippet-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--s2);
    padding: 2px 4px 2px 12px;
    background: var(--surface-2);
    font-size: 12px;
    color: var(--muted);
  }
  .snippet pre {
    margin: 0;
    padding: 10px 12px;
    overflow-x: auto;
    font: 12px/1.5 var(--mono);
  }
  .list {
    list-style: none;
    margin: 0;
    padding: 0 var(--s4);
  }
  .list > li {
    display: flex;
    gap: var(--s3);
    align-items: flex-start;
    padding: var(--s4) 0;
    border-top: 0.5px solid var(--separator);
  }
  .list > li:first-child {
    border-top: none;
  }
  .icon {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    display: grid;
    place-items: center;
    background: var(--fill);
    color: var(--text-2);
    flex: none;
  }
  .icon.on {
    background: var(--ready-fill);
    color: #fff;
  }
  .main {
    flex: 1;
    min-width: 0;
  }
  .name {
    font-weight: 600;
    font-size: 14px;
  }
  .ver {
    margin-left: 6px;
    font-weight: 400;
    color: var(--muted);
    font-size: 12px;
  }
  .sub {
    font-size: 12.5px;
    color: var(--muted);
    display: flex;
    gap: 4px;
    align-items: center;
  }
  .sub.ok {
    color: var(--ready);
  }
  .sub.warn {
    color: var(--warn);
  }
  .sub a {
    color: var(--accent-ink);
  }
  .files {
    margin: 6px 0 0;
    padding: 0;
    list-style: none;
    font-size: 11.5px;
    color: var(--text-2);
  }
  .chg {
    text-transform: capitalize;
    color: var(--muted);
  }
  code {
    font: 11px var(--mono);
  }
  .actions {
    display: flex;
    gap: 4px;
    align-items: center;
  }
</style>
