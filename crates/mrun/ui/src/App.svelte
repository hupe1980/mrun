<script lang="ts">
  // The window: a glass sidebar with six places, a toolbar with the model and its state, and
  // the privacy line at the bottom (FR-001, FR-019). Keyboard-complete (FR-003): ⌘1–6 places,
  // ⌘K palette, ⌘L models, ⌘, settings; the chat adds its own keys.
  import { ask } from "./lib/ask.svelte";
  import { onMount } from "svelte";
  import ContextMenu from "./components/ContextMenu.svelte";
  import FirstRun from "./components/FirstRun.svelte";
  import Palette, { type Command } from "./components/Palette.svelte";
  import Ask from "./components/Ask.svelte";
  import Symbol from "./components/Symbol.svelte";
  import { adoptToken, host } from "./lib/api";
  import { bytes, stateWord } from "./lib/format";
  import { jobs } from "./lib/jobs.svelte";
  import { live } from "./lib/live.svelte";
  import { tools } from "./lib/tools.svelte";
  import { isNative, post } from "./lib/native.svelte";
  import Agents from "./places/Agents.svelte";
  import Chat from "./places/Chat.svelte";
  import Models from "./places/Models.svelte";
  import Now from "./places/Now.svelte";
  import Pool from "./places/Pool.svelte";
  import Sessions from "./places/Sessions.svelte";
  import Settings from "./places/Settings.svelte";

  type Place = "now" | "chat" | "sessions" | "pool" | "models" | "agents";
  const places: { id: Place; title: string; icon: string }[] = [
    { id: "now", title: "Now", icon: "now" },
    { id: "chat", title: "Chat", icon: "chat" },
    { id: "sessions", title: "Sessions", icon: "sessions" },
    { id: "pool", title: "Pool", icon: "pool" },
    { id: "models", title: "Models", icon: "models" },
    { id: "agents", title: "Agents", icon: "agents" },
  ];

  let place = $state<Place>("now");
  let chatOpened = $state(false);
  $effect(() => {
    if (place === "chat") chatOpened = true;
  });
  let settings = $state(false);
  let settingsPane = $state<string | undefined>(undefined);
  let palette = $state<{ query: string } | null>(null);
  let firstRunDismissed = $state(false);
  let toast = $state<string | null>(null);
  let token = $state<string | null>(null);
  /** A request Now asked Sessions to show (025). */
  let focus = $state<number | null>(null);

  onMount(() => {
    token = adoptToken();
    live.start();
    window.addEventListener("mrun:menu", ((e: CustomEvent) => onMenu(e.detail)) as EventListener);
  });

  const firstRun = $derived(
    !firstRunDismissed && live.models !== null && !live.models.some((m) => m.pulled) && live.state === "stopped",
  );
  const srv = $derived(live.state);
  // A pool (026): prompts are processed on its Macs; the line says where.
  const poolInfo = $derived(live.status?.pool ?? null);
  const pooled = $derived(Boolean(poolInfo && (poolInfo.role === "member" || (poolInfo.role === "host" && poolInfo.members.length))));
  const poolLine = $derived(
    !poolInfo
      ? ""
      : poolInfo.role === "host"
        ? `Pool: this Mac serves ${poolInfo.members.length + 1} Macs' prompts${poolInfo.unencrypted ? " · not encrypted" : ""}`
        : `Pool: prompts are read on ${poolInfo.host.name} and processed on the pool's Macs${poolInfo.unencrypted ? " · not encrypted" : ""}`,
  );
  const used = $derived(
    live.status?.budget && live.status.cache
      ? live.status.budget.weights + live.status.budget.scratch + live.status.cache.buffer_bytes + live.status.cache.checkpoint_bytes
      : null,
  );

  function notify(message: string) {
    toast = message;
    setTimeout(() => (toast = null), 6000);
  }

  async function action(kind: string, target = "") {
    if (kind === "models") return void (place = "models");
    if (kind === "pool") return void (place = "pool");
    if (kind === "open-log") return void host.open("log");
    if (kind === "notify") return void notify(target);
    let confirm = false;
    if (kind === "stop" && live.app?.server.started_by === "other") {
      // Agents in a terminal may use that server; say so before ejecting it.
      confirm = await ask(`Eject ${live.status?.model?.id ?? "the model"}?`, "Eject", { detail: "It was started outside this window; your agents may be using it.", destructive: true });
      if (!confirm) return;
    }
    try {
      const job = await jobs.run(kind, target, confirm);
      if (job.state === "failed") notify(job.error ?? "That did not work.");
    } catch (e) {
      notify(String((e as Error).message));
    }
  }

  /** Items of the native menu bar menu arrive as events from the window shell. */
  function onMenu(detail: { item: string; target?: string }) {
    if (detail.item === "new-chat") place = "chat";
    else if (detail.item === "connect" && detail.target) void action("connect", detail.target);
    else if (detail.item === "quit") {
      void ask("Quit mrun?", "Quit", { detail: "A download is still running; it resumes next time." }).then((yes) => { if (yes) void host.quit(); });
    }
    else if (detail.item === "settings") ((settingsPane = detail.target), (settings = true));
    else if (detail.item === "stop") void action("stop");
    else if (detail.item === "place" && detail.target) place = detail.target as Place;
  }

  const commands = $derived<Command[]>([
    ...places.map((p, i) => ({ id: `go-${p.id}`, title: `Go to ${p.title}`, hint: `⌘${i + 1}`, icon: p.icon, run: () => (place = p.id) })),
    { id: "new-chat", title: "New Chat", hint: "⌘N", icon: "plus", run: () => (place = "chat") },
    { id: "settings", title: "Settings", hint: "⌘,", icon: "settings", run: () => (settings = true) },
    srv === "stopped"
      ? { id: "start", title: "Start Server", icon: "play", run: () => action("start") }
      : { id: "stop", title: "Stop Server", icon: "stop", run: () => action("stop") },
    ...(live.models ?? [])
      .filter((m) => m.pulled && !m.serving && m.fit !== "wont_fit")
      .map((m) => ({ id: `use-${m.name}`, title: `Use ${m.alias ?? m.name}`, hint: m.verdict, icon: "models", run: () => action("use", m.name) })),
    ...(["claude-code", "codex", "opencode"] as const).map((c) => ({
      id: `connect-${c}`,
      title: `Connect ${({ "claude-code": "Claude Code", codex: "Codex", opencode: "OpenCode" })[c]}`,
      icon: "agents",
      run: () => action("connect", c),
    })),
  ]);

  // The window has no title bar of its own (019 FR-011): empty toolbar space and the sidebar's
  // top strip move it, and a double-click does what the system setting says.
  function chrome(e: MouseEvent) {
    if (e.button !== 0 || !isNative()) return;
    const target = e.target as HTMLElement;
    if (target.closest("button, input, textarea, select, a, [role=button], [role=menuitem]")) return;
    e.preventDefault();
    post(e.detail === 2 ? "zoom" : "drag");
  }

  function onKey(e: KeyboardEvent) {
    const cmd = e.metaKey || e.ctrlKey;
    if (!cmd) return;
    const n = Number(e.key);
    if (n >= 1 && n <= places.length) {
      e.preventDefault();
      place = places[n - 1]!.id;
    } else if (e.key === ",") {
      e.preventDefault();
      settings = true;
    } else if (e.key.toLowerCase() === "k") {
      e.preventDefault();
      palette = { query: "" };
    } else if (e.key.toLowerCase() === "l") {
      e.preventDefault();
      palette = { query: "Use " };
    } else if (e.key.toLowerCase() === "n" && place !== "chat") {
      e.preventDefault();
      place = "chat";
    }
  }
</script>

<svelte:window onkeydown={onKey} />

{#if live.unauthorized || (!token && typeof sessionStorage !== "undefined" && !sessionStorage.getItem("mrun.token"))}
  <main class="locked">
    <Symbol name="lock" size={28} />
    <h1>Open mrun from the app</h1>
    <p>This page needs the key mrun gives its own window. Run <code>mrun ui</code> to open it.</p>
  </main>
{:else}
  <div class="window" class:remote={live.remote}>
    <nav class="sidebar glass" aria-label="Places">
      <!-- The traffic lights sit here in the window; it is also a handle to move the window. -->
      <div class="titlebar" role="presentation" onmousedown={chrome}></div>
      <div class="brand">
        <div class="logo" aria-hidden="true"><Symbol name="bolt" size={15} weight={2} /></div>
        <span>mrun</span>
      </div>
      <ul>
        {#each places as p, i (p.id)}
          <li>
            <button class:current={place === p.id} aria-current={place === p.id ? "page" : undefined} onclick={() => (place = p.id)} title="{p.title} (⌘{i + 1})">
              <Symbol name={p.icon} size={17} />
              <span>{p.title}</span>
              {#if p.id === "now" && srv !== "ready"}<i class="badge {srv}" aria-label={stateWord[srv]}></i>{/if}
            </button>
          </li>
        {/each}
      </ul>
      <div class="spacer"></div>
      <button class="side-settings" onclick={() => (settings = true)} title="Settings (⌘,)"><Symbol name="settings" size={16} /> Settings</button>
      <div class="privacy" class:remote={live.remote || pooled || tools.hosts.length > 0}>
        <Symbol name={live.remote ? "cloud" : pooled ? "pool" : tools.hosts.length ? "globe" : "lock"} size={13} />
        {#if pooled}
          <button class="privacy-link" onclick={() => (place = "pool")}>{poolLine}</button>
        {:else if live.remote}
          <span>Remote: prompts go to your RunPod pod{live.status?.pod?.cost_per_hr ? ` · $${live.status.pod.cost_per_hr.toFixed(2)}/h` : ""}</span>
        {:else if tools.hosts.length}
          <span>Tools: {tools.hosts.join(", ")}</span>
        {:else}
          <span>Runs on this Mac. Nothing leaves it.</span>
        {/if}
      </div>
    </nav>

    <main class="content">
      <!-- Empty toolbar space is the window's title bar (FR-011); the controls in it stay controls. -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <header class="toolbar glass" onmousedown={chrome}>
        <h1 class="place-title">{places.find((p) => p.id === place)?.title}</h1>
        <div class="chipbar">
          <button class="chip" onclick={() => (palette = { query: "Use " })} title="Choose a model (⌘L)">
            <i class="dot {srv}" aria-hidden="true"></i>
            <span class="model">{live.status?.model?.id ?? "No model"}</span>
            <span class="word">{stateWord[srv]}</span>
            {#if used !== null && live.status?.budget}<span class="mem num">{bytes(used)} / {bytes(live.status.budget.budget)}</span>{/if}
          </button>
          {#if live.app?.server.running && srv !== "stopped"}
            <!-- Eject: the model leaves memory (mrun stop); ⌘L or a click on the chip loads one. -->
            <button class="eject" onclick={() => action("stop")} disabled={jobs.running("stop", "")} title="Eject {live.status?.model?.id ?? "the model"}: free its memory (mrun stop)" aria-label="Eject {live.status?.model?.id ?? "the model"}"><Symbol name="eject" size={14} /></button>
          {/if}
        </div>
      </header>
      <div class="scroll" class:flush={place === "chat" || place === "sessions"}>
        <!-- Chat stays mounted once opened: its chats, draft and a streaming reply live in it, and
             unsaved chats (Keep chats off) would be lost when the place changes. -->
        {#if chatOpened}<div class="keep" hidden={place !== "chat"}><Chat active={place === "chat" && !settings && !palette} /></div>{/if}
        {#if place === "now"}<Now onAction={action} onshow={(id) => ((focus = id), (place = "sessions"))} />
        {:else if place === "sessions"}<Sessions {focus} onfocused={() => (focus = null)} />
        {:else if place === "pool"}<Pool onError={notify} />
        {:else if place === "models"}<Models onError={notify} onstarted={() => (place = "now")} onpool={() => (place = "pool")} />
        {:else if place === "agents"}<Agents onError={notify} />{/if}
      </div>
    </main>
  </div>

  {#if firstRun}
    <FirstRun ondone={() => ((firstRunDismissed = true), (place = "now"))} onchoose={() => ((firstRunDismissed = true), (place = "models"))} />
  {/if}
  {#if settings}<Settings pane={settingsPane} onclose={() => ((settings = false), (settingsPane = undefined))} />{/if}
  {#if palette}<Palette {commands} query={palette.query} onclose={() => (palette = null)} />{/if}
  <Ask />
  <ContextMenu />
  {#if toast}
    <div class="toast glass" role="status"><Symbol name="warning" size={14} />{toast}</div>
  {/if}
{/if}

<style>
  .window {
    display: flex;
    height: 100vh;
    min-width: 800px;
    min-height: 560px;
  }
  .sidebar {
    width: 210px;
    flex: none;
    display: flex;
    flex-direction: column;
    padding: 14px 10px 12px;
    border-right: 0.5px solid var(--separator);
  }
  .titlebar {
    display: none;
  }
  :global(html[data-shell="native"]) .titlebar {
    display: block;
    height: 38px;
    margin: -14px -10px 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 2px 8px 14px;
    font: 700 15px var(--display);
    letter-spacing: -0.01em;
  }
  .logo {
    width: 24px;
    height: 24px;
    border-radius: 7px;
    display: grid;
    place-items: center;
    color: #fff;
    background: linear-gradient(160deg, var(--accent), var(--generating-fill));
  }
  .sidebar ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 2px;
  }
  .sidebar li button,
  .side-settings {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 6px 9px;
    border: none;
    border-radius: 7px;
    background: none;
    font-size: 13px;
    color: var(--text);
    text-align: left;
  }
  .sidebar li button:hover,
  .side-settings:hover {
    background: var(--fill);
  }
  .sidebar li button.current {
    background: var(--fill-2);
    font-weight: 600;
  }
  .sidebar li button :global(svg) {
    color: var(--accent);
  }
  .badge {
    margin-left: auto;
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
  .spacer {
    flex: 1;
  }
  .side-settings {
    color: var(--text-2);
    margin-bottom: 8px;
  }
  .privacy {
    display: flex;
    gap: 6px;
    align-items: flex-start;
    padding: 8px 9px 0;
    border-top: 0.5px solid var(--separator);
    color: var(--muted);
    font-size: 11px;
    line-height: 1.35;
  }
  .privacy.remote {
    color: var(--remote);
  }
  .privacy-link {
    border: none;
    background: none;
    padding: 0;
    font: inherit;
    color: inherit;
    text-align: left;
  }
  .content {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    position: relative;
  }
  .toolbar {
    height: 52px;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px 0 24px;
    border-bottom: 0.5px solid var(--separator);
    position: sticky;
    top: 0;
    z-index: 5;
  }
  .remote .toolbar {
    box-shadow: inset 0 -2px 0 var(--remote-fill);
  }
  .place-title {
    margin: 0;
    font: 650 15px var(--display);
  }
  .keep {
    display: contents;
  }
  .keep[hidden] {
    display: none;
  }
  .chipbar {
    display: flex;
    gap: 4px;
    align-items: center;
    margin-left: auto;
  }
  .eject {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 50%;
    background: var(--fill);
    color: var(--text-2);
    cursor: default;
  }
  .eject:hover {
    background: var(--fill-2);
    color: var(--text);
  }
  .eject:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .chip {
    display: flex;
    align-items: center;
    gap: 7px;
    height: 28px;
    padding: 0 12px;
    border-radius: 999px;
    border: 0.5px solid var(--separator);
    background: var(--surface);
    font-size: 12px;
  }
  .chip .model {
    font-weight: 600;
  }
  .chip .word,
  .chip .mem {
    color: var(--muted);
  }
  .dot,
  .badge {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--stopped-fill);
  }
  .ready {
    background: var(--ready-fill);
  }
  .reading {
    background: var(--reading-fill);
    animation: breathe 1.6s var(--ease) infinite;
  }
  .generating {
    background: var(--generating-fill);
    animation: breathe 1s var(--ease) infinite;
  }
  .loading {
    background: var(--warn-fill);
    animation: breathe 1.6s var(--ease) infinite;
  }
  .failed {
    background: var(--failed-fill);
  }
  @keyframes breathe {
    50% {
      opacity: 0.35;
    }
  }
  .scroll {
    flex: 1;
    overflow-y: auto;
    padding: 24px 28px 32px;
    min-height: 0;
  }
  .scroll:not(.flush) > :global(*) {
    max-width: 920px;
    margin-inline: auto;
  }
  .scroll.flush {
    padding: 0;
    overflow: hidden;
  }
  .toast {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 80;
    display: flex;
    gap: 8px;
    align-items: center;
    max-width: min(560px, calc(100vw - 40px));
    padding: 10px 16px;
    border-radius: 14px;
    box-shadow: var(--shadow-sheet);
    color: var(--text);
  }
  .locked {
    height: 100vh;
    display: grid;
    place-content: center;
    justify-items: center;
    gap: 8px;
    text-align: center;
    color: var(--text-2);
  }
  .locked h1 {
    margin: 4px 0 0;
    font: 700 20px var(--display);
    color: var(--text);
  }
  .locked p {
    margin: 0;
  }
  .locked code {
    font-family: var(--mono);
  }
</style>
