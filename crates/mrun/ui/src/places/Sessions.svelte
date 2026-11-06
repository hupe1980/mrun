<script lang="ts">
  // Sessions (025): each coding-agent session as a skyline of its requests, live while the agent
  // works, every miss one click (or ⌥→) from its cause and the text that changed. Reads
  // `/mrun/sessions` once a second while open; text only on selection.
  import { onDestroy, onMount, tick as settled } from "svelte";
  import RequestInspector from "../components/RequestInspector.svelte";
  import Skyline from "../components/Skyline.svelte";
  import Symbol from "../components/Symbol.svelte";
  import { host } from "../lib/api";
  import { live } from "../lib/live.svelte";
  import { fitWidth, layout as lay, merge, move, summary, type Key } from "../lib/sessions";
  import type { ReplayRecord, Session } from "../lib/types";

  let { focus = null, onfocused }: { focus?: number | null; onfocused?: () => void } = $props();

  let sessions = $state<Session[]>([]);
  let records = $state<ReplayRecord[]>([]);
  let remote = $state(false);
  let loaded = $state(false);
  /** The session picked by hand; `null` follows the newest. */
  let picked = $state<string | null>(null);
  let selectedId = $state<number | null>(null);
  let inspecting = $state(false);
  let following = $state(true);
  let columnWidth = $state(6);
  let frame = $state<HTMLDivElement | null>(null);
  let timer: ReturnType<typeof setInterval> | null = null;

  const session = $derived(sessions.find((s) => s.id === picked) ?? sessions[0] ?? null);
  const layout = $derived(session ? lay(session, records, columnWidth) : null);
  const selected = $derived(layout && selectedId !== null ? layout.columns.findIndex((c) => c.record.id === selectedId) : -1);
  const record = $derived(layout && selected >= 0 ? layout.columns[selected]!.record : null);
  const running = $derived(
    session && session.id === sessions[0]?.id ? (live.status?.requests.find((r) => r.phase !== "queued") ?? null) : null,
  );

  async function poll() {
    try {
      const last = records.at(-1)?.id;
      let r = await host.sessions(last);
      // A restarted server counts from 1 again: start over.
      const count = (id: string) => records.filter((x) => x.session === id).length;
      if (records.length && (r.sessions.length === 0 || r.sessions.some((s) => count(s.id) > s.requests))) {
        records = [];
        r = await host.sessions();
      }
      sessions = r.sessions;
      records = merge(records, r.requests);
      remote = r.remote;
    } catch {
      sessions = [];
      records = [];
    }
    loaded = true;
    if (focus !== null) reveal(focus);
    if (following) void toEnd();
  }

  function reveal(id: number) {
    const r = records.find((x) => x.id === id);
    if (!r) return;
    picked = r.session;
    selectedId = id;
    inspecting = true;
    following = false;
    onfocused?.();
    void settled().then(() => frame?.querySelector(`#col-${id}`)?.scrollIntoView?.({ inline: "center", block: "nearest" }));
  }

  async function toEnd() {
    await settled();
    const sky = frame?.querySelector<HTMLElement>(".skyline");
    if (sky) sky.scrollLeft = sky.scrollWidth;
  }

  function onScroll(e: Event) {
    const sky = e.target as HTMLElement;
    if (!sky.classList?.contains("skyline")) return;
    if (sky.scrollLeft + sky.clientWidth < sky.scrollWidth - 8) following = false;
  }

  function select(index: number | null, open = inspecting) {
    if (!layout || index === null || !layout.columns[index]) return;
    selectedId = layout.columns[index]!.record.id;
    inspecting = open;
    void settled().then(() => frame?.querySelector(`#col-${selectedId}`)?.scrollIntoView?.({ inline: "nearest", block: "nearest" }));
  }

  function step(key: Key) {
    if (!layout) return;
    select(move(layout.columns, selected >= 0 ? selected : null, key), key === "next" || key === "previous" ? true : inspecting);
  }

  function follow() {
    following = true;
    void toEnd();
  }

  function zoom(factor: number | "fit") {
    if (!session) return;
    const room = (frame?.querySelector<HTMLElement>(".skyline")?.clientWidth ?? 800) - 140;
    columnWidth = factor === "fit" ? fitWidth(session, records, room) : Math.min(24, Math.max(2, Math.round(columnWidth * factor)));
  }

  function onKey(e: KeyboardEvent) {
    const keys: Record<string, Key> = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down", Home: "first", End: "last" };
    if (e.altKey && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
      e.preventDefault();
      step(e.key === "ArrowRight" ? "next" : "previous");
    } else if (keys[e.key]) {
      e.preventDefault();
      step(keys[e.key]!);
    } else if (e.key === "Enter" && selected >= 0) {
      e.preventDefault();
      inspecting = true;
    } else if (e.key === "Escape" && inspecting) {
      e.preventDefault();
      inspecting = false;
    } else if (e.key === " ") {
      e.preventDefault();
      if (following) following = false;
      else follow();
    }
  }

  function onWindowKey(e: KeyboardEvent) {
    if (!(e.metaKey || e.ctrlKey)) return;
    if (e.key === "=" || e.key === "+") zoom(1.5);
    else if (e.key === "-") zoom(1 / 1.5);
    else if (e.key === "0") zoom("fit");
    else return;
    e.preventDefault();
  }

  function onWheel(e: WheelEvent) {
    // A pinch arrives as a wheel event with the control key.
    if (!e.ctrlKey) return;
    e.preventDefault();
    zoom(e.deltaY < 0 ? 1.1 : 1 / 1.1);
  }

  function pick(e: Event) {
    picked = (e.target as HTMLSelectElement).value;
    selectedId = null;
    inspecting = false;
    follow();
  }

  function start() {
    if (!timer) timer = setInterval(() => void poll(), 1000);
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }
  const visibility = () => (document.hidden ? stop() : (void poll(), start()));

  onMount(() => {
    void poll();
    start();
    document.addEventListener("visibilitychange", visibility);
  });
  onDestroy(() => {
    stop();
    document.removeEventListener("visibilitychange", visibility);
  });
</script>

<svelte:window onkeydown={onWindowKey} />

<div class="place" bind:this={frame}>
  {#if remote || live.remote}
    <div class="empty">
      <Symbol name="cloud" size={26} />
      <p>Session replay covers models on this Mac.</p>
    </div>
  {:else if !session || !layout}
    <div class="empty">
      <Symbol name="sessions" size={26} />
      {#if loaded}
        <p>No requests yet. Run <code>mrun launch claude-code</code> and work as usual; each request appears here.</p>
      {/if}
    </div>
  {:else}
    <section class="main">
      <header class="head">
        <div class="title">
          <select aria-label="Session" value={session.id} onchange={pick}>
            {#each sessions as s (s.id)}
              <option value={s.id}>{s.label} · {s.id}</option>
            {/each}
          </select>
          <p class="summary num">{summary(session)}</p>
        </div>
        <div class="tools">
          <button class="btn" onclick={() => step("previous")} disabled={!session.attention} title="Previous miss worth fixing (⌥←)" aria-label="Previous miss worth fixing"><Symbol name="chevron" size={12} /></button>
          <button class="btn" onclick={() => step("next")} disabled={!session.attention} title="Next miss worth fixing (⌥→)">Next Miss</button>
        </div>
      </header>

      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div class="track" onscrollcapture={onScroll} onwheel={onWheel}>
        <Skyline {layout} selected={selected >= 0 ? selected : null} {running} label="Requests of {session.label}, {summary(session)}" onselect={(i) => select(i, true)} onkey={onKey} />
      </div>

      <footer class="legend">
        <span class="keys">
          <span><i class="sw reused"></i>Reused</span>
          <span><i class="sw computed"></i>Computed</span>
          <span><i class="sw attention"></i>Worth fixing</span>
          <span><i class="sw wait"></i>Wait for first token</span>
        </span>
        {#if following}
          <span class="follow on" role="status">● Following</span>
        {:else}
          <button class="follow" onclick={follow}>Paused · Space to follow</button>
        {/if}
        <span class="zoom">
          <button class="btn" onclick={() => zoom(1 / 1.5)} title="Zoom out (⌘−)" aria-label="Zoom out">−</button>
          <button class="btn" onclick={() => zoom("fit")} title="Fit the session (⌘0)">Fit</button>
          <button class="btn" onclick={() => zoom(1.5)} title="Zoom in (⌘+)" aria-label="Zoom in">+</button>
        </span>
      </footer>
    </section>
    {#if record && inspecting}
      <RequestInspector {record} onclose={() => (inspecting = false)} />
    {/if}
  {/if}
</div>

<style>
  .place {
    display: flex;
    height: 100%;
    min-height: 0;
  }
  .main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s3);
    padding: 20px 24px;
    overflow-y: auto;
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: var(--s4);
  }
  .title {
    display: grid;
    gap: 4px;
    min-width: 0;
  }
  select {
    justify-self: start;
    font: 650 15px var(--display);
    color: var(--text);
    border: none;
    background: none;
    padding: 2px 0;
  }
  .summary {
    margin: 0;
    font-size: 12px;
    color: var(--muted);
  }
  .tools {
    display: flex;
    gap: 6px;
    flex: none;
  }
  .tools .btn:first-child :global(svg) {
    transform: rotate(90deg);
  }
  .track {
    min-width: 0;
    padding: 12px 14px 6px;
    border-radius: var(--r-lg);
    background: var(--surface);
    box-shadow: var(--shadow);
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 14px;
    white-space: nowrap;
    font-size: 11px;
    color: var(--muted);
  }
  .keys {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 4px 14px;
  }
  .sw {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 2px;
    margin-right: 5px;
    vertical-align: -0.5px;
  }
  .sw.reused {
    background: var(--reused);
  }
  .sw.computed {
    background: var(--computed);
  }
  .sw.attention {
    background: var(--attention-fill);
  }
  .sw.wait {
    height: 2px;
    opacity: 0.6;
    background: var(--reading-fill);
  }
  .follow {
    border: none;
    background: none;
    font: inherit;
    color: var(--muted);
    padding: 0;
  }
  .follow.on {
    color: var(--ready);
    font-weight: 500;
  }
  .zoom {
    display: flex;
    gap: 4px;
  }
  .zoom .btn {
    min-width: 26px;
    padding: 2px 8px;
  }
  .empty {
    margin: auto;
    display: grid;
    justify-items: center;
    gap: 10px;
    max-width: 380px;
    text-align: center;
    color: var(--muted);
  }
  .empty p {
    margin: 0;
    line-height: 1.45;
  }
  code {
    font-family: var(--mono);
    font-size: 12px;
  }
</style>
