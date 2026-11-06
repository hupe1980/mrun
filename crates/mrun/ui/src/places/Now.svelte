<script lang="ts">
  // Now: what mrun is doing, at a glance (US1). The headline is a sentence; the budget is one
  // bar; running work has real progress; every recent request shows how much of its prompt was
  // reused, and why not.
  import BudgetBar from "../components/BudgetBar.svelte";
  import Lane from "../components/Lane.svelte";
  import ReuseRibbon from "../components/ReuseRibbon.svelte";
  import Symbol from "../components/Symbol.svelte";
  import { bytes, context, percent, perHour, seconds, spend, stopsIn, stopWord, tokens, uptime } from "../lib/format";
  import { jobs } from "../lib/jobs.svelte";
  import { live } from "../lib/live.svelte";
  import { host } from "../lib/api";
  import { COLORS, hostRing, owner, placeWords, ring, type RingMac } from "../lib/pool";
  import { onDestroy } from "svelte";

  let { onAction, onshow }: { onAction: (kind: string, target?: string) => void; onshow?: (id: number) => void } = $props();

  const s = $derived(live.status);
  const srv = $derived(live.state);
  const active = $derived(s?.requests.filter((r) => r.phase !== "queued") ?? []);
  const waiting = $derived(s?.requests.filter((r) => r.phase === "queued") ?? []);
  const recent = $derived((s?.finished ?? []).slice().reverse().slice(0, 12));
  // A share means something only with some history (019 FR-006).
  const lifetime = $derived(
    s?.cache && s.cache.prompt_tokens_total > 0 && (s.finished?.length ?? 0) >= 3
      ? { cached: s.cache.cached_tokens_total, total: s.cache.prompt_tokens_total }
      : null,
  );
  // The rented GPU (024 US2): what it costs, and when it stops by itself.
  const pod = $derived(s?.pod ?? null);
  const stopping = $derived(jobs.running("cloud-down", ""));
  const starting = $derived(pod ? undefined : jobs.active("cloud-up"));
  const autoStop = $derived(pod ? stopsIn(pod.idle_s, pod.idle_stop_s) : null);
  async function stopPod() {
    if (!window.confirm("Stop the pod? Billing ends; the model loads again next time.")) return;
    const job = await jobs.run("cloud-down", "", true);
    if (job.state === "failed") onAction("notify", job.error ?? "The pod did not stop.");
  }

  // The pool (026): the ring compactly, what it serves and, on a member, its place in the queue.
  const pool = $derived(s?.pool ?? null);
  const poolHost = $derived(pool?.role === "host" && pool.members.length ? pool : null);
  const poolMember = $derived(pool?.role === "member" ? pool : null);
  const poolMacs = $derived.by<RingMac[]>(() => {
    if (poolHost) {
      const first = poolHost.plan?.stages[0];
      return hostRing({ mac_id: first?.mac_id ?? "this", name: first?.name ?? "This Mac", chip: "", layers: 0 }, poolHost.members, poolHost.plan?.stages);
    }
    if (poolMember) {
      return [
        { mac_id: poolMember.host.mac_id, name: poolMember.host.name, chip: "", layers: 0, host: true },
        { mac_id: "this", name: "This Mac", chip: "", layers: 0, state: "serving" },
      ];
    }
    return [];
  });
  let place = $state<string | null>(null);
  let placeTimer: ReturnType<typeof setInterval> | null = null;
  $effect(() => {
    if (poolMember && !placeTimer) {
      const ask = () =>
        host
          .pool()
          .then((v) => (place = v.pool?.role === "member" ? placeWords(v.pool.place) : null))
          .catch(() => {});
      void ask();
      placeTimer = setInterval(ask, 2000);
    } else if (!poolMember && placeTimer) {
      clearInterval(placeTimer);
      placeTimer = null;
      place = null;
    }
  });
  onDestroy(() => placeTimer && clearInterval(placeTimer));
  const poolShape = $derived(ring(poolMacs.map((m) => ({ mac_id: m.mac_id, name: m.name, layers: m.layers || 1 })), 36, 36, 28));
  const poolWaiting = $derived(poolHost?.queue?.waiting.reduce((a, w) => a + w.waiting, 0) ?? 0);

  const headline = $derived.by(() => {
    switch (srv) {
      case "stopped":
        return { title: "mrun is not running", sub: "Start the server to serve your agents and chat." };
      case "loading":
        return { title: `Loading ${s?.model?.id ?? "the model"}`, sub: "Reading the weights into memory. This takes a moment the first time." };
      case "failed":
        return { title: "The model could not be loaded", sub: "See the log for the reason, or pick a model that fits this Mac." };
      default:
        return {
          title: `${s?.model?.id ?? "The model"} is ${srv === "ready" ? "ready" : "working"}`,
          sub: [live.remote ? "On a rented GPU" : s?.model?.family, s?.numerics?.backend, s?.budget ? `${context(s.budget.kv_capacity_tokens)} context` : null]
            .filter(Boolean)
            .join(" · "),
        };
    }
  });
</script>

<div class="now">
  <header class="hero">
    <div class="state-icon {srv}" aria-hidden="true">
      <Symbol name={srv === "stopped" ? "stop" : srv === "loading" ? "download" : "bolt"} size={22} />
    </div>
    <div class="titles">
      <h1>{headline.title}</h1>
      {#if headline.sub}<p>{headline.sub}</p>{/if}
    </div>
    <div class="actions">
      {#if srv === "stopped"}
        <button class="btn primary large" onclick={() => onAction("start")}><Symbol name="play" size={12} /> Start</button>
      {:else if srv === "failed"}
        <button class="btn large" onclick={() => onAction("open-log")}>Show Log</button>
        <button class="btn primary large" onclick={() => onAction("models")}>Pick a Model That Fits</button>
      {:else if lifetime}
        <div class="lifetime" title="Prompt tokens mrun did not have to compute since the server started">
          <span class="big num">{percent(lifetime.cached, lifetime.total)}</span>
          <span class="small">of all prompt tokens reused</span>
        </div>
      {/if}
    </div>
  </header>

  {#if pod}
    <section class="card pod" class:unreachable={!pod.reachable} aria-label="Rented GPU">
      <div class="pod-icon" aria-hidden="true"><Symbol name="cloud" size={20} /></div>
      <div class="pod-main">
        <div class="pod-title">{s?.model?.id ?? "A model"} on RunPod{pod.gpu ? ` · ${pod.gpu}` : ""}</div>
        <div class="pod-facts num">
          {#if pod.cost_per_hr}<span>{perHour(pod.cost_per_hr)}</span>{/if}
          {#if pod.uptime_s !== undefined}<span>up {uptime(pod.uptime_s)}</span>{/if}
          {#if pod.spend_usd !== undefined}<span class="spend">about {spend(pod.spend_usd)} so far</span>{/if}
          {#if !pod.reachable}
            <span class="warn">not answering; it may have stopped itself</span>
          {:else if autoStop !== null}
            <span>{autoStop === 0 ? "stops now unless used" : `stops in ${autoStop} min unless used`}</span>
          {/if}
        </div>
      </div>
      <button class="btn" disabled={stopping} onclick={stopPod}><Symbol name="stop" size={10} /> {stopping ? "Stopping…" : "Stop Pod"}</button>
    </section>
  {:else if starting}
    <section class="card pod" aria-label="Starting a rented GPU">
      <div class="pod-icon" aria-hidden="true"><Symbol name="cloud" size={20} /></div>
      <div class="pod-main">
        <div class="pod-title">Starting {starting.target.split("|")[0]} on RunPod</div>
        <div class="pod-facts" aria-live="polite"><span>{starting.progress?.text ?? "Asking RunPod for a GPU…"}</span></div>
      </div>
      <button class="btn" onclick={() => jobs.cancel("cloud-up", starting.target)}>Cancel Start</button>
    </section>
  {/if}

  {#if poolMacs.length > 1}
    <section class="card pool-now" aria-label="Pool">
      <svg class="pool-ring" class:running={active.length > 0} viewBox="0 0 72 72" aria-hidden="true">
        {#each poolShape.arcs as arc, i (arc.mac_id)}
          <path d={arc.d} class:lost={poolMacs[i]?.state === "lost"} style:stroke="var({COLORS[i % COLORS.length]})" />
        {/each}
      </svg>
      <div class="pool-main">
        <div class="pod-title">{poolHost ? `${poolHost.model ?? poolHost.name} on ${poolMacs.length} Macs · you host` : `Lending to ${poolMember?.name}`}</div>
        <div class="pod-facts num">
          {#if poolHost?.degraded}<span class="warn">{poolHost.degraded}</span>
          {:else if poolHost?.replanning}<span>Serving again on the Macs answering</span>
          {:else if poolHost}<span>{poolWaiting ? `${poolWaiting} waiting in turns` : "Nothing waiting"}</span>
          {:else if poolMember}<span>{place ?? `Your agents use the model on ${owner(poolMember.host.name)}'s Mac`}</span>{/if}
        </div>
      </div>
      <button class="btn" onclick={() => onAction("pool")}>Open Pool</button>
    </section>
  {/if}

  {#if s?.budget && s.cache}
    <section class="card">
      <BudgetBar
        budget={s.budget.budget}
        weights={s.budget.weights}
        scratch={s.budget.scratch}
        context={s.cache.buffer_bytes}
        checkpoints={s.cache.checkpoint_bytes}
      />
    </section>
  {/if}

  {#if srv !== "stopped" && srv !== "failed"}
    <section class="card">
      <h2 class="section-title">Working now</h2>
      {#if active.length || waiting.length}
        <ul class="lanes">
          {#each [...active, ...waiting] as r (r.id)}<Lane request={r} />{/each}
        </ul>
      {:else}
        <p class="empty">Nothing is running. Requests from your agents and chats appear here as they arrive.</p>
      {/if}
    </section>

    <section class="card">
      <h2 class="section-title">Recent requests</h2>
      {#if recent.length}
        <ul class="recent">
          {#each recent as f (f.id)}
            <li>
              <div class="row-head">
                <span class="num">{tokens(f.prompt_tokens)} prompt tokens → {tokens(f.generated)} generated</span>
                <span class="meta num">
                  {#if f.ttft_ms !== null}{seconds(f.ttft_ms)} to first token{/if}
                  {#if f.decode_tps} · {f.decode_tps.toFixed(1)} tok/s{/if}
                  {#if f.stop} · {stopWord(f.stop)}{/if}
                  {#if onshow}<button class="show" onclick={() => onshow(f.id)} title="Show in Sessions" aria-label="Show request {f.id} in Sessions"><Symbol name="sessions" size={12} /></button>{/if}
                </span>
              </div>
              <ReuseRibbon cached={f.cached} total={f.prompt_tokens} miss={f} />
            </li>
          {/each}
        </ul>
      {:else if s?.last}
        <ReuseRibbon cached={s.last.cached} total={s.last.cached + s.last.processed} miss={s.last} />
      {:else}
        <p class="empty">Each request will show how much of its prompt mrun reused from earlier turns, and why any part had to be computed again.</p>
      {/if}
    </section>
  {/if}

  {#if s?.budget}
    <p class="foot num">
      Budget {bytes(s.budget.budget)} · {tokens(s.budget.kv_capacity_tokens)} tokens of context · {s.lanes.configured}
      {s.lanes.configured === 1 ? "lane" : "lanes"}
    </p>
  {/if}
</div>

<style>
  .now {
    display: grid;
    gap: var(--s4);
  }
  .hero {
    display: flex;
    align-items: center;
    gap: var(--s4);
    padding: var(--s2) var(--s1) var(--s3);
  }
  .state-icon {
    width: 46px;
    height: 46px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    color: #fff;
    background: var(--ready-fill);
    flex: none;
  }
  .state-icon.reading {
    background: var(--reading-fill);
  }
  .state-icon.generating {
    background: var(--generating-fill);
  }
  .state-icon.loading {
    background: var(--warn-fill);
  }
  .state-icon.stopped {
    background: var(--stopped-fill);
  }
  .state-icon.failed {
    background: var(--failed-fill);
  }
  .titles {
    flex: 1;
    min-width: 0;
  }
  h1 {
    margin: 0;
    font: 700 24px/1.2 var(--display);
    letter-spacing: -0.015em;
  }
  .titles p {
    margin: 2px 0 0;
    color: var(--muted);
  }
  .actions {
    display: flex;
    gap: var(--s2);
  }
  .lifetime {
    text-align: right;
    line-height: 1.1;
  }
  .lifetime .big {
    display: block;
    font: 700 28px/1 var(--rounded);
  }
  .lifetime .small {
    font-size: 11px;
    color: var(--muted);
  }
  .lanes,
  .recent {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .recent li {
    padding: var(--s3) 0;
    border-top: 0.5px solid var(--separator);
  }
  .recent li:first-child {
    border-top: none;
    padding-top: 0;
  }
  .row-head {
    display: flex;
    justify-content: space-between;
    gap: var(--s3);
    margin-bottom: 6px;
  }
  .meta {
    color: var(--muted);
    font-size: 12px;
  }
  .pool-now {
    display: flex;
    align-items: center;
    gap: var(--s4);
  }
  .pool-ring {
    width: 56px;
    height: 56px;
    flex: none;
  }
  .pool-ring path {
    fill: none;
    stroke-width: 7;
    stroke-linecap: round;
  }
  .pool-ring path.lost {
    stroke: var(--muted) !important;
    opacity: 0.4;
  }
  .pool-ring.running {
    animation: turn 6s linear infinite;
  }
  @keyframes turn {
    to {
      transform: rotate(360deg);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .pool-ring.running {
      animation: none;
    }
  }
  .pool-main {
    flex: 1;
    display: grid;
    gap: 4px;
  }
  .show {
    margin-left: 6px;
    padding: 1px 4px;
    border: none;
    border-radius: var(--r-sm);
    background: none;
    color: var(--accent);
    vertical-align: -2px;
  }
  .show:hover {
    background: var(--accent-soft);
  }
  .empty {
    margin: 0;
    color: var(--muted);
  }
  .pod {
    display: flex;
    gap: var(--s3);
    align-items: center;
    border: 0.5px solid color-mix(in srgb, var(--remote-fill) 35%, transparent);
    background: linear-gradient(110deg, color-mix(in srgb, var(--remote-fill) 9%, var(--surface)), var(--surface) 60%);
  }
  .pod-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    flex: none;
    display: grid;
    place-items: center;
    color: #fff;
    background: var(--remote-fill);
  }
  .pod.unreachable .pod-icon {
    background: var(--stopped-fill);
  }
  .pod-main {
    flex: 1;
    min-width: 0;
  }
  .pod-title {
    font-weight: 600;
  }
  .pod-facts {
    display: flex;
    flex-wrap: wrap;
    gap: 0 var(--s2);
    color: var(--muted);
    font-size: 12px;
  }
  .pod-facts span + span::before {
    content: "·";
    margin-right: var(--s2);
  }
  .pod-facts .spend {
    color: var(--remote);
    font-weight: 600;
  }
  .pod-facts .warn {
    color: var(--attention);
  }
  .foot {
    margin: 0;
    text-align: center;
    color: var(--faint);
    font-size: 11px;
  }
</style>
