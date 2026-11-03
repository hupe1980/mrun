<script lang="ts">
  // Now: what mrun is doing, at a glance (US1). The headline is a sentence; the budget is one
  // bar; running work has real progress; every recent request shows how much of its prompt was
  // reused, and why not.
  import BudgetBar from "../components/BudgetBar.svelte";
  import Lane from "../components/Lane.svelte";
  import ReuseRibbon from "../components/ReuseRibbon.svelte";
  import Symbol from "../components/Symbol.svelte";
  import { bytes, context, percent, seconds, stopWord, tokens } from "../lib/format";
  import { live } from "../lib/live.svelte";

  let { onAction }: { onAction: (kind: string, target?: string) => void } = $props();

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
          sub: [s?.model?.family, s?.numerics?.backend, s?.budget ? `${context(s.budget.kv_capacity_tokens)} context` : null]
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
  .empty {
    margin: 0;
    color: var(--muted);
  }
  .foot {
    margin: 0;
    text-align: center;
    color: var(--faint);
    font-size: 11px;
  }
</style>
