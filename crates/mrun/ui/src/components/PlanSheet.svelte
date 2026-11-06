<script lang="ts">
  // The plan before anything loads (026 US2): the same ring, each Mac's arc labelled with its
  // layers and memory; the context the pool leaves, the speed for these links, who sets the pace,
  // where prompts go, and the command. Start loads every part at once.
  import { host } from "../lib/api";
  import { copyText } from "../lib/clipboard";
  import { bytes, context } from "../lib/format";
  import type { RingMac } from "../lib/pool";
  import type { PoolMember, PoolPlan } from "../lib/types";
  import PoolRing from "./PoolRing.svelte";
  import Symbol from "./Symbol.svelte";

  let {
    plan,
    members,
    hostName,
    busy = false,
    onstart,
    onclose,
  }: {
    plan: PoolPlan;
    members: PoolMember[];
    hostName: string;
    busy?: boolean;
    onstart: () => void;
    onclose: () => void;
  } = $props();

  let copied = $state(false);
  const macs = $derived<RingMac[]>(
    plan.stages.map((s, i) => {
      const m = members.find((x) => x.mac_id === s.mac_id);
      return {
        mac_id: s.mac_id,
        name: s.name,
        chip: m?.chip ?? "",
        layers: s.end - s.first,
        first: s.first,
        end: s.end,
        used: s.weights_bytes + s.state_bytes + s.scratch_bytes,
        budget: s.budget,
        power: m?.power,
        host: i === 0,
      };
    }),
  );
  const speed = $derived(`${Math.round(plan.speed.low)}–${Math.round(plan.speed.high)} tok/s`);
  const others = $derived(plan.stages.length - 1);

  async function copy() {
    const { text } = await host.command("pool-serve", plan.model);
    copied = await copyText(text);
    setTimeout(() => (copied = false), 1500);
  }
  function key(e: KeyboardEvent) {
    if (e.key === "Escape" && !busy) onclose();
  }
</script>

<svelte:window onkeydown={key} />

<div class="scrim" role="presentation" onclick={() => !busy && onclose()}></div>
<div class="sheet glass" role="dialog" aria-modal="true" aria-labelledby="plan-title">
  <header>
    <h1 id="plan-title">{plan.model} on {plan.stages.length} Macs</h1>
    <button class="icon" onclick={onclose} aria-label="Close" disabled={busy}><Symbol name="xmark" size={13} /></button>
  </header>

  <PoolRing
    {macs}
    title={plan.fits ? "Context" : "Does not fit"}
    subtitle={plan.fits ? context(plan.context_tokens) : ""}
    compact
    detail={plan.fits ? `about ${speed}` : (plan.refusal ?? "")}
    pace={plan.pace}
    label="The plan: {macs.map((m) => `${m.name}, layers ${m.first! + 1} to ${m.end}`).join('; ')}"
  />

  <dl class="facts">
    {#each plan.stages as s (s.mac_id)}
      <dt>{s.name}</dt>
      <dd class="num" class:short={!s.fits}>
        layers {s.first + 1}–{s.end} · {bytes(s.weights_bytes + s.state_bytes + s.scratch_bytes)} of {bytes(s.budget)}{#if !s.fits} · {bytes(s.short_bytes)} short{/if}{#if s.download_bytes} · downloads {bytes(s.download_bytes)} first{/if}
      </dd>
    {/each}
  </dl>

  {#if plan.fits}
    {#if plan.pace}
      <p class="note"><Symbol name="bolt" size={12} /> {plan.pace.name}{plan.pace.kind === "link" ? "'s link" : ""} sets the pace{plan.pace.kind === "link" ? "; a cable would help" : ""}.</p>
    {/if}
    {#if !plan.numerics_match}
      <p class="note">These Macs have different GPUs: replies may differ in the last digits from one Mac's.</p>
    {/if}
    <p class="privacy"><Symbol name="lock" size={12} /> Prompts are read on {hostName} and processed on {others === 1 ? "one more Mac" : `${others} more Macs`} in this pool.</p>
  {/if}

  <footer>
    <button class="btn plain" onclick={copy}><Symbol name={copied ? "check" : "copy"} size={13} /> {copied ? "Copied" : "Copy as Command"}</button>
    <span class="spacer"></span>
    <button class="btn large" onclick={onclose} disabled={busy}>Cancel</button>
    <button class="btn primary large" onclick={onstart} disabled={!plan.fits || busy}>{busy ? "Loading on every Mac…" : "Start"}</button>
  </footer>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    max-width: none !important;
    margin: 0 !important;
    z-index: 50;
    background: rgba(0, 0, 0, 0.22);
  }
  .sheet {
    position: fixed;
    z-index: 51;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: min(560px, calc(100vw - 32px));
    max-height: calc(100vh - 48px);
    overflow-y: auto;
    padding: var(--s4) var(--s5);
    border-radius: 22px;
    border: 0.5px solid var(--glass-edge);
    box-shadow: var(--shadow-sheet);
    display: grid;
    gap: var(--s3);
    animation: rise 380ms var(--ease) both;
  }
  @keyframes rise {
    from {
      opacity: 0;
      transform: translate(-50%, -46%) scale(0.98);
    }
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  h1 {
    margin: 0;
    font: 700 17px var(--display);
  }
  .icon {
    border: none;
    background: none;
    color: var(--muted);
    padding: 4px;
    border-radius: var(--r-sm);
  }
  .facts {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 14px;
    margin: 0;
    font-size: 12px;
  }
  dt {
    color: var(--text-2);
  }
  dd {
    margin: 0;
    text-align: right;
  }
  dd.short {
    color: var(--failed);
  }
  .note,
  .privacy {
    display: flex;
    gap: 6px;
    align-items: center;
    margin: 0;
    font-size: 12px;
    color: var(--text-2);
  }
  .privacy {
    color: var(--remote);
  }
  footer {
    display: flex;
    align-items: center;
    gap: var(--s2);
  }
  .spacer {
    flex: 1;
  }
</style>
