<script lang="ts">
  // The memory budget as one bar, like System Settings › Storage: what holds memory, in words
  // and sizes, inside the budget fixed at startup (D048). An overflow, when a request was
  // refused, is drawn past the end of the bar.
  import { bytes } from "../lib/format";

  let {
    budget,
    weights,
    scratch,
    context,
    checkpoints,
    overflow = 0,
  }: {
    budget: number;
    weights: number;
    scratch: number;
    context: number;
    checkpoints: number;
    overflow?: number;
  } = $props();

  const parts = $derived.by(() => {
    const used = weights + scratch + context + checkpoints;
    const free = Math.max(0, budget - used);
    return [
      { key: "weights", label: "Model", value: weights },
      { key: "scratch", label: "Working memory", value: scratch },
      { key: "context", label: "Conversations", value: context },
      { key: "checkpoints", label: "Checkpoints", value: checkpoints },
      { key: "free", label: "Free", value: free },
    ].filter((p) => p.key === "free" || p.value > 0);
  });
  const used = $derived(weights + scratch + context + checkpoints);
  const sentence = $derived(
    `Memory budget: ${bytes(used)} of ${bytes(budget)} in use. ` +
      parts.map((p) => `${p.label} ${bytes(p.value)}`).join(", ") +
      (overflow > 0 ? `. A request needed ${bytes(overflow)} more than the budget.` : "."),
  );
</script>

<div class="budget">
  <div class="head">
    <span class="title">Memory</span>
    <span class="figure num"><strong>{bytes(used)}</strong> of {bytes(budget)}</span>
  </div>
  <div class="track" role="img" aria-label={sentence}>
    {#each parts as p (p.key)}
      <span class="seg {p.key}" style:flex-grow={Math.max(p.value, budget * 0.004)}></span>
    {/each}
    {#if overflow > 0}
      <span class="over" style:width="{Math.min(30, (100 * overflow) / budget)}%"></span>
    {/if}
  </div>
  <ul class="legend" aria-hidden="true">
    {#each parts as p (p.key)}
      <li><i class="dot {p.key}"></i>{p.label}<span class="num">{bytes(p.value)}</span></li>
    {/each}
  </ul>
</div>

<style>
  .head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: var(--s2);
  }
  .title {
    font: 600 15px/1.2 var(--display);
  }
  .figure {
    color: var(--muted);
  }
  .figure strong {
    color: var(--text);
    font-weight: 600;
  }
  .track {
    position: relative;
    display: flex;
    gap: 2px;
    height: 14px;
    border-radius: 7px;
    overflow: visible;
  }
  .seg {
    flex-basis: 0;
    border-radius: 3px;
    transition: flex-grow 600ms var(--ease);
  }
  .seg:first-child {
    border-radius: 7px 3px 3px 7px;
  }
  .seg:last-of-type {
    border-radius: 3px 7px 7px 3px;
  }
  .weights {
    background: var(--mem-weights);
  }
  .scratch {
    background: var(--mem-scratch);
  }
  .context {
    background: var(--mem-context);
  }
  .checkpoints {
    background: var(--mem-checkpoints);
  }
  .free {
    background: var(--mem-free);
  }
  .over {
    position: absolute;
    left: 100%;
    top: 0;
    bottom: 0;
    margin-left: 3px;
    border-radius: 0 7px 7px 0;
    background: repeating-linear-gradient(
      -45deg,
      var(--failed-fill) 0 4px,
      transparent 4px 7px
    );
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s2) var(--s4);
    margin: var(--s3) 0 0;
    padding: 0;
    list-style: none;
    color: var(--text-2);
  }
  .legend li {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .legend .num {
    color: var(--muted);
  }
  .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
  }
  .dot.free {
    box-shadow: inset 0 0 0 1px var(--fill-2);
  }
</style>
