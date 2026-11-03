<script lang="ts">
  // mrun's signature view: a prompt as one thin bar, the part reused from earlier work solid,
  // the part computed now light, and at the split, why (D053) in words. Exactly the server's
  // numbers (FR-006).
  import { cause, percent, reuseSentence, tokens, tone } from "../lib/format";
  import type { Miss } from "../lib/types";

  let {
    cached,
    total,
    miss = null,
    compact = false,
  }: { cached: number; total: number; miss?: Miss | null; compact?: boolean } = $props();

  const share = $derived(total > 0 ? Math.min(1, cached / total) : 0);
  const computed = $derived(Math.max(0, total - cached));
  const why = $derived(miss?.cause && computed > 0 ? cause(miss) : null);
  // Amber only for a recompute nobody intended (019 FR-005); expected ones stay grey.
  const attention = $derived(computed > 0 && tone(miss) === "attention");
</script>

<div class="ribbon" class:compact class:attention role="img" aria-label={reuseSentence(cached, total, miss)}>
  <div class="bar" aria-hidden="true">
    <span class="reused" style:width="{share * 100}%"></span>
    {#if computed > 0}<span class="computed"></span>{/if}
  </div>
  {#if !compact}
    <div class="caption" aria-hidden="true">
      <span class="num"><strong>{percent(cached, total)}</strong> reused · {tokens(cached)} of {tokens(total)}</span>
      {#if why}
        <span class="why">{tokens(computed)} computed · {why}</span>
      {:else if computed > 0}
        <span class="why num">{tokens(computed)} new</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .ribbon {
    min-width: 0;
  }
  .bar {
    display: flex;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    background: var(--fill);
  }
  .compact .bar {
    height: 4px;
    width: 64px;
  }
  .reused {
    background: var(--reused);
    transition: width 500ms var(--ease);
  }
  .computed {
    flex: 1;
    background: var(--computed);
  }
  .caption {
    display: flex;
    justify-content: space-between;
    gap: var(--s3);
    margin-top: 5px;
    font-size: 12px;
    color: var(--muted);
  }
  .attention .why {
    color: var(--attention);
    font-weight: 500;
  }
  .caption strong {
    color: var(--text);
    font-weight: 600;
  }
  .attention .computed {
    background: var(--attention-fill);
  }
  .why {
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
