<script lang="ts">
  // A download's real progress: bytes, percent, rate and time left (FR-012).
  import { bytes, seconds } from "../lib/format";
  import type { Job } from "../lib/types";

  let { job }: { job: Job } = $props();
  const p = $derived(job.progress);
  const fraction = $derived(p && p.total > 0 ? p.bytes / p.total : 0);
</script>

<div class="pull">
  <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(fraction * 100)} aria-label="Download progress">
    <span style:width="{fraction * 100}%"></span>
  </div>
  <div class="caption num">
    {#if p}
      <span>{bytes(p.bytes)} of {bytes(p.total)} · {Math.round(fraction * 100)} %</span>
      <span>{p.rate ? `${bytes(p.rate)}/s` : ""}{p.eta_s ? ` · ${seconds(p.eta_s * 1000)} left` : ""}</span>
    {:else}
      <span>Preparing…</span>
    {/if}
  </div>
</div>

<style>
  .bar {
    height: 6px;
    border-radius: 3px;
    background: var(--fill);
    overflow: hidden;
  }
  .bar span {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 400ms linear;
  }
  .caption {
    display: flex;
    justify-content: space-between;
    gap: var(--s3);
    margin-top: 4px;
    font-size: 11.5px;
    color: var(--muted);
  }
</style>
