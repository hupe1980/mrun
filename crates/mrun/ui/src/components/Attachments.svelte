<script lang="ts">
  // Files waiting to be sent (020 US1): name and token count, each removable; refusals stay in
  // view with their reason until dismissed (SC-003: never silent).
  import { tokens } from "../lib/format";
  import type { Attachment } from "../lib/types";
  import Symbol from "./Symbol.svelte";

  let {
    items,
    refused,
    onremove,
    ondismiss,
  }: {
    items: Attachment[];
    refused: { name: string; reason: string }[];
    onremove: (index: number) => void;
    ondismiss: () => void;
  } = $props();
</script>

{#if items.length || refused.length}
  <div class="attachments" aria-label="Attached files">
    {#each items as a, i (a.name + i)}
      <span class="chip" title="{a.name} · {a.bytes.toLocaleString('en-US')} bytes">
        <Symbol name="doc" size={13} />
        <span class="name">{a.name}</span>
        <span class="count num">{a.estimated ? "about " : ""}{tokens(a.tokens)} tokens</span>
        <button class="x" onclick={() => onremove(i)} aria-label="Remove {a.name}"><Symbol name="xmark" size={10} weight={2} /></button>
      </span>
    {/each}
    {#each refused as r, i (r.name + i)}
      <span class="chip refused" role="alert">
        <Symbol name="warning" size={13} />
        <span class="name">{r.name}</span>
        <span class="count">{r.reason}</span>
      </span>
    {/each}
    {#if refused.length}
      <button class="btn plain small" onclick={ondismiss}>Dismiss</button>
    {/if}
  </div>
{/if}

<style>
  .attachments {
    flex-basis: 100%;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    padding: 0 0 8px;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 100%;
    padding: 4px 6px 4px 9px;
    border-radius: 8px;
    background: var(--surface);
    border: 0.5px solid var(--separator);
    font-size: 12px;
  }
  .name {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 220px;
  }
  .count {
    color: var(--muted);
  }
  .refused {
    border-color: color-mix(in srgb, var(--attention-fill) 60%, transparent);
    padding-right: 9px;
  }
  .refused .count {
    color: var(--attention);
  }
  .x {
    display: grid;
    place-items: center;
    width: 18px;
    height: 18px;
    border: none;
    border-radius: 50%;
    background: var(--fill);
    color: var(--muted);
  }
  .x:hover {
    background: var(--fill-2);
    color: var(--text);
  }
  .small {
    height: 24px;
    font-size: 12px;
  }
</style>
