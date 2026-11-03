<script lang="ts">
  // The "…" menu on a row: its actions, and always "Copy as Command" with the exact CLI command
  // the UI runs (FR-004): the UI teaches the CLI and never does what the CLI cannot.
  import { host } from "../lib/api";
  import Symbol from "./Symbol.svelte";

  type Item = { label: string; run: () => void; danger?: boolean; disabled?: boolean };
  let {
    items = [],
    kind,
    target = "",
    label = "More",
  }: { items?: Item[]; kind: string; target?: string; label?: string } = $props();

  let open = $state(false);
  let copied = $state(false);
  let root: HTMLElement | undefined = $state();

  async function copyCommand() {
    try {
      const { text } = await host.command(kind, target);
      await navigator.clipboard.writeText(text);
      copied = true;
      setTimeout(() => ((copied = false), (open = false)), 900);
    } catch {
      open = false;
    }
  }

  function outside(e: MouseEvent) {
    if (open && root && !root.contains(e.target as Node)) open = false;
  }
</script>

<svelte:window onclick={outside} onkeydown={(e) => e.key === "Escape" && (open = false)} />

<div class="menu-root" bind:this={root}>
  <button class="btn plain dots" aria-haspopup="menu" aria-expanded={open} aria-label={label} onclick={() => (open = !open)}>
    <Symbol name="more" size={16} weight={2.4} />
  </button>
  {#if open}
    <div class="menu glass" role="menu">
      {#each items as item (item.label)}
        <button role="menuitem" class:danger={item.danger} disabled={item.disabled} onclick={() => ((open = false), item.run())}>{item.label}</button>
      {/each}
      {#if items.length}<hr />{/if}
      <button role="menuitem" onclick={copyCommand}>
        <Symbol name={copied ? "check" : "copy"} size={13} />{copied ? "Copied" : "Copy as Command"}
      </button>
    </div>
  {/if}
</div>

<style>
  .menu-root {
    position: relative;
  }
  .dots {
    width: 28px;
    padding: 0;
    justify-content: center;
    color: var(--muted);
  }
  .menu {
    position: absolute;
    right: 0;
    top: calc(100% + 4px);
    z-index: 20;
    min-width: 190px;
    padding: 5px;
    border-radius: var(--r);
    border: 0.5px solid var(--separator);
    box-shadow: var(--shadow);
    display: grid;
  }
  .menu button {
    display: flex;
    gap: 8px;
    align-items: center;
    text-align: left;
    border: none;
    background: none;
    padding: 4px 10px;
    border-radius: 5px;
  }
  .menu button:hover:not(:disabled) {
    background: var(--accent);
    color: #fff;
  }
  .menu button.danger {
    color: var(--failed);
  }
  .menu button.danger:hover {
    color: #fff;
    background: var(--failed-fill);
  }
  .menu button:disabled {
    opacity: 0.4;
  }
  hr {
    border: none;
    border-top: 0.5px solid var(--separator);
    margin: 4px 6px;
  }
</style>
