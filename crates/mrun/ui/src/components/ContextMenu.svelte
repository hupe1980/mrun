<script lang="ts">
  // The page's context menu, for browsers and tests (019 FR-012); the window shows a native one.
  // Arrow keys move, Return picks, Esc or a click outside dismisses.
  import { tick } from "svelte";
  import { pageMenu } from "../lib/native.svelte";

  let menu: HTMLElement | undefined = $state();
  const open = $derived(pageMenu.open);

  $effect(() => {
    if (open) void tick().then(() => menu?.querySelector<HTMLElement>("button:not(:disabled)")?.focus());
  });

  function move(e: KeyboardEvent) {
    const buttons = [...(menu?.querySelectorAll<HTMLElement>("button:not(:disabled)") ?? [])];
    const at = buttons.indexOf(document.activeElement as HTMLElement);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const next = (at + (e.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next]?.focus();
    } else if (e.key === "Escape" || e.key === "Tab") {
      e.preventDefault();
      pageMenu.close(null);
    }
  }

  // Keep the menu inside the window.
  const left = $derived(open ? Math.min(open.x, window.innerWidth - 230) : 0);
  const top = $derived(open ? Math.min(open.y, window.innerHeight - 40 - open.items.length * 26) : 0);
</script>

{#if open}
  <div class="catcher" role="presentation" onmousedown={() => pageMenu.close(null)} oncontextmenu={(e) => (e.preventDefault(), pageMenu.close(null))}></div>
  <div class="menu glass" role="menu" tabindex="-1" bind:this={menu} style:left="{left}px" style:top="{top}px" onkeydown={move}>
    {#each open.items as item, i (i)}
      {#if item === "-"}
        <hr />
      {:else}
        <button role="menuitem" class:danger={item.destructive} disabled={item.enabled === false} onclick={() => pageMenu.close(item.id)}>{item.title}</button>
      {/if}
    {/each}
  </div>
{/if}

<style>
  .catcher {
    position: fixed;
    inset: 0;
    z-index: 60;
  }
  .menu {
    position: fixed;
    z-index: 61;
    min-width: 200px;
    max-width: 280px;
    padding: 5px;
    border-radius: var(--r);
    border: 0.5px solid var(--separator);
    box-shadow: var(--shadow);
    display: grid;
    outline: none;
  }
  button {
    text-align: left;
    border: none;
    background: none;
    padding: 4px 10px;
    border-radius: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  button:hover:not(:disabled),
  button:focus-visible {
    background: var(--accent);
    color: #fff;
    outline: none;
  }
  button.danger {
    color: var(--failed);
  }
  button.danger:hover,
  button.danger:focus-visible {
    color: #fff;
    background: var(--failed-fill);
  }
  button:disabled {
    opacity: 0.4;
  }
  hr {
    border: none;
    border-top: 0.5px solid var(--separator);
    margin: 4px 6px;
  }
</style>
