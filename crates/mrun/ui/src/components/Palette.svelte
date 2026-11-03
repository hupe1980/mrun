<script lang="ts">
  // ⌘K: every place and every action by name, like Spotlight for the window (FR-003). ⌘L opens
  // it filtered to models (the model picker).
  import { tick } from "svelte";
  import Symbol from "./Symbol.svelte";

  export type Command = { id: string; title: string; hint?: string; icon: string; run: () => void };
  let { commands, query = "", onclose }: { commands: Command[]; query?: string; onclose: () => void } = $props();

  // svelte-ignore state_referenced_locally
  let q = $state(query);
  let index = $state(0);
  let input: HTMLInputElement | undefined = $state();
  const shown = $derived(
    commands.filter((c) => !q || `${c.title} ${c.hint ?? ""}`.toLowerCase().includes(q.toLowerCase())).slice(0, 12),
  );

  $effect(() => {
    void tick().then(() => input?.focus());
  });

  function key(e: KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      index = Math.min(shown.length - 1, index + 1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      index = Math.max(0, index - 1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const c = shown[index];
      if (c) {
        onclose();
        c.run();
      }
    } else if (e.key === "Escape") {
      onclose();
    }
  }
</script>

<div class="scrim" role="presentation" onclick={onclose}></div>
<div class="palette glass" role="dialog" aria-modal="true" aria-label="Command palette">
  <div class="field">
    <Symbol name="search" size={16} />
    <input bind:this={input} bind:value={q} oninput={() => (index = 0)} onkeydown={key} placeholder="Search actions, places and models" aria-label="Search" role="combobox" aria-expanded="true" aria-controls="palette-list" />
  </div>
  <ul id="palette-list" role="listbox">
    {#each shown as c, i (c.id)}
      <li role="option" aria-selected={i === index}>
        <button class:sel={i === index} onmouseenter={() => (index = i)} onclick={() => (onclose(), c.run())}>
          <Symbol name={c.icon} size={15} /><span class="t">{c.title}</span>{#if c.hint}<span class="h">{c.hint}</span>{/if}
        </button>
      </li>
    {:else}
      <li class="none">No matches</li>
    {/each}
  </ul>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    z-index: 70;
  }
  .palette {
    position: fixed;
    z-index: 71;
    left: 50%;
    top: 18%;
    transform: translateX(-50%);
    width: min(560px, calc(100vw - 32px));
    border-radius: 16px;
    border: 0.5px solid var(--glass-edge);
    box-shadow: var(--shadow-sheet);
    overflow: hidden;
  }
  .field {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    color: var(--muted);
    border-bottom: 0.5px solid var(--separator);
  }
  .field input {
    flex: 1;
    border: none;
    background: none;
    font-size: 17px;
    outline: none;
    padding: 0;
    color: var(--text);
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 6px;
    max-height: 360px;
    overflow-y: auto;
  }
  li button {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border: none;
    border-radius: 8px;
    background: none;
    text-align: left;
  }
  li button.sel {
    background: var(--accent);
    color: #fff;
  }
  .t {
    flex: 1;
  }
  .h {
    font-size: 11px;
    opacity: 0.7;
  }
  .none {
    padding: 10px;
    color: var(--muted);
  }
</style>
