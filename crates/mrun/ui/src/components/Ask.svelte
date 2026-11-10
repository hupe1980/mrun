<script lang="ts">
  // The window's confirmation sheet (see lib/ask): a question, Cancel and the action. Enter
  // confirms, Esc cancels.
  import { asking } from "../lib/ask.svelte";

  const q = $derived(asking.question);

  function key(e: KeyboardEvent) {
    if (!q) return;
    if (e.key === "Escape") (e.preventDefault(), q.answer(false));
    else if (e.key === "Enter") (e.preventDefault(), q.answer(true));
  }
</script>

<svelte:window onkeydown={key} />

{#if q}
  <div class="scrim" role="presentation" onclick={() => q.answer(false)}></div>
  <div class="ask glass" role="alertdialog" aria-modal="true" aria-labelledby="ask-title" aria-describedby={q.detail ? "ask-detail" : undefined}>
    <h2 id="ask-title">{q.title}</h2>
    {#if q.detail}<p id="ask-detail">{q.detail}</p>{/if}
    <div class="buttons">
      <button class="btn" onclick={() => q.answer(false)}>Cancel</button>
      <!-- svelte-ignore a11y_autofocus -->
      <button class="btn" class:primary={!q.destructive} class:danger={q.destructive} autofocus onclick={() => q.answer(true)}>{q.ok}</button>
    </div>
  </div>
{/if}

<style>
  .scrim {
    position: fixed;
    inset: 0;
    z-index: 80;
    background: rgba(0, 0, 0, 0.18);
  }
  .ask {
    position: fixed;
    z-index: 81;
    left: 50%;
    top: 96px;
    transform: translateX(-50%);
    width: min(400px, calc(100vw - 32px));
    padding: 20px 20px 16px;
    border-radius: var(--r-xl);
    border: 0.5px solid var(--glass-edge);
    box-shadow: var(--shadow-sheet);
  }
  h2 {
    margin: 0 0 6px;
    font: 650 15px var(--display);
  }
  p {
    margin: 0;
    font-size: 13px;
    color: var(--text-2);
    line-height: 1.4;
  }
  .buttons {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 18px;
  }
</style>
