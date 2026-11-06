<script lang="ts">
  // Joining a pool (026 US1), like pairing AirDrop or Bluetooth: both Macs show the same six
  // digits. The member confirms they match what the host shows; the host admits the Mac it sees.
  // A Mac in between would show other digits. What lending means is said before anyone agrees.
  import { bytes } from "../lib/format";
  import { owner } from "../lib/pool";
  import Symbol from "./Symbol.svelte";

  let {
    side,
    code,
    pool,
    other,
    detail = "",
    unencrypted = false,
    busy = false,
    onyes,
    onno,
  }: {
    /** `member`: this Mac asked to join; `host`: another Mac asks to join this one's pool. */
    side: "member" | "host";
    code: string;
    pool: string;
    /** The other Mac's name. */
    other: string;
    /** The other Mac's chip and memory (host side). */
    detail?: string;
    /** The pool's model traffic is not encrypted (its host's choice). */
    unencrypted?: boolean;
    busy?: boolean;
    onyes: () => void;
    onno: () => void;
  } = $props();

  const digits = $derived(code.split(""));
  function key(e: KeyboardEvent) {
    if (e.key === "Escape") onno();
  }
</script>

<svelte:window onkeydown={key} />

<div class="scrim" role="presentation" onclick={onno}></div>
<div class="sheet glass" role="dialog" aria-modal="true" aria-labelledby="join-title" aria-describedby="join-code">
  <div class="mark" aria-hidden="true"><Symbol name="pool" size={30} weight={1.6} /></div>
  {#if side === "member"}
    <h1 id="join-title">Join {pool}?</h1>
    <p class="lead">Check that {other} shows the same code.</p>
  {:else}
    <h1 id="join-title">{other} wants to join</h1>
    <p class="lead">{detail}</p>
  {/if}

  <div class="code" id="join-code" aria-label="Code {code.split('').join(' ')}">
    {#each digits as d, i (i)}<span class:gap={i === 3}>{d}</span>{/each}
  </div>

  {#if side === "member"}
    <ul class="terms">
      <li><Symbol name="bolt" size={13} /> This Mac lends its memory and computes part of the model for everyone in {pool}.</li>
      <li><Symbol name="lock" size={13} /> Your prompts are read on {other}, where the model's endpoint is; the other Macs see only numbers between layers.</li>
      {#if unencrypted}<li><Symbol name="warning" size={13} /> {pool} does not encrypt those numbers between the Macs; prompts and replies stay encrypted.</li>{/if}
      <li><Symbol name="xmark" size={13} /> Leave any time from Pool or the menu bar.</li>
    </ul>
  {:else}
    <p class="terms-host">Admit only a Mac you expect: {owner(other)} sees this code on their screen.</p>
  {/if}

  <footer>
    <button class="btn large" onclick={onno} disabled={busy}>{side === "member" ? "Cancel" : "Decline"}</button>
    <button class="btn primary large" onclick={onyes} disabled={busy}>
      {#if busy}{side === "member" ? "Waiting for the host…" : "Admitting…"}{:else}{side === "member" ? "The Codes Match" : "Admit"}{/if}
    </button>
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
    width: min(440px, calc(100vw - 32px));
    padding: var(--s5) var(--s5) var(--s4);
    border-radius: 22px;
    border: 0.5px solid var(--glass-edge);
    box-shadow: var(--shadow-sheet);
    display: grid;
    justify-items: center;
    gap: var(--s3);
    text-align: center;
    animation: rise 380ms var(--ease) both;
  }
  @keyframes rise {
    from {
      opacity: 0;
      transform: translate(-50%, -46%) scale(0.98);
    }
  }
  .mark {
    width: 56px;
    height: 56px;
    border-radius: 16px;
    display: grid;
    place-items: center;
    color: #fff;
    background: linear-gradient(160deg, var(--pool-1), var(--pool-2));
  }
  h1 {
    margin: 0;
    font: 700 18px var(--display);
  }
  .lead {
    margin: -4px 0 0;
    color: var(--text-2);
    font-size: 13px;
  }
  .code {
    display: flex;
    gap: 6px;
    margin: 6px 0;
    font: 600 38px var(--rounded);
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.02em;
  }
  .code span {
    width: 34px;
    padding: 6px 0;
    border-radius: 10px;
    background: var(--fill);
  }
  .code span.gap {
    margin-left: 12px;
  }
  .terms {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 8px;
    text-align: left;
    font-size: 12.5px;
    color: var(--text-2);
  }
  .terms li {
    display: grid;
    grid-template-columns: 18px 1fr;
    gap: 4px;
  }
  .terms :global(svg) {
    margin-top: 2px;
    color: var(--accent);
  }
  .terms-host {
    margin: 0;
    font-size: 12.5px;
    color: var(--text-2);
  }
  footer {
    display: flex;
    gap: var(--s2);
    justify-content: center;
    margin-top: var(--s2);
  }
</style>
