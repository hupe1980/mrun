<script lang="ts">
  // The first run, as one sheet like pairing AirPods (US3 AS1, FR-014): the model that fits
  // this Mac is already chosen; Continue downloads and starts it; Connect sets up an agent. At
  // most three decisions, and the terminal's commands one click away.
  import { bytes } from "../lib/format";
  import { jobs } from "../lib/jobs.svelte";
  import { live } from "../lib/live.svelte";
  import type { ModelEntry } from "../lib/types";
  import PullProgress from "./PullProgress.svelte";
  import Symbol from "./Symbol.svelte";

  let { ondone, onchoose }: { ondone: () => void; onchoose: () => void } = $props();

  type Step = "pick" | "pulling" | "starting" | "connect" | "failed";
  let step = $state<Step>("pick");
  let error = $state<string | null>(null);
  let terminal = $state(false);

  const model = $derived<ModelEntry | undefined>(
    live.models?.find((m) => m.recommended && m.fit !== "wont_fit") ?? live.models?.find((m) => m.fit === "fits"),
  );
  const name = $derived(model?.alias ?? model?.name ?? "");

  async function go() {
    if (!model) return;
    error = null;
    if (!model.pulled) {
      step = "pulling";
      const job = await jobs.run("pull", model.name);
      if (job.state !== "done") {
        if (job.state === "cancelled") return void (step = "pick");
        error = job.error ?? "The download did not finish.";
        return void (step = "failed");
      }
    }
    step = "starting";
    const job = await jobs.run("use", model.name, true);
    if (job.state !== "done") {
      error = job.error ?? "The server did not start.";
      return void (step = "failed");
    }
    step = "connect";
  }

  async function connect() {
    const job = await jobs.run("connect", "claude-code");
    if (job.state === "done") ondone();
    else error = job.error ?? "Claude Code could not be connected.";
  }
</script>

<div class="scrim"></div>
<div class="sheet glass" role="dialog" aria-modal="true" aria-labelledby="first-title">
  <div class="mark" aria-hidden="true"><Symbol name="bolt" size={30} weight={1.8} /></div>

  {#if step === "connect"}
    <h1 id="first-title">{name} is ready</h1>
    <p>Connect a coding agent, or chat with the model right away.</p>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    <div class="buttons">
      <button class="btn primary large" onclick={connect}>Connect Claude Code</button>
      <button class="btn large" onclick={ondone}>Done</button>
    </div>
  {:else}
    <h1 id="first-title">Set up mrun on this Mac</h1>
    {#if model}
      <div class="pick">
        <div class="model">
          <strong>{name}</strong>
          <span>{bytes(model.size)} · {model.verdict}</span>
          <small>The largest model that leaves room for long agent sessions on this Mac.</small>
        </div>
      </div>
    {:else}
      <p>Looking for a model that fits this Mac…</p>
    {/if}

    {#if step === "pulling"}
      {@const job = jobs.of("pull", model?.name ?? "")}
      {#if job}<PullProgress {job} />{/if}
      <div class="buttons"><button class="btn large" onclick={() => jobs.cancel("pull", model?.name ?? "")}>Cancel</button></div>
    {:else if step === "starting"}
      <p class="status">Loading {name} into memory…</p>
    {:else}
      {#if error}<p class="error" role="alert">{error}</p>{/if}
      <div class="buttons">
        <button class="btn primary large" disabled={!model} onclick={go}>{model?.pulled ? "Start" : "Continue"}</button>
        <button class="btn plain" onclick={onchoose}>Choose Another Model</button>
      </div>
    {/if}

    <button class="terminal btn plain" aria-expanded={terminal} onclick={() => (terminal = !terminal)}>Prefer the terminal?</button>
    {#if terminal}
      <pre class="commands">mrun pull {name}
mrun serve --detach
mrun connect claude-code --write</pre>
    {/if}
  {/if}
  <p class="privacy"><Symbol name="lock" size={12} /> Runs on this Mac. Nothing leaves it except this download.</p>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
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
    padding: var(--s6) var(--s6) var(--s4);
    border-radius: 24px;
    border: 0.5px solid var(--glass-edge);
    box-shadow: var(--shadow-sheet);
    text-align: center;
    display: grid;
    gap: var(--s3);
    animation: rise 420ms var(--ease) both;
  }
  @keyframes rise {
    from {
      opacity: 0;
      transform: translate(-50%, -46%) scale(0.98);
    }
  }
  .mark {
    width: 64px;
    height: 64px;
    margin: 0 auto;
    border-radius: 18px;
    display: grid;
    place-items: center;
    color: #fff;
    background: linear-gradient(160deg, var(--accent), var(--generating-fill));
    box-shadow: 0 10px 24px color-mix(in srgb, var(--accent) 35%, transparent);
  }
  h1 {
    margin: 0;
    font: 700 21px/1.25 var(--display);
  }
  p {
    margin: 0;
    color: var(--text-2);
  }
  .pick {
    background: var(--surface);
    border-radius: var(--r-lg);
    padding: var(--s3) var(--s4);
    text-align: left;
  }
  .model {
    display: grid;
    gap: 2px;
  }
  .model strong {
    font-size: 15px;
  }
  .model span {
    color: var(--ready);
  }
  .model small {
    color: var(--muted);
  }
  .buttons {
    display: grid;
    gap: var(--s2);
    justify-items: stretch;
  }
  .buttons .btn.large {
    justify-content: center;
  }
  .status {
    color: var(--muted);
  }
  .error {
    color: var(--failed);
  }
  .terminal {
    justify-self: center;
    font-size: 12px;
  }
  .commands {
    text-align: left;
    margin: 0;
    padding: var(--s3);
    border-radius: var(--r);
    background: var(--surface-2);
    font: 12px/1.6 var(--mono);
    user-select: all;
  }
  .privacy {
    display: inline-flex;
    gap: 5px;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 11px;
    margin-top: var(--s2);
  }
</style>
