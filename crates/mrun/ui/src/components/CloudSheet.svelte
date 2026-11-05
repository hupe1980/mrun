<script lang="ts" module>
  /** What the sheet starts: a catalog offer or a model the user added (024). */
  export interface CloudPick {
    repo: string;
    weights_bytes: number;
    context?: number;
    note?: string;
    /** The catalog's fitting GPU ids; else `need` GB decides. */
    fits?: string[];
    need: number;
  }
</script>

<script lang="ts">
  // Run a model on a rented GPU (024 US1), one sheet: the GPU with its price, Community Cloud
  // only when asked, what leaves the Mac and when billing ends, then a four-step stepper while
  // `mrun cloud up` works. The key is asked here once, if none is set, and kept in the Keychain.
  import { host } from "../lib/api";
  import { bytes, context, fittingGpus, perHour, type Cloud } from "../lib/format";
  import { jobs } from "../lib/jobs.svelte";
  import { live } from "../lib/live.svelte";
  import Symbol from "./Symbol.svelte";

  let { pick, onclose, onstarted }: { pick: CloudPick; onclose: () => void; onstarted: () => void } = $props();

  const listing = $derived(live.cloud);
  const hasKey = $derived(Boolean(listing?.status.key));
  let community = $state(false);
  const cloud = $derived<Cloud>(community ? "community" : "secure");
  const choices = $derived(fittingGpus(listing?.gpus ?? [], cloud, { fits: pick.fits, need: pick.need }));
  const secureBest = $derived(fittingGpus(listing?.gpus ?? [], "secure", { fits: pick.fits, need: pick.need })[0]);
  const communityBest = $derived(fittingGpus(listing?.gpus ?? [], "community", { fits: pick.fits, need: pick.need })[0]);
  // Community Cloud is offered only when it saves money (clarified: Secure by default).
  const saving = $derived(secureBest && communityBest && communityBest.price < secureBest.price ? secureBest.price - communityBest.price : 0);
  let chosenId = $state<string | null>(null);
  let showAll = $state(false);
  const chosen = $derived(choices.find((c) => c.gpu.id === chosenId) ?? choices[0]);
  const shown = $derived(showAll ? choices : choices.slice(0, 4));
  const target = $derived(chosen ? `${pick.repo}|${chosen.gpu.id}|${cloud}` : "");

  // A pod already rented: mrun keeps one per account.
  const other = $derived(listing?.status.pods.find((p) => p.model !== pick.repo));
  const localServer = $derived(live.app?.server.running && !live.remote ? (live.status?.model?.id ?? "the local server") : null);

  type Stage = "key" | "choose" | "starting" | "failed";
  let stage = $state<Stage>("choose");
  let secret = $state("");
  let keyError = $state<string | null>(null);
  let checkingKey = $state(false);
  let error = $state<string | null>(null);
  let copied = $state(false);
  let startTarget = $state("");

  $effect(() => {
    if (stage === "choose" && listing && !hasKey) stage = "key";
  });

  const job = $derived(startTarget ? jobs.of("cloud-up", startTarget) : undefined);
  const steps = [
    { id: "pod", title: "Rent GPU" },
    { id: "agent", title: "Start" },
    { id: "loading", title: "Load model" },
    { id: "ready", title: "Serve" },
  ] as const;
  const current = $derived(Math.max(0, steps.findIndex((s) => s.id === job?.progress?.step)));
  const loadFraction = $derived(job?.progress?.step === "loading" && job.progress.total > 0 ? Math.min(1, job.progress.bytes / job.progress.total) : null);

  async function saveKey(e: SubmitEvent) {
    e.preventDefault();
    const key = secret.trim();
    if (!key) return;
    checkingKey = true;
    keyError = null;
    // The key leaves page state as soon as it is sent (024 FR-005).
    secret = "";
    try {
      const done = await jobs.run("cloud-login", "", false, key);
      if (done.state === "done") {
        await live.refreshCloud();
        stage = "choose";
      } else keyError = done.error ?? "RunPod did not accept this key.";
    } catch (err) {
      keyError = String((err as Error).message);
    } finally {
      checkingKey = false;
    }
  }

  async function start() {
    if (!chosen || other) return;
    error = null;
    startTarget = target;
    stage = "starting";
    try {
      const done = await jobs.run("cloud-up", startTarget, true);
      if (done.state === "done") onstarted();
      else if (done.state === "cancelled") stage = "choose";
      else {
        error = done.error ?? "The pod did not start.";
        stage = "failed";
      }
    } catch (err) {
      error = String((err as Error).message);
      stage = "failed";
    }
  }

  async function copyCommand() {
    if (!target) return;
    const { text } = await host.command("cloud-up", target);
    await navigator.clipboard?.writeText(text).catch(() => {});
    copied = true;
    setTimeout(() => (copied = false), 1200);
  }

  async function stopOther() {
    if (!window.confirm("Stop the pod? Billing ends; the model loads again next time.")) return;
    await jobs.run("cloud-down", "", true);
  }

  function key(e: KeyboardEvent) {
    if (e.key === "Escape" && stage !== "starting") onclose();
    if (e.key === "Enter" && stage === "choose" && chosen && !other && !(e.target instanceof HTMLButtonElement)) {
      e.preventDefault();
      void start();
    }
  }

  const short = $derived(pick.repo.split("/").pop() ?? pick.repo);
</script>

<svelte:window onkeydown={key} />

<div class="scrim" role="presentation" onclick={() => stage !== "starting" && onclose()}></div>
<div class="sheet glass" role="dialog" aria-modal="true" aria-labelledby="cloud-title">
  <header>
    <div class="mark" aria-hidden="true"><Symbol name="cloud" size={26} weight={1.8} /></div>
    <div>
      <h1 id="cloud-title">Run {short} on a rented GPU</h1>
      <p class="sub num">{pick.repo} · {bytes(pick.weights_bytes)}{pick.context ? ` · ${context(pick.context)} context` : ""}</p>
    </div>
  </header>

  {#if stage === "key"}
    <form class="keyform" onsubmit={saveKey}>
      <label for="runpod-key">Your RunPod API key</label>
      <!-- svelte-ignore a11y_autofocus -->
      <input id="runpod-key" type="password" bind:value={secret} placeholder="rpa_…" autocomplete="off" spellcheck="false" autofocus />
      <small>Checked with RunPod, then kept in your Keychain. <button type="button" class="link" onclick={() => host.open("url", "https://www.runpod.io/console/user/settings")}>Create a key</button></small>
      {#if keyError}<p class="error" role="alert"><Symbol name="warning" size={13} /> {keyError}</p>{/if}
      <div class="buttons">
        <button type="button" class="btn" onclick={onclose}>Cancel</button>
        <button type="submit" class="btn primary" disabled={!secret.trim() || checkingKey}>{checkingKey ? "Checking…" : "Continue"}</button>
      </div>
    </form>
  {:else if stage === "choose"}
    {#if other}
      <p class="notice"><Symbol name="warning" size={14} /> A pod already serves {other.model ?? "another model"}. mrun rents one at a time.
        <button class="btn" onclick={stopOther} disabled={jobs.running("cloud-down", "")}>{jobs.running("cloud-down", "") ? "Stopping…" : "Stop Pod"}</button></p>
    {/if}
    {#if !listing}
      <p class="muted">Asking RunPod for prices…</p>
    {:else if !choices.length}
      <p class="notice"><Symbol name="warning" size={14} /> RunPod offers no GPU with {pick.need} GB or more in {community ? "Community" : "Secure"} Cloud right now.
        <button class="btn" onclick={() => live.refreshCloud()}>Check Again</button></p>
    {:else}
      <fieldset class="gpus">
        <legend>GPU</legend>
        {#each shown as c (c.gpu.id)}
          <label class="gpu" class:sel={c === chosen}>
            <input type="radio" name="gpu" value={c.gpu.id} checked={c === chosen} onchange={() => (chosenId = c.gpu.id)} />
            <span class="gname">{c.gpu.name}</span>
            <span class="num muted">{c.gpu.memory_gb} GB</span>
            <span class="num price">{perHour(c.price)}</span>
            <span class="stock {c.stock?.toLowerCase() ?? 'none'}">{c.stock ? `${c.stock.toLowerCase()} stock` : "sold out"}</span>
          </label>
        {/each}
        {#if choices.length > 4}
          <button class="link more" onclick={() => (showAll = !showAll)}>{showAll ? "Fewer GPUs" : `All ${choices.length} GPUs that fit`}</button>
        {/if}
      </fieldset>
    {/if}
    {#if saving > 0 || community}
      <label class="community">
        <input type="checkbox" bind:checked={community} onchange={() => (chosenId = null)} />
        <span>Cheaper on Community Cloud{communityBest ? `: ${perHour(communityBest.price)}` : ""}<small>Runs on machines other people host.</small></span>
      </label>
    {/if}
    <ul class="facts">
      <li><Symbol name="cloud" size={13} /> Your prompts and code go to RunPod while it runs.</li>
      <li><Symbol name="stop" size={11} /> Stops itself after 30 idle minutes, at most 8 hours. Billed per second.</li>
      {#if localServer}<li class="warn"><Symbol name="warning" size={13} /> Replaces {localServer} on this Mac while it runs.</li>{/if}
    </ul>
    <div class="footer">
      <button class="btn plain" disabled={!chosen} onclick={copyCommand}><Symbol name={copied ? "check" : "copy"} size={13} /> {copied ? "Copied" : "Copy as Command"}</button>
      <span class="grow"></span>
      <button class="btn" onclick={onclose}>Cancel</button>
      <button class="btn primary" disabled={!chosen || Boolean(other)} onclick={start}>
        {localServer ? "Replace and Start" : "Start"}{chosen ? ` · ${perHour(chosen.price)}` : ""}
      </button>
    </div>
  {:else}
    <ol class="stepper" aria-label="Starting">
      {#each steps as s, i (s.id)}
        {@const state = stage === "failed" ? (i < current ? "done" : i === current ? "failed" : "todo") : i < current ? "done" : i === current ? "now" : "todo"}
        <li class={state} aria-current={state === "now" ? "step" : undefined}>
          <span class="dot" aria-hidden="true">
            {#if state === "done"}<Symbol name="check" size={11} weight={2.4} />{:else if state === "failed"}<Symbol name="xmark" size={11} weight={2.4} />{:else}{i + 1}{/if}
          </span>
          <span class="label">{s.title}</span>
        </li>
      {/each}
    </ol>
    {#if stage === "starting"}
      <p class="line" aria-live="polite">{job?.progress?.text ?? "Asking RunPod for a GPU…"}</p>
      {#if loadFraction !== null && job?.progress}
        <div class="bar" role="progressbar" aria-label="Weights loaded" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(loadFraction * 100)}>
          <span style:width="{loadFraction * 100}%"></span>
        </div>
      {/if}
      <p class="muted small">Billing runs from the first step{chosen ? ` at ${perHour(chosen.price)}` : ""}. Cancel ends it.</p>
      <div class="footer">
        <span class="grow"></span>
        <button class="btn" onclick={onclose}>Hide</button>
        <button class="btn" onclick={() => jobs.cancel("cloud-up", startTarget)}>Cancel Start</button>
      </div>
    {:else}
      <p class="error" role="alert"><Symbol name="warning" size={14} /> {error}</p>
      <p class="muted small">A failed start leaves no pod running.</p>
      <div class="footer">
        <span class="grow"></span>
        <button class="btn" onclick={onclose}>Close</button>
        <button class="btn primary" onclick={() => (stage = "choose")}>Try Again</button>
      </div>
    {/if}
  {/if}
</div>

<style>
  /* Not capped like the place's other direct children (App's .scroll rule). */
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
    width: min(500px, calc(100vw - 32px));
    max-height: calc(100vh - 64px);
    overflow-y: auto;
    padding: var(--s5);
    border-radius: 22px;
    border: 0.5px solid var(--glass-edge);
    box-shadow: var(--shadow-sheet);
    display: grid;
    gap: var(--s3);
    animation: rise 380ms var(--ease) both;
  }
  @keyframes rise {
    from {
      opacity: 0;
      transform: translate(-50%, -46%) scale(0.98);
    }
  }
  header {
    display: flex;
    gap: var(--s3);
    align-items: center;
  }
  .mark {
    width: 48px;
    height: 48px;
    border-radius: 13px;
    flex: none;
    display: grid;
    place-items: center;
    color: #fff;
    background: linear-gradient(160deg, var(--remote-fill), var(--generating-fill));
    box-shadow: 0 8px 20px color-mix(in srgb, var(--remote-fill) 35%, transparent);
  }
  h1 {
    margin: 0;
    font: 700 17px/1.25 var(--display);
  }
  .sub {
    margin: 2px 0 0;
    color: var(--muted);
    font-size: 12px;
  }
  .keyform {
    display: grid;
    gap: 6px;
  }
  .keyform label {
    font-weight: 600;
  }
  .keyform input {
    font-family: var(--mono);
  }
  .keyform small,
  .muted {
    color: var(--muted);
  }
  .small {
    font-size: 11.5px;
    margin: 0;
  }
  .link {
    border: none;
    background: none;
    padding: 0;
    color: var(--accent-ink);
    font: inherit;
    cursor: pointer;
  }
  .gpus {
    border: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 2px;
    background: var(--surface);
    border-radius: var(--r-lg);
    padding: 4px;
  }
  legend {
    position: absolute;
    clip: rect(0 0 0 0);
  }
  .gpu {
    display: grid;
    grid-template-columns: auto 1fr auto auto 82px;
    gap: var(--s3);
    align-items: center;
    padding: 8px 10px;
    border-radius: var(--r);
    cursor: pointer;
  }
  .gpu.sel {
    background: var(--accent-soft);
  }
  .gname {
    font-weight: 550;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .price {
    font-weight: 600;
  }
  .stock {
    font-size: 11px;
    text-align: right;
    color: var(--muted);
  }
  .stock.high {
    color: var(--ready);
  }
  .stock.low {
    color: var(--warn);
  }
  .stock.none {
    color: var(--failed);
  }
  .more {
    justify-self: start;
    padding: 4px 10px;
    font-size: 12px;
  }
  .community {
    display: flex;
    gap: var(--s2);
    align-items: flex-start;
  }
  .community small {
    display: block;
    color: var(--muted);
    font-size: 11.5px;
  }
  .facts {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 4px;
    color: var(--text-2);
    font-size: 12px;
  }
  .facts li {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .facts .warn {
    color: var(--attention);
  }
  .notice {
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: wrap;
    margin: 0;
    padding: var(--s2) var(--s3);
    border-radius: var(--r);
    background: color-mix(in srgb, var(--warn-fill) 12%, transparent);
  }
  .footer,
  .buttons {
    display: flex;
    gap: var(--s2);
    align-items: center;
    justify-content: flex-end;
    margin-top: var(--s1);
  }
  .grow {
    flex: 1;
  }
  .error {
    display: flex;
    gap: 6px;
    align-items: flex-start;
    margin: 0;
    color: var(--failed);
  }
  .stepper {
    list-style: none;
    margin: var(--s2) 0 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
  }
  .stepper li {
    display: grid;
    justify-items: center;
    gap: 6px;
    position: relative;
    font-size: 11.5px;
    color: var(--muted);
  }
  /* The rail between two steps, filled once the left one is done. */
  .stepper li + li::before {
    content: "";
    position: absolute;
    top: 11px;
    right: 50%;
    width: 100%;
    height: 2px;
    background: var(--fill);
    z-index: -1;
  }
  .stepper li.done + li::before {
    background: var(--remote-fill);
  }
  .dot {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font: 600 11px/1 var(--rounded);
    background: var(--surface);
    border: 1.5px solid var(--fill);
  }
  .done .dot {
    background: var(--remote-fill);
    border-color: var(--remote-fill);
    color: #fff;
  }
  .now .dot {
    border-color: var(--remote-fill);
    color: var(--remote);
    animation: pulse 1.6s ease-in-out infinite;
  }
  .now .label {
    color: var(--text);
    font-weight: 600;
  }
  .failed .dot {
    background: var(--failed-fill);
    border-color: var(--failed-fill);
    color: #fff;
  }
  @keyframes pulse {
    50% {
      box-shadow: 0 0 0 5px color-mix(in srgb, var(--remote-fill) 22%, transparent);
    }
  }
  .line {
    margin: 0;
    text-align: center;
    color: var(--text-2);
    font-size: 12.5px;
    min-height: 1.4em;
  }
  .bar {
    height: 6px;
    border-radius: 3px;
    background: var(--fill);
    overflow: hidden;
  }
  .bar span {
    display: block;
    height: 100%;
    background: var(--remote-fill);
    transition: width 400ms linear;
  }
  @media (prefers-reduced-motion: reduce) {
    .sheet,
    .now .dot {
      animation: none;
    }
    .bar span {
      transition: none;
    }
  }
</style>
