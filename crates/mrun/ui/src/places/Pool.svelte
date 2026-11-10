<script lang="ts">
  // Pool (026): Macs on the local network running one model together, colleagues' included.
  // No pool: start one, or join one nearby. Hosting: the ring of Macs, joins to admit, a model
  // to plan and start. A member: its pool and Leave. Every action is its `mrun pool` command.
  import { onDestroy, onMount } from "svelte";
  import JoinSheet from "../components/JoinSheet.svelte";
  import PlanSheet from "../components/PlanSheet.svelte";
  import PoolRing from "../components/PoolRing.svelte";
  import Symbol from "../components/Symbol.svelte";
  import { host } from "../lib/api";
  import { bytes } from "../lib/format";
  import { jobs } from "../lib/jobs.svelte";
  import { live } from "../lib/live.svelte";
  import { backMacs, hostRing, memberWord, owner, placeWords, takePlan, type RingMac } from "../lib/pool";
  import type { PoolPlan, PoolView } from "../lib/types";

  let { onError }: { onError: (message: string) => void } = $props();

  let view = $state<PoolView | null>(null);
  let nearby = $state<NonNullable<PoolView["nearby"]>>([]);
  let model = $state("");
  let plan = $state<PoolPlan | null>(null);
  let planning = $state(false);
  let declined = $state<string[]>([]);
  // A pool encrypts its model traffic unless its host turns that off (constitution IV).
  let encrypt = $state(true);
  let timer: ReturnType<typeof setInterval> | null = null;
  let browsing = false;

  const pool = $derived(view?.pool ?? null);
  const hosting = $derived(pool?.role === "host" ? pool : null);
  const member = $derived(pool?.role === "member" ? pool : null);
  const waiting = $derived(hosting?.waiting.find((w) => !declined.includes(w.mac_id)) ?? null);
  const running = $derived((live.status?.requests ?? []).some((r) => r.phase !== "queued"));
  const pulled = $derived((live.models ?? []).filter((m) => m.pulled && !m.part).map((m) => m.alias ?? m.name));
  const back = $derived(hosting && !hosting.replanning ? backMacs(hosting.members, hosting.plan?.stages) : []);
  const recovery = $derived(hosting?.recovery ?? null);

  async function refresh() {
    try {
      view = await host.pool();
      if (!view.pool && !browsing) {
        browsing = true;
        host
          .pool(true)
          .then((v) => (nearby = v.nearby ?? []))
          .catch(() => {})
          .finally(() => (browsing = false));
      }
    } catch {
      view = null;
    }
  }

  async function act(kind: string, target = "") {
    try {
      const job = await jobs.run(kind, target);
      if (job.state === "failed") onError(job.error ?? "That did not work.");
    } catch (e) {
      onError(String((e as Error).message));
    }
    await refresh();
  }

  function admit(id: string) {
    void act("pool-admit", id);
  }

  function decline(id: string) {
    declined = [...declined, id];
    void act("pool-decline", id);
  }

  async function showPlan() {
    if (!model) return;
    planning = true;
    try {
      plan = await host.poolPlan(model);
    } catch (e) {
      onError(String((e as Error).message));
    }
    planning = false;
  }

  async function start() {
    if (!plan) return;
    await act("pool-serve", plan.model);
    plan = null;
  }

  const ringMacs = $derived.by<RingMac[]>(() => {
    if (!view) return [];
    const me: RingMac = { mac_id: view.this.mac_id ?? "this", name: view.this.name, chip: view.this.chip, layers: 0, host: true, budget: view.this.memory_bytes };
    if (hosting) return hostRing(me, hosting.members, hosting.plan?.stages);
    if (member) {
      return [
        { mac_id: member.host.mac_id, name: member.host.name, chip: "", layers: 0, host: true },
        { ...me, host: false, state: "serving" },
      ];
    }
    return [me];
  });

  const tps = $derived(live.status?.requests.find((r) => r.decode_tps)?.decode_tps ?? null);
  const speed = $derived(tps ? `${tps.toFixed(1)} tok/s` : running ? "Reading" : "Ready");
  const together = $derived(ringMacs.reduce((a, m) => a + (m.budget ?? 0), 0));

  onMount(() => {
    // Models asked for this model's plan.
    const asked = takePlan();
    if (asked) {
      model = asked;
      void showPlan();
    }
    void refresh();
    timer = setInterval(() => void refresh(), 1500);
  });
  onDestroy(() => timer && clearInterval(timer));
</script>

{#if view?.joining}
  <JoinSheet
    side="member"
    code={view.joining.code}
    pool={view.joining.pool}
    unencrypted={view.joining.unencrypted ?? false}
    other={view.joining.host}
    busy={jobs.running("pool-confirm", view.joining.code)}
    onyes={() => act("pool-confirm", view?.joining?.code ?? "")}
    onno={() => act("pool-leave")}
  />
{:else if waiting}
  <JoinSheet
    side="host"
    code={waiting.code}
    pool={hosting?.name ?? ""}
    other={waiting.name}
    detail="{waiting.chip.replace(/^Apple /, '')} · {bytes(waiting.memory_bytes)}{waiting.power === 'battery' ? ' · on battery' : ''}"
    busy={jobs.running("pool-admit", waiting.mac_id)}
    onyes={() => admit(waiting!.mac_id)}
    onno={() => decline(waiting!.mac_id)}
  />
{/if}

{#if plan && hosting}
  <PlanSheet {plan} members={hosting.members} hostName={view?.this.name ?? "this Mac"} busy={jobs.running("pool-serve", plan.model)} onstart={start} onclose={() => (plan = null)} />
{/if}

<div class="pool">
  {#if !pool && view?.rejoin}
    <div class="banner calm" role="status">
      <Symbol name="pool" size={14} />
      <span><strong>This Mac was in {owner(view.rejoin.host)}'s pool.</strong> Rejoin it without a new code.</span>
      <button class="btn primary" onclick={() => act("pool-rejoin")} disabled={jobs.running("pool-rejoin", "")}>Rejoin</button>
    </div>
  {/if}
  {#if !pool}
    <section class="hero">
      <PoolRing
        macs={ringMacs}
        title="This Mac"
        subtitle={bytes(view?.this.memory_bytes ?? 0)}
        detail="Add Macs to run a model none of them fits alone"
        label="This Mac alone"
      />
      <div class="pitch">
        <h2>One model, every Mac on the desk</h2>
        <p>Pool this Mac with others on your network. Each Mac runs part of the model's layers, and everyone's agents use it from their own Mac.</p>
        <p class="fine">A pool makes bigger models fit; it does not make a single reply faster. A cable or wired network keeps it quick.</p>
        <label class="encrypt">
          <input type="checkbox" bind:checked={encrypt} />
          <span>Encrypt the model's traffic between the Macs{#if !encrypt}<small> Off: the numbers between layers cross the network readable, for a cable you trust. Prompts and replies stay encrypted, and members are told.</small>{/if}</span>
        </label>
        <button class="btn primary large" onclick={() => act("pool-start", encrypt ? "" : "unencrypted")} disabled={jobs.running("pool-start", encrypt ? "" : "unencrypted")}>
          <Symbol name="plus" size={12} /> Start a Pool
        </button>
      </div>
    </section>

    <section class="card">
      <h2 class="section-title">Pools nearby</h2>
      {#if nearby.length}
        <ul class="nearby">
          {#each nearby as p (p.host_id)}
            <li>
              <span class="glyphs" aria-hidden="true">
                {#each Array(Math.min(p.members, 4)) as _, i (i)}<Symbol name="laptop" size={16} />{/each}
              </span>
              <span class="what">
                <strong>{p.name}</strong>
                <small>{p.members} {p.members === 1 ? "Mac" : "Macs"}{p.model ? ` · ${p.model}` : ""} · hosted by {owner(p.host)}</small>
              </span>
              <button class="btn" onclick={() => act("pool-join", p.addresses[0] ?? p.name)} disabled={jobs.running("pool-join", p.addresses[0] ?? p.name)}>Join</button>
            </li>
          {/each}
        </ul>
      {:else}
        <p class="empty">Looking on this network… When a colleague starts a pool, it appears here.</p>
      {/if}
    </section>
  {:else}
    {#if hosting?.replanning}
      <div class="banner calm" role="status">
        <Symbol name="pool" size={14} />
        <span><strong>Serving again on {ringMacs.filter((m) => m.state !== "lost").length} Macs.</strong> Each Mac reloads its layers; agents wait meanwhile.</span>
      </div>
    {:else if hosting?.degraded}
      <div class="banner" role="alert">
        <Symbol name="warning" size={14} />
        {#if recovery?.wait}
          <span><strong>{hosting.degraded}.</strong> The rest cannot hold {hosting.model}; the pool serves again when {owner(recovery.wait)}'s Mac is back.</span>
        {:else}
          <span><strong>{hosting.degraded}.</strong> Its requests ended.</span>
        {/if}
        {#if recovery?.smaller}
          <button class="btn primary" onclick={() => act("pool-serve", recovery.smaller ?? "")} disabled={jobs.running("pool-serve", recovery.smaller ?? "")}>Run {recovery.smaller} instead</button>
        {:else}
          <button class="btn" onclick={() => act("pool-leave")}>End Pool</button>
        {/if}
      </div>
    {:else if back.length}
      <div class="banner calm" role="status">
        <Symbol name="laptop" size={14} />
        <span><strong>{back.join(" and ")} {back.length === 1 ? "is" : "are"} back.</strong> The pool serves without {back.length === 1 ? "it" : "them"} until you re-plan.</span>
        <button class="btn primary" onclick={() => act("pool-replan")} disabled={jobs.running("pool-replan", "")}>Re-plan</button>
      </div>
    {/if}

    <section class="stage">
      <PoolRing
        macs={ringMacs}
        title={hosting?.model ?? member?.name ?? "Pool"}
        subtitle={hosting?.model ? speed : `${ringMacs.length} Macs`}
        detail={hosting ? `${bytes(together)} together` : member ? `hosted by ${owner(member.host.name)}` : ""}
        pace={hosting?.plan?.pace ?? null}
        {running}
        label="{pool.name}: {ringMacs.map((m) => m.name).join(', ')}"
      />
    </section>

    {#if hosting}
      <section class="card">
        <h2 class="section-title">Run a model across the pool</h2>
        <div class="choose">
          <select aria-label="Model" bind:value={model}>
            <option value="" disabled>Choose a model…</option>
            {#each pulled as m (m)}<option value={m}>{m}</option>{/each}
          </select>
          <button class="btn primary" onclick={showPlan} disabled={!model || planning}>{planning ? "Planning…" : "Plan"}</button>
        </div>
        <p class="fine">The plan shows each Mac's layers and memory, the context and the speed before anything loads.</p>
      </section>

      <section class="card">
        <h2 class="section-title">Macs in {hosting.name}</h2>
        <ul class="members">
          <li><span><strong>{view?.this.name}</strong> <small>host · this Mac</small></span></li>
          {#each hosting.members as m (m.mac_id)}
            <li>
              <span><strong>{m.name}</strong> <small>{m.chip.replace(/^Apple /, "")} · {bytes(m.memory_bytes)} · {memberWord(m.state)}</small></span>
              <button class="btn plain" onclick={() => act("pool-remove", m.mac_id)}>Remove</button>
            </li>
          {:else}
            <li class="empty">No Mac has joined yet. On another Mac: Pool → {hosting.name} → Join.</li>
          {/each}
        </ul>
        <footer><button class="btn" onclick={() => act("pool-leave")}>End Pool</button></footer>
      </section>
    {:else if member}
      <section class="card">
        <p class="lead">This Mac lends its memory to <strong>{member.name}</strong>. Your agents use the pool's model from this Mac as usual.</p>
        {#if placeWords(member.place)}<p class="fine num">{placeWords(member.place)}</p>{/if}
        <footer><button class="btn" onclick={() => act("pool-leave")}>Leave Pool</button></footer>
      </section>
    {/if}
  {/if}
</div>

<style>
  .pool {
    display: grid;
    gap: var(--s4);
  }
  .hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--s5);
    align-items: center;
  }
  .pitch {
    display: grid;
    gap: var(--s3);
    justify-items: start;
  }
  .pitch h2 {
    margin: 0;
    font: 700 22px var(--display);
    letter-spacing: -0.01em;
  }
  .pitch p {
    margin: 0;
    color: var(--text-2);
    line-height: 1.45;
  }
  .fine {
    font-size: 12px;
    color: var(--muted) !important;
  }
  .encrypt {
    display: flex;
    gap: var(--s2);
    align-items: flex-start;
    font-size: 13px;
  }
  .encrypt small {
    display: block;
    margin-top: 2px;
  }
  .card {
    padding: var(--s4);
    border-radius: var(--r-lg);
    background: var(--surface);
    box-shadow: var(--shadow);
    display: grid;
    gap: var(--s3);
  }
  .section-title {
    margin: 0;
    font: 650 13px var(--display);
  }
  .nearby,
  .members {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 2px;
  }
  .nearby li,
  .members li {
    display: flex;
    align-items: center;
    gap: var(--s3);
    padding: 8px 4px;
    border-top: 0.5px solid var(--separator);
  }
  .nearby li:first-child,
  .members li:first-child {
    border-top: none;
  }
  .glyphs {
    display: flex;
    gap: 1px;
    color: var(--accent);
    width: 70px;
  }
  .what,
  .members span {
    flex: 1;
    display: grid;
    gap: 1px;
  }
  small {
    color: var(--muted);
    font-size: 12px;
  }
  .empty {
    margin: 0;
    color: var(--muted);
  }
  .choose {
    display: flex;
    gap: var(--s2);
  }
  .choose select {
    flex: 1;
    font: inherit;
    color: var(--text);
    background: var(--surface);
    border: 0.5px solid var(--separator);
    border-radius: var(--r);
    padding: 7px 10px;
  }
  .choose select:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 1px;
  }
  .banner {
    display: flex;
    gap: var(--s2);
    align-items: center;
    padding: 10px 14px;
    border-radius: var(--r);
    background: var(--attention-fill);
    color: var(--text);
  }
  .banner span {
    flex: 1;
  }
  .banner.calm {
    background: var(--accent-soft);
  }
  .lead {
    margin: 0;
  }
  footer {
    display: flex;
    justify-content: flex-end;
  }
  @media (max-width: 900px) {
    .hero {
      grid-template-columns: 1fr;
    }
  }
</style>
