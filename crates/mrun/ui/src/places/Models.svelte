<script lang="ts">
  // Models, like Xcode's components: mrun's aliases and pulled models, each with the CLI's own
  // fit verdict for this Mac (FR-011), one marked Best for this Mac, and Pull, Use, Remove. A
  // model that won't fit cannot be loaded from here (US3 AS5).
  import ActionMenu from "../components/ActionMenu.svelte";
  import PullProgress from "../components/PullProgress.svelte";
  import Symbol from "../components/Symbol.svelte";
  import { host } from "../lib/api";
  import { bytes, repoFromInput } from "../lib/format";
  import { jobs } from "../lib/jobs.svelte";
  import { live } from "../lib/live.svelte";
  import { contextMenu } from "../lib/native.svelte";
  import type { ModelEntry, Plan } from "../lib/types";

  let { onError }: { onError: (message: string) => void } = $props();

  // The server's own word on what it serves wins over a listing a moment old (019 FR-008).
  const served = $derived(live.state === "stopped" ? null : (live.status?.model?.id ?? null));
  const models = $derived(
    (live.models ?? [])
      .map((m) => (served && [m.name, m.alias, m.repo].includes(served) ? { ...m, serving: true } : live.state === "stopped" ? { ...m, serving: false } : m))
      .sort((a, b) => Number(b.recommended) - Number(a.recommended) || Number(b.pulled) - Number(a.pulled)),
  );
  const best = $derived(models.find((m) => m.recommended && m.fit !== "wont_fit") ?? models.find((m) => m.fit === "fits"));
  let confirmRemove = $state<ModelEntry | null>(null);

  async function act(kind: string, m: ModelEntry) {
    let confirm = false;
    if (kind === "use" && live.app?.server.started_by === "other") {
      confirm = window.confirm("This server was started from a terminal. Switch it to " + m.name + " anyway?");
      if (!confirm) return;
    }
    try {
      const job = await jobs.run(kind, m.name, confirm);
      if (job.state === "failed" && job.error) onError(job.error);
    } catch (e) {
      onError(String((e as Error).message));
    }
  }

  function menu(e: MouseEvent | KeyboardEvent, m: ModelEntry) {
    const items = [
      !m.pulled
        ? { id: "pull", title: "Download", enabled: m.fit !== "wont_fit" }
        : m.serving
          ? { id: "stop", title: "Stop Serving" }
          : { id: "use", title: "Use", enabled: m.fit !== "wont_fit" },
      "-" as const,
      { id: "copy", title: "Copy as Command" },
      ...(m.pulled && !m.serving ? ["-" as const, { id: "remove", title: "Remove…", destructive: true }] : []),
    ];
    void contextMenu(e, items).then(async (id) => {
      if (id === "remove") confirmRemove = m;
      else if (id === "stop") await act("stop", m);
      else if (id === "use" || id === "pull") await act(id, m);
      else if (id === "copy") {
        const shown = await host.command(!m.pulled ? "pull" : m.serving ? "stop" : "use", m.name);
        await navigator.clipboard?.writeText(shown.text).catch(() => {});
      }
    });
  }

  // Add a model by name (020 US2): the CLI's own plan before a byte downloads.
  let addInput = $state("");
  let adding = $state<{ repo: string; plan: Plan | null; error: string | null; busy: boolean } | null>(null);

  async function check(e: SubmitEvent) {
    e.preventDefault();
    const repo = repoFromInput(addInput);
    if (!repo) {
      adding = { repo: addInput, plan: null, error: "Enter a model as org/name, or paste its Hugging Face link.", busy: false };
      return;
    }
    adding = { repo, plan: null, error: null, busy: true };
    try {
      const plan = await host.plan(repo);
      if (adding?.repo === repo) adding = { repo, plan, error: null, busy: false };
    } catch (err) {
      if (adding?.repo === repo) adding = { repo, plan: null, error: String((err as Error).message), busy: false };
    }
  }

  async function download(repo: string) {
    try {
      const job = await jobs.run("pull", repo);
      if (job.state === "failed" && job.error) onError(job.error);
      else if (job.state === "done") adding = null;
    } catch (err) {
      onError(String((err as Error).message));
    }
  }

  const verdictClass = (m: ModelEntry) =>
    m.fit === "fits" ? "fits" : m.fit === "tight" ? "tight" : m.fit === "wont_fit" ? "wont" : "unknown";
</script>

<div class="models">
  <p class="page-lede">Each verdict comes from the same memory rule the server applies. A model that won’t fit is never loaded halfway.</p>

  {#if live.models === null}
    <ul class="list card" aria-busy="true" aria-label="Loading models">
      {#each [0, 1, 2] as i (i)}
        <li aria-hidden="true">
          <div class="icon"></div>
          <div class="main"><span class="skeleton" style:width="34%"></span><span class="skeleton" style:width="52%" style:margin-top="8px"></span><span class="skeleton" style:width="40%" style:margin-top="8px"></span></div>
        </li>
      {/each}
    </ul>
  {:else}
    <ul class="list card">
      {#each models as m (m.name)}
        {@const pull = jobs.of("pull", m.name)}
        <!-- Right-click and Shift-F10 reach the row's actions; its buttons stay the primary way. -->
        <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
        <li class:serving={m.serving} oncontextmenu={(e) => menu(e, m)} onkeydown={(e) => (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) && menu(e, m)}>
          <div class="icon" aria-hidden="true"><Symbol name="models" size={20} /></div>
          <div class="main">
            <div class="name">
              <span>{m.alias ?? m.name}</span>
              {#if m === best}<span class="tag best">Best for this Mac</span>{/if}
              {#if m.preview}<span class="tag">Preview</span>{/if}
            </div>
            <div class="sub">{m.alias ? `${m.repo} · ` : ""}{bytes(m.size)}{m.pulled ? " · Downloaded" : ""}</div>
            <div class="verdict {verdictClass(m)}">
              <Symbol name={m.fit === "fits" ? "check" : m.fit === "wont_fit" ? "xmark" : "warning"} size={12} weight={2} />
              <span>{m.verdict}</span>
            </div>
            {#if pull?.state === "running"}<PullProgress job={pull} />{/if}
          </div>
          <div class="actions">
            {#if pull?.state === "running"}
              <button class="btn" onclick={() => jobs.cancel("pull", m.name)}>Cancel</button>
            {:else if !m.pulled}
              <button class="btn" disabled={m.fit === "wont_fit"} onclick={() => act("pull", m)}>
                <Symbol name="download" size={13} /> Download
              </button>
            {:else if m.serving}
              <span class="current"><Symbol name="check" size={13} weight={2} /> In Use</span>
            {:else if m.fit === "wont_fit"}
              {#if best}<button class="btn plain" onclick={() => best && act(best.pulled ? "use" : "pull", best)}>Use {best.alias ?? best.name} Instead</button>{/if}
            {:else}
              <!-- One filled button per list: the model mrun recommends (019 FR-009). -->
              <button class="btn" class:primary={m === best} disabled={jobs.running("use", m.name)} onclick={() => act("use", m)}>
                {jobs.running("use", m.name) ? "Switching…" : "Use"}
              </button>
            {/if}
            <ActionMenu
              kind={m.pulled ? (m.serving ? "stop" : "use") : "pull"}
              target={m.name}
              items={m.pulled && !m.serving ? [{ label: "Remove…", danger: true, run: () => (confirmRemove = m) }] : []}
              label="More for {m.name}"
            />
          </div>
        </li>
      {/each}
    </ul>
  {/if}

  <section class="add card" aria-labelledby="add-title">
    <h2 id="add-title" class="section-title">Add a model</h2>
    <form class="add-form" onsubmit={check}>
      <input type="text" bind:value={addInput} placeholder="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit or a Hugging Face link" aria-label="Model to add" spellcheck="false" autocomplete="off" />
      <button class="btn" type="submit" disabled={!addInput.trim() || adding?.busy}>{adding?.busy ? "Checking…" : "Check"}</button>
    </form>
    {#if adding?.error}
      <p class="add-error" role="alert"><Symbol name="warning" size={13} /> {adding.error}</p>
    {:else if adding?.plan}
      {@const p = adding.plan}
      {@const pull = jobs.of("pull", adding.repo)}
      {@const fit = p.fit === "fits" ? "fits" : p.fit === "tight" ? "tight" : p.fit === "wont_fit" ? "wont" : "unknown"}
      <div class="plan">
        <div class="main">
          <div class="name">{p.repo}</div>
          <div class="sub">{bytes(p.bytes)} · {p.files} files · family {p.family}{p.revision ? ` · ${p.revision.slice(0, 7)}` : ""}</div>
          <div class="verdict {fit}">
            <Symbol name={p.fit === "fits" ? "check" : p.fit === "wont_fit" ? "xmark" : "warning"} size={12} weight={2} />
            <span>{p.verdict}</span>
          </div>
          {#if pull?.state === "running"}<PullProgress job={pull} />{/if}
        </div>
        <div class="actions">
          {#if pull?.state === "running"}
            <button class="btn" onclick={() => jobs.cancel("pull", adding!.repo)}>Cancel</button>
          {:else}
            <button class="btn" disabled={p.fit === "wont_fit"} onclick={() => download(adding!.repo)}><Symbol name="download" size={13} /> Download</button>
          {/if}
          <ActionMenu kind="plan" target={adding.repo} label="More for {p.repo}" />
        </div>
      </div>
    {/if}
  </section>
</div>

{#if confirmRemove}
  {@const m = confirmRemove}
  <div class="scrim" role="presentation" onclick={() => (confirmRemove = null)}></div>
  <div class="alert glass" role="alertdialog" aria-labelledby="rm-title" aria-describedby="rm-text">
    <div class="alert-icon" aria-hidden="true"><Symbol name="trash" size={26} /></div>
    <h2 id="rm-title">Remove {m.alias ?? m.name}?</h2>
    <p id="rm-text">This frees {bytes(m.size)} on this Mac. You can download it again later.</p>
    <div class="alert-buttons">
      <button class="btn" onclick={() => (confirmRemove = null)}>Cancel</button>
      <button class="btn primary danger-fill" onclick={() => { confirmRemove = null; void act("remove", m); }}>Remove</button>
    </div>
  </div>
{/if}

<style>
  .models {
    display: grid;
  }
  .list {
    list-style: none;
    margin: 0;
    padding: 0 var(--s4);
  }
  li {
    display: flex;
    gap: var(--s3);
    align-items: flex-start;
    padding: var(--s4) 0;
    border-top: 0.5px solid var(--separator);
  }
  li:first-child {
    border-top: none;
  }
  .icon {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    display: grid;
    place-items: center;
    background: var(--fill);
    color: var(--text-2);
    flex: none;
  }
  .serving .icon {
    background: var(--accent);
    color: #fff;
  }
  .main {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 3px;
  }
  .name {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    font-weight: 600;
    font-size: 14px;
  }
  .tag {
    font: 600 10.5px/1 var(--font);
    padding: 3px 6px;
    border-radius: 5px;
    background: var(--fill);
    color: var(--text-2);
  }
  .tag.best {
    background: color-mix(in srgb, var(--ready-fill) 18%, transparent);
    color: var(--ready);
  }
  .tag.in-use {
    background: var(--accent-soft);
    color: var(--accent-ink);
  }
  .sub {
    color: var(--muted);
    font-size: 12px;
  }
  .add {
    margin-top: var(--s4);
  }
  .add-form {
    display: flex;
    gap: var(--s2);
  }
  .add-form input {
    flex: 1;
    min-width: 0;
    font-family: var(--mono);
    font-size: 12px;
  }
  .add-error {
    display: flex;
    gap: 6px;
    align-items: center;
    margin: var(--s3) 0 0;
    color: var(--attention);
  }
  .plan {
    display: flex;
    gap: var(--s3);
    margin-top: var(--s3);
    padding-top: var(--s3);
    border-top: 0.5px solid var(--separator);
  }
  .verdict {
    display: flex;
    gap: 5px;
    align-items: center;
    font-size: 12.5px;
  }
  .verdict.fits {
    color: var(--ready);
  }
  .verdict.tight {
    color: var(--warn);
  }
  .verdict.wont {
    color: var(--failed);
  }
  .verdict.unknown {
    color: var(--muted);
  }
  .actions {
    display: flex;
    gap: 4px;
    align-items: center;
  }
  .current {
    display: inline-flex;
    gap: 4px;
    align-items: center;
    color: var(--accent-ink);
    font-weight: 500;
    padding: 0 8px;
  }
  .empty {
    color: var(--muted);
  }
  .scrim {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.18);
    z-index: 40;
  }
  .alert {
    position: fixed;
    z-index: 41;
    left: 50%;
    top: 30%;
    transform: translate(-50%, -30%);
    width: 300px;
    padding: var(--s5) var(--s4) var(--s4);
    border-radius: var(--r-xl);
    box-shadow: var(--shadow-sheet);
    text-align: center;
  }
  .alert-icon {
    display: grid;
    place-items: center;
    color: var(--failed);
  }
  .alert h2 {
    font: 650 14px/1.3 var(--display);
    margin: var(--s2) 0 4px;
  }
  .alert p {
    margin: 0 0 var(--s4);
    color: var(--text-2);
    font-size: 12px;
  }
  .alert-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--s2);
  }
  .alert-buttons .btn {
    justify-content: center;
  }
  .danger-fill {
    background: var(--failed-fill) !important;
  }
</style>
