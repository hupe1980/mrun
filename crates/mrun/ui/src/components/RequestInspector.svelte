<script lang="ts">
  // One request of a session (025 US2): its numbers, why any part was computed, the text at the
  // divergence (cached, then sent) with only the change marked, and what to do about it.
  import { host } from "../lib/api";
  import { copyText } from "../lib/clipboard";
  import { cause, hint, percent, seconds, stopWord, tokens } from "../lib/format";
  import { changed } from "../lib/sessions";
  import type { Excerpt, ReplayRecord } from "../lib/types";
  import Symbol from "./Symbol.svelte";

  let { record, onclose }: { record: ReplayRecord; onclose: () => void } = $props();

  let excerpt = $state<Excerpt | null>(null);
  let missing = $state(false);
  let copied = $state(false);

  $effect(() => {
    const id = record.id;
    excerpt = null;
    missing = false;
    if (!record.excerpt) return;
    host.excerpt(id).then(
      (e) => {
        if (record.id === id) excerpt = e;
      },
      () => {
        if (record.id === id) missing = true;
      },
    );
  });

  const computed = $derived(Math.max(0, record.prompt_tokens - record.cached));
  const diff = $derived(excerpt ? changed(excerpt.before, excerpt.after) : null);

  async function copy() {
    copied = await copyText(JSON.stringify(record, null, 2));
    setTimeout(() => (copied = false), 1500);
  }
</script>

<aside class="inspector" aria-label="Request {record.id}">
  <header>
    <h2>Request {record.id}</h2>
    <button class="icon" onclick={onclose} title="Close (Esc)" aria-label="Close the inspector"><Symbol name="xmark" size={13} /></button>
  </header>
  <dl class="numbers num">
    <dt>Prompt</dt><dd>{tokens(record.prompt_tokens)} tokens</dd>
    <dt>Reused</dt><dd>{tokens(record.cached)} · {percent(record.cached, record.prompt_tokens)}</dd>
    <dt>Computed</dt><dd>{tokens(computed)}</dd>
    <dt>Generated</dt><dd>{tokens(record.generated)}{record.decode_tps ? ` · ${record.decode_tps.toFixed(1)} tok/s` : ""}</dd>
    {#if record.ttft_ms !== null}<dt>First token</dt><dd>{seconds(record.ttft_ms)}</dd>{/if}
    {#if record.stop}<dt>Ended</dt><dd>{stopWord(record.stop)}</dd>{/if}
  </dl>

  {#if record.cause && computed > 0}
    <section class="why" class:attention={record.attention}>
      <p class="cause">
        {#if record.attention}<Symbol name="warning" size={13} />{/if}
        <span>{cause(record)[0]?.toUpperCase()}{cause(record).slice(1)}</span>
      </p>
      {#if excerpt && diff}
        <div class="excerpt">
          <figure>
            <figcaption>Cached</figcaption>
            <pre><span class="ctx">{excerpt.context}</span><span class="same">{excerpt.before.slice(0, diff.shared)}</span><del>{excerpt.before.slice(diff.shared, diff.before)}</del><span class="same">{excerpt.before.slice(diff.before)}</span></pre>
          </figure>
          <figure>
            <figcaption>Sent</figcaption>
            <pre><span class="ctx">{excerpt.context}</span><span class="same">{excerpt.after.slice(0, diff.shared)}</span><ins>{excerpt.after.slice(diff.shared, diff.after)}</ins><span class="same">{excerpt.after.slice(diff.after)}</span></pre>
          </figure>
        </div>
      {:else if record.excerpt && !missing}
        <div class="excerpt loading" aria-hidden="true"><span></span><span></span></div>
      {:else if missing}
        <p class="note">The text is no longer available (another model is loaded).</p>
      {/if}
      <p class="hint">{hint(record)}</p>
    </section>
  {:else}
    <p class="hint">{hint(record)}</p>
  {/if}

  <footer>
    <button class="btn" onclick={copy}><Symbol name={copied ? "check" : "copy"} size={12} /> {copied ? "Copied" : "Copy as JSON"}</button>
  </footer>
</aside>

<style>
  .inspector {
    width: 300px;
    flex: none;
    display: flex;
    flex-direction: column;
    gap: var(--s3);
    padding: var(--s4);
    border-left: 0.5px solid var(--separator);
    background: var(--surface);
    overflow-y: auto;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  h2 {
    margin: 0;
    font: 650 14px var(--display);
  }
  .icon {
    border: none;
    background: none;
    color: var(--muted);
    padding: 4px;
    border-radius: var(--r-sm);
  }
  .icon:hover {
    background: var(--fill);
  }
  .numbers {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    margin: 0;
    font-size: 12px;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    text-align: right;
  }
  .why {
    display: grid;
    gap: var(--s2);
    padding-top: var(--s3);
    border-top: 0.5px solid var(--separator);
  }
  .cause {
    display: flex;
    gap: 6px;
    align-items: center;
    margin: 0;
    font-weight: 600;
    font-size: 13px;
  }
  .attention .cause {
    color: var(--attention);
  }
  .excerpt {
    display: grid;
    gap: var(--s2);
  }
  figure {
    margin: 0;
  }
  figcaption {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 3px;
  }
  pre {
    margin: 0;
    padding: 8px 10px;
    border-radius: var(--r-sm);
    background: var(--surface-2);
    font: 11px/1.5 var(--mono);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 140px;
    overflow: hidden;
  }
  .ctx,
  .same {
    color: var(--muted);
  }
  del,
  ins {
    text-decoration: none;
    border-radius: 3px;
    padding: 0 1px;
    color: var(--text);
    font-weight: 600;
  }
  del {
    background: var(--fill-2);
    text-decoration: line-through;
    text-decoration-color: var(--muted);
  }
  ins {
    background: var(--attention-fill);
  }
  .loading span {
    display: block;
    height: 46px;
    border-radius: var(--r-sm);
    background: var(--fill);
    animation: breathe 1.4s var(--ease) infinite;
  }
  @keyframes breathe {
    50% {
      opacity: 0.5;
    }
  }
  .hint,
  .note {
    margin: 0;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-2);
  }
  footer {
    margin-top: auto;
  }
</style>
