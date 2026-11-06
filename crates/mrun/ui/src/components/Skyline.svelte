<script lang="ts">
  // A session as a skyline (025): one band per row, one column per request in time order, its
  // reused part solid and the computed cap light (amber when worth fixing), a wait strip under
  // each column, idle breaks between bursts. The listbox owns the keys; columns are its options,
  // named by their reuse sentence.
  import { cause, reuseSentence, seconds } from "../lib/format";
  import type { Layout } from "../lib/sessions";
  import type { Request } from "../lib/types";

  let {
    layout,
    selected,
    running = null,
    onselect,
    onkey,
    label,
  }: {
    layout: Layout;
    selected: number | null;
    /** A request in progress, drawn after the last column. */
    running?: Request | null;
    onselect: (index: number) => void;
    onkey: (e: KeyboardEvent) => void;
    label: string;
  } = $props();

  const STRIP = 2;
  const w = $derived(layout.columnWidth);
  const tallest = $derived(Math.max(1, ...layout.columns.map((c) => c.record.prompt_tokens)));
  const runX = $derived(layout.width + (layout.columns.length ? 10 : 0));
  const width = $derived(runX + (running ? w + 16 : 0) + 8);
  const height = $derived(layout.height + 4);
  const optionId = (i: number) => `col-${layout.columns[i]?.record.id ?? i}`;

  function name(i: number): string {
    const r = layout.columns[i]!.record;
    const parts = [reuseSentence(r.cached, r.prompt_tokens, r)];
    if (r.attention) parts.push("Worth fixing.");
    if (r.ttft_ms !== null) parts.push(`${seconds(r.ttft_ms)} to first token.`);
    if (r.stop === "stopped") parts.push("Stopped.");
    return parts.join(" ");
  }

  function tip(i: number): string {
    const r = layout.columns[i]!.record;
    const why = r.cause && r.cached < r.prompt_tokens ? `\n${cause(r)}` : "";
    const wait = r.ttft_ms !== null ? `\n${seconds(r.ttft_ms)} to first token` : "";
    return `Request ${r.id} · ${r.prompt_tokens.toLocaleString("en-US")} tokens${why}${wait}`;
  }
</script>

<div class="frame">
<div class="labels" aria-hidden="true" style:height="{height}px">
  {#each layout.bands as band (band.id)}
    <span style:top="{band.y}px" style:height="{band.height}px">{band.label}</span>
  {/each}
</div>
<div
  class="skyline"
  role="listbox"
  tabindex="0"
  aria-label={label}
  aria-activedescendant={selected !== null ? optionId(selected) : undefined}
  onkeydown={onkey}
>
  <svg {width} {height} role="presentation">
    <defs aria-hidden="true">
      <pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="4" height="4" class="cap" />
        <line x1="0" y1="0" x2="0" y2="4" class="hatch" />
      </pattern>
    </defs>
    <g aria-hidden="true">
      {#each layout.bands as band, i (band.id)}
        {#if i > 0}<line class="rule" x1="0" x2={width} y1={band.y} y2={band.y} />{/if}
      {/each}
    </g>
    <g role="presentation">
      {#each layout.breaks as b (b.x)}
        <g class="break" aria-hidden="true">
          <line x1={b.x + 32} x2={b.x + 32} y1="0" y2={layout.height} />
          <text x={b.x + 32} y="11" text-anchor="middle">{b.minutes} min idle</text>
        </g>
      {/each}
      {#each layout.columns as c, i (c.record.id)}
        {@const band = layout.bands[c.row]!}
        {@const base = band.y + band.height - STRIP - 4}
        <!-- The listbox owns focus and keys (aria-activedescendant). -->
        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_interactive_supports_focus -->
        <g
          id={optionId(i)}
          role="option"
          aria-selected={selected === i}
          aria-label={name(i)}
          class="column"
          class:attention={c.record.attention}
          class:stopped={c.record.stop === "stopped" || c.record.stop === "error"}
          onclick={() => onselect(i)}
        >
          <title>{tip(i)}</title>
          <rect class="hit" x={c.x - 1} y={band.y} width={w + 2} height={band.height} />
          <rect class="cap" x={c.x} y={base - c.height} width={w} height={Math.max(0, c.height - c.reused)} rx={w > 4 ? 1 : 0} />
          <rect class="body" x={c.x} y={base - c.reused} width={w} height={c.reused} />
          <rect class="wait" x={c.x} y={base + 2} width={w} height={STRIP} style:opacity={0.1 + 0.55 * c.wait} />
          {#if selected === i}
            <rect class="ring" x={c.x - 2.5} y={base - c.height - 2.5} width={w + 5} height={c.height + STRIP + 7} rx="3" />
          {/if}
        </g>
      {/each}
      {#if running && layout.bands[0]}
        {@const band = layout.bands[0]}
        {@const base = band.y + band.height - STRIP - 4}
        {@const total = Math.min(64, (running.prompt_tokens / tallest) * 64)}
        {@const done = running.prompt_tokens ? (total * running.progress) / running.prompt_tokens : 0}
        <!-- Now says the running request's progress in words; here it is only drawn. -->
        <g class="running" aria-hidden="true">
          <rect class="outline" x={runX} y={base - total} width={Math.max(w, 4)} height={total} rx="1" />
          <rect class="progress" x={runX} y={base - done} width={Math.max(w, 4)} height={done} />
        </g>
      {/if}
    </g>
  </svg>
</div>
</div>

<style>
  .frame {
    display: flex;
    min-width: 0;
  }
  .labels {
    position: relative;
    flex: none;
    width: 92px;
    margin-top: 8px;
    border-right: 0.5px solid var(--separator);
    margin-right: 10px;
  }
  .labels span {
    position: absolute;
    left: 0;
    right: 8px;
    display: flex;
    align-items: flex-end;
    padding-bottom: 6px;
    box-sizing: border-box;
    font: 500 11px var(--font);
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .skyline {
    flex: 1;
    min-width: 0;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 8px 0 4px;
    border-radius: var(--r);
    outline: none;
    scrollbar-width: thin;
  }
  .skyline:focus-visible {
    box-shadow: 0 0 0 3px var(--focus);
  }
  svg {
    display: block;
    overflow: visible;
  }
  .rule {
    stroke: var(--separator);
    stroke-width: 0.5;
  }
  .break line {
    stroke: var(--faint);
    stroke-dasharray: 2 3;
  }
  .break text {
    font: 500 10px var(--font);
    fill: var(--muted);
    paint-order: stroke;
    stroke: var(--surface);
    stroke-width: 4px;
  }
  .column {
    cursor: pointer;
  }
  .hit {
    fill: transparent;
  }
  .column:hover .hit {
    fill: var(--fill);
  }
  .body {
    fill: var(--reused);
  }
  .cap {
    fill: var(--computed);
  }
  .attention .cap {
    fill: var(--attention-fill);
  }
  .stopped .cap {
    fill: url(#hatch);
  }
  .hatch {
    stroke: var(--text-2);
    stroke-width: 1;
  }
  .wait {
    fill: var(--reading-fill);
  }
  .ring {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.5;
  }
  .running .outline {
    fill: none;
    stroke: var(--reading-fill);
    stroke-dasharray: 2 2;
  }
  .running .progress {
    fill: var(--reading-fill);
    animation: pulse 1.4s var(--ease) infinite;
  }
  @keyframes pulse {
    50% {
      opacity: 0.45;
    }
  }
  .column {
    animation: rise 320ms var(--ease) both;
  }
  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
  }
  @media (prefers-contrast: more) {
    .cap,
    .body {
      stroke: var(--text);
      stroke-width: 0.75;
    }
  }
</style>
