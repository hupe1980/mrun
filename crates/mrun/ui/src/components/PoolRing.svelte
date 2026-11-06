<script lang="ts">
  // The pool as what it is (026 R7): a ring. Each Mac is a tile on the circle; the model's
  // layers are the circle itself, one coloured arc per Mac sized by its layers; link chips sit
  // where one Mac hands over to the next. While a request runs, a light travels the ring, Mac to
  // Mac; with Reduce Motion the arcs are filled instead.
  import { bytes } from "../lib/format";
  import { COLORS, linkLabel, memberWord, ring, type RingMac } from "../lib/pool";
  import Symbol from "./Symbol.svelte";


  let {
    macs,
    title,
    subtitle = "",
    detail = "",
    pace = null,
    running = false,
    compact = false,
    label,
  }: {
    macs: RingMac[];
    title: string;
    subtitle?: string;
    detail?: string;
    /** The Mac or link setting the pace. */
    pace?: { kind: string; mac_id: string } | null;
    running?: boolean;
    /** Smaller, for a sheet. */
    compact?: boolean;
    label: string;
  } = $props();

  const SIZE = 420;
  const R = 150;
  const C = SIZE / 2;
  const shape = $derived(ring(macs.map((m) => ({ mac_id: m.mac_id, name: m.name, layers: m.layers || 1 })), C, C, R));
  const color = (i: number) => `var(${COLORS[i % COLORS.length]})`;

  function name(m: RingMac): string {
    const parts = [m.name, m.chip];
    if (m.layers && m.first !== undefined && m.end !== undefined) parts.push(`layers ${m.first + 1} to ${m.end}`);
    if (m.used !== undefined && m.budget) parts.push(`${bytes(m.used)} of ${bytes(m.budget)}`);
    if (m.link) parts.push(linkLabel(m.link.kind, m.link.ms));
    if (m.power === "battery") parts.push("on battery");
    parts.push(m.host ? "host" : memberWord(m.state ?? ""));
    return parts.join(", ");
  }
</script>

<figure class="ring" class:running class:compact aria-label={label}>
  <svg viewBox="0 0 {SIZE} {SIZE}" aria-hidden="true">
    <circle class="track" cx={C} cy={C} r={R} />
    {#each shape.arcs as arc, i (arc.mac_id)}
      <path
        class="arc"
        class:lost={macs[i]?.state === "lost"}
        class:pace={pace?.kind === "stage" && pace.mac_id === arc.mac_id}
        d={arc.d}
        style:stroke={color(i)}
        style:--delay="{(i * 0.9) / macs.length}s"
      />
    {/each}
    {#if running}
      <circle class="orbit" cx={C} cy={C} r={R} />
    {/if}
  </svg>

  <div class="center">
    <strong>{title}</strong>
    {#if subtitle}<span class="num">{subtitle}</span>{/if}
    {#if detail}<small>{detail}</small>{/if}
  </div>

  <ul class="tiles">
    {#each macs as m, i (m.mac_id)}
      {@const t = shape.tiles[i]!}
      <li
        class="tile"
        class:lost={m.state === "lost"}
        class:busy={m.state === "downloading" || m.state === "loading" || m.state === "connecting"}
        style:left="{(t.x / SIZE) * 100}%"
        style:top="{(t.y / SIZE) * 100}%"
        style:--mac={color(i)}
        aria-label={name(m)}
      >
        <span class="glyph"><Symbol name="laptop" size={18} /></span>
        <span class="who">{m.name}</span>
        <span class="what num">
          {m.chip.replace(/^Apple /, "")}{#if m.layers && m.first !== undefined}{m.chip ? " · " : ""}layers {m.first + 1}–{m.end}{/if}
        </span>
        {#if m.link}
          <span class="link num" class:pace={pace?.kind === "link" && pace.mac_id === m.mac_id} class:wifi={m.link.kind === "wifi"}>{linkLabel(m.link.kind, m.link.ms)}</span>
        {/if}
        {#if m.used !== undefined && m.budget}
          <span class="gauge" aria-hidden="true"><i style:width="{Math.min(100, (m.used / m.budget) * 100)}%"></i></span>
        {/if}
        {#if m.state === "lost"}<span class="flag">Stopped answering</span>
        {:else if m.state === "downloading" || m.state === "loading" || m.state === "connecting"}<span class="flag soft">{memberWord(m.state)}</span>
        {:else if m.power === "battery"}<span class="flag soft">On battery</span>{/if}
      </li>
    {/each}
  </ul>


</figure>

<style>
  .ring {
    position: relative;
    width: min(100%, 460px);
    aspect-ratio: 1;
    margin: 0 auto;
  }
  .ring.compact {
    width: min(100%, 340px);
    margin-block: 24px;
  }
  .compact .tile {
    width: 128px;
    padding: 6px;
  }
  svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
  }
  .track {
    fill: none;
    stroke: var(--fill);
    stroke-width: 14;
  }
  .arc {
    fill: none;
    stroke-width: 14;
    stroke-linecap: round;
    transition: opacity 400ms var(--ease);
  }
  .arc.lost {
    opacity: 0.25;
    stroke-dasharray: 4 6;
  }
  .arc.pace {
    filter: drop-shadow(0 0 4px var(--attention-fill));
  }
  .running .arc {
    animation: glow 0.9s var(--ease) infinite;
    animation-delay: var(--delay);
  }
  @keyframes glow {
    50% {
      opacity: 0.55;
    }
  }
  .orbit {
    fill: none;
    stroke: #fff;
    stroke-width: 6;
    stroke-linecap: round;
    stroke-dasharray: 18 2000;
    opacity: 0.85;
    transform-origin: center;
    animation: orbit 1.6s linear infinite;
    mix-blend-mode: overlay;
  }
  @keyframes orbit {
    to {
      transform: rotate(360deg);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .orbit {
      display: none;
    }
    .running .arc {
      opacity: 1;
      stroke-width: 18;
    }
  }
  @media (prefers-contrast: more) {
    .arc {
      stroke-width: 16;
    }
    .tile {
      border-width: 1.5px;
    }
  }
  .center {
    position: absolute;
    inset: 30%;
    display: grid;
    place-content: center;
    justify-items: center;
    gap: 3px;
    text-align: center;
  }
  .center strong {
    font: 650 14px var(--display);
    color: var(--text-2);
  }
  .center span {
    font: 650 26px var(--rounded);
  }
  .center small {
    font-size: 11px;
    color: var(--muted);
    max-width: 150px;
  }
  .tiles {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .tile {
    position: absolute;
    transform: translate(-50%, -50%);
    width: 150px;
    display: grid;
    justify-items: center;
    gap: 2px;
    padding: 8px 8px 7px;
    border-radius: var(--r-lg);
    background: var(--surface);
    border: 0.5px solid var(--separator);
    box-shadow: var(--shadow);
    text-align: center;
  }
  .tile .glyph {
    color: var(--mac);
  }
  .who {
    font-size: 11.5px;
    font-weight: 600;
    line-height: 1.25;
    max-width: 100%;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .what {
    font-size: 10.5px;
    color: var(--muted);
  }
  .gauge {
    width: 70%;
    height: 3px;
    border-radius: 2px;
    background: var(--fill);
    overflow: hidden;
    margin-top: 2px;
  }
  .gauge i {
    display: block;
    height: 100%;
    background: var(--mac);
  }
  .flag {
    font-size: 10px;
    font-weight: 600;
    color: var(--failed);
  }
  .flag.soft {
    color: var(--muted);
    font-weight: 500;
  }
  .tile.lost {
    opacity: 0.6;
    filter: grayscale(1);
  }
  .tile.busy .glyph {
    animation: glow 1.2s var(--ease) infinite;
  }
  .link {
    font-size: 10px;
    color: var(--text-2);
  }
  .link.wifi {
    color: var(--warn);
  }
  .link.pace {
    color: var(--attention);
    font-weight: 650;
  }
</style>
