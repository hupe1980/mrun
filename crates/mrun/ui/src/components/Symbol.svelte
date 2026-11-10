<script lang="ts" module>
  // Monochrome glyphs in the spirit of SF Symbols (1.6 stroke on a 20-point grid). Decorative
  // by default: the word next to a symbol carries the meaning (FR-002).
  const paths: Record<string, string> = {
    now: "M3.5 13.5a6.5 6.5 0 1 1 13 0 M10 13.5l3.2-4.2 M10 13.5h0",
    chat: "M4 5.5A2.5 2.5 0 0 1 6.5 3h7A2.5 2.5 0 0 1 16 5.5v5a2.5 2.5 0 0 1-2.5 2.5H9l-3.5 3v-3A2.5 2.5 0 0 1 4 10.5z",
    pool: "M10 3.2a2.2 2.2 0 1 0 0 4.4 2.2 2.2 0 0 0 0-4.4z M4.2 12.4a2.2 2.2 0 1 0 0 4.4 2.2 2.2 0 0 0 0-4.4z M15.8 12.4a2.2 2.2 0 1 0 0 4.4 2.2 2.2 0 0 0 0-4.4z M8.6 7.2l-3 5.3 M11.4 7.2l3 5.3 M6.4 14.6h7.2",
    laptop: "M5 5.5h10v7H5z M3 14.5h14l-1 1.5H4z",
    sessions: "M3.5 16.5h13 M5.5 14V10 M8.5 14V6 M11.5 14V8.5 M14.5 14V4",
    models: "M10 2.8l6.2 3.5v7.2L10 17l-6.2-3.5V6.3z M3.9 6.4L10 9.9l6.1-3.5 M10 9.9V17",
    agents: "M8.2 11.8l3.6-3.6 M6.6 9.5l-1.4 1.4a2.7 2.7 0 0 0 3.9 3.9l1.4-1.4 M13.4 10.5l1.4-1.4a2.7 2.7 0 0 0-3.9-3.9l-1.4 1.4",
    settings: "M10 12.6a2.6 2.6 0 1 0 0-5.2 2.6 2.6 0 0 0 0 5.2z M10 2.5v2 M10 15.5v2 M2.5 10h2 M15.5 10h2 M4.7 4.7l1.4 1.4 M13.9 13.9l1.4 1.4 M4.7 15.3l1.4-1.4 M13.9 6.1l1.4-1.4",
    lock: "M5.5 9h9v7.5h-9z M7 9V6.5a3 3 0 0 1 6 0V9",
    cloud: "M6 15.5h8.2a3.3 3.3 0 0 0 .5-6.6A4.8 4.8 0 0 0 5.6 8 3.8 3.8 0 0 0 6 15.5z",
    stop: "M6 6h8v8H6z",
    eject: "M10 4.5l5.5 6.5h-11z M4.5 14.5h11",
    send: "M10 16V4.5 M5.5 9L10 4.5 14.5 9",
    download: "M10 3.5v9 M6 8.8l4 4 4-4 M4.5 16.5h11",
    check: "M4.5 10.5l3.5 3.5 7.5-8",
    xmark: "M5.5 5.5l9 9 M14.5 5.5l-9 9",
    sparkles: "M9 3l1.3 3.7L14 8l-3.7 1.3L9 13l-1.3-3.7L4 8l3.7-1.3z M15 12.5l.6 1.7 1.7.6-1.7.6-.6 1.7-.6-1.7-1.7-.6 1.7-.6z",
    inspector: "M3.5 4.5h13v11h-13z M12 4.5v11",
    search: "M8.8 14.2a5.4 5.4 0 1 0 0-10.8 5.4 5.4 0 0 0 0 10.8z M12.8 12.8l3.7 3.7",
    more: "M5 10h.01 M10 10h.01 M15 10h.01",
    copy: "M7 7h8.5v9.5H7z M4.5 13V3.5H13",
    play: "M7 4.5l8 5.5-8 5.5z",
    plus: "M10 4v12 M4 10h12",
    chevron: "M6 8l4 4 4-4",
    warning: "M10 3.5l7 12.5H3z M10 8.5v3.5 M10 14.3h0",
    bolt: "M11 2.8L5 11h4.5L9 17.2l6-8.2h-4.5z",
    trash: "M4.5 6h11 M8 6V4h4v2 M6 6l.8 10.5h6.4L14 6",
    reload: "M15.5 10a5.5 5.5 0 1 1-1.6-3.9 M15.5 3.5V7h-3.5",
    doc: "M5.5 2.8h6l3 3v11.4h-9z M11.5 2.8v3h3 M7.8 10h4.4 M7.8 13h4.4",
    clip: "M14.6 9.6l-5 5a3 3 0 0 1-4.2-4.2l6-6a2 2 0 0 1 2.8 2.8l-6 6a1 1 0 0 1-1.4-1.4l5-5",
    pencil: "M12.8 4.2l3 3L7.5 15.5 4 16l.5-3.5z M11 6l3 3",
    globe: "M10 17a7 7 0 1 0 0-14 7 7 0 0 0 0 14z M3 10h14 M10 3a10.7 10.7 0 0 1 0 14 M10 3a10.7 10.7 0 0 0 0 14",
  };
  export type SymbolName = keyof typeof paths;
</script>

<script lang="ts">
  let {
    name,
    size = 16,
    label = undefined,
    weight = 1.6,
  }: { name: string; size?: number; label?: string; weight?: number } = $props();
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 20 20"
  fill="none"
  stroke="currentColor"
  stroke-width={weight}
  stroke-linecap="round"
  stroke-linejoin="round"
  role={label ? "img" : undefined}
  aria-label={label}
  aria-hidden={label ? undefined : "true"}
  class:fill={name === "stop" || name === "play"}
>
  <path d={paths[name] ?? ""} />
</svg>

<style>
  svg {
    flex: none;
    display: block;
  }
  svg.fill path {
    fill: currentColor;
  }
</style>
