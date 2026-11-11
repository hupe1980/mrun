<script lang="ts">
  // One tool call in a reply (028 FR-004): a sentence ("Run cargo test", "Edit src/a.rs") with
  // the server, then while asking what exactly will happen (the command, the diff, the host it
  // reaches) and Deny / Always / Allow, ⏎ allowing and Esc denying. A call the sandbox stopped
  // shows what it was denied and "Allow …" (FR-009). Done calls fold to one line; a click shows
  // the result.
  import { describe, tools } from "../lib/tools.svelte";
  import type { ToolCall } from "../lib/types";
  import Symbol from "./Symbol.svelte";

  let {
    call,
    on = [],
    ondecide,
    onallow,
  }: {
    call: ToolCall;
    /** The servers on in this chat. */
    on?: string[];
    ondecide?: (id: string, decision: "once" | "always" | "deny" | "retry" | "continue") => void;
    onallow?: (id: string, index: number) => void;
  } = $props();

  const found = $derived(tools.resolve(call.name, on));
  const server = $derived(found?.server);
  const d = $derived(describe(call, server, found?.tool ?? call.name.split("__").slice(1).join("__")));
  let open = $state(false);
  let allowBtn: HTMLButtonElement | undefined = $state();

  $effect(() => {
    if (call.state === "asking" || call.state === "blocked") queueMicrotask(() => allowBtn?.focus());
  });

  function key(e: KeyboardEvent) {
    if (call.state !== "asking") return;
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      ondecide?.(call.id, "deny");
    } else if (e.key === "Enter" && e.altKey) {
      e.preventDefault();
      ondecide?.(call.id, "always");
    }
  }

  const stateWord: Record<ToolCall["state"], string> = {
    pending: "",
    asking: "Waiting for you",
    running: "Running…",
    blocked: "Stopped by the sandbox",
    done: "",
    failed: "Failed",
    denied: "Denied",
  };
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div class="card {call.state}" role="group" aria-label="{d.verb} {d.subject}" onkeydown={key}>
  <button class="line" onclick={() => (open = !open)} aria-expanded={open} disabled={call.state === "asking" || call.state === "pending"}>
    <span class="glyph" aria-hidden="true">
      {#if call.state === "running"}<span class="spin"></span>
      {:else if call.state === "done"}<Symbol name="check" size={12} weight={2} />
      {:else if call.state === "failed" || call.state === "blocked"}<Symbol name="warning" size={12} />
      {:else if call.state === "denied"}<Symbol name="xmark" size={12} />
      {:else}<Symbol name="tool" size={12} />{/if}
    </span>
    <span class="verb">{d.verb}</span>
    {#if d.subject}<code class="subject">{d.subject}</code>{/if}
    <span class="meta">
      {#if d.reaches}<span class="reach"><Symbol name="globe" size={11} />{d.reaches}</span>{/if}
      {#if stateWord[call.state]}<span>{stateWord[call.state]}</span>{/if}
      {#if call.auto}<span title="Always allowed in this chat">Always</span>{/if}
      <span class="server">{server?.name ?? call.name.split("__")[0]}</span>
    </span>
  </button>

  {#if call.state === "asking"}
    {#if d.body?.kind === "diff"}
      <pre class="body diff"><del>{d.body.old}</del><ins>{d.body.new}</ins></pre>
    {:else if d.body?.text}
      <pre class="body">{d.body.text}</pre>
    {/if}
    {#if d.reaches}<p class="note"><Symbol name="globe" size={12} /> This sends the call to {d.reaches}, beyond this Mac.</p>{/if}
    <div class="buttons">
      <span class="hint">⏎ Allow · ⌥⏎ Always · Esc Deny</span>
      <button class="btn" onclick={() => ondecide?.(call.id, "deny")}>Deny</button>
      <button class="btn" onclick={() => ondecide?.(call.id, "always")} title="Allow {d.verb.toLowerCase()} without asking again in this chat">Always Allow</button>
      <button class="btn primary" bind:this={allowBtn} onclick={() => ondecide?.(call.id, "once")}>Allow</button>
    </div>
  {:else if call.state === "blocked"}
    {#each call.denials ?? [] as den, i (den.text)}
      <div class="denial">
        <Symbol name="shield" size={13} />
        <span>{den.text}</span>
        {#if den.label}<button class="btn" onclick={() => onallow?.(call.id, i)}>{den.label}</button>{/if}
      </div>
    {/each}
    <div class="buttons">
      <button class="btn" onclick={() => ondecide?.(call.id, "continue")}>Continue Without</button>
      <button class="btn primary" bind:this={allowBtn} onclick={() => ondecide?.(call.id, "retry")}>Run Again</button>
    </div>
  {:else if open && call.result !== undefined}
    <pre class="body result">{call.result || "(no output)"}</pre>
  {/if}
</div>

<style>
  .card {
    border-radius: var(--r);
    background: var(--surface);
    border: 0.5px solid var(--separator);
    font-size: 13px;
    overflow: hidden;
  }
  .card.asking,
  .card.blocked {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .card.blocked {
    border-color: var(--warn-fill);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--warn-fill) 18%, transparent);
  }
  .line {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border: none;
    background: none;
    text-align: left;
    color: var(--text);
    min-width: 0;
  }
  .line:disabled {
    opacity: 1;
    cursor: default;
  }
  .glyph {
    width: 20px;
    height: 20px;
    border-radius: 6px;
    display: grid;
    place-items: center;
    flex: none;
    background: var(--fill);
    color: var(--text-2);
  }
  .done .glyph {
    background: color-mix(in srgb, var(--ready-fill) 18%, transparent);
    color: var(--ready);
  }
  .failed .glyph,
  .denied .glyph {
    background: color-mix(in srgb, var(--failed-fill) 14%, transparent);
    color: var(--failed);
  }
  .asking .glyph {
    background: var(--accent);
    color: #fff;
  }
  .blocked .glyph {
    background: var(--warn-fill);
    color: #fff;
  }
  .spin {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 1.6px solid var(--accent);
    border-right-color: transparent;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(1turn);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .spin {
      animation: none;
    }
  }
  .verb {
    font-weight: 600;
    flex: none;
  }
  .subject {
    font: 12px var(--mono);
    color: var(--text-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .meta {
    margin-left: auto;
    display: flex;
    gap: 8px;
    flex: none;
    color: var(--muted);
    font-size: 11px;
  }
  .reach {
    display: inline-flex;
    gap: 3px;
    align-items: center;
    color: var(--remote);
  }
  .server {
    padding: 0 6px;
    border-radius: 4px;
    background: var(--fill);
  }
  .body {
    margin: 0 10px 8px;
    padding: 8px 10px;
    max-height: 260px;
    overflow: auto;
    border-radius: var(--r-sm);
    background: var(--surface-2);
    font: 12px/1.45 var(--mono);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .diff del,
  .diff ins {
    display: block;
    text-decoration: none;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .diff del {
    background: color-mix(in srgb, var(--failed-fill) 14%, transparent);
  }
  .diff del::before {
    content: "− ";
    color: var(--failed);
  }
  .diff ins {
    margin-top: 2px;
    background: color-mix(in srgb, var(--ready-fill) 16%, transparent);
  }
  .diff ins::before {
    content: "+ ";
    color: var(--ready);
  }
  .note {
    margin: 0 10px 8px;
    display: flex;
    gap: 6px;
    align-items: center;
    color: var(--remote);
    font-size: 12px;
  }
  .denial {
    margin: 0 10px 6px;
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 12px;
    color: var(--text-2);
  }
  .denial span {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .buttons {
    display: flex;
    gap: 6px;
    justify-content: flex-end;
    align-items: center;
    padding: 0 10px 10px;
  }
  .hint {
    margin-right: auto;
    color: var(--muted);
    font-size: 11px;
  }
</style>
