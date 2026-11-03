<script lang="ts">
  // One chat message. Assistant replies carry a quiet stats line, reuse first (FR-008): it is
  // what makes mrun's chat different, every follow-up reuses the conversation. Reasoning shows
  // its progress while it runs and collapses to "Thought for N s" (FR-009; 019 FR-001). Copy,
  // Edit and Regenerate appear on hover and focus and in the context menu (019 FR-002, FR-003).
  import { copyText } from "../lib/clipboard";
  import { seconds, stopWord, tokens } from "../lib/format";
  import { contextMenu, type MenuItem } from "../lib/native.svelte";
  import type { Message } from "../lib/types";
  import Markdown from "./Markdown.svelte";
  import ReuseRibbon from "./ReuseRibbon.svelte";
  import Symbol from "./Symbol.svelte";

  let {
    message,
    streaming = false,
    last = false,
    editing = false,
    onedit,
    oncommit,
    oncancel,
    onregenerate,
  }: {
    message: Message;
    streaming?: boolean;
    /** The chat's last reply: only it can be regenerated. */
    last?: boolean;
    editing?: boolean;
    onedit?: () => void;
    oncommit?: (text: string) => void;
    oncancel?: () => void;
    onregenerate?: () => void;
  } = $props();

  let showRibbon = $state(false);
  let showReasoning = $state(false);
  let copied = $state<"ok" | "failed" | null>(null);
  let draft = $state("");
  let field: HTMLTextAreaElement | undefined = $state();
  let now = $state(Date.now());
  const st = $derived(message.stats);
  const thinking = $derived(streaming && Boolean(message.reasoning) && !message.text);
  // The newest line of the thoughts, as a quiet preview while they run.
  const preview = $derived(
    thinking ? (message.reasoning ?? "").trimEnd().split("\n").filter((l) => l.trim()).at(-1)?.trim() ?? "" : "",
  );

  // A one-second ticker only while reasoning runs (SC-001).
  $effect(() => {
    if (!thinking) return;
    now = Date.now();
    const timer = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(timer);
  });

  $effect(() => {
    if (editing) {
      draft = message.text;
      queueMicrotask(() => {
        field?.focus();
        field?.setSelectionRange(draft.length, draft.length);
      });
    }
  });

  /** Whole seconds while thinking runs: "23 s", then "1 min 5 s". */
  function elapsed(ms: number): string {
    return ms < 60_000 ? `${Math.floor(Math.max(0, ms) / 1000)} s` : seconds(ms);
  }

  const thinkingLine = $derived(
    ["Thinking", elapsed(now - (message.reasoning_started ?? now)), message.reasoning_tokens ? `${tokens(message.reasoning_tokens)} tokens` : null]
      .filter(Boolean)
      .join(" · "),
  );

  async function copy() {
    copied = (await copyText(message.text)) ? "ok" : "failed";
    setTimeout(() => (copied = null), 1500);
  }

  function menu(e: MouseEvent | KeyboardEvent) {
    if (streaming || editing) return;
    const items: MenuItem[] = [{ id: "copy", title: "Copy" }];
    if (message.role === "user" && onedit) items.push({ id: "edit", title: "Edit" });
    if (message.role === "assistant" && last && onregenerate) items.push({ id: "regenerate", title: "Regenerate" });
    void contextMenu(e, items).then((id) => {
      if (id === "copy") void copy();
      else if (id === "edit") onedit?.();
      else if (id === "regenerate") onregenerate?.();
    });
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) menu(e);
  }

  function editKey(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (draft.trim()) oncommit?.(draft.trim());
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      oncancel?.();
    }
  }
</script>

{#snippet actions()}
  {#if !streaming && !editing}
    <div class="actions">
      <button class="act" onclick={copy} aria-label={message.role === "user" ? "Copy question" : "Copy reply"} title="Copy">
        <Symbol name={copied === "ok" ? "check" : "copy"} size={13} />
        {#if copied}<span>{copied === "ok" ? "Copied" : "Couldn’t copy"}</span>{/if}
      </button>
      {#if message.role === "user" && onedit}
        <button class="act" onclick={onedit} aria-label="Edit question" title="Edit"><Symbol name="pencil" size={13} /></button>
      {/if}
      {#if message.role === "assistant" && last && onregenerate}
        <button class="act" onclick={onregenerate} aria-label="Regenerate reply" title="Regenerate (⌘R)"><Symbol name="reload" size={13} /></button>
      {/if}
    </div>
  {/if}
{/snippet}

{#if message.role === "user"}
  <!-- Right-click and Shift-F10 reach the message's actions; its buttons stay the primary way. -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="user" oncontextmenu={menu} onkeydown={onKey}>
    {#if editing}
      <form class="edit" onsubmit={(e) => { e.preventDefault(); if (draft.trim()) oncommit?.(draft.trim()); }}>
        <textarea bind:this={field} bind:value={draft} aria-label="Edit question" rows={Math.min(8, draft.split("\n").length + 1)} onkeydown={editKey}></textarea>
        <div class="edit-buttons">
          <button type="button" class="btn" onclick={oncancel}>Cancel</button>
          <button type="submit" class="btn primary" disabled={!draft.trim()}>Send</button>
        </div>
      </form>
    {:else}
      <div class="stack">
        <!-- One line: the bubble keeps white space (pre-wrap). -->
        <div class="bubble">{#if message.attachments?.length}<span class="files">{#each message.attachments as a, i (a.name + i)}<span class="file"><Symbol name="doc" size={12} />{a.name}</span>{/each}</span>{/if}{message.text}</div>
        {@render actions()}
      </div>
    {/if}
  </div>
{:else}
  <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
  <article class="assistant" aria-busy={streaming} oncontextmenu={menu} onkeydown={onKey}>
    {#if message.reasoning}
      <button class="thought" class:live={thinking} aria-expanded={showReasoning} onclick={() => (showReasoning = !showReasoning)}>
        <Symbol name="sparkles" size={13} />
        {#if thinking}
          <span class="num">{thinkingLine}</span>
        {:else}
          <span>Thought for {seconds(message.reasoning_ms ?? 0)}</span>
        {/if}
        <span class="chev" class:open={showReasoning}><Symbol name="chevron" size={12} /></span>
      </button>
      {#if thinking && !showReasoning && preview}
        <p class="preview" aria-hidden="true">{preview}</p>
      {/if}
      {#if showReasoning}<div class="reasoning" class:live={thinking}>{message.reasoning}</div>{/if}
    {/if}
    {#if message.error}
      <div class="error" role="alert"><Symbol name="warning" size={14} />{message.error}</div>
    {/if}
    {#if message.text}<Markdown text={message.text} />{:else if streaming && !message.reasoning}<span class="caret" aria-label="Waiting for the reply"></span>{/if}
    {#if st && !streaming}
      <div class="foot">
        <button class="stats num" onclick={() => (showRibbon = !showRibbon)} aria-expanded={showRibbon}>
          <ReuseRibbon cached={st.cached_tokens} total={st.prompt_tokens} compact miss={st.miss} />
          <span>{tokens(st.cached_tokens)} of {tokens(st.prompt_tokens)} reused</span>
          {#if st.ttft_ms !== null}<span>· {seconds(st.ttft_ms)} to first token</span>{/if}
          {#if st.decode_tps}<span>· {st.decode_tps.toFixed(1)} tok/s</span>{/if}
          <span>· {stopWord(st.stop)}</span>
        </button>
        {@render actions()}
      </div>
      {#if showRibbon}
        <div class="ribbon"><ReuseRibbon cached={st.cached_tokens} total={st.prompt_tokens} miss={st.miss} /></div>
      {/if}
    {:else if !streaming && message.text}
      <div class="foot">{@render actions()}</div>
    {/if}
  </article>
{/if}

<style>
  .user {
    display: flex;
    justify-content: flex-end;
  }
  .stack {
    display: flex;
    flex-direction: row-reverse;
    align-items: center;
    max-width: 80%;
    gap: 6px;
  }
  .stack .bubble {
    max-width: 100%;
  }
  .files {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin: 0 0 4px;
  }
  .file {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 1px 7px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.2);
    font-size: 12px;
  }
  .edit {
    width: min(560px, 80%);
    display: grid;
    gap: 6px;
  }
  .edit textarea {
    width: 100%;
    resize: vertical;
    font-size: 14px;
    border-radius: 12px;
    padding: 8px 12px;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
    outline: none;
  }
  .edit-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 6px;
  }
  .foot {
    display: flex;
    align-items: center;
    gap: var(--s2);
    flex-wrap: wrap;
  }
  .actions {
    display: flex;
    gap: 2px;
    opacity: 0;
    transition: opacity 120ms var(--ease);
  }
  .user:hover .actions,
  .assistant:hover .actions,
  .actions:focus-within {
    opacity: 1;
  }
  @media (hover: none) {
    .actions {
      opacity: 1;
    }
  }
  .act {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 24px;
    min-width: 24px;
    justify-content: center;
    padding: 0 5px;
    border: none;
    border-radius: 6px;
    background: none;
    color: var(--muted);
    font-size: 11.5px;
  }
  .act:hover {
    background: var(--fill);
    color: var(--text);
  }
  .preview {
    margin: -2px 0 8px 12px;
    max-width: 640px;
    color: var(--faint);
    font-size: 12.5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    -webkit-mask-image: linear-gradient(90deg, #000 70%, transparent);
    mask-image: linear-gradient(90deg, #000 70%, transparent);
  }
  .thought.live :global(svg) {
    animation: twinkle 1.6s var(--ease) infinite;
  }
  @keyframes twinkle {
    50% {
      opacity: 0.35;
    }
  }
  .reasoning.live {
    max-height: 240px;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
  }
  .bubble {
    max-width: 72%;
    padding: 8px 13px;
    border-radius: 18px 18px 5px 18px;
    background: var(--accent);
    color: #fff;
    font-size: 14px;
    line-height: 1.45;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .assistant {
    max-width: 760px;
  }
  .thought {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: none;
    background: var(--fill);
    color: var(--text-2);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 12px;
    margin-bottom: 8px;
  }
  .chev {
    transition: transform 200ms var(--ease);
    transform: rotate(-90deg);
  }
  .chev.open {
    transform: none;
  }
  .reasoning {
    border-left: 2px solid var(--fill-2);
    padding: 2px 0 2px 12px;
    margin: 0 0 10px;
    color: var(--muted);
    white-space: pre-wrap;
    font-size: 13px;
  }
  .error {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 8px 12px;
    border-radius: var(--r);
    background: color-mix(in srgb, var(--failed-fill) 12%, transparent);
    color: var(--failed);
    margin-bottom: 8px;
  }
  .caret {
    display: inline-block;
    width: 8px;
    height: 16px;
    border-radius: 2px;
    background: var(--accent);
    animation: blink 1s steps(2) infinite;
  }
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }
  .foot {
    margin-top: 8px;
  }
  .stats {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    padding: 2px 0;
    border: none;
    background: none;
    color: var(--muted);
    font-size: 11.5px;
    text-align: left;
  }
  .stats:hover {
    color: var(--text-2);
  }
  .ribbon {
    margin-top: 6px;
    max-width: 420px;
  }
</style>
