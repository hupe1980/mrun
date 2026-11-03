<script lang="ts">
  // The chat's inspector (⌥⌘I), a panel like Keynote's: the system prompt, thinking, and the
  // full sampling panel (FR-010a). Every setting shows the model's own default; only changed
  // settings are sent, as fields of the public API request.
  import type { Chat, Sampling } from "../lib/types";
  import Symbol from "./Symbol.svelte";

  let {
    chat = $bindable(),
    defaults,
    thinkingFamily,
    onclose,
  }: {
    chat: Chat;
    defaults: Required<Pick<Sampling, "temperature" | "top_p" | "top_k" | "min_p">> & Sampling;
    thinkingFamily: boolean;
    onclose: () => void;
  } = $props();

  type Key = keyof Sampling;
  const fields: { key: Key; label: string; min: number; max: number; step: number; help: string }[] = [
    { key: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05, help: "Higher is more varied; 0 always picks the most likely token." },
    { key: "top_p", label: "Top P", min: 0, max: 1, step: 0.01, help: "Keeps the smallest set of tokens whose probabilities add up to this." },
    { key: "top_k", label: "Top K", min: 0, max: 200, step: 1, help: "Keeps only this many most likely tokens; 0 keeps all." },
    { key: "min_p", label: "Min P", min: 0, max: 1, step: 0.01, help: "Drops tokens less likely than this share of the most likely one." },
    { key: "repetition_penalty", label: "Repetition Penalty", min: 0.5, max: 2, step: 0.01, help: "Above 1 discourages repeating the last 20 tokens." },
    { key: "max_tokens", label: "Max Reply Length", min: 1, max: 131072, step: 1, help: "Tokens; the server clamps it to what fits." },
    { key: "seed", label: "Seed", min: 0, max: 2 ** 32 - 1, step: 1, help: "The same seed and settings give the same reply." },
  ];

  const fallback: Record<Key, number | undefined> = $derived({
    temperature: defaults.temperature,
    top_p: defaults.top_p,
    top_k: defaults.top_k,
    min_p: defaults.min_p,
    repetition_penalty: defaults.repetition_penalty ?? 1,
    max_tokens: defaults.max_tokens,
    seed: undefined,
  });
  const changed = $derived(Object.keys(chat.sampling).length);

  function set(key: Key, raw: string) {
    const value = Number(raw);
    const next = { ...chat.sampling };
    if (raw === "" || !Number.isFinite(value) || value === fallback[key]) delete next[key];
    else next[key] = value;
    chat.sampling = next;
  }

  function reset() {
    chat.sampling = {};
  }
</script>

<aside class="inspector" aria-label="Chat settings">
  <header>
    <span>Chat Settings</span>
    <button class="btn plain" onclick={onclose} aria-label="Close settings (⌥⌘I)"><Symbol name="xmark" size={13} /></button>
  </header>

  <label class="block">
    <span class="label">System Prompt</span>
    <textarea id="system-prompt" rows="4" placeholder="Optional. Tells the model how to behave in this chat." bind:value={chat.system}></textarea>
  </label>

  {#if thinkingFamily}
    <label class="switch">
      <span><span class="label">Thinking</span><small>Reasoning shown collapsed above the reply.</small></span>
      <input type="checkbox" role="switch" checked={chat.thinking ?? true} onchange={(e) => (chat.thinking = (e.currentTarget as HTMLInputElement).checked)} />
    </label>
  {/if}

  <div class="group-head">
    <span class="label">Sampling</span>
    <button class="btn plain small" onclick={reset} disabled={changed === 0}>Reset to Model Defaults</button>
  </div>
  {#each fields as f (f.key)}
    {@const value = chat.sampling[f.key]}
    {@const def = fallback[f.key]}
    <label class="field" class:changed={value !== undefined} title={f.help}>
      <span class="row">
        <span>{f.label}{#if value !== undefined}<i class="mark" aria-label="changed"></i>{/if}</span>
        <input
          type="number"
          min={f.min}
          max={f.max}
          step={f.step}
          value={value ?? ""}
          placeholder={def === undefined ? "Default" : `Default (${def})`}
          oninput={(e) => set(f.key, (e.currentTarget as HTMLInputElement).value)}
        />
      </span>
      {#if f.max <= 2}
        <input
          class="slider"
          type="range"
          min={f.min}
          max={f.max}
          step={f.step}
          value={value ?? def ?? f.min}
          aria-label={f.label}
          oninput={(e) => set(f.key, (e.currentTarget as HTMLInputElement).value)}
        />
      {/if}
    </label>
  {/each}
  <p class="note">Settings belong to this chat and apply from the next reply. Agents use the model's defaults.</p>
</aside>

<style>
  .inspector {
    width: 288px;
    flex: none;
    border-left: 0.5px solid var(--separator);
    background: var(--surface-2);
    overflow-y: auto;
    padding: 0 var(--s4) var(--s4);
  }
  header {
    position: sticky;
    top: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--s3) 0;
    font-weight: 600;
    background: var(--surface-2);
  }
  .label {
    font-weight: 600;
    font-size: 12px;
    display: block;
    margin-bottom: 4px;
  }
  .block {
    display: block;
    margin-bottom: var(--s4);
  }
  textarea {
    width: 100%;
    resize: vertical;
  }
  .switch {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--s3);
    margin-bottom: var(--s4);
  }
  .switch small {
    color: var(--muted);
    font-size: 11px;
  }
  .group-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--s2);
  }
  .small {
    height: 22px;
    font-size: 11px;
    padding: 0 6px;
  }
  .field {
    display: block;
    padding: 8px 0;
    border-top: 0.5px solid var(--separator);
  }
  .row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--s2);
  }
  .row input {
    width: 112px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .mark {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    margin-left: 6px;
    vertical-align: middle;
  }
  .slider {
    width: 100%;
    margin: 6px 0 0;
    accent-color: var(--accent);
  }
  .note {
    color: var(--muted);
    font-size: 11px;
    margin: var(--s3) 0 0;
  }
</style>
