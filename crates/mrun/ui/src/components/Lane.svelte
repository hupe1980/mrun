<script lang="ts">
  // One running or waiting request: real progress, never an indefinite spinner (US1 AS2).
  import { phaseLine, seconds } from "../lib/format";
  import type { Request } from "../lib/types";
  import ReuseRibbon from "./ReuseRibbon.svelte";

  let { request }: { request: Request } = $props();

  const fraction = $derived(
    request.phase === "prefill" && request.prompt_tokens > 0
      ? request.progress / request.prompt_tokens
      : request.phase === "generating"
        ? 1
        : 0,
  );
  const api = $derived(
    ({ messages: "Messages", chat: "Chat Completions", responses: "Responses" } as Record<string, string>)[
      request.api
    ] ?? request.api,
  );
</script>

<li class="lane {request.phase}">
  <span class="pulse" aria-hidden="true"></span>
  <div class="body">
    <div class="line">
      <span class="what">{phaseLine(request)}</span>
      <span class="meta num">{api} · lane {request.lane + 1} · {seconds(request.age_ms)}</span>
    </div>
    {#if request.phase === "prefill"}
      <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax={request.prompt_tokens} aria-valuenow={request.progress} aria-label="Reading the prompt">
        <span style:width="{fraction * 100}%"></span>
      </div>
    {:else if request.cached !== null}
      <ReuseRibbon cached={request.cached} total={request.prompt_tokens} miss={request.cause ? { cached: request.cached, processed: request.prompt_tokens - request.cached, cause: request.cause, divergence: null, message: null, unreused: request.unreused ?? 0 } : null} />
    {/if}
  </div>
</li>

<style>
  .lane {
    display: flex;
    gap: var(--s3);
    align-items: flex-start;
    padding: var(--s3) 0;
    border-top: 0.5px solid var(--separator);
  }
  .lane:first-child {
    border-top: none;
  }
  .pulse {
    width: 8px;
    height: 8px;
    margin-top: 5px;
    border-radius: 50%;
    flex: none;
    background: var(--stopped-fill);
  }
  .prefill .pulse {
    background: var(--reading-fill);
    animation: breathe 1.6s var(--ease) infinite;
  }
  .generating .pulse {
    background: var(--generating-fill);
    animation: breathe 1s var(--ease) infinite;
  }
  @keyframes breathe {
    50% {
      opacity: 0.35;
      transform: scale(0.8);
    }
  }
  .body {
    flex: 1;
    min-width: 0;
  }
  .line {
    display: flex;
    justify-content: space-between;
    gap: var(--s3);
    margin-bottom: 6px;
  }
  .what {
    font-weight: 500;
  }
  .meta {
    color: var(--muted);
    font-size: 12px;
    white-space: nowrap;
  }
  .progress {
    height: 6px;
    border-radius: 3px;
    background: var(--fill);
    overflow: hidden;
  }
  .progress span {
    display: block;
    height: 100%;
    background: var(--reading-fill);
    transition: width 900ms linear;
  }
</style>
