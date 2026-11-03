<script lang="ts">
  // Markdown from the model, rendered as elements from marked's token tree: no HTML string is
  // ever inserted, so model text cannot inject markup (research R3). Raw HTML in the text shows
  // as text. Links open in the browser through the host.
  import { marked, type Token, type Tokens } from "marked";
  import { copyText } from "../lib/clipboard";
  import { host } from "../lib/api";
  import Symbol from "./Symbol.svelte";

  let { text }: { text: string } = $props();

  const tokens = $derived(marked.lexer(text, { gfm: true, breaks: false }));
  let copied = $state<string | null>(null);

  let failed = $state<string | null>(null);

  async function copy(code: string) {
    if (await copyText(code)) {
      copied = code;
      setTimeout(() => (copied = null), 1200);
    } else {
      failed = code;
      setTimeout(() => (failed = null), 2000);
    }
  }

  function open(event: MouseEvent, href: string) {
    event.preventDefault();
    if (/^https?:\/\//.test(href)) void host.open("url", href);
  }
</script>

{#snippet inline(list: Token[] | undefined)}
  {#each list ?? [] as t, i (i)}
    {#if t.type === "strong"}<strong>{@render inline((t as Tokens.Strong).tokens)}</strong>
    {:else if t.type === "em"}<em>{@render inline((t as Tokens.Em).tokens)}</em>
    {:else if t.type === "del"}<del>{@render inline((t as Tokens.Del).tokens)}</del>
    {:else if t.type === "codespan"}<code>{(t as Tokens.Codespan).text}</code>
    {:else if t.type === "br"}<br />
    {:else if t.type === "link"}<a href={(t as Tokens.Link).href} onclick={(e) => open(e, (t as Tokens.Link).href)}>{@render inline((t as Tokens.Link).tokens)}</a>
    {:else if t.type === "image"}<a href={(t as Tokens.Image).href} onclick={(e) => open(e, (t as Tokens.Image).href)}>{(t as Tokens.Image).text || (t as Tokens.Image).href}</a>
    {:else if t.type === "text" && "tokens" in t && t.tokens}{@render inline(t.tokens)}
    {:else if t.type === "escape" || t.type === "text" || t.type === "html"}{decode((t as Tokens.Text).text)}
    {:else}{"raw" in t ? t.raw : ""}{/if}
  {/each}
{/snippet}

{#snippet blocks(list: Token[])}
  {#each list as t, i (i)}
    {#if t.type === "paragraph"}<p>{@render inline((t as Tokens.Paragraph).tokens)}</p>
    {:else if t.type === "heading"}
      <div class="h h{(t as Tokens.Heading).depth}" role="heading" aria-level={(t as Tokens.Heading).depth + 2}>{@render inline((t as Tokens.Heading).tokens)}</div>
    {:else if t.type === "code"}
      {@const code = t as Tokens.Code}
      <div class="code">
        <div class="code-bar">
          <span>{code.lang || "code"}</span>
          <button class="btn plain" onclick={() => copy(code.text)} aria-label="Copy code">
            <Symbol name={copied === code.text ? "check" : "copy"} size={13} />{copied === code.text ? "Copied" : failed === code.text ? "Couldn’t copy" : "Copy"}
          </button>
        </div>
        <pre><code>{code.text}</code></pre>
      </div>
    {:else if t.type === "list"}
      {@const l = t as Tokens.List}
      {#if l.ordered}
        <ol start={typeof l.start === "number" ? l.start : 1}>{#each l.items as item, j (j)}<li>{@render blocks(item.tokens)}</li>{/each}</ol>
      {:else}
        <ul>{#each l.items as item, j (j)}<li>{@render blocks(item.tokens)}</li>{/each}</ul>
      {/if}
    {:else if t.type === "blockquote"}<blockquote>{@render blocks((t as Tokens.Blockquote).tokens)}</blockquote>
    {:else if t.type === "hr"}<hr />
    {:else if t.type === "table"}
      {@const tb = t as Tokens.Table}
      <div class="table"><table>
        <thead><tr>{#each tb.header as c, j (j)}<th>{@render inline(c.tokens)}</th>{/each}</tr></thead>
        <tbody>{#each tb.rows as row, j (j)}<tr>{#each row as c, k (k)}<td>{@render inline(c.tokens)}</td>{/each}</tr>{/each}</tbody>
      </table></div>
    {:else if t.type === "text"}<p>{@render inline("tokens" in t && t.tokens ? t.tokens : [t])}</p>
    {:else if t.type === "html"}<p>{(t as Tokens.HTML).text}</p>
    {:else if t.type === "space"}{""}
    {/if}
  {/each}
{/snippet}

<script lang="ts" module>
  /** marked keeps entities such as `&amp;` in text tokens; show them as characters, as text. */
  function decode(s: string): string {
    return s.replace(/&(amp|lt|gt|quot|#39);/g, (_, e: string) =>
      ({ amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'" })[e] ?? _,
    );
  }
</script>

<div class="md">{@render blocks(tokens)}</div>

<style>
  .md {
    line-height: 1.55;
    font-size: 14px;
    overflow-wrap: anywhere;
  }
  .md :global(p) {
    margin: 0 0 0.7em;
  }
  .md > :global(:last-child) {
    margin-bottom: 0;
  }
  .h {
    font-weight: 650;
    margin: 0.9em 0 0.4em;
  }
  .h1 {
    font-size: 18px;
  }
  .h2 {
    font-size: 16px;
  }
  .h3,
  .h4,
  .h5,
  .h6 {
    font-size: 14px;
  }
  .md :global(code) {
    font: 12.5px/1.5 var(--mono);
    background: var(--fill);
    border-radius: 4px;
    padding: 1px 4px;
  }
  .code {
    margin: 0.4em 0 0.8em;
    border-radius: var(--r);
    background: var(--surface-2);
    border: 0.5px solid var(--separator);
    overflow: hidden;
  }
  .code-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2px 4px 2px 12px;
    color: var(--muted);
    font-size: 11px;
    border-bottom: 0.5px solid var(--separator);
  }
  .code-bar .btn {
    height: 22px;
    font-size: 11px;
  }
  pre {
    margin: 0;
    padding: 10px 12px;
    overflow-x: auto;
  }
  pre code {
    background: none !important;
    padding: 0 !important;
  }
  ul,
  ol {
    margin: 0 0 0.7em;
    padding-left: 1.4em;
  }
  blockquote {
    margin: 0 0 0.7em;
    padding-left: 12px;
    border-left: 3px solid var(--fill-2);
    color: var(--text-2);
  }
  hr {
    border: none;
    border-top: 0.5px solid var(--separator);
    margin: 1em 0;
  }
  a {
    color: var(--accent-ink);
    text-decoration: none;
  }
  a:hover {
    text-decoration: underline;
  }
  .table {
    overflow-x: auto;
    margin-bottom: 0.8em;
  }
  table {
    border-collapse: collapse;
    font-size: 13px;
  }
  th,
  td {
    border-bottom: 0.5px solid var(--separator);
    padding: 4px 10px;
    text-align: left;
  }
</style>
