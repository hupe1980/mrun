+++
title = "The mrun window"
description = "mrun's window and menu bar item: server state, chat, session replay, models and agent setup."
weight = 15
+++

`mrun ui` opens mrun's window: the memory budget, what each request is doing and how much of each
prompt was reused. It also replays agent sessions, chats with the model, downloads models and
connects your agents.

```sh
mrun ui              # the window and a menu bar item
mrun ui --browser    # the same in your default browser
```

<figure class="screenshot"><a href="../../screenshots/now.png" title="Open full size"><img src="../../screenshots/now.png" alt="Now: the memory budget, a prompt being read, and the reuse of recent requests" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Now: the memory budget, a prompt being read, and the reuse of recent requests</figcaption></figure>

## Now

A line that says what mrun is doing, for example *qwen3.6 is ready*. Under it:

- **Memory**: one bar with the model, working memory, conversations, checkpoints and what is
  free, inside the budget fixed at startup.
- **Working now**: each running or waiting request with its progress ("Reading prompt · 12,400 of
  30,000 tokens") and speed.
- **Recent requests**: one ribbon per request, the same numbers as `mrun ps`. The reused part of
  the prompt is solid; the computed part is lighter and labelled with why.
  - **Grey**, expected: a new conversation, new tokens at the end, your own edit (*edited at
    message 12, user*), a resume from a checkpoint, or a re-rendered earlier reply (*reply
    re-rendered at message 2, assistant*).
  - **Amber**, worth fixing: *dropped from memory earlier*, *the cache is off*, or *client changed
    message 7 (tool)*, where an agent re-sent an earlier message differently.

  The share of all prompt tokens reused appears at the top once a few requests have finished,
  and under it the tokens per pass of [speculative decoding](@/docs/how-it-works.md#speculative-decoding)
  once it has drafted; a request that drafted shows its own.

## Chat

Chat with the served model through the same API your agents use. <kbd>Esc</kbd> stops a reply
and keeps what arrived. Under each reply a line shows the reuse, time to first token, decode speed,
tokens per pass when it was drafted, and stop reason; choosing it opens the reply's ribbon. While the model thinks, the reply shows
*Thinking · 23 s · 1,104 tokens* and the newest line of its thoughts; afterwards *Thought for
41 s*, which expands.

<figure class="screenshot"><a href="../../screenshots/chat.png" title="Open full size"><img src="../../screenshots/chat.png" alt="Chat: each reply says how much of the conversation was reused, how fast it started and why it stopped" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Chat: each reply says how much of the conversation was reused, how fast it started and why it stopped</figcaption></figure>

**Files**: drag, paste or attach a file. It goes into your message whole, as text, and
follow-ups reuse it from the cache. The chip shows its token count. A file that would overflow the
context is refused with both numbers. Images and PDFs are refused; the models read text only.

Hover over a message, or right-click it, for its actions:
- **Copy**: copies a reply or a question; code blocks have their own Copy.
- **Regenerate** (<kbd>⌘R</kbd>): asks again for the last reply.
- **Edit**: changes a question in place, also with <kbd>↑</kbd> in an empty message field.
  <kbd>Return</kbd> sends it, the chat continues from there, and the replies after it are
  dropped. The new reply shows that everything before your edit was reused.

The inspector (<kbd>⌥⌘I</kbd>) holds the system prompt, the thinking switch and the sampling
settings (temperature, top-p, top-k, min-p, repetition penalty, reply length, seed), each with the
model's default; Reset restores them.

Search finds words in titles and inside messages and files. Right-click a chat to copy it as
Markdown or save it to Downloads.

The first chat asks whether to keep chats on this Mac. Kept chats live in `~/.mrun/chats`.
Settings › General changes the answer and can delete them all.

## Sessions

Each coding-agent session as a skyline: one column per request, in order, its height the prompt.
The reused part is green, the computed cap grey, or amber when it is worth fixing. The main
conversation, each subagent and the side requests (titles, compaction) get their own rows; a pause
of over 30 seconds shows as *4 min idle*. The thin strip under each column darkens with the wait
for the first token.

<figure class="screenshot"><a href="../../screenshots/sessions.png" title="Open full size"><img src="../../screenshots/sessions.png" alt="Sessions: a Claude Code session with its main row, subagents and side requests, and a selected miss with the text that changed" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Sessions: a selected miss shows the cached and the sent text, with the change marked</figcaption></figure>

Select a column for its numbers and cause. When the prompt changed, the inspector shows the text
at the change twice, as cached and as sent, with only the change marked. The text is read from
memory when you select the request; it is never stored or logged. <kbd>⌥→</kbd> jumps to the next
miss worth fixing.

The view follows the newest request; scrolling back pauses it, <kbd>Space</kbd> follows again.
Sessions keep the last 2,000 requests in memory until the server stops. `mrun ps --sessions`
prints the same. A recent request on Now opens here with its icon. With a rented GPU, Sessions
stays empty: the remote server reports no causes.

## Pool

Macs on your network running one model together: start a pool, join one nearby with the code
both Macs show, and plan a model across them. The pool is drawn as a ring: each Mac a tile with its
layers and link, the model's layers the circle. See [Mac pool](@/docs/pool.md).

## Models

The aliases and pulled models, each with the verdict `mrun ls` prints: *Fits*, *Tight* (less
context than coding agents need) or *Won't fit*. One model is marked **Best for this Mac**, the
served one **In Use**. Download can be cancelled and resumes later. Use restarts the server with
that model. A model that won't fit is never loaded. Each model that fits offers a context
(automatic, or 16K up to what fits) with the memory it uses; Use serves it with that context, and
for the model in use Apply restarts it (`mrun serve --context`).

<figure class="screenshot"><a href="../../screenshots/models.png" title="Open full size"><img src="../../screenshots/models.png" alt="Models: the fit verdict for this Mac, a context per model, the model in use with Eject" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Models: the fit verdict for this Mac, a context per model, the model in use with Eject</figcaption></figure>

**Add a model**: paste a Hugging Face repo (`org/name`) or its link to see what would be
downloaded, whether the model is supported and the verdict (`mrun pull <repo> --dry-run`).

**On a rented GPU**: models too big for this Mac, on RunPod (opt-in; see
[Remote GPU](../cloud/)). Run… shows the fitting GPUs with price and stock, asks for the RunPod
key once if none is set, and follows the start step by step. While a pod serves, Now shows its
cost and when it stops itself, and the menu bar shows the spend.

With no model pulled, a first-run sheet offers the best model for this Mac and to connect Claude
Code.

## Agents

Claude Code, Codex and OpenCode, each with whether it is installed and uses mrun. Connect runs
`mrun connect <agent> --write` (with a backup); Disconnect restores the settings.

**Your code**: examples for all three APIs in curl, Python and TypeScript. **Other apps**: the
settings for Zed and Crush, as `mrun connect zed` prints them.

<figure class="screenshot"><a href="../../screenshots/agents.png" title="Open full size"><img src="../../screenshots/agents.png" alt="Agents: Claude Code connected, Codex one click away" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Agents: Claude Code connected, Codex one click away</figcaption></figure>

## The CLI underneath

Every action in the window is an `mrun` command. The … menu and the context menu of a model,
chat or agent (or <kbd>⇧F10</kbd>) have **Copy as Command**, for example `mrun pull qwen3.6`.
Settings › Server shows the server's settings with their sources, opens `config.toml` and opens
the server log in Console, which follows it live (`mrun logs -f` in a terminal).

<figure class="screenshot"><a href="../../screenshots/settings-server.png" title="Open full size"><img src="../../screenshots/settings-server.png" alt="Settings, Server: each setting with where it comes from" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Settings › Server: each setting with where it comes from</figcaption></figure>

## Keyboard

| Keys | Does |
| --- | --- |
| <kbd>⌘1</kbd>–<kbd>⌘6</kbd> | Now, Chat, Sessions, Pool, Models, Agents |
| <kbd>⌘K</kbd> | Every action and place by name |
| <kbd>⌘L</kbd> | Choose a model (<kbd>⏏</kbd> in the toolbar ejects it) |
| <kbd>⌘N</kbd> | New chat |
| <kbd>⌘E</kbd> | The chat's system prompt |
| <kbd>⌘R</kbd> | Regenerate the last reply |
| <kbd>↑</kbd> (empty message field) | Edit your last question |
| <kbd>Esc</kbd> or <kbd>⌘.</kbd> | Stop the reply |
| <kbd>⌥⌘I</kbd> | Chat settings |
| <kbd>⌘,</kbd> | Settings |
| <kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> <kbd>↓</kbd> | Sessions: the previous or next request, the row above or below |
| <kbd>⌥←</kbd> <kbd>⌥→</kbd> | Sessions: the previous or next miss worth fixing |
| <kbd>⌘+</kbd> <kbd>⌘−</kbd> <kbd>⌘0</kbd> | Sessions: zoom in, out, fit the session |

## On the Mac

The model chip at the top of every place shows what is served and its memory; a click (or
<kbd>⌘L</kbd>) switches models, and <kbd>⏏</kbd> beside it ejects the model, freeing its memory
(`mrun stop`). The menu bar menu has **Stop Serving** too.

Closing the window keeps mrun in the menu bar; without the menu bar item, closing quits. Quit
ejects the model the window started; a server started in a terminal keeps serving its agents.

The window reopens where you left it. While mrun reads a prompt, the menu bar bolt fills from
the bottom (still with Reduce Motion). Increase Contrast strengthens separators, bars and focus
rings.

## Privacy

The window talks only to mrun on this Mac and makes no network requests except downloads you
start. No account, telemetry or update check. Its controls need a key mrun gives only to its own
window. In remote mode (`mrun cloud`) the window shows a distinct colour and says that prompts go
to your pod.
