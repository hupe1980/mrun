+++
title = "The mrun window"
description = "mrun's window and menu bar item: what the server is doing, a chat with the model, models that fit your Mac, and one-click agent setup. Everything stays on your Mac."
weight = 15
+++

`mrun ui` opens mrun's window. It shows what the terminal can't: how full the memory budget is,
what each request is doing, and, for every prompt, how much mrun reused instead of computing it
again. It also chats with the model, downloads models that fit, and connects your agents.

```sh
mrun ui              # the window and a menu bar item
mrun ui --browser    # the same in your default browser
```

<figure class="screenshot"><a href="../../screenshots/now.png" title="Open full size"><img src="../../screenshots/now.png" alt="Now: the memory budget, a prompt being read, and the reuse of recent requests" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Now: the memory budget, a prompt being read, and the reuse of recent requests</figcaption></figure>

## Now

A sentence that says what mrun is doing, for example *qwen3.6 is ready*. Under it:

- **Memory**: one bar, like System Settings › Storage. It shows the model, working memory,
  conversations, checkpoints and what is free, inside the budget mrun fixed at startup.
- **Working now**: each running or waiting request with its real progress ("Reading prompt ·
  12,400 of 30,000 tokens"), its lane and its speed.
- **Recent requests**: one thin ribbon per request. The reused part of the prompt is solid, and
  the computed part is lighter, labelled with why. These are the server's own numbers, the same
  as `mrun ps`.
  - **Expected recomputes are grey**: a new conversation, new tokens at the end, your own edit
    (*edited at message 12, user*), a resume from a checkpoint, or the few tokens where the
    chat template re-renders an earlier reply (*reply re-rendered at message 2, assistant*).
  - **Amber marks what nobody intended**, which is the one thing worth fixing: *dropped from
    memory earlier*, *the cache is off*, or *client changed message 7 (tool)*, where an agent
    re-sent an earlier message differently.

  The share of all prompt tokens reused appears at the top once a few requests have finished.

## Chat

Talk to the model you are serving through the same API your agents use. Replies stream in.
<kbd>Esc</kbd> stops a reply and keeps what arrived. Under each reply one quiet line says how much
of the conversation was reused, the time to first token, the decode speed and why it stopped.
Choosing that line opens the reply's ribbon.

While a model thinks, the reply reads *Thinking · 23 s · 1,104 tokens* with the newest line of
its thoughts. Afterwards it collapses to *Thought for 41 s*, and choosing it shows the thoughts.

<figure class="screenshot"><a href="../../screenshots/chat.png" title="Open full size"><img src="../../screenshots/chat.png" alt="Chat: each reply says how much of the conversation was reused, how fast it started and why it stopped" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Chat: each reply says how much of the conversation was reused, how fast it started and why it stopped</figcaption></figure>

**Files**: drag a file onto the chat, paste it, or choose the paperclip. It goes into your message
whole, as text, and each follow-up reuses it from the cache instead of reading it again. The chip
shows its token count from the server. A file that would overflow the context is refused with
both numbers. Images and PDFs are refused too: mrun's models read text only for now.

Hover over a message, or right-click it, for its actions:
- **Copy**: copies a reply or a question; code blocks have their own Copy.
- **Regenerate** (<kbd>⌘R</kbd>): asks again for the last reply.
- **Edit**: changes a question in place, also with <kbd>↑</kbd> in an empty message field.
  <kbd>Return</kbd> sends it, the chat continues from there, and the replies after it are
  dropped. The new reply shows that everything before your edit was reused.

The inspector (<kbd>⌥⌘I</kbd>) holds the chat's system prompt, the thinking switch for models
that think, and every sampling setting: temperature, top-p, top-k, min-p, repetition penalty,
reply length and seed. Each setting shows the model's own default, and Reset puts them all back.

Search finds words in titles and inside messages and files. Right-click a chat to copy it as
Markdown or save it to Downloads.

The first chat asks once whether to keep chats on this Mac. Kept chats live in `~/.mrun/chats`
and nowhere else. Settings › General changes the answer and can delete them all.

## Models

mrun's aliases and the models you pulled, each with a verdict for this Mac: *Fits*, *Tight*
(fits, but with less context than coding agents need) or *Won't fit*. The verdict is the one
`mrun ls` prints, from the same memory rule the server applies. One model is marked **Best for
this Mac**. Download shows its progress and can be cancelled (a later download resumes). Use
restarts the server with that model. The model being served reads **In Use**. A model that won't fit is never loaded halfway.

<figure class="screenshot"><a href="../../screenshots/models.png" title="Open full size"><img src="../../screenshots/models.png" alt="Models: the fit verdict for this Mac, the model in use and the best one" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Models: the fit verdict for this Mac, the model in use and the best one</figcaption></figure>

**Add a model**: paste a Hugging Face repo (`org/name`) or its link. mrun shows what it would
download, whether it supports the model, and the verdict for this Mac, before a byte downloads.
This is `mrun pull <repo> --dry-run`.

The first time you open mrun with no model, one sheet offers the best model for this Mac:
Continue downloads and starts it, and Connect sets up Claude Code.

## Agents

Claude Code, Codex and OpenCode, each with whether it is installed and whether it uses mrun.
Connect writes the agent's settings exactly as `mrun connect <agent> --write` does, with a backup.
Disconnect puts them back.

**Your code**: the server's address and model in ready-to-paste examples for all three APIs, in
curl, Python and TypeScript. **Other apps**: the settings to paste into Zed and Crush, the same
as `mrun connect zed` prints.

<figure class="screenshot"><a href="../../screenshots/agents.png" title="Open full size"><img src="../../screenshots/agents.png" alt="Agents: Claude Code connected, Codex one click away" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Agents: Claude Code connected, Codex one click away</figcaption></figure>

## The CLI underneath

Every action in the window is an `mrun` command. Its menu (…), and the menu you get by
right-clicking a model, chat or agent (or <kbd>⇧F10</kbd>), has **Copy as Command**, which copies
the exact command the window ran, for example `mrun pull qwen3.6`. Settings › Server
shows the server's settings with their sources and opens `config.toml`, which mrun never writes.

## Keyboard

| Keys | Does |
| --- | --- |
| <kbd>⌘1</kbd>–<kbd>⌘4</kbd> | Now, Chat, Models, Agents |
| <kbd>⌘K</kbd> | Every action and place by name |
| <kbd>⌘L</kbd> | Choose a model |
| <kbd>⌘N</kbd> | New chat |
| <kbd>⌘E</kbd> | The chat's system prompt |
| <kbd>⌘R</kbd> | Regenerate the last reply |
| <kbd>↑</kbd> (empty message field) | Edit your last question |
| <kbd>Esc</kbd> or <kbd>⌘.</kbd> | Stop the reply |
| <kbd>⌥⌘I</kbd> | Chat settings |
| <kbd>⌘,</kbd> | Settings |

## On the Mac

The window opens where you left it, at the same size. Its toolbar is the title bar: drag it to
move the window, and double-click it to zoom (or whatever System Settings › Desktop & Dock says).
While mrun reads a prompt, the bolt in the menu bar fills from the bottom; it stays still with
Reduce Motion. Increase Contrast strengthens separators, bars and focus rings.

## Privacy

The window talks only to mrun on this Mac. It makes no network requests of its own except the
downloads you start. There is no account, no telemetry and no update check. Its controls need a
key that mrun gives only to its own window, so other pages and apps can't use them. In remote
mode (`mrun cloud`) the window shows a distinct colour and says that prompts go to your pod.
