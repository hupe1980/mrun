<script lang="ts">
  // Chat with the served model through the public Messages API, as any agent would (FR-007).
  // A flat list of chats, a reply that streams and stops on Esc keeping what arrived, a stats
  // line under each reply, and the inspector for the system prompt and sampling (FR-010a).
  import { onMount, tick } from "svelte";
  import Attachments from "../components/Attachments.svelte";
  import Inspector from "../components/Inspector.svelte";
  import Message from "../components/Message.svelte";
  import Symbol from "../components/Symbol.svelte";
  import ToolsPicker from "../components/ToolsPicker.svelte";
  import { content, estimate, fence, readAttachment } from "../lib/attach";
  import { countTokens, host, streamMessages } from "../lib/api";
  import { exportMarkdown, searchHit, tokens } from "../lib/format";
  import { copyText } from "../lib/clipboard";
  import { live } from "../lib/live.svelte";
  import { contextMenu } from "../lib/native.svelte";
  import { tools } from "../lib/tools.svelte";
  import type { Attachment, Chat, Message as Msg, Sampling, Stats, ToolCall } from "../lib/types";

  let { active }: { active: boolean } = $props();

  const fresh = (): Chat => {
    const now = new Date().toISOString();
    return {
      id: Array.from(crypto.getRandomValues(new Uint8Array(8)), (b) => b.toString(16).padStart(2, "0")).join(""),
      title: "New Chat",
      created: now,
      updated: now,
      model: null,
      system: null,
      sampling: {},
      thinking: null,
      messages: [],
    };
  };

  let chats = $state<Chat[]>([fresh()]);
  // svelte-ignore state_referenced_locally
  let currentId = $state(chats[0]!.id);
  let query = $state("");
  let draft = $state("");
  let streaming = $state<AbortController | null>(null);
  /** The running reply, so Edit and Regenerate can stop it and wait (019 FR-004). */
  let running: Promise<void> | null = null;
  /** The index of the question being edited in place. */
  let editing = $state<number | null>(null);
  /** Files waiting for the next message, and the ones refused with their reason (020 US1). */
  let pending = $state<Attachment[]>([]);
  let refused = $state<{ name: string; reason: string }[]>([]);
  let picker: HTMLInputElement | undefined = $state();
  let dragging = $state(false);
  let inspector = $state(false);
  let defaults = $state<Record<string, any>>({ temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0 });
  let thinkingFamily = $state(false);
  let scroller: HTMLElement | undefined = $state();
  /** Tools (028): the picker, the calls waiting for the user, and whether a reply's tools run. */
  let toolsOpen = $state(false);
  let toolsBusy = $state(false);
  const decisions = new Map<string, (d: Decision) => void>();
  type Decision = "once" | "always" | "deny" | "retry" | "continue";
  /** The most tool rounds one reply may take (FR-005). */
  const ROUNDS = 20;
  let composer: HTMLTextAreaElement | undefined = $state();

  const current = $derived(chats.find((c) => c.id === currentId) ?? chats[0]!);
  // Titles and message text, with the matching excerpt (020 FR-010).
  const visible = $derived(
    chats
      .map((c) => ({ c, hit: searchHit(c, query) }))
      .filter((x) => x.hit !== null)
      .sort((a, b) => b.c.updated.localeCompare(a.c.updated)),
  );
  const model = $derived(live.status?.model?.id ?? null);
  const keep = $derived(live.app?.prefs?.keep_chats ?? null);
  const ask = $derived(keep === null && current.messages.length === 0 && chats.length === 1);
  const ready = $derived(live.state !== "stopped" && live.state !== "loading" && model !== null);
  const toolsOn = $derived(current.tools ?? []);
  // The privacy line names the remote hosts this chat's tools reach (FR-008).
  $effect(() => {
    tools.hosts = toolsOn.map((n) => tools.server(n)).filter((s) => s?.kind === "http" && s.host).map((s) => s!.host!);
  });
  $effect(() => {
    if (toolsOn.length && !tools.list) void tools.load("start");
  });

  onMount(async () => {
    if (live.app?.prefs?.keep_chats) {
      try {
        const list = await host.chats();
        const loaded = await Promise.all(list.map((c) => host.chat(c.id)));
        if (loaded.length) {
          chats = [...loaded, ...chats.filter((c) => c.messages.length)];
          if (!chats.length) chats = [fresh()];
          currentId = chats[0]!.id;
        }
      } catch {
        /* kept chats unreadable: start empty */
      }
    }
  });

  // The model's own sampling defaults and whether it thinks (from the public models route).
  $effect(() => {
    if (!model) return;
    host
      .modelInfo(model)
      .then((info) => {
        const d = (info as any).defaults;
        if (d) defaults = d;
        thinkingFamily = Boolean((info as any).thinking);
      })
      .catch(() => {});
  });

  async function persist(chat: Chat) {
    if (live.app?.prefs?.keep_chats) await host.saveChat($state.snapshot(chat) as Chat).catch(() => {});
  }

  function newChat() {
    const c = fresh();
    chats = [c, ...chats.filter((x) => x.messages.length > 0)];
    currentId = c.id;
    void tick().then(() => composer?.focus());
  }

  async function choose(keepChats: boolean) {
    await host.prefs({ keep_chats: keepChats }).catch(() => {});
    await live.tick();
  }

  function request(chat: Chat): Record<string, unknown> {
    const sampling: Sampling = chat.sampling;
    const body: Record<string, unknown> = {
      model,
      max_tokens: sampling.max_tokens ?? 8192,
      messages: chat.messages.filter((m) => !m.error || m.text || m.tools?.length).flatMap(turn),
    };
    const defs = tools.definitions(chat.tools ?? []);
    if (defs.length) body.tools = defs;
    if (chat.system) body.system = chat.system;
    for (const k of ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed"] as const) {
      if (sampling[k] !== undefined) body[k] = sampling[k];
    }
    if (thinkingFamily && chat.thinking !== false) body.thinking = { type: "enabled", budget_tokens: 8192 };
    return body;
  }

  /** A message as the API's turns: a reply that called tools is followed by their results. */
  function turn(m: Msg): Record<string, unknown>[] {
    if (m.role === "user") return [{ role: "user", content: content(m) }];
    if (!m.tools?.length) return [{ role: "assistant", content: m.text }];
    return [
      {
        role: "assistant",
        content: [...(m.text ? [{ type: "text", text: m.text }] : []), ...m.tools.map((t) => ({ type: "tool_use", id: t.id, name: t.name, input: t.input }))],
      },
      {
        role: "user",
        content: m.tools.map((t) => ({
          type: "tool_result",
          tool_use_id: t.id,
          content: t.result ?? "The call did not run.",
          is_error: t.state !== "done",
        })),
      },
    ];
  }

  async function send(text = draft.trim()) {
    if ((!text && !pending.length) || streaming || toolsBusy || !ready) return;
    const chat = current;
    const files = pending;
    draft = "";
    pending = [];
    refused = [];
    if (chat.messages.length === 0) chat.title = (text.split("\n")[0] || files[0]!.name).slice(0, 60);
    chat.model = model;
    chat.messages.push(files.length ? { role: "user", text, attachments: files } : { role: "user", text });
    await reply(chat);
  }

  // The server's count for a file alone: one user message with it, minus an empty one (the
  // template's own tokens; research R2), per model.
  let overhead: { model: string; tokens: number } | null = null;
  async function fileTokens(a: Attachment): Promise<number> {
    if (overhead?.model !== model) overhead = { model: model!, tokens: await countTokens({ model, messages: [{ role: "user", content: "" }] }) };
    const all = await countTokens({ model, messages: [{ role: "user", content: fence(a.name, a.text) }] });
    return Math.max(0, all - overhead.tokens);
  }

  /** Tokens the chat would send now, with the files waiting; `null` without a server. */
  async function conversationTokens(files: Attachment[]): Promise<number | null> {
    if (!ready) return null;
    const draftChat = { ...current, messages: [...current.messages, { role: "user" as const, text: draft, attachments: files }] };
    const body = request(draftChat);
    return countTokens({ model: body.model, system: body.system, messages: body.messages });
  }

  /** Reads, counts and checks each file against what is left of the context (FR-001–FR-003). */
  async function addFiles(files: Iterable<File>) {
    const capacity = live.status?.budget?.kv_capacity_tokens ?? null;
    for (const file of files) {
      const read = await readAttachment(file);
      if (!read.ok) {
        refused = [...refused, { name: read.name, reason: read.reason }];
        continue;
      }
      const a = read.attachment;
      try {
        if (ready) {
          a.tokens = await fileTokens(a);
          a.estimated = false;
        }
        const used = await conversationTokens(pending);
        if (capacity !== null && used !== null && used + a.tokens > capacity) {
          const left = Math.max(0, capacity - used);
          refused = [...refused, { name: a.name, reason: `${tokens(a.tokens)} tokens; ${tokens(left)} are left in this chat's context.` }];
          continue;
        }
      } catch {
        a.tokens = estimate(a.bytes);
        a.estimated = true;
      }
      pending = [...pending, a];
    }
  }

  // Counts belong to a model: take them again when the served model changes (Edge Cases).
  let countedFor: string | null = null;
  $effect(() => {
    const m = model;
    if (!m || m === countedFor) return;
    countedFor = m;
    const files = pending;
    if (!files.length) return;
    void Promise.all(files.map(async (a) => ({ ...a, tokens: await fileTokens(a).catch(() => estimate(a.bytes)), estimated: false }))).then(
      (recounted) => (pending = recounted),
    );
  });

  function onDrop(e: DragEvent) {
    dragging = false;
    if (!e.dataTransfer?.files.length) return;
    e.preventDefault();
    void addFiles(e.dataTransfer.files);
  }

  function onPaste(e: ClipboardEvent) {
    const files = e.clipboardData?.files;
    if (files?.length) {
      e.preventDefault();
      void addFiles(files);
    }
  }

  async function exportChat(c: Chat, save: boolean) {
    const text = exportMarkdown(c, fence);
    if (!save) {
      if (!(await copyText(text))) refused = [{ name: c.title, reason: "Couldn't copy." }];
      return;
    }
    await host.exportChat(c.title, text).catch((e) => (refused = [{ name: c.title, reason: String((e as Error).message) }]));
  }

  function reply(chat: Chat): Promise<void> {
    running = converse(chat).finally(() => (running = null));
    return running;
  }

  /** Replies, runs the tools a reply calls (each allowed first), and replies again (FR-005). */
  async function converse(chat: Chat) {
    for (let round = 1; ; round++) {
      const msg = await replyNow(chat);
      if (!msg.tools?.length || msg.error || msg.stats?.stop === "stopped") return;
      if (!(await runTools(chat, msg))) return;
      await persist(chat);
      if (round >= ROUNDS) {
        msg.error = `Stopped after ${ROUNDS} tool rounds. Send a message to continue.`;
        return;
      }
    }
  }

  /** Waits for the card's buttons (or the keyboard) on one call. */
  function decide(call: ToolCall, state: "asking" | "blocked"): Promise<Decision> {
    // Stopped meanwhile: nobody will answer, so the answer is no.
    if (!toolsBusy) return Promise.resolve("deny");
    call.state = state;
    void scrollDown();
    return new Promise((resolve) => decisions.set(call.id, resolve));
  }

  function onDecide(id: string, d: Decision) {
    decisions.get(id)?.(d);
    decisions.delete(id);
  }

  /** "Allow …" on a denial: the server's block gains it; Run Again then uses it. */
  async function onAllow(id: string, index: number) {
    const call = current.messages.flatMap((m) => m.tools ?? []).find((t) => t.id === id);
    const found = call && tools.resolve(call.name, toolsOn);
    const den = call?.denials?.[index];
    if (!call || !found || !den?.fix) return;
    try {
      await host.mcpAllow(found.server.name, den.fix);
      call.denials = call.denials!.map((d, i) => (i === index ? { text: `Allowed: ${d.label?.replace(/^Allow /, "")}`, fix: undefined } : d));
    } catch (e) {
      call.denials = [...call.denials!, { text: (e as Error).message }];
    }
  }

  /** Runs a reply's calls in order; false when the user stopped. */
  async function runTools(chat: Chat, msg: Msg): Promise<boolean> {
    toolsBusy = true;
    try {
      for (const call of msg.tools!) {
        const found = tools.resolve(call.name, chat.tools ?? []);
        if (!found) {
          call.state = "failed";
          call.result = `No tool ${call.name} is on in this chat.`;
          continue;
        }
        let d: Decision = (chat.always ?? []).includes(call.name) ? "once" : await decide(call, "asking");
        if (!toolsBusy) return false;
        call.auto = d === "once" && (chat.always ?? []).includes(call.name);
        if (d === "deny") {
          call.state = "denied";
          call.result = "The user declined this call.";
          continue;
        }
        if (d === "always") chat.always = [...(chat.always ?? []), call.name];
        for (;;) {
          call.state = "running";
          const r = await host.mcpCall(found.server.name, found.tool, call.input).catch((e) => ({ text: String((e as Error).message), is_error: true, denials: [] }));
          call.result = r.text;
          call.denials = r.denials;
          call.state = r.is_error ? "failed" : "done";
          if (!toolsBusy) return false;
          if (!r.denials?.some((x) => x.fix)) break;
          d = await decide(call, "blocked");
          if (!toolsBusy) return false;
          if (d !== "retry") {
            call.state = r.is_error ? "failed" : "done";
            break;
          }
        }
      }
      return true;
    } finally {
      toolsBusy = false;
    }
  }

  async function replyNow(chat: Chat): Promise<Msg> {
    const body = request(chat);
    chat.messages.push({ role: "assistant", text: "" });
    const msg = chat.messages[chat.messages.length - 1] as Msg;
    const controller = new AbortController();
    streaming = controller;
    const started = performance.now();
    let first: number | null = null;
    let thinkStart: number | null = null;
    let output = 0;
    const calls = new Map<number, ToolCall>();
    const stats: Stats = { prompt_tokens: 0, cached_tokens: 0, ttft_ms: null, decode_tps: null, stop: "end" };
    void scrollDown();
    try {
      for await (const ev of streamMessages(body, controller.signal)) {
        switch (ev.type) {
          case "meta":
            stats.miss = ev.miss;
            stats.cached_tokens = ev.miss.cached;
            stats.prompt_tokens = ev.miss.cached + ev.miss.processed;
            break;
          case "usage":
            if (!stats.prompt_tokens) {
              stats.prompt_tokens = ev.input;
              stats.cached_tokens = ev.cached;
            }
            break;
          case "thinking":
            first ??= performance.now();
            thinkStart ??= performance.now();
            msg.reasoning_started ??= Date.now();
            // mrun streams one delta per token (019 research R2).
            msg.reasoning_tokens = (msg.reasoning_tokens ?? 0) + 1;
            msg.reasoning = (msg.reasoning ?? "") + ev.text;
            break;
          case "text":
            first ??= performance.now();
            if (thinkStart !== null && msg.reasoning_ms === undefined) msg.reasoning_ms = performance.now() - thinkStart;
            msg.text += ev.text;
            void scrollDown();
            break;
          case "output":
            output = ev.tokens;
            break;
          case "stop":
            stats.stop = ev.reason === "max_tokens" ? "length" : ev.reason === "tool_use" ? "tools" : "end";
            break;
          case "speculation":
            stats.speculation = ev.speculation;
            break;
          case "error":
            msg.error = ev.message;
            stats.stop = "error";
            break;
          case "tool_start": {
            first ??= performance.now();
            msg.tools = [...(msg.tools ?? []), { id: ev.id, name: ev.name, input: {}, raw: "", state: "pending" }];
            calls.set(ev.index, msg.tools[msg.tools.length - 1]!);
            void scrollDown();
            break;
          }
          case "tool_args": {
            const c = calls.get(ev.index);
            if (c) c.raw = (c.raw ?? "") + ev.json;
            break;
          }
        }
      }
    } catch (e) {
      if (controller.signal.aborted) stats.stop = "stopped";
      else {
        msg.error = "The connection to mrun was lost.";
        stats.stop = "error";
      }
    }
    const end = performance.now();
    if (first !== null) {
      stats.ttft_ms = first - started;
      if (output > 1 && end > first) stats.decode_tps = (output - 1) / ((end - first) / 1000);
    }
    if (thinkStart !== null && msg.reasoning_ms === undefined) msg.reasoning_ms = end - thinkStart;
    for (const c of msg.tools ?? []) {
      try {
        c.input = c.raw ? JSON.parse(c.raw) : {};
      } catch {
        c.state = "failed";
        c.result = "The model's arguments were not valid JSON.";
      }
      delete c.raw;
    }
    // A stopped reply's calls never run.
    if (stats.stop === "stopped") msg.tools = msg.tools?.filter((c) => c.state !== "pending");
    msg.stats = stats;
    streaming = null;
    chat.updated = new Date().toISOString();
    await persist(chat);
    return msg;
  }

  function stop() {
    streaming?.abort();
    // Stop while tools wait or run: deny what is asked and end the turn.
    toolsBusy = false;
    for (const id of [...decisions.keys()]) onDecide(id, "deny");
  }

  /** Stops a running reply (as Stop does) and waits until it has settled. */
  async function settle() {
    if (!streaming && !toolsBusy) return;
    stop();
    await running;
  }

  async function regenerate() {
    await settle();
    const chat = current;
    if (!ready || chat.messages.at(-1)?.role !== "assistant") return;
    chat.messages.pop();
    await reply(chat);
  }

  function startEdit(index: number) {
    if (current.messages[index]?.role === "user") editing = index;
  }

  /** Sends an edited question: the chat continues from it, the later turns are dropped. */
  async function commitEdit(index: number, text: string) {
    await settle();
    const chat = current;
    editing = null;
    if (!ready) return;
    const defaultTitle = index === 0 && chat.title === chat.messages[0]!.text.split("\n")[0]!.slice(0, 60);
    // The question keeps its files; only its text was edited.
    const files = chat.messages[index]!.attachments;
    chat.messages = chat.messages.slice(0, index);
    chat.messages.push(files?.length ? { role: "user", text, attachments: files } : { role: "user", text });
    if (defaultTitle) chat.title = text.split("\n")[0]!.slice(0, 60);
    await reply(chat);
  }

  function chatMenu(e: MouseEvent | KeyboardEvent, c: Chat) {
    const some = c.messages.length > 0;
    void contextMenu(e, [
      { id: "open", title: "Open" },
      "-",
      { id: "copy-md", title: "Copy as Markdown", enabled: some },
      { id: "save-md", title: "Save as Markdown…", enabled: some },
      "-",
      { id: "delete", title: "Delete Chat", destructive: true, enabled: some },
    ]).then((id) => {
      if (id === "open") currentId = c.id;
      else if (id === "copy-md" || id === "save-md") void exportChat(c, id === "save-md");
      else if (id === "delete") void remove(c.id);
    });
  }

  async function remove(id: string) {
    chats = chats.filter((c) => c.id !== id);
    if (!chats.length) chats = [fresh()];
    if (currentId === id) currentId = chats[0]!.id;
    if (live.app?.prefs?.keep_chats) await host.deleteChat(id).catch(() => {});
  }

  async function scrollDown() {
    await tick();
    scroller?.scrollTo({ top: scroller.scrollHeight });
  }

  function onKey(e: KeyboardEvent) {
    if (!active) return;
    const cmd = e.metaKey || e.ctrlKey;
    if ((e.key === "Escape" || (cmd && e.key === ".")) && (streaming || toolsBusy)) {
      e.preventDefault();
      stop();
    } else if (cmd && e.key.toLowerCase() === "n" && !e.shiftKey) {
      e.preventDefault();
      newChat();
    } else if (cmd && e.key.toLowerCase() === "r") {
      e.preventDefault();
      void regenerate();
    } else if (cmd && e.key.toLowerCase() === "e") {
      e.preventDefault();
      inspector = true;
      void tick().then(() => document.getElementById("system-prompt")?.focus());
    } else if (cmd && e.altKey && e.key.toLowerCase() === "i") {
      e.preventDefault();
      inspector = !inspector;
    }
  }

  function onComposerKey(e: KeyboardEvent) {
    if (e.key === "ArrowUp" && !draft && !streaming) {
      // ↑ in an empty composer edits the last question, as in Messages.
      const last = current.messages.map((m) => m.role).lastIndexOf("user");
      if (last >= 0) {
        e.preventDefault();
        startEdit(last);
      }
    } else if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      void send();
    }
  }

  export function focusComposer() {
    composer?.focus();
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="chat">
  <nav class="list" aria-label="Chats">
    <div class="list-head">
      <input type="search" placeholder="Search" aria-label="Search chats" bind:value={query} />
      <button class="btn plain" onclick={newChat} aria-label="New chat (⌘N)" title="New Chat (⌘N)"><Symbol name="plus" size={15} /></button>
    </div>
    <ul>
      {#each visible as { c, hit } (c.id)}
        <li>
          <button class="item" class:current={c.id === currentId} onclick={() => ((currentId = c.id), (editing = null))} oncontextmenu={(e) => chatMenu(e, c)} onkeydown={(e) => (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) && chatMenu(e, c)}>
            <span class="title">{c.title}</span>
            {#if hit?.excerpt}
              <span class="sub excerpt">{hit.excerpt}</span>
            {:else}
              <span class="sub">{c.messages.length ? `${Math.ceil(c.messages.length / 2)} ${c.messages.length > 2 ? "turns" : "turn"}` : "Empty"}</span>
            {/if}
          </button>
          {#if c.messages.length}
            <button class="del" aria-label="Delete chat {c.title}" onclick={() => remove(c.id)}><Symbol name="trash" size={12} /></button>
          {/if}
        </li>
      {/each}
    </ul>
    {#if keep === false}<p class="hint">Chats are not kept. Settings › General changes this.</p>{/if}
  </nav>

  <!-- Files dropped anywhere on the conversation are attached (020 FR-001). -->
  <section
    class="conversation"
    class:dragging
    aria-label="Conversation"
    ondragover={(e) => {
      if (e.dataTransfer?.types.includes("Files")) {
        e.preventDefault();
        dragging = true;
      }
    }}
    ondragleave={(e) => e.currentTarget === e.target && (dragging = false)}
    ondrop={onDrop}
  >
    <div class="scroll" bind:this={scroller}>
      {#if current.messages.length === 0}
        <div class="welcome">
          <div class="mark" aria-hidden="true"><Symbol name="chat" size={28} /></div>
          <h2>Chat with {model ?? "the local model"}</h2>
          <p>Everything stays on this Mac. Each reply shows how much of the conversation mrun reused instead of reading it again.</p>
        </div>
      {:else}
        <div class="messages">
          {#each current.messages as m, i (i)}
            <Message
              message={m}
              streaming={streaming !== null && i === current.messages.length - 1}
              last={i === current.messages.length - 1}
              editing={editing === i}
              onedit={() => startEdit(i)}
              oncommit={(text) => void commitEdit(i, text)}
              oncancel={() => ((editing = null), composer?.focus())}
              onregenerate={() => void regenerate()}
              tools={toolsOn}
              ondecide={onDecide}
              onallow={(id, i) => void onAllow(id, i)}
            />
          {/each}
        </div>
      {/if}
    </div>

    {#if ask}
      <div class="ask" role="group" aria-label="Keep chats">
        <Symbol name="lock" size={14} />
        <span>Keep chats on this Mac?</span>
        <button class="btn" onclick={() => choose(false)}>Don’t Keep</button>
        <button class="btn primary" onclick={() => choose(true)}>Keep</button>
      </div>
    {/if}

    <form class="composer" class:has-files={pending.length || refused.length} onsubmit={(e) => { e.preventDefault(); void send(); }}>
      <Attachments items={pending} {refused} onremove={(i) => (pending = pending.filter((_, j) => j !== i))} ondismiss={() => (refused = [])} />
      <button type="button" class="btn plain icon" onclick={() => picker?.click()} aria-label="Attach files" title="Attach Files">
        <Symbol name="clip" size={16} />
      </button>
      <input bind:this={picker} type="file" multiple hidden tabindex="-1" aria-label="Files to attach" onchange={(e) => { const f = e.currentTarget.files; if (f) void addFiles([...f]); e.currentTarget.value = ""; }} />
      <textarea
        bind:this={composer}
        bind:value={draft}
        rows="1"
        placeholder={ready ? `Message ${model}` : "Start the server to chat"}
        aria-label="Message"
        onkeydown={onComposerKey}
        onpaste={onPaste}
        disabled={!ready}
      ></textarea>
      <span class="tools-anchor">
        <button type="button" class="btn plain icon" class:on={toolsOn.length > 0} onclick={() => (toolsOpen = !toolsOpen)} aria-label="Tools ({toolsOn.length} on)" aria-expanded={toolsOpen} title="Tools">
          <Symbol name="tool" size={16} />
          {#if toolsOn.length}<i class="badge num">{toolsOn.length}</i>{/if}
        </button>
        {#if toolsOpen}
          <ToolsPicker
            bind:on={() => toolsOn, (v) => (current.tools = v)}
            {model}
            onclose={() => (toolsOpen = false)}
            onmanage={() => ((toolsOpen = false), window.dispatchEvent(new CustomEvent("mrun:menu", { detail: { item: "settings", target: "tools" } })))}
          />
        {/if}
      </span>
      <button type="button" class="btn plain icon" class:on={inspector} onclick={() => (inspector = !inspector)} aria-label="Chat settings (⌥⌘I)" title="Chat Settings (⌥⌘I)">
        <Symbol name="inspector" size={16} />
      </button>
      {#if streaming || toolsBusy}
        <button type="button" class="send stop" onclick={stop} aria-label="Stop (Esc)" title="Stop (Esc)"><Symbol name="stop" size={12} /></button>
      {:else}
        <button type="submit" class="send" disabled={(!draft.trim() && !pending.length) || !ready} aria-label="Send (Return)"><Symbol name="send" size={15} weight={2} /></button>
      {/if}
    </form>
  </section>

  {#if inspector}
    <Inspector bind:chat={chats[chats.findIndex((c) => c.id === currentId)]!} defaults={defaults as any} {thinkingFamily} onclose={() => (inspector = false)} />
  {/if}
</div>

<style>
  .chat {
    display: flex;
    height: 100%;
    min-height: 0;
  }
  .list {
    width: 220px;
    flex: none;
    border-right: 0.5px solid var(--separator);
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .list-head {
    display: flex;
    gap: 4px;
    padding: var(--s3);
  }
  .list-head input {
    flex: 1;
    min-width: 0;
  }
  .list ul {
    list-style: none;
    margin: 0;
    padding: 0 var(--s2);
    overflow-y: auto;
    flex: 1;
  }
  .list li {
    position: relative;
  }
  .item {
    width: 100%;
    display: grid;
    text-align: left;
    padding: 7px 10px;
    border: none;
    border-radius: var(--r-sm);
    background: none;
  }
  .item.current {
    background: var(--accent-soft);
  }
  .item .title {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 18px;
  }
  .item .sub {
    font-size: 11px;
    color: var(--muted);
  }
  .del {
    position: absolute;
    right: 6px;
    top: 9px;
    border: none;
    background: none;
    color: var(--muted);
    opacity: 0;
  }
  li:hover .del,
  .del:focus-visible {
    opacity: 1;
  }
  .hint {
    color: var(--muted);
    font-size: 11px;
    padding: 0 var(--s3);
  }
  .conversation {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  .conversation.dragging {
    box-shadow: inset 0 0 0 2px var(--accent);
    background: var(--accent-soft);
  }
  .excerpt {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .scroll {
    flex: 1;
    overflow-y: auto;
    padding: var(--s5) var(--s5) var(--s3);
  }
  .messages {
    display: grid;
    gap: var(--s5);
    max-width: 800px;
    margin: 0 auto;
  }
  .welcome {
    max-width: 420px;
    margin: 12vh auto 0;
    text-align: center;
  }
  .welcome .mark {
    width: 56px;
    height: 56px;
    border-radius: 16px;
    display: grid;
    place-items: center;
    margin: 0 auto var(--s3);
    color: #fff;
    background: linear-gradient(160deg, var(--accent), var(--generating-fill));
  }
  .welcome h2 {
    font: 650 18px/1.3 var(--display);
    margin: 0 0 6px;
  }
  .welcome p {
    color: var(--muted);
    margin: 0;
  }
  .ask {
    display: flex;
    align-items: center;
    gap: var(--s2);
    margin: 0 auto var(--s2);
    padding: 6px 8px 6px 14px;
    border-radius: 999px;
    background: var(--surface);
    box-shadow: var(--shadow);
  }
  .ask span {
    margin-right: var(--s2);
  }
  .composer {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: var(--s2);
    width: min(800px, calc(100% - 2 * var(--s5)));
    margin: 0 auto var(--s4);
    padding: 6px 6px 6px 14px;
    border-radius: 20px;
    background: var(--surface);
    border: 0.5px solid var(--separator);
    box-shadow: var(--shadow);
  }
  .composer textarea {
    flex: 1;
    min-width: 0;
    border: none;
    background: none;
    resize: none;
    padding: 6px 0;
    font-size: 14px;
    max-height: 200px;
    field-sizing: content;
    outline: none;
  }
  .icon {
    width: 30px;
    height: 30px;
    padding: 0;
    justify-content: center;
    color: var(--muted);
  }
  .icon.on {
    color: var(--accent-ink);
  }
  .tools-anchor {
    position: relative;
    display: inline-flex;
  }
  .badge {
    position: absolute;
    top: 1px;
    right: 0;
    min-width: 14px;
    height: 14px;
    padding: 0 3px;
    border-radius: 7px;
    background: var(--accent);
    color: #fff;
    font: 600 9px/14px var(--font);
    font-style: normal;
    text-align: center;
  }
  .send {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    border: none;
    display: grid;
    place-items: center;
    background: var(--accent);
    color: #fff;
  }
  .send:disabled {
    background: var(--fill-2);
  }
  .send.stop {
    background: var(--text);
    color: var(--surface);
  }
</style>
