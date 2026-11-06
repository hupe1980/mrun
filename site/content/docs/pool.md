+++
title = "Mac pool"
description = "Opt-in: several Macs on one network, colleagues' included, run one model none of them fits alone."
weight = 66
+++

A pool joins Macs on one network so they run a model none of them holds alone: four 36–48 GB
MacBooks can run a 125 GB model. Every Mac runs part of the model's layers. Everyone in the pool
uses it: each Mac's agents keep talking to their own `http://127.0.0.1:8484`, which forwards to the
Mac that hosts the pool.

<figure class="screenshot"><a href="../../screenshots/pool.png" title="Open full size"><img src="../../screenshots/pool.png" alt="Pool: four MacBooks on a ring, each with its layers and link; Dana's Wi-Fi sets the pace" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Four MacBooks serving one model; each tile shows its layers and link</figcaption></figure>

## What it does and does not do

- **Bigger models, not faster tokens.** Each token passes every Mac in turn, so a pool decodes at
  best as fast as one Mac with enough memory, minus the network's delay per hop.
- **The network decides.** Thunderbolt and wired Ethernet keep a hop well under a millisecond;
  Wi-Fi costs milliseconds per hop. The plan shows the speed for your links and which one sets
  the pace.
- **Exact.** A pool's replies are those of the same model on one Mac with the same GPUs, and
  follow-ups reuse their prefix as on one Mac. Macs with different GPUs may differ in the last
  digits; the plan says so.

## Start, join, serve

```sh
mrun pool start                 # on the Mac that hosts: announce a pool on this network
mrun pool start --unencrypted   # the same, without encrypting the model's traffic (see Privacy)
mrun pool join "Ben's pool"     # on another Mac: shows a code to compare
mrun pool admit                 # on the host: admit the Mac whose code matches
mrun pool plan qwen3.8-235b     # layers, memory, context and speed per Mac
mrun pool serve qwen3.8-235b    # every Mac loads its layers
mrun pool leave                 # a member leaves; on the host, ends the pool
mrun pool rejoin                # a member back after a loss or a restart, without a code
mrun pool replan                # on the host: serve again on the Macs answering now
```

In the window: **Pool** (<kbd>⌘4</kbd>). Joining works like pairing AirDrop: both Macs show the
same six digits, the host admits, the joining Mac confirms. A Mac in between would show other
digits. Models marks a model too big for this Mac that the pool's Macs together hold: **Fits the
pool**, which opens the plan.

A member that lacks the model downloads only the files of its own layers when the pool starts;
the plan says how much. They stay in its store as a part: `mrun ls` shows `layers 25–48 · for
pools`, `mrun rm` removes it, and `mrun pull` completes it into the whole model with the files it
has. A part serves only in a pool.

Every Mac needs the same mrun version; joining refuses another one (E063). Pools are tested across
processes on one Mac; [hardware reports](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports)
from real pools are welcome.

## Privacy

- Nothing listens beyond loopback until you start or join a pool.
- Joining needs both owners. Each membership has its own key in the Keychain; traffic between the
  Macs is authenticated and encrypted.
- A host may start its pool **unencrypted** (`--unencrypted`, or "Encrypt the model's traffic"
  off) for a cable or network it trusts: the numbers between layers then cross readable but
  authenticated; prompts, replies, joining and control stay encrypted. Members see this before
  they join, and every privacy line says "not encrypted".
- The host reads every member's prompts as text, where the model's endpoint is. The other Macs
  see only numbers between layers. Each Mac's window says so, and a member's request text (in
  Sessions) is shown only on that member's Mac.
- A Mac lending its memory stays awake while the pool serves; Leave is one click.

## When a Mac leaves

A Mac that leaves, is removed or stops answering (lid closed, cable pulled; shown within 5
seconds) ends the requests it was part of, with error E060 naming it. If the other Macs still hold
the model, the pool serves it again on them by itself, with the same replies; agents need no
restart. If they cannot, the pool waits for that Mac and offers the largest pulled model that fits
the rest. A Mac that comes back joins with **Rejoin**, no new code, and **Re-plan** on the host
takes it back in. When the host ends the pool, each member gets its own model back.
