# Risk Register

Risk IDs are stable. This register names failure modes and mitigations, not another task backlog.
Discharge requires evidence; design text does not close a risk. Work sequencing belongs to the
[roadmap entry point](ROADMAP.md) and owning feature plan/tasks.

| ID | Risk and impact | Mitigation/evidence boundary | Status |
| --- | --- | --- | --- |
| R001 | Nonpreemptible Metal calls miss interactive cancellation | Bounded synchronized chunks and measured safe release | Open: native trials unrun |
| R002 | Sharing mutates another branch or restores stale state | Exact identities/tokens, immutable membership and fault cases | Open: native state proof unrun |
| R003 | Native scratch escapes accounting; cell eviction does not free physical buffers | Unique ledger, preallocation gate, allocator peaks and OS/device observations | Open: peaks/cleanup unrun |
| R004 | Template, grammar and actual client schemas disagree | Pinned client capture, native rendering and independent final validation | Open: interoperability unrun |
| R005 | Batch/precision changes harm numerics or coding quality | Exact reference, preregistered tolerances and paired task outcomes | Open: quality lanes unrun |
| R006 | A fast small model fails useful coding tasks | Objective tasks and quality-matched competitors; no size-only admission | Open: coding quality unrun |
| R007 | GPU faults leave unsafe state or unobservable cleanup | Invalidate uncertain contexts and qualify truthful unavailable/restart behavior | Open: fault behavior unrun |
| R008 | Cache-off wins are presented as product leadership | Optimized comparison matrix retaining losing cells | Open: comparisons unrun |
| R009 | Ignored planning disappears on clone/workstation loss | Explicit local backup/provisioning; no public/build dependency | Ongoing operational risk |

R001..R008 identify what source inspection cannot prove, not demonstrated defects in the candidate.
R009 is a deliberate consequence of local-only planning. Close a row only when the adverse
scenario is covered by identified evidence and its limits are stated.