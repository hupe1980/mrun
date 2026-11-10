// Context length per model (027): what a context costs, as `mrun ls --context` computes it, and
// the action target that serves it (`model|tokens`, `serve --context`).
import type { ModelEntry } from "./types";

/** Choices up to what fits: 16K, 32K, 64K, … (`0` is Automatic, the most that fits). */
export function contextChoices(maxContext: number | null): number[] {
  const out: number[] = [];
  for (let t = 16 * 1024; maxContext && t <= maxContext; t *= 2) out.push(t);
  return out;
}

/** Bytes `tokens` of context need for this model: `fixed + tokens × per_token`. */
export function memoryAt(m: Pick<ModelEntry, "memory">, tokens: number): number | null {
  return m.memory ? m.memory.fixed + tokens * m.memory.per_token : null;
}

/** The `use`/`start` target: the model, with `|tokens` when a context is chosen. */
export function contextTarget(name: string, tokens: number | null): string {
  return tokens ? `${name}|${tokens}` : name;
}

/** The model of a target that may carry a context (`last_model`). */
export function targetModel(target: string): string {
  return target.split("|")[0]!;
}
