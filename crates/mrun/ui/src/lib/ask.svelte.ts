// Confirmations in the window's own sheet: the native window's webview has no `confirm()` (it
// returns false at once), so every question goes through here, in the browser too.

export interface Question {
  title: string;
  detail?: string;
  /** The confirming button's label, e.g. "Eject". */
  ok: string;
  /** Red, for what cannot be undone or ends something others use. */
  destructive?: boolean;
  answer: (yes: boolean) => void;
}

export const asking = $state<{ question: Question | null }>({ question: null });

/** Asks and resolves `true` on the confirming button, `false` on Cancel or Esc. */
export function ask(title: string, ok: string, options: { detail?: string; destructive?: boolean } = {}): Promise<boolean> {
  return new Promise((resolve) => {
    asking.question = {
      title,
      ok,
      ...options,
      answer: (yes) => {
        asking.question = null;
        resolve(yes);
      },
    };
  });
}
