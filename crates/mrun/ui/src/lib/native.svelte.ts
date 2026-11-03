// The native shell (019 contract): in `mrun ui`'s window the page can ask the shell to drag or
// zoom the window and to show a native context menu. In a browser the same menu renders in the
// page, so every caller writes one `contextMenu(...)` and never checks where it runs.

export type MenuItem = { id: string; title: string; enabled?: boolean; destructive?: boolean } | "-";

declare global {
  interface Window {
    ipc?: { postMessage(message: string): void };
  }
}

export function isNative(): boolean {
  return document.documentElement.dataset.shell === "native" && typeof window.ipc?.postMessage === "function";
}

export function post(op: "drag" | "zoom"): void {
  if (isNative()) window.ipc!.postMessage(JSON.stringify({ op }));
}

/** The page-rendered menu, when there is no native shell (`ContextMenu.svelte` shows it). */
class PageMenu {
  open = $state<{ items: MenuItem[]; x: number; y: number; resolve: (id: string | null) => void } | null>(null);

  close(id: string | null): void {
    const open = this.open;
    this.open = null;
    open?.resolve(id);
  }
}

export const pageMenu = new PageMenu();

/**
 * Opens a context menu for a right-click, or for Shift-F10 / the context-menu key at the focused
 * element; resolves with the picked id, or `null` when dismissed.
 */
export function contextMenu(event: MouseEvent | KeyboardEvent, items: MenuItem[]): Promise<string | null> {
  event.preventDefault();
  event.stopPropagation();
  let x: number;
  let y: number;
  if (event instanceof MouseEvent && event.type === "contextmenu" && (event.clientX || event.clientY)) {
    ({ clientX: x, clientY: y } = event);
  } else {
    const box = (event.currentTarget as HTMLElement | null)?.getBoundingClientRect();
    x = box ? box.left + 24 : 24;
    y = box ? box.top + box.height / 2 : 24;
  }
  pageMenu.close(null);
  if (isNative()) {
    return new Promise((resolve) => {
      const pick = (e: Event) => resolve((e as CustomEvent<{ id: string | null }>).detail?.id ?? null);
      window.addEventListener("mrun:menu-pick", pick, { once: true });
      const enabled = items.map((i) => (i === "-" ? i : { ...i, enabled: i.enabled ?? true }));
      window.ipc!.postMessage(JSON.stringify({ op: "menu", items: enabled.slice(0, 32), x, y }));
    });
  }
  return new Promise((resolve) => (pageMenu.open = { items, x, y, resolve }));
}
