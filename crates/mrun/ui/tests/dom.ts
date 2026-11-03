// Shared by the component tests: mount, settle, and check the rendered page with axe (the WCAG
// rules a DOM stand-in can judge: names, roles, ARIA, labels; not contrast or layout).

import axe from "axe-core";
import { flushSync, mount, unmount, type Component } from "svelte";

export function render<P extends Record<string, any>>(component: Component<P>, props: P) {
  document.body.innerHTML = "";
  const target = document.createElement("main");
  document.body.appendChild(target);
  const instance = mount(component, { target, props });
  flushSync();
  return { target, destroy: () => unmount(instance) };
}

export const settle = async (n = 6) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
  flushSync();
};

export async function violations(root: Element = document.body): Promise<string[]> {
  const { vi } = await import("vitest");
  vi.useRealTimers();
  const r = await axe.run(root as HTMLElement, {
    rules: {
      "color-contrast": { enabled: false },
      region: { enabled: false },
      "landmark-one-main": { enabled: false },
      "page-has-heading-one": { enabled: false },
    },
    resultTypes: ["violations"],
  });
  return r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
}
