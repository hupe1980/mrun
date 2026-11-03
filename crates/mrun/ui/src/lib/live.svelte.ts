// Live state the whole window shares: the server's status and the host's state, polled once a
// second while the window is visible (FR-005), paused while it is hidden.

import { HostError, host } from "./api";
import { serverState } from "./format";
import type { AgentEntry, AppState, ModelEntry, ServerState, Status } from "./types";

class Live {
  status = $state<Status | null>(null);
  app = $state<AppState | null>(null);
  reachable = $state(false);
  /** The host refused the page's token (an old tab): it must be reopened from mrun. */
  unauthorized = $state(false);
  models = $state<ModelEntry[] | null>(null);
  /** Kept across visits, so a place shows its last rows at once (019 US4). */
  agents = $state<AgentEntry[] | null>(null);
  #timer: ReturnType<typeof setInterval> | null = null;

  get state(): ServerState {
    const loading = this.app?.server?.running === true && !this.status?.model;
    if (this.app?.server && !this.app.server.running) return "stopped";
    return serverState(this.status, this.reachable, loading);
  }

  get remote(): boolean {
    return Boolean(this.status?.pod);
  }

  async tick(): Promise<void> {
    try {
      this.app = await host.state();
      this.unauthorized = false;
    } catch (e) {
      if (e instanceof HostError && e.status === 401) this.unauthorized = true;
      return;
    }
    if (!this.app?.server?.running) {
      this.reachable = false;
      this.status = null;
      return;
    }
    try {
      this.status = await host.status();
      this.reachable = true;
    } catch {
      this.reachable = false;
    }
  }

  async refreshModels(): Promise<void> {
    try {
      const first = await host.models();
      this.models = first.value;
      if (first.stale) this.models = (await host.models(true)).value;
    } catch {
      /* keep the last list */
    }
  }

  async refreshAgents(): Promise<void> {
    const first = await host.agents();
    this.agents = first.value;
    if (first.stale) this.agents = (await host.agents(true)).value;
  }

  start(): void {
    const run = () => void this.tick();
    run();
    void this.refreshModels();
    this.#timer = setInterval(run, 1000);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && this.#timer) {
        clearInterval(this.#timer);
        this.#timer = null;
      } else if (!document.hidden && !this.#timer) {
        run();
        this.#timer = setInterval(run, 1000);
      }
    });
  }
}

export const live = new Live();
