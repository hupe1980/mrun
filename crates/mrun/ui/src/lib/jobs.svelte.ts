// Actions the user starts (pull, use, connect, …) run as the CLI's own commands in the UI host
// (research R2); the page starts them and follows their jobs.

import { host } from "./api";
import { live } from "./live.svelte";
import type { Job } from "./types";

class Jobs {
  /** The latest job per `kind:target`. */
  byKey = $state<Record<string, Job>>({});

  key(kind: string, target: string): string {
    return `${kind}:${target}`;
  }

  of(kind: string, target: string): Job | undefined {
    return this.byKey[this.key(kind, target)];
  }

  running(kind: string, target: string): boolean {
    return this.of(kind, target)?.state === "running";
  }

  /** Starts an action and follows it until it ends; resolves with the final job. */
  async run(kind: string, target: string, confirm = false): Promise<Job> {
    let { job } = await host.act(kind, target, confirm);
    const key = this.key(kind, target);
    this.byKey[key] = job;
    while (job.state === "running") {
      await new Promise((r) => setTimeout(r, 400));
      job = await host.job(job.id);
      this.byKey[key] = job;
    }
    await live.tick();
    if (kind === "pull" || kind === "remove" || kind === "use" || kind === "start" || kind === "stop") await live.refreshModels();
    if (kind === "connect" || kind === "disconnect") await live.refreshAgents().catch(() => {});
    return job;
  }

  async cancel(kind: string, target: string): Promise<void> {
    const job = this.of(kind, target);
    if (job?.state === "running") this.byKey[this.key(kind, target)] = await host.cancel(job.id);
  }
}

export const jobs = new Jobs();
