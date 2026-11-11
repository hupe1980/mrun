// `npm run shots` (019 FR-018): every place, light and dark, against fixed stub data, written to
// shots/. No model, no server, no network beyond loopback: the built page is served with stub
// host routes, and a locally installed Chrome renders it (puppeteer-core never downloads one).

import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { createServer } from "node:http";
import { execFileSync } from "node:child_process";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import * as data from "./fixtures.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const dist = join(root, "dist");
const out = join(root, "shots");
const token = "0".repeat(64);

function chrome() {
  const candidates = [
    process.env.CHROME,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ];
  for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]) {
    try {
      candidates.push(execFileSync("which", [name]).toString().trim());
    } catch {
      /* not installed */
    }
  }
  const found = candidates.find((c) => c && existsSync(c));
  if (!found) throw new Error("No Chrome or Chromium found; set CHROME to its path.");
  return found;
}

const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };

function json(res, body) {
  res.writeHead(200, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

// The second half of the run shows a rented GPU serving (024).
let remote = false;
// Session replay (025): the 140-request session, none, or 2,000 requests (SC-002 timing).
let replay = "session";
// Mac pool (026): which pool state the stub serves.
let poolMode = "empty";

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");
  const path = url.pathname;
  if (path === "/app/state") return json(res, data.appState);
  if (path === "/mrun/status") {
    if (remote) return json(res, data.podStatus);
    // The pool serving (026): Now shows it compactly.
    if (poolMode === "serving") return json(res, { ...data.status, pool: { ...data.poolView("serving").pool, queue: { running: null, waiting: [{ member: "anna000000000000", waiting: 1 }] } } });
    return json(res, data.status);
  }
  if (path === "/mrun/sessions") {
    const all = replay === "none" ? { kept: 2000, dropped: 0, remote: false, now_ms: 0, sessions: [], requests: [] } : data.sessions(replay === "big" ? 1940 : 140);
    const since = Number(url.searchParams.get("since") ?? -1);
    return json(res, { ...all, requests: all.requests.filter((r) => r.id > since) });
  }
  if (path.endsWith("/excerpt")) return json(res, data.excerpt);
  if (path === "/mrun/pool") return json(res, data.poolView(poolMode));
  if (path === "/mrun/pool/plan") return json(res, data.poolPlan);
  if (path === "/app/cloud") return json(res, data.cloud);
  if (path === "/app/actions" && req.method === "POST") return json(res, { job: data.cloudJob });
  if (path.startsWith("/app/jobs/")) return json(res, data.cloudJob);
  if (path === "/app/command") return json(res, { argv: [], text: "" });
  if (path === "/health") return json(res, { status: "ok" });
  if (path === "/app/models") return json(res, data.models);
  if (path === "/app/agents") return json(res, data.agents);
  if (path === "/app/chats") return json(res, []);
  if (path === "/app/config") return json(res, data.configView);
  if (path.startsWith("/v1/models/")) return json(res, data.modelInfo);
  if (path === "/app/plan") return json(res, data.plan);
  if (path === "/app/apps") return json(res, data.apps);
  if (path === "/app/mcp") return json(res, data.mcpList);
  if (path === "/app/mcp/call") return json(res, data.testRun);
  if (path === "/v1/messages/count_tokens") {
    let body = "";
    for await (const chunk of req) body += chunk;
    const counted = JSON.parse(body);
    return json(res, { input_tokens: Math.ceil(JSON.stringify([counted.messages, counted.tools ?? []]).length / 3.6) });
  }
  if (path === "/v1/messages") {
    let body = "";
    for await (const chunk of req) body += chunk;
    const last = JSON.parse(body).messages.at(-1)?.content ?? "";
    res.writeHead(200, { "content-type": "text/event-stream" });
    const frames = typeof last === "string" && last.includes("tests") ? data.toolUse() : last.includes("think") ? data.thinking() : data.reply();
    // Paced like a real stream, so times and rates read as they would.
    for (const f of frames) {
      res.write(f);
      await new Promise((r) => setTimeout(r, 24));
    }
    if (!last.includes("think")) res.end();
    return;
  }
  if (path.startsWith("/app/")) return json(res, {});
  const file = normalize(join(dist, path === "/" ? "index.html" : path));
  if (!file.startsWith(dist) || !existsSync(file)) {
    res.writeHead(404);
    return res.end();
  }
  res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream" });
  res.end(readFileSync(file));
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}`;
mkdirSync(out, { recursive: true });

const browser = await puppeteer.launch({ executablePath: chrome(), headless: true, args: ["--no-proxy-server"] });
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const written = [];
try {
  for (const scheme of ["light", "dark"]) {
    const page = await browser.newPage();
    await page.setViewport({ width: 1100, height: 720, deviceScaleFactor: 2 });
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: scheme }]);
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${base}/#t=${token}`, { waitUntil: "networkidle0" });
    const shot = async (name) => {
      await pause(400);
      const file = join(out, `${scheme}-${name}.png`);
      await page.screenshot({ path: file });
      written.push(file);
    };
    const go = (title) =>
      page.evaluate((t) => [...document.querySelectorAll("nav[aria-label=Places] button")].find((b) => b.textContent.trim().startsWith(t))?.click(), title);
    const ask = async (text) => {
      await page.type("textarea[aria-label=Message]", text);
      await page.keyboard.press("Enter");
    };

    await shot("1-now");
    const attach = join(mkdtempSync(join(tmpdir(), "mrun-shots-")), "parser.rs");
    writeFileSync(attach, data.sourceFile);
    await go("Chat");
    await ask("Write a haiku about caches, then explain it.");
    await page.waitForSelector(".assistant .stats", { timeout: 15000 });
    await page.hover(".assistant");
    await shot("2-chat");
    await ask("Now think about why the prefix matters");
    await pause(1500);
    await shot("3-chat-thinking");
    await page.click("button[aria-label='Stop (Esc)']");
    // Tools (028): the picker, then a call waiting for Allow.
    await page.keyboard.down("Meta");
    await page.keyboard.press("n");
    await page.keyboard.up("Meta");
    await page.click("button[aria-label^=Tools]");
    await page.waitForSelector("[aria-label='Tools for this chat'] input[role=switch]", { timeout: 5000 });
    await page.click("[aria-label='Tools for this chat'] input[role=switch]");
    await shot("4b-chat-tools");
    await page.keyboard.press("Escape");
    await ask("Why do the parser tests fail?");
    await page.waitForSelector(".card.asking", { timeout: 10000 });
    await shot("4c-chat-tool-call");
    await page.keyboard.press("Enter");
    await page.waitForSelector(".card.failed", { timeout: 10000 });
    await (await page.$("input[type=file]")).uploadFile(attach);
    await page.waitForSelector(".attachments .chip", { timeout: 5000 });
    await page.type("textarea[aria-label=Message]", "Why does this panic on empty input?");
    await shot("4-chat-file");
    await go("Models");
    await shot("5-models");
    await page.type("input[aria-label='Model to add']", "https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit");
    await page.keyboard.press("Enter");
    await page.waitForSelector(".plan", { timeout: 5000 });
    await page.evaluate(() => document.querySelector(".add")?.scrollIntoView({ block: "end" }));
    await shot("6-models-add");
    await page.evaluate(() => document.querySelector(".cloud")?.scrollIntoView({ block: "start" }));
    await shot("6b-models-cloud");
    await page.evaluate(() => [...document.querySelectorAll(".cloud button")].find((b) => b.textContent.trim() === "Run…")?.click());
    await page.waitForSelector(".gpus", { timeout: 5000 });
    await shot("6c-cloud-sheet");
    await page.evaluate(() => document.querySelector(".sheet .footer .primary")?.click());
    await page.waitForSelector(".stepper", { timeout: 5000 });
    await shot("6d-cloud-starting");
    await page.evaluate(() => [...document.querySelectorAll(".sheet button")].find((b) => b.textContent.trim() === "Hide")?.click());
    await go("Agents");
    await shot("7-agents");
    await page.evaluate(() => document.querySelector(".code")?.scrollIntoView({ block: "start" }));
    await shot("8-agents-code");
    await page.keyboard.down("Meta");
    await page.keyboard.press(",");
    await page.keyboard.up("Meta");
    await shot("9-settings");
    await page.evaluate(() => [...document.querySelectorAll("[role=tab]")].find((t) => t.textContent?.includes("Server"))?.click());
    await shot("9-settings-server");
    await page.evaluate(() => [...document.querySelectorAll("[role=tab]")].find((t) => t.textContent?.includes("Tools"))?.click());
    await shot("9-settings-tools");
    await page.keyboard.press("Escape");
    await page.keyboard.down("Meta");
    await page.keyboard.press("k");
    await page.keyboard.up("Meta");
    await shot("10-palette");
    await page.keyboard.press("Escape");
    await page.evaluate(() => document.querySelector("button[aria-label='Close settings']")?.click());
    await go("Sessions");
    await page.waitForSelector("[role=option]", { timeout: 5000 });
    await shot("13-sessions-follow");
    await page.evaluate(() => {
      const option = [...document.querySelectorAll("[role=option]")].find((o) => o.getAttribute("aria-label")?.includes("Worth fixing"));
      option?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      option?.scrollIntoView({ inline: "center", block: "nearest" });
    });
    await page.waitForSelector("aside ins", { timeout: 5000 });
    await shot("14-sessions");
    replay = "none";
    await go("Now");
    await go("Sessions");
    await pause(1200);
    await shot("15-sessions-empty");
    if (scheme === "light") {
      // SC-002: 2,000 requests render and scroll without stutter.
      replay = "big";
      await go("Now");
      const started = Date.now();
      await go("Sessions");
      await page.waitForFunction(() => document.querySelectorAll("[role=option]").length >= 2000, { timeout: 10000 });
      const shown = Date.now() - started;
      const frames = await page.evaluate(async () => {
        const sky = document.querySelector(".skyline");
        const times = [];
        let last = performance.now();
        for (let i = 0; i < 60; i++) {
          sky.scrollLeft = (sky.scrollWidth * i) / 60;
          await new Promise((r) => requestAnimationFrame(r));
          const now = performance.now();
          times.push(now - last);
          last = now;
        }
        return times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)];
      });
      console.log(`2,000 requests: shown in ${shown} ms, 95th percentile frame while scrolling ${frames.toFixed(1)} ms`);
    }
    replay = "session";
    // Mac pool (026).
    poolMode = "empty";
    await go("Pool");
    await pause(1200);
    await shot("16-pool-empty");
    poolMode = "join";
    await pause(1800);
    await shot("17-pool-join");
    await page.keyboard.press("Escape");
    poolMode = "open";
    await pause(1700);
    await page.select("select[aria-label=Model]", "qwen3.6");
    await page.evaluate(() => [...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "Plan")?.click());
    await page.waitForSelector("[aria-labelledby=plan-title]", { timeout: 5000 });
    await shot("18-pool-plan");
    await page.keyboard.press("Escape");
    poolMode = "serving";
    await pause(1700);
    await shot("19-pool-serving");
    await go("Now");
    await pause(1500);
    await shot("21-now-pool");
    await go("Pool");
    poolMode = "lost";
    await pause(1700);
    await shot("20-pool-lost");
    poolMode = "empty";
    remote = true;
    await go("Now");
    await pause(1500);
    await shot("11-now-pod");
    await page.keyboard.down("Meta");
    await page.keyboard.press(",");
    await page.keyboard.up("Meta");
    await shot("12-settings-key");
    await page.keyboard.press("Escape");
    remote = false;
    if (errors.length) throw new Error(`page errors (${scheme}): ${errors.join("; ")}`);
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}
console.log(`${written.length} screenshots in ${out}`);
