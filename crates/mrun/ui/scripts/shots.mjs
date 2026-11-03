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

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");
  const path = url.pathname;
  if (path === "/app/state") return json(res, data.appState);
  if (path === "/mrun/status") return json(res, data.status);
  if (path === "/health") return json(res, { status: "ok" });
  if (path === "/app/models") return json(res, data.models);
  if (path === "/app/agents") return json(res, data.agents);
  if (path === "/app/chats") return json(res, []);
  if (path === "/app/config") return json(res, { sources: {} });
  if (path.startsWith("/v1/models/")) return json(res, data.modelInfo);
  if (path === "/app/plan") return json(res, data.plan);
  if (path === "/app/apps") return json(res, data.apps);
  if (path === "/v1/messages/count_tokens") {
    let body = "";
    for await (const chunk of req) body += chunk;
    return json(res, { input_tokens: Math.ceil(JSON.stringify(JSON.parse(body).messages).length / 3.6) });
  }
  if (path === "/v1/messages") {
    let body = "";
    for await (const chunk of req) body += chunk;
    const last = JSON.parse(body).messages.at(-1)?.content ?? "";
    res.writeHead(200, { "content-type": "text/event-stream" });
    const frames = last.includes("think") ? data.thinking() : data.reply();
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
    await go("Agents");
    await shot("7-agents");
    await page.evaluate(() => document.querySelector(".code")?.scrollIntoView({ block: "start" }));
    await shot("8-agents-code");
    await page.keyboard.down("Meta");
    await page.keyboard.press(",");
    await page.keyboard.up("Meta");
    await shot("9-settings");
    await page.keyboard.press("Escape");
    await page.keyboard.down("Meta");
    await page.keyboard.press("k");
    await page.keyboard.up("Meta");
    await shot("10-palette");
    if (errors.length) throw new Error(`page errors (${scheme}): ${errors.join("; ")}`);
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}
console.log(`${written.length} screenshots in ${out}`);
