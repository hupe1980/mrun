// "Your code" (020 FR-007): the three APIs in curl, Python and TypeScript, filled from the live
// port and model. mrun listens on loopback only and ignores API keys; the SDKs just need one.
// `tests/ui_snippets.rs` checks that every path below is a route the server serves (SC-005).

export type Api = "messages" | "chat" | "responses";
export type Lang = "curl" | "python" | "typescript";

export const apis: { id: Api; title: string; path: string }[] = [
  { id: "messages", title: "Anthropic Messages", path: "/v1/messages" },
  { id: "chat", title: "OpenAI Chat Completions", path: "/v1/chat/completions" },
  { id: "responses", title: "OpenAI Responses", path: "/v1/responses" },
];

export const langs: { id: Lang; title: string }[] = [
  { id: "curl", title: "curl" },
  { id: "python", title: "Python" },
  { id: "typescript", title: "TypeScript" },
];

export function snippet(api: Api, lang: Lang, port: number, model: string): string {
  const base = `http://127.0.0.1:${port}`;
  const m = JSON.stringify(model);
  // No apostrophes: the curl examples quote their JSON in single quotes.
  const ask = "Explain the build of this repository in one paragraph.";
  switch (`${api}/${lang}`) {
    case "messages/curl":
      return `curl ${base}/v1/messages \\
  -H 'content-type: application/json' \\
  -d '{"model": ${m}, "max_tokens": 1024,
       "messages": [{"role": "user", "content": "${ask}"}]}'`;
    case "chat/curl":
      return `curl ${base}/v1/chat/completions \\
  -H 'content-type: application/json' \\
  -d '{"model": ${m},
       "messages": [{"role": "user", "content": "${ask}"}]}'`;
    case "responses/curl":
      return `curl ${base}/v1/responses \\
  -H 'content-type: application/json' \\
  -d '{"model": ${m}, "input": "${ask}"}'`;
    case "messages/python":
      return `# pip install anthropic
import anthropic

client = anthropic.Anthropic(base_url="${base}", api_key="mrun")
reply = client.messages.create(
    model=${m},
    max_tokens=1024,
    messages=[{"role": "user", "content": "${ask}"}],
)
print(reply.content[-1].text)`;
    case "chat/python":
      return `# pip install openai
from openai import OpenAI

client = OpenAI(base_url="${base}/v1", api_key="mrun")
reply = client.chat.completions.create(
    model=${m},
    messages=[{"role": "user", "content": "${ask}"}],
)
print(reply.choices[0].message.content)`;
    case "responses/python":
      return `# pip install openai
from openai import OpenAI

client = OpenAI(base_url="${base}/v1", api_key="mrun")
reply = client.responses.create(model=${m}, input="${ask}")
print(reply.output_text)`;
    case "messages/typescript":
      return `// npm install @anthropic-ai/sdk
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({ baseURL: "${base}", apiKey: "mrun" });
const reply = await client.messages.create({
  model: ${m},
  max_tokens: 1024,
  messages: [{ role: "user", content: "${ask}" }],
});
console.log(reply.content.at(-1));`;
    case "chat/typescript":
      return `// npm install openai
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${base}/v1", apiKey: "mrun" });
const reply = await client.chat.completions.create({
  model: ${m},
  messages: [{ role: "user", content: "${ask}" }],
});
console.log(reply.choices[0].message.content);`;
    default:
      return `// npm install openai
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${base}/v1", apiKey: "mrun" });
const reply = await client.responses.create({ model: ${m}, input: "${ask}" });
console.log(reply.output_text);`;
  }
}
