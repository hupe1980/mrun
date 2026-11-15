import { describe as group, expect, it } from "vitest";
import { browserReach, cageWords, describe, serviceHost } from "../src/lib/tools.svelte";
import type { McpServer, ToolCall } from "../src/lib/types";

const browser = { name: "browser", kind: "stdio", command: "mrun mcp serve browser", env: [], sandbox: false } as unknown as McpServer;
const call = (input: Record<string, unknown>) => ({ id: "t", name: "browser__x", input }) as unknown as ToolCall;

group("mrun's browser tools (pageon)", () => {
  it("reads each call as a sentence and names the site it reaches", () => {
    expect(describe(call({ url: "https://shop.example/cart" }), browser, "browser_open")).toEqual({ verb: "Open", subject: "https://shop.example/cart", reaches: "shop.example" });
    expect(describe(call({ ref: "e12" }), browser, "browser_click")).toMatchObject({ verb: "Click", subject: "e12" });
    expect(describe(call({ ref: "e5", text: "ann@example.com", submit: true }), browser, "browser_type")).toMatchObject({ verb: "Type and submit", subject: "into e5", body: { kind: "code", text: "ann@example.com" } });
    expect(describe(call({ action: "new", url: "https://a.example/" }), browser, "browser_tabs")).toMatchObject({ verb: "Open a tab", reaches: "a.example" });
    expect(describe(call({}), browser, "browser_reset")).toMatchObject({ verb: "Rewind", subject: "one step" });
  });

  it("says the pages run sandboxed", () => {
    expect(cageWords(browser)).toBe("Uses websites · pages sandboxed");
  });
});

group("mrun's search tool (029)", () => {
  const search = (command: string) => ({ name: "search", kind: "stdio", command, env: [], sandbox: false }) as unknown as McpServer;
  it("names the query and the service that receives it", () => {
    const exa = search("mrun mcp serve search --provider exa");
    expect(describe(call({ query: "tokio 1.48" }), exa, "search")).toEqual({ verb: "Search", subject: "“tokio 1.48”", reaches: "mcp.exa.ai" });
    expect(cageWords(exa)).toBe("Searches the web · mcp.exa.ai");
    expect(serviceHost(search("mrun mcp serve search --provider brave"))).toBe("api.search.brave.com");
    expect(serviceHost(search("mrun mcp serve search --provider searxng --url http://127.0.0.1:8888"))).toBe("127.0.0.1:8888");
    expect(serviceHost(browser)).toBeUndefined();
  });
});

group("the browser's reach", () => {
  it("says which sites the browser may use", () => {
    expect(browserReach("mrun mcp serve browser")).toBe("Uses websites · pages sandboxed");
    expect(browserReach("mrun mcp serve browser --allow-local")).toBe("Uses any site and local servers · sandboxed");
    expect(browserReach("mrun mcp serve browser --origin https://github.com --origin http://localhost:3000 --allow-local")).toBe(
      "Uses only github.com, localhost:3000 · sandboxed",
    );
  });
});
