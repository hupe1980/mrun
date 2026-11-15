import { describe as group, expect, it } from "vitest";
import { cageWords, describe } from "../src/lib/tools.svelte";
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
