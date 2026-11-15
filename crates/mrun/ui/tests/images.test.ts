import { describe, expect, it } from "vitest";
import { blocks, content, imageTokens, readAttachment } from "../src/lib/attach";

describe("images in the chat (009)", () => {
  it("counts tokens as the server's resize_v1 does", () => {
    expect(imageTokens(1568, 992)).toBe(31 * 49);
    expect(imageTokens(4000, 3000)).toBeLessThanOrEqual(2560);
    expect(imageTokens(10, 10)).toBeGreaterThanOrEqual(64);
  });

  it("sends images as image blocks ahead of the text, and files stay text", () => {
    const image = { name: "shot.png", text: "", bytes: 3, tokens: 64, estimated: false, image: { media: "image/png", data: "AAA=" } };
    const file = { name: "a.rs", text: "fn a() {}", bytes: 9, tokens: 3, estimated: false };
    const m = { text: "what's wrong?", attachments: [image, file] };
    const b = blocks(m) as Record<string, any>[];
    expect(b[0]).toEqual({ type: "image", source: { type: "base64", media_type: "image/png", data: "AAA=" } });
    expect(b[1]!.type).toBe("text");
    expect(b[1]!.text).toContain("fn a() {}");
    expect(content(m)).not.toContain("AAA=");
    expect(blocks({ text: "hi" })).toBe("hi");
  });

  it("refuses images when the model reads none", async () => {
    const file = new File([new Uint8Array([137, 80, 78, 71])], "x.png", { type: "image/png" });
    const read = await readAttachment(file, false);
    expect(read.ok).toBe(false);
    if (!read.ok) expect(read.reason).toBe("The served model reads no images.");
    const bmp = new File([new Uint8Array([1])], "x.bmp", { type: "image/bmp" });
    const r2 = await readAttachment(bmp, true);
    expect(r2.ok).toBe(false);
  });
});
