import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Components mounted in a DOM stand-in; nothing here ships.
export default defineConfig({
  plugins: [svelte()],
  resolve: { conditions: ["browser"] },
  test: {
    environment: "happy-dom",
    include: ["tests/**/*.test.ts"],
  },
});
