import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Flat, stable names: the binary embeds dist/ as it is, and the UI host serves a file by its name
// or nothing (no directory walk to defend).
export default defineConfig({
  plugins: [svelte()],
  base: "/",
  build: {
    outDir: "dist",
    assetsDir: "",
    emptyOutDir: true,
    sourcemap: false,
    cssCodeSplit: false,
    // WebKit of macOS 15 and later (the window), and current browsers (`mrun ui --browser`).
    cssTarget: ["safari17", "chrome120", "firefox128"],
    target: ["safari17", "chrome120", "firefox128"],
    rollupOptions: {
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "[name].js",
        assetFileNames: "[name][extname]",
      },
    },
  },
  server: {
    // `npm run dev`: the host's routes come from a running `mrun ui --browser`.
    proxy: {
      "/app": "http://127.0.0.1:8485",
      "/v1": "http://127.0.0.1:8485",
      "/mrun": "http://127.0.0.1:8485",
      "/health": "http://127.0.0.1:8485",
    },
  },
});
