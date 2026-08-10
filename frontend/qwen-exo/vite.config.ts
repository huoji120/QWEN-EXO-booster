import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/qwen-exo/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "../../python/qwen_exo_booster/static/app",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    proxy: {
      "^/qwen-exo/(status|health|telemetry|recall-trace|knowledge|policydata|tensor-bank|service-config)":
        "http://127.0.0.1:30000",
      "/v1": "http://127.0.0.1:30000",
    },
  },
});
