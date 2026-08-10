import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs: the same bundle is served from the site root by
  // Firebase Hosting *and* from /static/ by the Cloud Run proxy (which is what
  // the Playwright suite exercises). An absolute base would 404 under /static/.
  base: "./",
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
