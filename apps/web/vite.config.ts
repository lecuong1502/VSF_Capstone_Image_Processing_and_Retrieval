import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API and thumbnails to `python -m pipeline.review serve`.
const REVIEW_SERVER = "http://127.0.0.1:8765";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": REVIEW_SERVER, "/thumbs": REVIEW_SERVER },
  },
});
