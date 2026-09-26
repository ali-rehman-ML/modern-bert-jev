import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Hugging Face serves a static Space from the root of its own subdomain, but
// relative asset paths survive either way.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { target: "es2022" },
});
