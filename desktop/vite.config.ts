import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// base must be relative: Electron loads the built renderer over file://,
// and an absolute /assets/ path resolves to the filesystem root there.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { port: 5273, strictPort: true },
  build: { outDir: "dist", emptyOutDir: true },
})
