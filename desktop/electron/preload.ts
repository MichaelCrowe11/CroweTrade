import { contextBridge, ipcRenderer } from "electron"

/**
 * The renderer runs sandboxed with context isolation on, so it reaches the main
 * process only through what is explicitly exposed here.
 *
 * Nothing is exposed yet. When the app moves off public HTTP feeds and onto our
 * own pipeline, the feed subscription belongs here rather than in the renderer:
 * a websocket held in the renderer dies on reload and takes its backpressure
 * state with it, and any signing key must never be reachable from page context.
 */
contextBridge.exposeInMainWorld("crowetrade", {
  platform: process.platform,
  /** 1-minute OHLCV for a pool, [ts, o, h, l, c, v] ascending. [] on failure. */
  candles: (pool: string): Promise<number[][]> => ipcRenderer.invoke("candles", pool),
  /** Ask the Analyst. Returns the answer plus which engine endpoints it read. */
  ask: (question: string): Promise<{ text: string; consulted: string[] }> =>
    ipcRenderer.invoke("ask", question),
})

export {}
