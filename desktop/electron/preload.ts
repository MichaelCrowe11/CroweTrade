import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron"

/**
 * The renderer runs sandboxed with context isolation on, so it reaches the main
 * process only through what is explicitly exposed here.
 *
 * The browser slice controls WebContentsViews that live entirely in the main
 * process: the renderer names a panel id and a rectangle, and the main process
 * decides whether a view exists, what it may load, and where popups go. No
 * WebContents object ever crosses this bridge.
 */

interface BrowserViewState {
  id: string
  url: string
  canGoBack: boolean
  canGoForward: boolean
  loading: boolean
}

contextBridge.exposeInMainWorld("crowetrade", {
  platform: process.platform,
  /** 1-minute OHLCV for a pool, [ts, o, h, l, c, v] ascending. [] on failure. */
  candles: (pool: string): Promise<number[][]> => ipcRenderer.invoke("candles", pool),
  setTheme: (theme: string): Promise<void> => ipcRenderer.invoke("theme:set", theme),
  engineGates: (mints: string[], detail?: string): Promise<Record<string, unknown>> =>
    ipcRenderer.invoke("engine-gates", mints, detail),
  /** Ask the Analyst. Resolves with the whole answer; deltas stream as events. */
  ask: (question: string): Promise<{ text: string; consulted: string[] }> =>
    ipcRenderer.invoke("ask", question),
  /** Live text of the in-flight answer, one fragment per call. */
  onAskDelta: (cb: (delta: string) => void): (() => void) => {
    const handler = (_e: IpcRendererEvent, delta: string) => cb(delta)
    ipcRenderer.on("analyst:delta", handler)
    return () => ipcRenderer.removeListener("analyst:delta", handler)
  },
  /** Engine reads as the Analyst makes them, before the answer lands. */
  onAskTool: (cb: (name: string) => void): (() => void) => {
    const handler = (_e: IpcRendererEvent, name: string) => cb(name)
    ipcRenderer.on("analyst:tool", handler)
    return () => ipcRenderer.removeListener("analyst:tool", handler)
  },
  onAskReasoning: (cb: (text: string) => void): (() => void) => {
    const handler = (_e: IpcRendererEvent, text: string): void => cb(text)
    ipcRenderer.on("analyst:reasoning", handler)
    return () => ipcRenderer.removeListener("analyst:reasoning", handler)
  },

  orchestrator: {
    /** Run the agent loop toward a goal. Resolves with its final prose. */
    ask: (goal: string): Promise<{ text: string }> => ipcRenderer.invoke("orch:ask", goal),
    /** Kill the loop and whatever command is running. */
    stop: (): Promise<void> => ipcRenderer.invoke("orch:stop"),
    /** Live events: assistant text, tool calls, terminal output, panel actions. */
    onEvent: (cb: (e: Record<string, unknown>) => void): (() => void) => {
      const handler = (_e: IpcRendererEvent, evt: Record<string, unknown>) => cb(evt)
      ipcRenderer.on("orch:event", handler)
      return () => ipcRenderer.removeListener("orch:event", handler)
    },
  },

  workflows: {
    list: (): Promise<unknown[]> => ipcRenderer.invoke("wf:list"),
    delete: (id: string): Promise<void> => ipcRenderer.invoke("wf:delete", id),
    /** Replays the steps visibly; resolves with the run result line. */
    run: (id: string): Promise<string> => ipcRenderer.invoke("wf:run", id),
  },

  browser: {
    ensure: (id: string, url: string): Promise<boolean> =>
      ipcRenderer.invoke("browser:ensure", id, url),
    setBounds: (
      id: string,
      bounds: { x: number; y: number; width: number; height: number },
    ): Promise<void> => ipcRenderer.invoke("browser:bounds", id, bounds),
    navigate: (id: string, url: string): Promise<void> =>
      ipcRenderer.invoke("browser:navigate", id, url),
    back: (id: string): Promise<void> => ipcRenderer.invoke("browser:back", id),
    forward: (id: string): Promise<void> => ipcRenderer.invoke("browser:forward", id),
    reload: (id: string): Promise<void> => ipcRenderer.invoke("browser:reload", id),
    dispose: (id: string): Promise<void> => ipcRenderer.invoke("browser:dispose", id),
    onState: (cb: (state: BrowserViewState) => void): (() => void) => {
      const handler = (_e: IpcRendererEvent, state: BrowserViewState) => cb(state)
      ipcRenderer.on("browser:state", handler)
      return () => ipcRenderer.removeListener("browser:state", handler)
    },
  },
})

export {}
