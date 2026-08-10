/**
 * The preload bridge, as the renderer sees it.
 *
 * One declaration for the whole app: TypeScript refuses two declarations of
 * the same Window property with different shapes, so every panel that talks to
 * the main process describes its slice of the bridge here rather than inline.
 * Everything is optional because the renderer must degrade honestly when the
 * preload did not run (vite dev in a plain browser, or a broken build).
 */

interface BrowserViewState {
  id: string
  url: string
  canGoBack: boolean
  canGoForward: boolean
  loading: boolean
}

interface Window {
  crowetrade?: {
    platform?: string
    candles?: (pool: string) => Promise<number[][]>
    /** Keeps the NATIVE window background in step with the CSS theme. */
    setTheme?: (theme: string) => Promise<void>
    /** Gate snapshot fields from the engine, which resolves LP lock, holder
     *  spread and deployer history this app cannot see. Empty on failure. */
    engineGates?: (
      mints: string[],
      detail?: string,
    ) => Promise<Record<string, {
      mintAuthority?: string | null
      freezeAuthority?: string | null
      topHolderShare?: number
      deployerPriorMints?: number
      deployerPriorRugs?: number
    }>>
    ask?: (question: string) => Promise<{ text: string; consulted: string[] }>
    onAskDelta?: (cb: (delta: string) => void) => () => void
    onAskTool?: (cb: (name: string) => void) => () => void
    /** The model's working-out, streamed separately from the answer. */
    onAskReasoning?: (cb: (text: string) => void) => () => void
    orchestrator?: {
      ask: (goal: string) => Promise<{ text: string }>
      stop: () => Promise<void>
      onEvent: (cb: (e: Record<string, unknown>) => void) => () => void
    }
    workflows?: {
      list: () => Promise<unknown[]>
      delete: (id: string) => Promise<void>
      run: (id: string) => Promise<string>
    }
    browser?: {
      ensure: (id: string, url: string) => Promise<boolean>
      setBounds: (
        id: string,
        bounds: { x: number; y: number; width: number; height: number },
      ) => Promise<void>
      navigate: (id: string, url: string) => Promise<void>
      back: (id: string) => Promise<void>
      forward: (id: string) => Promise<void>
      reload: (id: string) => Promise<void>
      dispose: (id: string) => Promise<void>
      onState: (cb: (state: BrowserViewState) => void) => () => void
    }
  }
}
