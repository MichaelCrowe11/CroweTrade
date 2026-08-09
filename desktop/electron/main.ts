import { app, BrowserWindow, WebContentsView, ipcMain, session, shell } from "electron"
import { execFileSync } from "node:child_process"
import * as fs from "node:fs"
import * as path from "node:path"
import { createSseParser } from "./sse"
import { runOrchestrator, stopOrchestrator } from "./orchestrator"

/**
 * Candle fetch lives in the main process for two reasons: GeckoTerminal sends
 * no CORS headers so the sandboxed renderer cannot call it, and feed I/O
 * belongs on this side of the bridge anyway (the preload said so from day one).
 *
 * A 30s per-pool cache keeps us inside the public rate limit when the operator
 * flips between tokens quickly.
 */
const FOUNDRY =
  "https://crowelm-prod-eastus2.services.ai.azure.com/api/projects/crowelm-foundry"
const ANALYST_MODEL = "gpt-5.6-sol"

/**
 * Ask the CroweTrade Analyst.
 *
 * Lives in the main process because it needs an Azure token, and a credential
 * reachable from page context is a credential you have published. The renderer
 * gets an answer, never a key.
 *
 * The token comes from `az account get-access-token`, so the operator's own
 * Azure login is the auth: no key is stored in the app at all.
 */
/**
 * Streaming: the response arrives as SSE and every text delta is forwarded to
 * the renderer the moment it lands, so the Analyst speaks the way Cortex's
 * conversation surface does instead of holding its breath for the whole
 * answer. Tool invocations stream too (response.output_item.added), which is
 * grounding made visible in real time: the operator watches the Analyst read
 * the engine before it speaks. Event vocabulary verified against the live
 * endpoint: output_text.delta carries text; output_item.added carries
 * openapi_call items; completed closes the stream.
 */
async function askAnalyst(
  question: string,
  onDelta: (text: string) => void,
  onTool: (name: string) => void,
): Promise<{ text: string; consulted: string[] }> {
  const token = execFileSync(
    "az",
    ["account", "get-access-token", "--resource", "https://ai.azure.com", "--query", "accessToken", "-o", "tsv"],
    { encoding: "utf8" },
  ).trim()

  const root = path.join(__dirname, "../../analyst")
  const res = await fetch(`${FOUNDRY}/openai/v1/responses`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: ANALYST_MODEL,
      instructions: fs.readFileSync(path.join(root, "agent/instructions.md"), "utf8"),
      tools: [{
        type: "openapi",
        openapi: {
          name: "crowetrade_engine_read",
          description: "Read-only access to the live CroweTrade engine.",
          auth: { type: "anonymous" },
          spec: JSON.parse(fs.readFileSync(path.join(root, "config/engine-openapi.json"), "utf8")),
        },
      }],
      input: question,
      stream: true,
    }),
  })
  if (!res.ok || !res.body) {
    throw new Error(`analyst ${res.status}: ${(await res.text()).slice(0, 160)}`)
  }

  const parser = createSseParser()
  const decoder = new TextDecoder()
  const consulted: string[] = []
  let text = ""
  let failure: string | null = null

  for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
    for (const evt of parser.push(decoder.decode(chunk, { stream: true }))) {
      if (evt.type === "response.output_text.delta") {
        const delta = evt["delta"]
        if (typeof delta === "string") {
          text += delta
          onDelta(delta)
        }
      } else if (evt.type === "response.output_item.added") {
        const item = evt["item"] as { type?: string; name?: string } | undefined
        if (item && item.type !== "message" && item.type !== "reasoning") {
          const name = item.name ?? item.type ?? "tool"
          consulted.push(name)
          onTool(name)
        }
      } else if (evt.type === "response.failed" || evt.type === "error") {
        failure = JSON.stringify(evt).slice(0, 200)
      }
    }
  }

  if (failure) throw new Error(`analyst stream failed: ${failure}`)
  return { text, consulted }
}

const OHLCV = "https://api.geckoterminal.com/api/v2/networks/solana/pools"
const candleCache = new Map<string, { at: number; rows: number[][] }>()
const CANDLE_TTL_MS = 30_000

async function fetchCandles(pool: string): Promise<number[][]> {
  const hit = candleCache.get(pool)
  if (hit && Date.now() - hit.at < CANDLE_TTL_MS) return hit.rows
  const res = await fetch(`${OHLCV}/${pool}/ohlcv/minute?aggregate=1&limit=120`)
  if (!res.ok) throw new Error(`ohlcv -> ${res.status}`)
  const body = (await res.json()) as {
    data?: { attributes?: { ohlcv_list?: number[][] } }
  }
  // Upstream returns newest-first; the chart wants time ascending.
  const rows = (body.data?.attributes?.ohlcv_list ?? []).slice().reverse()
  candleCache.set(pool, { at: Date.now(), rows })
  return rows
}

/* ── In-app browser ──────────────────────────────────────────────────────────
 *
 * Each browser panel in the renderer is backed by a WebContentsView owned
 * here. Not an iframe (modern sites refuse embedding; Cortex documented the
 * same conclusion) and not a webview tag (deprecated posture, weaker process
 * story). The renderer is only a control surface: it names a panel id and a
 * rectangle; this process decides whether a view exists, what it may load,
 * and where popups go. Views are sandboxed, context-isolated, node-free, and
 * live in their own persistent session partition so nothing they load shares
 * state with the terminal's renderer.
 */
const BROWSER_PARTITION = "persist:crowetrade-browser"
/** Panel ids come from the renderer's store; anything else is refused. */
const BROWSER_PANEL_ID = /^browser-[A-Za-z0-9-]+$/
const browserViews = new Map<string, WebContentsView>()

function isHttpUrl(u: unknown): u is string {
  if (typeof u !== "string") return false
  try {
    const p = new URL(u).protocol
    return p === "https:" || p === "http:"
  } catch {
    return false
  }
}

function pushBrowserState(id: string, view: WebContentsView): void {
  const wc = view.webContents
  const send = () => {
    win?.webContents.send("browser:state", {
      id,
      url: wc.getURL(),
      canGoBack: wc.navigationHistory.canGoBack(),
      canGoForward: wc.navigationHistory.canGoForward(),
      loading: wc.isLoading(),
    })
  }
  wc.on("did-navigate", send)
  wc.on("did-navigate-in-page", send)
  wc.on("did-start-loading", send)
  wc.on("did-stop-loading", send)
}

function ensureBrowserView(id: string, url: string): boolean {
  if (!win) return false
  if (browserViews.has(id)) return true
  const view = new WebContentsView({
    webPreferences: {
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      partition: BROWSER_PARTITION,
    },
  })
  // Invisible until the renderer reports where its panel sits.
  view.setBounds({ x: 0, y: 0, width: 0, height: 0 })
  // target=_blank and window.open land in the operator's real browser, where
  // their sessions live; a chrome-less child window here would be neither.
  view.webContents.setWindowOpenHandler(({ url: target }) => {
    if (isHttpUrl(target)) void shell.openExternal(target)
    return { action: "deny" }
  })
  // In-view navigation stays on the web; file:, about:, custom schemes are
  // refused before they start.
  view.webContents.on("will-navigate", (e, target) => {
    if (!isHttpUrl(target)) e.preventDefault()
  })
  pushBrowserState(id, view)
  win.contentView.addChildView(view)
  browserViews.set(id, view)
  // Failures surface through did-stop-loading state, not a rejection.
  void view.webContents.loadURL(url).catch(() => {})
  return true
}

function disposeBrowserView(id: string): void {
  const view = browserViews.get(id)
  if (!view) return
  browserViews.delete(id)
  win?.contentView.removeChildView(view)
  view.webContents.close()
}

// Presence of the dev server URL is what selects the dev renderer, NOT
// app.isPackaged. Running `electron .` against a production bundle is
// unpackaged too, so keying off isPackaged sends it to a dev server that is not
// there and paints an empty window with no error.
const DEV_URL = process.env["VITE_DEV_SERVER_URL"]

let win: BrowserWindow | null = null

function createWindow(): void {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1120,
    minHeight: 720,
    // The instrument look starts at the frame: no stock title bar, and the
    // traffic lights inset so they sit inside our own header rule.
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 18, y: 18 },
    // Must match --clm-cream (dark) or the window flashes white on open.
    backgroundColor: "#0a0a0b",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })

  // Paint only once the renderer has something to show. A shot run appears
  // WITHOUT taking focus and lets every click fall through: these runs happen
  // on a shared, actively-used desktop, and a window that eats a click meant
  // for whatever is beneath it both disrupts the operator and mutates the
  // seeded state it was launched to photograph. (Observed, not hypothetical:
  // a stray click closed two seeded panels mid-shot.)
  const isShotRun = Boolean(process.env["CROWETRADE_SHOT"])
  win.once("ready-to-show", () => {
    if (isShotRun) {
      win?.setIgnoreMouseEvents(true)
      win?.showInactive()
    } else {
      win?.show()
    }
  })

  // Dev self-capture: CROWETRADE_SHOT=/path/out.png makes the window write a
  // PNG of itself shortly after load, then carry on running. Exists because
  // driving `screencapture` at a window someone else is actively using steals
  // focus and races macOS Spaces; the window photographing itself does neither.
  const shotPath = process.env["CROWETRADE_SHOT"]
  const shotState = process.env["CROWETRADE_SHOT_STATE"]
  let shotSeeded = false
  if (shotPath) {
    win.webContents.on("did-finish-load", () => {
      // Optional state seeding, dev-only like the shot itself: the env var is
      // a JSON object of localStorage key to value; each value is stored
      // stringified and the page reloads once. One seed can carry the
      // workspace layout AND an engine fixture, so states like an open
      // breaker are photographable without waiting for the live engine.
      if (shotState && !shotSeeded) {
        shotSeeded = true
        try {
          const entries = Object.entries(JSON.parse(shotState) as Record<string, unknown>)
          const js = entries
            .map(
              ([k, v]) =>
                `localStorage.setItem(${JSON.stringify(k)}, ${JSON.stringify(JSON.stringify(v))});`,
            )
            .join("")
          void win?.webContents.executeJavaScript(`${js}location.reload()`)
        } catch (e) {
          console.error("CROWETRADE_SHOT_STATE is not valid JSON:", e)
        }
        return
      }
      // Optional page-context driver: CROWETRADE_SHOT_JS runs after the app
      // mounts, so a shot can exercise a real interaction (click a suggestion
      // chip, submit a form) and photograph the result. Page-level, so it
      // works even though shot windows ignore OS mouse events.
      const shotJs = process.env["CROWETRADE_SHOT_JS"]
      if (shotJs) {
        setTimeout(() => {
          void win?.webContents
            .executeJavaScript(shotJs)
            .catch((e: unknown) => console.error("shot-js failed:", e))
        }, 2_000)
      }
      setTimeout(() => {
        win?.webContents
          .capturePage()
          .then((img) => fs.promises.writeFile(shotPath, img.toPNG()))
          .catch((e: unknown) => console.error("self-capture failed:", e))
        // capturePage on the window does NOT composite child WebContentsViews,
        // so each browser view photographs itself alongside.
        let n = 0
        for (const view of browserViews.values()) {
          const p = shotPath.replace(/\.png$/, n === 0 ? "-view.png" : `-view${n}.png`)
          n++
          view.webContents
            .capturePage()
            .then((img) => fs.promises.writeFile(p, img.toPNG()))
            .catch((e: unknown) => console.error("view self-capture failed:", e))
        }
      }, 12_000)
    })
  }

  // Anything targeting a new window is an external link; hand it to the OS
  // browser rather than opening a chrome-less Electron window with no way back.
  win.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: "deny" }
  })

  if (DEV_URL) {
    void win.loadURL(DEV_URL)
  } else {
    void win.loadFile(path.join(__dirname, "../dist/index.html"))
  }

  // A renderer that fails to load otherwise shows an empty window and says
  // nothing, which is how the previous version of this file wasted a launch.
  win.webContents.on("did-fail-load", (_e, code, desc, url) => {
    console.error(`renderer failed to load: ${code} ${desc} (${url})`)
  })

  win.on("closed", () => {
    // The views die with the window's view hierarchy; only the map survives.
    browserViews.clear()
    win = null
  })
}

void app.whenReady().then(() => {
  // Dev launches get the real dock icon too; packaged builds carry it in the
  // bundle via electron-builder, and the PNG is not in the packaged files
  // list, so existsSync is the dev/prod guard here.
  const dockIcon = path.join(__dirname, "../build/icon-1024.png")
  if (process.platform === "darwin" && fs.existsSync(dockIcon)) {
    app.dock?.setIcon(dockIcon)
  }

  // Failures resolve to an empty array rather than rejecting across the bridge:
  // the chart renders its honest "no data" state and the app stays quiet.
  ipcMain.handle("ask", async (_e, question: unknown) => {
    if (typeof question !== "string" || !question.trim()) {
      return { text: "empty question", consulted: [] }
    }
    try {
      return await askAnalyst(
        question,
        (delta) => win?.webContents.send("analyst:delta", delta),
        (name) => win?.webContents.send("analyst:tool", name),
      )
    } catch (e) {
      // Surface the real reason: "analyst unavailable" sends someone hunting
      // when the answer is usually that `az login` expired.
      return { text: e instanceof Error ? e.message : String(e), consulted: [] }
    }
  })

  ipcMain.handle("orch:ask", async (_e, goal: unknown) => {
    if (typeof goal !== "string" || !goal.trim()) return { text: "empty goal" }
    try {
      return await runOrchestrator(goal, (evt) => win?.webContents.send("orch:event", evt))
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      win?.webContents.send("orch:event", { kind: "error", message })
      return { text: message }
    }
  })

  ipcMain.handle("orch:stop", () => {
    stopOrchestrator()
  })

  ipcMain.handle("candles", async (_e, pool: unknown) => {
    if (typeof pool !== "string" || !/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(pool)) return []
    try {
      return await fetchCandles(pool)
    } catch {
      return []
    }
  })

  // The embedded browser renders arbitrary web pages beside a trading surface,
  // so its session gets no permissions at all: no camera, no clipboard-read,
  // no notifications. A block explorer needs none of them.
  session
    .fromPartition(BROWSER_PARTITION)
    .setPermissionRequestHandler((_wc, _permission, cb) => cb(false))

  ipcMain.handle("browser:ensure", (_e, id: unknown, url: unknown) => {
    if (typeof id !== "string" || !BROWSER_PANEL_ID.test(id)) return false
    return ensureBrowserView(id, isHttpUrl(url) ? url : "https://solscan.io")
  })

  ipcMain.handle("browser:bounds", (_e, id: unknown, bounds: unknown) => {
    if (typeof id !== "string") return
    const view = browserViews.get(id)
    const b = bounds as { x?: unknown; y?: unknown; width?: unknown; height?: unknown } | null
    if (!view || !b) return
    const { x, y, width, height } = b
    if (
      typeof x !== "number" || typeof y !== "number" ||
      typeof width !== "number" || typeof height !== "number" ||
      ![x, y, width, height].every(Number.isFinite)
    ) {
      return
    }
    view.setBounds({
      x: Math.max(0, Math.round(x)),
      y: Math.max(0, Math.round(y)),
      width: Math.max(0, Math.round(width)),
      height: Math.max(0, Math.round(height)),
    })
  })

  ipcMain.handle("browser:navigate", (_e, id: unknown, url: unknown) => {
    if (typeof id !== "string" || !isHttpUrl(url)) return
    void browserViews.get(id)?.webContents.loadURL(url).catch(() => {})
  })

  ipcMain.handle("browser:back", (_e, id: unknown) => {
    if (typeof id !== "string") return
    const wc = browserViews.get(id)?.webContents
    if (wc?.navigationHistory.canGoBack()) wc.navigationHistory.goBack()
  })

  ipcMain.handle("browser:forward", (_e, id: unknown) => {
    if (typeof id !== "string") return
    const wc = browserViews.get(id)?.webContents
    if (wc?.navigationHistory.canGoForward()) wc.navigationHistory.goForward()
  })

  ipcMain.handle("browser:reload", (_e, id: unknown) => {
    if (typeof id !== "string") return
    browserViews.get(id)?.webContents.reload()
  })

  ipcMain.handle("browser:dispose", (_e, id: unknown) => {
    if (typeof id !== "string") return
    disposeBrowserView(id)
  })

  createWindow()
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit()
})
