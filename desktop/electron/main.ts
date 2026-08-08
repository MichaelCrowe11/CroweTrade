import { app, BrowserWindow, ipcMain, shell } from "electron"
import * as fs from "node:fs"
import * as path from "node:path"

/**
 * Candle fetch lives in the main process for two reasons: GeckoTerminal sends
 * no CORS headers so the sandboxed renderer cannot call it, and feed I/O
 * belongs on this side of the bridge anyway (the preload said so from day one).
 *
 * A 30s per-pool cache keeps us inside the public rate limit when the operator
 * flips between tokens quickly.
 */
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

  // Paint only once the renderer has something to show.
  win.once("ready-to-show", () => win?.show())

  // Dev self-capture: CROWETRADE_SHOT=/path/out.png makes the window write a
  // PNG of itself shortly after load, then carry on running. Exists because
  // driving `screencapture` at a window someone else is actively using steals
  // focus and races macOS Spaces; the window photographing itself does neither.
  const shotPath = process.env["CROWETRADE_SHOT"]
  if (shotPath) {
    win.webContents.once("did-finish-load", () => {
      setTimeout(() => {
        win?.webContents
          .capturePage()
          .then((img) => fs.promises.writeFile(shotPath, img.toPNG()))
          .catch((e: unknown) => console.error("self-capture failed:", e))
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
    win = null
  })
}

void app.whenReady().then(() => {
  // Failures resolve to an empty array rather than rejecting across the bridge:
  // the chart renders its honest "no data" state and the app stays quiet.
  ipcMain.handle("candles", async (_e, pool: unknown) => {
    if (typeof pool !== "string" || !/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(pool)) return []
    try {
      return await fetchCandles(pool)
    } catch {
      return []
    }
  })

  createWindow()
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit()
})
