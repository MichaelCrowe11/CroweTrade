import { app, BrowserWindow, ipcMain, shell } from "electron"
import { execFileSync } from "node:child_process"
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
async function askAnalyst(question: string): Promise<{ text: string; consulted: string[] }> {
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
    }),
  })
  if (!res.ok) throw new Error(`analyst ${res.status}: ${(await res.text()).slice(0, 160)}`)

  const body = (await res.json()) as { output?: { type?: string; name?: string; content?: { text?: string }[] }[] }
  const out = body.output ?? []
  return {
    consulted: out
      .filter((o) => o.type !== "message" && o.type !== "reasoning")
      .map((o) => o.name ?? o.type ?? "tool"),
    text: out
      .filter((o) => o.type === "message")
      .flatMap((o) => (o.content ?? []).map((c) => c.text ?? ""))
      .join("\n"),
  }
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
  ipcMain.handle("ask", async (_e, question: unknown) => {
    if (typeof question !== "string" || !question.trim()) {
      return { text: "empty question", consulted: [] }
    }
    try {
      return await askAnalyst(question)
    } catch (e) {
      // Surface the real reason: "analyst unavailable" sends someone hunting
      // when the answer is usually that `az login` expired.
      return { text: e instanceof Error ? e.message : String(e), consulted: [] }
    }
  })

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
