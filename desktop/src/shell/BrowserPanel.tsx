import { useEffect, useRef, useState, type FormEvent } from "react"
import { normalizeUrl } from "./url.js"

/**
 * In-app browser chrome. The page itself is a WebContentsView owned by the
 * main process (Cortex learned the hard way that an iframe is not a browser:
 * modern sites refuse embedding). This component is the control surface: URL
 * bar, back/forward/reload, and the rectangle the native view is told to fill.
 *
 * The host div and the native view are kept in sync three ways: a
 * ResizeObserver for size changes, a window resize listener, and a settle poll
 * for position-only moves (the analyst drawer and panel entry animations
 * translate this panel without resizing it, which ResizeObserver cannot see).
 * The poll is cheap because bounds are only sent when the rectangle changed.
 */

interface NavState {
  canGoBack: boolean
  canGoForward: boolean
  loading: boolean
}

function NavIcon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path
        d={d}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function BrowserPanel({ panelId, initialUrl }: { panelId: string; initialUrl: string }) {
  const hostRef = useRef<HTMLDivElement>(null)
  const editingRef = useRef(false)
  const lastRectRef = useRef("")
  // ensure() seeds from the mount-time URL only; later selection changes must
  // not yank a page the operator is reading.
  const initialRef = useRef(initialUrl)
  const [draft, setDraft] = useState(initialUrl)
  const [invalid, setInvalid] = useState(false)
  const [nav, setNav] = useState<NavState>({ canGoBack: false, canGoForward: false, loading: true })

  const api = window.crowetrade?.browser

  useEffect(() => {
    if (!api) return
    void api.ensure(panelId, initialRef.current)

    const report = () => {
      const el = hostRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const bounds = {
        x: Math.round(r.x),
        y: Math.round(r.y),
        width: Math.round(r.width),
        height: Math.round(r.height),
      }
      const key = `${bounds.x},${bounds.y},${bounds.width},${bounds.height}`
      if (key === lastRectRef.current) return
      lastRectRef.current = key
      void api.setBounds(panelId, bounds)
    }

    report()
    const ro = new ResizeObserver(report)
    if (hostRef.current) ro.observe(hostRef.current)
    window.addEventListener("resize", report)
    const settle = window.setInterval(report, 300)

    const offState = api.onState((s) => {
      if (s.id !== panelId) return
      setNav({ canGoBack: s.canGoBack, canGoForward: s.canGoForward, loading: s.loading })
      if (!editingRef.current) {
        setDraft(s.url)
        setInvalid(false)
      }
    })

    return () => {
      ro.disconnect()
      window.removeEventListener("resize", report)
      window.clearInterval(settle)
      offState()
      void api.dispose(panelId)
    }
  }, [api, panelId])

  const submit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!api) return
    const url = normalizeUrl(draft)
    if (!url) {
      setInvalid(true)
      return
    }
    setInvalid(false)
    void api.navigate(panelId, url)
    // Hand focus back to the page so the address bar stops pinning the draft.
    ;(e.currentTarget.elements[0] as HTMLElement | undefined)?.blur?.()
  }

  if (!api) {
    return (
      <div className="browser">
        <p className="empty">
          Browser bridge unavailable. Rebuild the Electron main process and relaunch.
        </p>
      </div>
    )
  }

  return (
    <div className="browser">
      <div className="browser__bar">
        <button
          type="button"
          className="browser__nav"
          disabled={!nav.canGoBack}
          onClick={() => void api.back(panelId)}
          aria-label="Back"
        >
          <NavIcon d="M14 6l-6 6 6 6" />
        </button>
        <button
          type="button"
          className="browser__nav"
          disabled={!nav.canGoForward}
          onClick={() => void api.forward(panelId)}
          aria-label="Forward"
        >
          <NavIcon d="M10 6l6 6-6 6" />
        </button>
        <button
          type="button"
          className="browser__nav"
          onClick={() => void api.reload(panelId)}
          aria-label="Reload"
        >
          <NavIcon d="M5 12a7 7 0 107-7H8m0 0l2.6-2.6M8 5l2.6 2.6" />
        </button>
        <form className="browser__form" onSubmit={submit}>
          <input
            className="browser__url mono"
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value)
              setInvalid(false)
            }}
            onFocus={(e) => {
              editingRef.current = true
              e.target.select()
            }}
            onBlur={() => {
              editingRef.current = false
            }}
            aria-label="Address"
            aria-invalid={invalid || undefined}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
          />
        </form>
        {nav.loading && <span className="browser__loading mono">loading</span>}
      </div>
      {/* The native view covers this surface once it has bounds; the hint only
          shows if the view failed to appear, which makes the failure visible
          instead of a silent black rectangle. */}
      <div ref={hostRef} className="browser__host">
        <p className="browser__hint">page loads over this surface</p>
      </div>
    </div>
  )
}
