/**
 * Theme selection.
 *
 * Dependency-free on purpose: `node --test --experimental-strip-types` cannot
 * resolve the `.js`-specifier imports the app files use, so anything that
 * needs a test lives in a module with no imports. That constraint has already
 * shaped url.ts, migrate.ts, standing.ts and events.ts in this codebase.
 *
 * Dark is the default because a trading surface is usually read in a dim room.
 * Light exists for daylight and for screenshots that have to sit inside a
 * document, where near-black panels dominate the page.
 */

export type Theme = "dark" | "light"

export const THEME_KEY = "crowetrade-theme"

/** Anything unrecognised, absent or corrupt resolves to dark. A theme is a
 *  preference, never a reason to fail to paint. */
export function parseTheme(raw: string | null | undefined): Theme {
  return raw === "light" ? "light" : "dark"
}

export function nextTheme(current: Theme): Theme {
  return current === "dark" ? "light" : "dark"
}

/**
 * The attribute the token sheet keys off. Dark deliberately writes NO
 * attribute rather than `data-theme="dark"`: the dark values live on bare
 * `:root`, so an absent attribute is already correct and stripping it keeps
 * the default path free of state.
 */
export function applyTheme(root: { setAttribute(k: string, v: string): void; removeAttribute(k: string): void }, theme: Theme): void {
  if (theme === "light") root.setAttribute("data-theme", "light")
  else root.removeAttribute("data-theme")
}
