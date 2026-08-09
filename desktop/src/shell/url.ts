/**
 * URL normalization for the in-app browser bar.
 *
 * Dependency-free on purpose: node --test with type stripping cannot resolve
 * the `.js`-specifier imports the app modules use between themselves, so
 * test-critical logic lives in modules that import nothing.
 *
 * Only http and https come back. Everything else returns null rather than a
 * best guess, because the result of this function is handed to a live
 * WebContents: a scheme like file: or javascript: is not a typo to repair, it
 * is a request the browser must refuse.
 */
export function normalizeUrl(input: string): string | null {
  const raw = input.trim()
  if (!raw) return null
  const candidate = /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(raw) ? raw : `https://${raw}`
  let url: URL
  try {
    url = new URL(candidate)
  } catch {
    return null
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") return null
  // A pasted "not a url at all" parses as https://not%20a%20url... only when
  // the host survives; URL() already rejected it above. Hostnames with spaces
  // cannot exist, so anything that parsed has a plausible host.
  return url.href
}
