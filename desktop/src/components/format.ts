/** Compact age, e.g. "4m", "3h", "12d". Age is the dominant variable here. */
export function age(from: number | null, now: number): string {
  if (from === null) return "unknown"
  const secs = Math.max(0, Math.floor((now - from) / 1000))
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

/** Abbreviated USD. Full precision on a six-figure number is noise on a panel. */
export function usd(v: number | null): string {
  if (v === null) return "unknown"
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`
  if (v >= 1) return `$${v.toFixed(2)}`
  // Sub-dollar prices are where meme coins actually live, so significant digits
  // matter more than a fixed decimal count. toPrecision keeps them.
  return `$${v.toPrecision(3)}`
}

/** Middle-truncated mint address: both ends carry identity, the middle does not. */
export function shortMint(mint: string): string {
  return mint.length <= 16 ? mint : `${mint.slice(0, 6)}...${mint.slice(-6)}`
}
