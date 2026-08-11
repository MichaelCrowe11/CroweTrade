/**
 * Capability tiers for the engine's authenticated surface.
 *
 * The engine has two kinds of caller. The OPERATOR changes what the engine
 * does with money: kill, veto, force a tick, spend raw inference. A
 * COLLABORATOR only reads: the corpus, the fitted model, the gates, the
 * Analyst. Until this split existed there was one credential, so a second
 * person could be given everything or nothing, and "everything" included the
 * kill switch and unmetered inference.
 *
 * What lives in this module is the DECISION, not the comparison. Constant-time
 * token comparison in the Worker is `crypto.subtle.timingSafeEqual`, a workerd
 * extension Node does not implement, so it cannot run under `node --test`. The
 * part worth testing is what a match MEANS, because that is where a mistake is
 * silent: a research token quietly satisfying /api/kill looks like nothing at
 * all right up until someone uses it.
 */

export type Tier = "admin" | "research"

/**
 * Which secrets a presented token matched. Both false is the normal case for
 * an unauthenticated request, and both true would mean the two secrets were
 * set to the same string, which the operator should avoid but which is not
 * this module's business to police.
 */
export interface TokenMatch {
  admin: boolean
  research: boolean
}

/**
 * Does a caller holding `match` satisfy a route requiring `required`?
 *
 * Admin satisfies BOTH tiers. That direction is deliberate: the split exists
 * to constrain a second person, and it must never lock the operator out of a
 * surface they already had, nor force the installed terminal to start carrying
 * two credentials to do what one used to do.
 *
 * Research satisfies ONLY research. That asymmetry is the entire boundary. If
 * it ever becomes symmetric the split is decorative, which is worse than no
 * split at all, because it reads as protection while providing none.
 */
export function tierSatisfied(required: Tier, match: TokenMatch): boolean {
  if (match.admin) return true
  return required === "research" && match.research
}
