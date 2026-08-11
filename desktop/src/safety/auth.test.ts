import { test } from "node:test"
import assert from "node:assert/strict"
import { tierSatisfied, type TokenMatch } from "../../../shared/auth.ts"

/**
 * An authorization boundary is tested for REFUSAL first, the same way the
 * preflight guard is. The expensive failure here is a wrongly-permitted
 * caller, and unlike a wrongly-refused one it produces no complaint: nobody
 * reports being allowed to do something they should not have been able to do.
 *
 * The single case that matters most is the last one in this file.
 */

const NOTHING: TokenMatch = { admin: false, research: false }
const ADMIN: TokenMatch = { admin: true, research: false }
const RESEARCH: TokenMatch = { admin: false, research: true }

test("an unauthenticated caller satisfies neither tier", () => {
  assert.equal(tierSatisfied("admin", NOTHING), false)
  assert.equal(tierSatisfied("research", NOTHING), false)
})

test("the operator keeps every surface they had before the split", () => {
  assert.equal(tierSatisfied("admin", ADMIN), true)
  assert.equal(tierSatisfied("research", ADMIN), true)
})

test("a research token reaches the research tier", () => {
  assert.equal(tierSatisfied("research", RESEARCH), true)
})

test("a research token NEVER reaches an admin route", () => {
  // If this ever passes, kill, veto, tick and raw inference are open to every
  // holder of the collaborator credential, and the split is decorative.
  assert.equal(tierSatisfied("admin", RESEARCH), false)
})

test("both secrets set to the same string still resolves as admin", () => {
  // Not a supported configuration, but it must fail toward the operator's own
  // identity rather than into some third state.
  assert.equal(tierSatisfied("admin", { admin: true, research: true }), true)
})
