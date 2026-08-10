/**
 * Sign the live envelope with the trading wallet.
 *
 * Produces the two values LIVE_DUST_POLICY is missing: `signer` (the wallet
 * address) and `signature` (that wallet's Ed25519 signature over the
 * envelope's canonical hash). Paste both into shared/policy.ts, then deploy.
 *
 * Why this exists even for a sole operator. The signature is what makes a fill
 * traceable to a specific set of limits that somebody agreed to, rather than
 * to whatever config happened to be deployed at the time. Six months from now,
 * looking at a trade, the question "what was I allowing when this happened"
 * has an answer that cannot be edited after the fact. Preflight refuses an
 * unsigned live envelope for exactly that reason.
 *
 * It signs the hash of the envelope with `signature: null`, which is the same
 * canonical form policyHash() produces, so verification is a straight compare.
 *
 * Usage:
 *   node --experimental-strip-types engine/scripts/sign-policy.mjs ~/.crowetrade-dust.json
 */
import { readFileSync } from "node:fs"
import { policyHash, LIVE_DUST_POLICY } from "../../shared/policy.ts"
import { parseKeypair, base58 } from "../../shared/signer.ts"

const keyPath = process.argv[2]
if (!keyPath) {
  console.error("usage: sign-policy.mjs <keypair.json>")
  process.exit(1)
}

const { seed, publicKey } = parseKeypair(readFileSync(keyPath, "utf8"))
const signer = base58(publicKey)

// Hash the envelope as it will be deployed, with the signer filled in and the
// signature still null. Signing a DIFFERENT object than the one deployed would
// make the signature meaningless, which is the failure mode worth avoiding.
const toSign = { ...LIVE_DUST_POLICY, signer, signature: null }
const hash = await policyHash(toSign)

const pkcs8 = Uint8Array.from([
  0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70,
  0x04, 0x22, 0x04, 0x20, ...seed,
])
const key = await crypto.subtle.importKey("pkcs8", pkcs8, { name: "Ed25519" }, false, ["sign"])
const sig = new Uint8Array(
  await crypto.subtle.sign({ name: "Ed25519" }, key, new TextEncoder().encode(hash)),
)

console.log("\nEnvelope hash (this is what was signed):")
console.log("  " + hash)
console.log("\nPaste these into LIVE_DUST_POLICY in shared/policy.ts:\n")
console.log(`  signer: ${JSON.stringify(signer)},`)
console.log(`  signature: ${JSON.stringify(base58(sig))},`)
console.log("\nAlso set expiresAt to a near-future date, e.g.:")
console.log(`  expiresAt: "<48 hours from when you deploy>",`)
console.log(
  "\nNOTE: changing ANY field after signing invalidates this signature,\n" +
  "because the hash covers the whole envelope. Re-run this if you edit caps.\n",
)
