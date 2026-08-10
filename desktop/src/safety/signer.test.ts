import { test } from "node:test"
import assert from "node:assert/strict"
import {
  parseKeypair, readCompactU16, layoutOf, fromBase64, toBase64,
  signTransaction, base58, base58Decode, verifyPolicySignature, SIG_LEN, KEYPAIR_LEN,
} from "../../../shared/signer.ts"

/**
 * Signing is the last place to accept "probably correct", so these tests check
 * the bytes rather than the happy path. The property that matters most:
 * signing changes EXACTLY the 64 signature bytes and nothing else, because
 * that is what makes "we signed what we simulated" verifiable instead of
 * assumed.
 */

/** A deterministic 64-byte keypair file: seed 0..31, then 32 pubkey bytes. */
const KEYPAIR = JSON.stringify([...Array(KEYPAIR_LEN).keys()])

/** Build a single-signer transaction: [1][64 zero bytes][message]. */
function fakeTx(message: number[]): Uint8Array {
  return Uint8Array.from([1, ...new Array(SIG_LEN).fill(0), ...message])
}

test("a well-formed keypair splits into seed and public key", () => {
  const { seed, publicKey } = parseKeypair(KEYPAIR)
  assert.equal(seed.length, 32)
  assert.equal(publicKey.length, 32)
  assert.equal(seed[0], 0)
  assert.equal(publicKey[0], 32)
})

test("a malformed keypair throws rather than returning a partial key", () => {
  assert.throws(() => parseKeypair("not json"), /valid JSON/)
  assert.throws(() => parseKeypair('{"a":1}'), /array of bytes/)
  assert.throws(() => parseKeypair("[1,2,3]"), /64 bytes/)
  assert.throws(() => parseKeypair(JSON.stringify([...Array(63).keys(), 999])), /not a byte/)
  assert.throws(() => parseKeypair(JSON.stringify([...Array(63).keys(), -1])), /not a byte/)
})

test("compact-u16 decodes single and multi-byte lengths", () => {
  assert.deepEqual(readCompactU16(Uint8Array.from([1])), { value: 1, length: 1 })
  assert.deepEqual(readCompactU16(Uint8Array.from([127])), { value: 127, length: 1 })
  // 0x80 0x01 = 128, the first two-byte value.
  assert.deepEqual(readCompactU16(Uint8Array.from([0x80, 0x01])), { value: 128, length: 2 })
  assert.throws(() => readCompactU16(Uint8Array.from([0x80])), /truncated/)
})

test("layout finds the message after the signature slots", () => {
  const tx = fakeTx([9, 9, 9])
  const l = layoutOf(tx)
  assert.equal(l.signatureCount, 1)
  assert.equal(l.signaturesAt, 1)
  assert.equal(l.messageAt, 1 + SIG_LEN)
})

test("a transaction with no message or no signatures is refused", () => {
  assert.throws(() => layoutOf(Uint8Array.from([0])), /no signatures/)
  // Declares one signature but the buffer ends inside the slot.
  assert.throws(() => layoutOf(Uint8Array.from([1, 0, 0])), /truncated/)
})

test("base64 round-trips arbitrary bytes, including high ones", () => {
  const bytes = Uint8Array.from([0, 1, 127, 128, 254, 255])
  assert.deepEqual(fromBase64(toBase64(bytes)), bytes)
})

test("signing changes EXACTLY the signature bytes and nothing else", async () => {
  const message = [1, 2, 3, 4, 5, 6, 7, 8]
  const tx = fakeTx(message)
  const signedB64 = await signTransaction(toBase64(tx), KEYPAIR)
  const signed = fromBase64(signedB64)

  assert.equal(signed.length, tx.length, "length must not change")
  assert.equal(signed[0], 1, "signature count untouched")
  // The message is byte-identical: this is the whole guarantee.
  assert.deepEqual(
    Array.from(signed.subarray(1 + SIG_LEN)),
    message,
    "message bytes must be passed through untouched",
  )
  // And the signature slot is now populated.
  const sig = signed.subarray(1, 1 + SIG_LEN)
  assert.equal(sig.length, SIG_LEN)
  assert.ok(sig.some((b) => b !== 0), "signature slot must be filled")
})

test("signing is deterministic: Ed25519 over the same message and key", async () => {
  const tx = toBase64(fakeTx([42, 42]))
  const a = await signTransaction(tx, KEYPAIR)
  const b = await signTransaction(tx, KEYPAIR)
  assert.equal(a, b)
})

test("a different message produces a different signature", async () => {
  const a = await signTransaction(toBase64(fakeTx([1])), KEYPAIR)
  const b = await signTransaction(toBase64(fakeTx([2])), KEYPAIR)
  assert.notEqual(a, b)
})

test("a multi-signer transaction is REFUSED, not partially signed", async () => {
  // Two declared signers; this signer holds one key. Signing slot 0 anyway
  // would produce a transaction that can never land, after paying a fee.
  const tx = Uint8Array.from([2, ...new Array(SIG_LEN * 2).fill(0), 7, 7])
  await assert.rejects(() => signTransaction(toBase64(tx), KEYPAIR), /one key/)
})

test("base58 encodes addresses, and leading zero bytes become leading ones", () => {
  // The rule that is easy to omit: a leading zero BYTE is a leading '1'
  // CHARACTER, so keys starting with zero would otherwise encode short.
  assert.equal(base58(Uint8Array.from([0, 0, 1])), "112")
  assert.equal(base58(Uint8Array.from([])), "")
  // A known vector: 32 zero bytes is the system program address.
  assert.equal(base58(new Uint8Array(32)), "1".repeat(32))
})

test("base58 round-trips a real derived address length", () => {
  // Solana addresses are 32 bytes and render as 32-44 base58 characters.
  const addr = base58(Uint8Array.from({ length: 32 }, (_, i) => (i * 7 + 13) % 256))
  assert.ok(addr.length >= 32 && addr.length <= 44, `unexpected length ${addr.length}`)
})

test("base58 decode is the exact inverse of encode", () => {
  for (const bytes of [
    Uint8Array.from([0, 0, 1]),
    Uint8Array.from([255, 254, 1, 0]),
    Uint8Array.from({ length: 32 }, (_, i) => (i * 11 + 5) % 256),
  ]) {
    assert.deepEqual(base58Decode(base58(bytes)), bytes)
  }
})

test("base58 decode returns null on junk rather than throwing", () => {
  // Operator-pasted input; 0, O, I and l are deliberately absent from the
  // alphabet precisely because they are misread.
  assert.equal(base58Decode("0OIl"), null)
  assert.equal(base58Decode("not valid!"), null)
})

test("a policy signature VERIFIES, and any tampering fails it", async () => {
  // A REAL pair, not the synthetic KEYPAIR fixture above. That fixture's
  // "public key" bytes are not derived from its seed, which is fine for
  // signing (signing uses only the seed) and useless for verification. The
  // first version of this test used it and failed correctly.
  const pair = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"])
  const publicKey = new Uint8Array(await crypto.subtle.exportKey("raw", pair.publicKey))
  const hash = "a".repeat(64)
  const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, pair.privateKey,
    new TextEncoder().encode(hash)))

  assert.equal(await verifyPolicySignature(hash, base58(publicKey), base58(sig)), true)
  // A different hash means the envelope changed after signing.
  assert.equal(await verifyPolicySignature("b".repeat(64), base58(publicKey), base58(sig)), false)
  // A different signer did not consent to this.
  assert.equal(await verifyPolicySignature(hash, base58(new Uint8Array(32)), base58(sig)), false)
})

test("a made-up signature string does NOT pass, which is the whole point", async () => {
  const publicKey = parseKeypair(KEYPAIR).publicKey
  for (const fake of ["anything", "1".repeat(88), ""]) {
    assert.equal(await verifyPolicySignature("a".repeat(64), base58(publicKey), fake), false, fake)
  }
})
