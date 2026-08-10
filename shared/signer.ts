/**
 * Transaction signing.
 *
 * No runtime imports: `crypto.subtle` is a global in both workerd and Node, so
 * this whole module is unit-testable under `node --test
 * --experimental-strip-types`, which cannot resolve the `.js` specifiers the
 * engine files use. The same constraint shaped trajectory.ts and preflight.ts.
 * Signing is the last place we would accept "probably correct".
 *
 * WHAT A SOLANA TRANSACTION LOOKS LIKE ON THE WIRE
 *
 *   [compact-u16 signature count][64-byte signature]...[message bytes]
 *
 * The signature is over the MESSAGE BYTES ONLY, not over the whole envelope.
 * So signing is: find where the message starts, sign that slice, write the
 * signature into its slot. We never rebuild or re-serialize the message.
 *
 * That last point is load-bearing and was learned the expensive way earlier in
 * this project: Jupiter requires the quote to be echoed back VERBATIM because
 * a re-serialized quote is a different quote. The same logic applies harder to
 * a transaction. Any byte we regenerate is a byte that can differ from what
 * was simulated, and then the thing we signed is not the thing we checked.
 */

/** Ed25519 signatures are always exactly this long. */
export const SIG_LEN = 64
/** A Solana keypair file is [32-byte seed][32-byte public key]. */
export const SEED_LEN = 32
export const KEYPAIR_LEN = 64

/**
 * Decode a `solana-keygen` JSON keypair.
 *
 * Throws on anything malformed rather than returning a partial key: a key that
 * is subtly wrong produces signatures that fail on chain after a fee has been
 * paid, and the error would surface far from its cause.
 */
export function parseKeypair(json: string): { seed: Uint8Array; publicKey: Uint8Array } {
  let arr: unknown
  try {
    arr = JSON.parse(json)
  } catch {
    throw new Error("keypair is not valid JSON")
  }
  if (!Array.isArray(arr)) throw new Error("keypair must be a JSON array of bytes")
  if (arr.length !== KEYPAIR_LEN) {
    throw new Error(`keypair must be ${KEYPAIR_LEN} bytes, got ${arr.length}`)
  }
  for (const b of arr) {
    if (typeof b !== "number" || !Number.isInteger(b) || b < 0 || b > 255) {
      throw new Error("keypair contains a value that is not a byte")
    }
  }
  const bytes = Uint8Array.from(arr as number[])
  return { seed: bytes.slice(0, SEED_LEN), publicKey: bytes.slice(SEED_LEN) }
}

/**
 * Decode a compact-u16 (shortvec) at `offset`.
 *
 * Solana's own length prefix: 7 bits per byte, low group first, continuation
 * bit set while more follow. Values under 128 are a single byte, which covers
 * every transaction we build, but decoding it properly costs three lines and
 * removes an assumption.
 */
export function readCompactU16(bytes: Uint8Array, offset = 0): { value: number; length: number } {
  let value = 0
  let length = 0
  for (;;) {
    if (offset + length >= bytes.length) throw new Error("truncated compact-u16")
    const byte = bytes[offset + length] as number
    value |= (byte & 0x7f) << (7 * length)
    length += 1
    if ((byte & 0x80) === 0) break
    if (length > 3) throw new Error("compact-u16 too long")
  }
  return { value, length }
}

export interface TxLayout {
  /** How many signature slots the transaction carries. */
  signatureCount: number
  /** Byte offset of the first signature slot. */
  signaturesAt: number
  /** Byte offset where the signed message begins. */
  messageAt: number
}

/**
 * Locate the signature slots and the message inside a serialized transaction.
 *
 * Refuses a transaction that declares more signers than we can satisfy: this
 * signer holds exactly one key, and quietly signing slot 0 of a two-signer
 * transaction would produce something that can never land, after a fee.
 */
export function layoutOf(tx: Uint8Array): TxLayout {
  const { value: signatureCount, length } = readCompactU16(tx, 0)
  if (signatureCount < 1) throw new Error("transaction declares no signatures")
  const signaturesAt = length
  const messageAt = signaturesAt + signatureCount * SIG_LEN
  if (messageAt >= tx.length) throw new Error("transaction is truncated: no message after signatures")
  return { signatureCount, signaturesAt, messageAt }
}

/** Base64 to bytes, without Buffer so this runs in a Worker unchanged. */
export function fromBase64(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

export function toBase64(bytes: Uint8Array): string {
  let bin = ""
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin)
}

/** Wrap a raw Ed25519 seed in the PKCS8 envelope WebCrypto expects. */
function pkcs8(seed: Uint8Array): Uint8Array {
  if (seed.length !== SEED_LEN) throw new Error("seed must be 32 bytes")
  return Uint8Array.from([
    0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70,
    0x04, 0x22, 0x04, 0x20, ...seed,
  ])
}

/**
 * Sign a serialized transaction and return it, still serialized.
 *
 * The message bytes are passed through untouched; only the signature slot
 * changes. What comes back is byte-identical to what went in except for 64
 * bytes, which is the property that makes "we signed exactly what we
 * simulated" checkable rather than assumed.
 */
export async function signTransaction(txBase64: string, keypairJson: string): Promise<string> {
  const { seed } = parseKeypair(keypairJson)
  const tx = fromBase64(txBase64)
  const layout = layoutOf(tx)
  if (layout.signatureCount !== 1) {
    throw new Error(
      `transaction needs ${layout.signatureCount} signers; this signer holds one key`,
    )
  }

  const message = tx.subarray(layout.messageAt)
  const key = await crypto.subtle.importKey("pkcs8", pkcs8(seed), { name: "Ed25519" }, false, ["sign"])
  const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, key, message))
  if (sig.length !== SIG_LEN) throw new Error(`unexpected signature length ${sig.length}`)

  const signed = new Uint8Array(tx)
  signed.set(sig, layout.signaturesAt)
  return toBase64(signed)
}
