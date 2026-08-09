// Verify the 402 wire format against the x402 v2 spec without deploying real
// payment config: a placeholder payTo on a live endpoint could be paid.
const { paymentRequired, ROUTES, SOLANA_MAINNET, USDC_MINT } = await import("../src/x402.ts")
const req = new Request("https://engine.example/api/v1/safety/So11111111111111111111111111111111111111112")
const res = paymentRequired(req, ROUTES["/api/v1/safety"], "PLACEHOLDER_WALLET")

const checks = []
const ok = (n, c) => checks.push({ check: n, pass: !!c })

ok("status is 402", res.status === 402)
const hdr = res.headers.get("PAYMENT-REQUIRED")
ok("PAYMENT-REQUIRED header present", hdr)
const decoded = JSON.parse(Buffer.from(hdr, "base64").toString("utf8"))
ok("x402Version === 2", decoded.x402Version === 2)
ok("resource.url present", typeof decoded.resource?.url === "string")
ok("accepts is a non-empty array", Array.isArray(decoded.accepts) && decoded.accepts.length > 0)
const a = decoded.accepts[0]
ok("scheme exact", a.scheme === "exact")
ok("network is Solana CAIP-2", a.network === SOLANA_MAINNET)
ok("amount is an atomic-unit STRING", typeof a.amount === "string")
ok("asset is USDC mint", a.asset === USDC_MINT)
ok("payTo present", typeof a.payTo === "string")
ok("maxTimeoutSeconds is a number", typeof a.maxTimeoutSeconds === "number")
ok("body mirrors header (human-readable 402)", JSON.parse(await res.text()).x402Version === 2)

for (const c of checks) console.log(`  ${c.pass ? "ok  " : "FAIL"} ${c.check}`)
console.log(checks.every(c => c.pass) ? "\nSPEC COMPLIANT" : "\nNON-COMPLIANT")
process.exit(checks.every(c => c.pass) ? 0 : 1)
