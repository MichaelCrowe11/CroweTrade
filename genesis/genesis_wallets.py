"""Second-mover replay: follow the informed wallets instead of the launch.

Every passive replay so far loses because a passive entrant is the exit
liquidity for whoever was first. This asks the other question: build a
reputation per wallet from what happened AFTER its buys on earlier launches,
reconstruct the holder table of each new launch from its own trades, and
enter when the informed cohort or a healthy structure shows up.

Point in time by construction: launches are processed in creation order and
a wallet's reputation for launch N is built from launches 1..N-1 only. The
global hit rate used for shrinkage is the running one. Nothing here sees the
future of the launch it is scoring.

Optimism, stated: we enter at the reserves immediately after the trade we
follow. A real follower lands one to three seconds later with other trades in
between. Fills use the curve's own arithmetic (constant product on virtual
reserves, 1% fee each way), so entry and exit are exact for a live curve.
"""
import collections, statistics as st

FEE = 0.01
SIZE = 0.1
HIT_MULT = 1.3        # a wallet "hit" when the launch printed +30% within 120s of its first buy
SHRINK = 5.0          # pseudo-observations pulled toward the running base rate
# (name, take-profit %, stop %, hold s). A stop on the curve is executable at the
# curve price at any moment, which the minute engine's stops never were.
EXITS = [("tp20/60", 20, None, 60), ("tp30/120", 30, None, 120), ("hold60", None, None, 60), ("tp50/300", 50, None, 300),
         ("tp20/sl10/60", 20, 10, 60), ("tp30/sl15/120", 30, 15, 120), ("tp50/sl20/300", 50, 20, 300), ("sl15/120", None, 15, 120)]
LATENCY_MS = 2000     # a real follower lands after the trade it follows; enter at the curve state this long later

def buy(vs, vt, sol):
    s = sol * (1 - FEE); out = vt - (vs * vt) / (vs + s); return out, vs + s, vt - out
def sell(vs, vt, tok):
    out = vs - (vs * vt) / (vt + tok); return out * (1 - FEE)

class Wallet:
    __slots__ = ("n", "hits", "early", "fast_sells", "trades", "net_sol")
    def __init__(self): self.n = 0; self.hits = 0; self.early = 0; self.fast_sells = 0; self.trades = 0; self.net_sol = 0.0
    def hit_rate(self, base): return (self.hits + SHRINK * base) / (self.n + SHRINK)

def informed(w, base, min_n=5, margin=0.10):
    """Informed = beats the running base hit rate by `margin` after shrinkage."""
    if w is None or w.n < min_n: return False
    if w.hit_rate(base) < base + margin: return False
    if w.fast_sells / max(1, w.n) >= 0.5: return False          # flips within ten seconds: a sniper
    if w.trades / max(1, w.n) >= 20: return False                # volume bot
    return True

def price_after(tr, i): return tr[i]["vsol"] / tr[i]["vtok"]

def exit_return(tr, i, tp, sl, hold, latency_ms=LATENCY_MS):
    """Follow trade i: enter at the curve state `latency_ms` after it; return net % under (tp, sl, hold)."""
    t_follow = tr[i]["ts"] + latency_ms
    j = i
    while j + 1 < len(tr) and tr[j + 1]["ts"] <= t_follow: j += 1
    vs, vt = tr[j]["vsol"], tr[j]["vtok"]
    got, _, _ = buy(vs, vt, SIZE)
    entry_px = vs / vt
    t0 = t_follow
    last_vs, last_vt = vs, vt
    for r in tr[j + 1:]:
        if r["ts"] > t0 + hold * 1000: break
        last_vs, last_vt = r["vsol"], r["vtok"]
        px = r["vsol"] / r["vtok"]
        if tp is not None and px >= entry_px * (1 + tp / 100):
            return (sell(r["vsol"], r["vtok"], got) / SIZE - 1) * 100
        if sl is not None and px <= entry_px * (1 - sl / 100):
            return (sell(r["vsol"], r["vtok"], got) / SIZE - 1) * 100
    return (sell(last_vs, last_vt, got) / SIZE - 1) * 100

def holders_at(tr, t, dev):
    """Holder table from trades at or before t."""
    bal = collections.defaultdict(float); buyers = set(); sells = 0; dev_sold = False
    for r in tr:
        if r["ts"] > t: break
        if r["is_buy"]: bal[r["user"]] += r["tokens"]; buyers.add(r["user"])
        else:
            bal[r["user"]] -= r["tokens"]; sells += 1
            if r["user"] == dev: dev_sold = True
    held = {u: b for u, b in bal.items() if b > 0}
    total = sum(held.values())
    top = max(held.values()) / total if total > 0 else 0.0
    return dict(buyers=len(buyers), holders=len(held), top_share=top, sells=sells, dev_sold=dev_sold)

def run(con, tokens):
    """tokens: rows with mint, created_at, creator, dev_tokens; in creation order."""
    reps = {}
    base_hits = 0; base_n = 0
    rules = collections.defaultdict(list)     # rule -> list of net returns
    diag = dict(tokens=0, with_trades=0, informed_first_buys=0, s2_entries=0, s3_entries=0, first_informed_lag_s=[])
    for t in sorted(tokens, key=lambda x: x["created_at"]):
        tr = con.execute("SELECT ts, is_buy, sol, tokens, user, vsol, vtok FROM trades WHERE mint=? ORDER BY ts, id", (t["mint"],)).fetchall()
        tr = [dict(r) for r in tr]
        diag["tokens"] += 1
        if not tr: continue
        diag["with_trades"] += 1
        c0 = t["created_at"]; dev = t["creator"]
        base = base_hits / base_n if base_n else 0.05
        # ---- second-mover entries on THIS launch, using reputations from earlier launches
        seen_informed = []
        s1_done = s2_done = False
        for i, r in enumerate(tr):
            if not r["is_buy"] or r["user"] == dev: continue
            w = reps.get(r["user"])
            if not informed(w, base): continue
            if r["user"] not in seen_informed: seen_informed.append(r["user"])
            if not s1_done:
                s1_done = True; diag["informed_first_buys"] += 1; diag["first_informed_lag_s"].append((r["ts"] - c0) / 1000)
                for name, tp, sl, hold in EXITS: rules[f"S1 follow first informed buyer {name}"].append(exit_return(tr, i, tp, sl, hold))
                if informed(w, base, min_n=10, margin=0.20):
                    for name, tp, sl, hold in EXITS: rules[f"S1s follow strict-informed buyer {name}"].append(exit_return(tr, i, tp, sl, hold))
            if not s2_done and len(seen_informed) >= 2 and r["ts"] <= c0 + 60_000:
                s2_done = True; diag["s2_entries"] += 1
                for name, tp, sl, hold in EXITS: rules[f"S2 two informed buyers in 60s {name}"].append(exit_return(tr, i, tp, sl, hold))
        # ---- structure entry at +10s
        h = holders_at(tr, c0 + 10_000, dev)
        if h["buyers"] >= 5 and h["top_share"] <= 0.2 and h["sells"] == 0 and not h["dev_sold"]:
            idx = max((i for i, r in enumerate(tr) if r["ts"] <= c0 + 10_000), default=None)
            if idx is not None:
                diag["s3_entries"] += 1
                for name, tp, sl, hold in EXITS: rules[f"S3 healthy structure at +10s {name}"].append(exit_return(tr, idx, tp, sl, hold, latency_ms=0))
        # ---- update reputations from what happened on this launch (visible to LATER launches only)
        first_buy = {}; first_sell = {}; per_wallet_trades = collections.Counter(); net = collections.defaultdict(float)
        for i, r in enumerate(tr):
            per_wallet_trades[r["user"]] += 1
            net[r["user"]] += (-r["sol"] if r["is_buy"] else r["sol"])
            if r["is_buy"]: first_buy.setdefault(r["user"], i)
            else: first_sell.setdefault(r["user"], i)
        for u, i in first_buy.items():
            if u == dev: continue
            w = reps.setdefault(u, Wallet())
            p = price_after(tr, i); t_buy = tr[i]["ts"]
            mx = max((r["vsol"] / r["vtok"] for r in tr[i + 1:] if r["ts"] <= t_buy + 120_000), default=p)
            hit = mx >= HIT_MULT * p
            w.n += 1; w.hits += hit; w.trades += per_wallet_trades[u]; w.net_sol += net[u]
            if t_buy - c0 <= 10_000: w.early += 1
            j = first_sell.get(u)
            if j is not None and tr[j]["ts"] - t_buy <= 10_000: w.fast_sells += 1
            base_hits += hit; base_n += 1
    # ---- summarise
    def desc(v):
        if len(v) < 10: return None
        s = sorted(v)
        return dict(n=len(v), mean=round(st.mean(v), 2), exbest=round(st.mean(s[:-1]), 2), median=round(st.median(v), 2),
                    win=round(100 * sum(1 for x in v if x > 0) / len(v), 1), p90=round(s[int(0.9 * len(s))], 1))
    grid = []
    for rule, v in rules.items():
        d = desc(v)
        if d: grid.append(dict(rule=rule, **d))
    grid.sort(key=lambda g: -g["exbest"])
    wallets = list(reps.values())
    seasoned = [w for w in wallets if w.n >= 5]
    base = base_hits / base_n if base_n else None
    informed_n = sum(1 for w in seasoned if informed(w, base or 0.05))
    strict_n = sum(1 for w in seasoned if informed(w, base or 0.05, min_n=10, margin=0.20))
    hr = sorted(w.hit_rate(base or 0.05) for w in seasoned)
    diag["wallets_strict"] = strict_n
    diag["hit_rate_quartiles"] = [round(hr[int(q * (len(hr) - 1))], 3) for q in (0.25, 0.5, 0.75, 0.9)] if hr else None
    diag["wallets_seen"] = len(wallets); diag["wallets_seasoned"] = len(seasoned); diag["wallets_informed"] = informed_n
    diag["base_hit_rate"] = round(base, 4) if base is not None else None
    diag["first_informed_lag_s"] = round(st.median(diag["first_informed_lag_s"]), 1) if diag["first_informed_lag_s"] else None
    lines = [f"second-mover: {diag['wallets_seen']} wallets seen, {diag['wallets_seasoned']} with 5+ prior launches, {informed_n} informed (base+10%), {strict_n} strict (base+20%, 10+ launches); base hit rate {diag['base_hit_rate']}, seasoned hit-rate quartiles {diag['hit_rate_quartiles']}; follower latency {LATENCY_MS} ms"]
    for g in grid[:4]: lines.append(f"second-mover {g['rule']}: mean {g['mean']:+.1f}% exBest {g['exbest']:+.1f}% win {g['win']}% (n={g['n']})")
    if not grid: lines.append("second-mover: no rule reached 10 entries yet")
    return dict(diag=diag, grid=grid, lines=lines)
