#!/usr/bin/env python3
"""Genesis replay: is there an edge in the first seconds after a pump.fun creation?

Reads ~/crowetrade-genesis/genesis.db (creations + first 15 minutes of trades at
transaction resolution). Replays a 0.1 SOL paper entry at creation + k seconds
using the bonding curve's own arithmetic (constant product on virtual reserves,
1% fee each way), exits by hold or take-profit on the reserves observed at that
moment. Prints a report, writes JSON, optionally POSTs it to the engine.
"""
import json, sqlite3, statistics as st, sys, os, subprocess, datetime, collections
DIR=os.path.expanduser("~/crowetrade-genesis"); DB=f"{DIR}/genesis.db"
FEE=0.01; SIZE=0.1
HOURS=float(os.environ.get("GENESIS_HOURS","24"))
def buy(vsol,vtok,sol):
    s=sol*(1-FEE); out=vtok-(vsol*vtok)/(vsol+s); return out,vsol+s,vtok-out
def sell(vsol,vtok,tokens):
    out=vsol-(vsol*vtok)/(vtok+tokens); return out*(1-FEE)
con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
now=int(datetime.datetime.now().timestamp()*1000); since=now-int(HOURS*3600000)
hb=json.load(open(f"{DIR}/heartbeat.json")) if os.path.exists(f"{DIR}/heartbeat.json") else {}
tokens=con.execute("SELECT * FROM tokens WHERE created_at>=? AND created_at<=? ORDER BY created_at",(since,now-16*60000)).fetchall()
creators=collections.Counter(t["creator"] for t in con.execute("SELECT creator FROM tokens WHERE creator IS NOT NULL"))
ENTRIES=[2,5,10,20,30,60,120]; EXITS=[("hold30",None,30),("hold60",None,60),("hold120",None,120),("hold300",None,300),("tp20/300",20,300),("tp50/300",50,300),("tp30/120",30,120)]
rows=[]; n_with_trades=0; n_no_trades=0
for t in tokens:
    tr=con.execute("SELECT ts, is_buy, sol, tokens, user, vsol, vtok FROM trades WHERE mint=? ORDER BY ts, id",(t["mint"],)).fetchall()
    if not tr: n_no_trades+=1; continue
    n_with_trades+=1
    c0=t["created_at"]; vsol0=t["vsol0"] or 30.0; vtok0=t["vtok0"] or 1_073_000_000.0
    def state_at(ms):
        vs,vt=vsol0,vtok0
        for r in tr:
            if r["ts"]<=ms: vs,vt=r["vsol"],r["vtok"]
            else: break
        return vs,vt
    buyers=set(r["user"] for r in tr if r["is_buy"]); dev=t["creator"]
    dev_sold=any((not r["is_buy"]) and r["user"]==dev for r in tr)
    for k in ENTRIES:
        te=c0+k*1000
        if tr[-1]["ts"]<te-1000 and (tr[-1]["ts"]-c0)<k*1000: pass  # no trades after entry: curve unchanged, exit at same reserves
        vs,vt=state_at(te)
        got,vs1,vt1=buy(vs,vt,SIZE)
        entry_px=vs/vt
        early=[r for r in tr if r["ts"]<=te]
        feats=dict(dev_sol=t["dev_sol"] or 0.0, buys_before=sum(1 for r in early if r["is_buy"]), buyers_before=len(set(r["user"] for r in early if r["is_buy"])),
                   sol_in_before=sum(r["sol"] for r in early if r["is_buy"]), sells_before=sum(1 for r in early if not r["is_buy"]),
                   creator_launches=creators.get(dev,0), dev_sold_before=any((not r["is_buy"]) and r["user"]==dev for r in early), source=t["source"])
        for name,tp,hold in EXITS:
            tx=te+hold*1000; exit_vs,exit_vt=state_at(tx); how="time"
            if tp is not None:
                for r in tr:
                    if r["ts"]<=te: continue
                    if r["ts"]>tx: break
                    if (r["vsol"]/r["vtok"])>=entry_px*(1+tp/100): exit_vs,exit_vt=r["vsol"],r["vtok"]; how="tp"; break
            out=sell(exit_vs,exit_vt,got); ret=(out/SIZE-1)*100
            rows.append(dict(mint=t["mint"],k=k,exit=name,ret=ret,how=how,**feats))
def desc(rs):
    v=[r["ret"] for r in rs]
    if len(v)<10: return None
    return dict(n=len(v),mean=round(st.mean(v),2),exbest=round(st.mean(sorted(v)[:-1]),2),median=round(st.median(v),2),win=round(100*sum(1 for x in v if x>0)/len(v),1),p90=round(sorted(v)[int(0.9*len(v))],1))
report={"generatedAt":now,"windowHours":HOURS,"tokens":len(tokens),"withTrades":n_with_trades,"noTrades":n_no_trades,"collector":hb,"grid":[],"strata":[]}
for k in ENTRIES:
    for name,_,_ in EXITS:
        d=desc([r for r in rows if r["k"]==k and r["exit"]==name])
        if d: report["grid"].append(dict(k=k,exit=name,**d))
best=sorted(report["grid"],key=lambda g:-g["exbest"])[:5]
STRATA=[("dev_sol<0.5",lambda r:r["dev_sol"]<0.5),("dev_sol 0.5-2",lambda r:0.5<=r["dev_sol"]<2),("dev_sol>=2",lambda r:r["dev_sol"]>=2),
        ("buyers_before>=3",lambda r:r["buyers_before"]>=3),("buyers_before==0",lambda r:r["buyers_before"]==0),("sells_before==0",lambda r:r["sells_before"]==0),
        ("creator first launch (this db)",lambda r:r["creator_launches"]<=1),("creator >3 launches (this db)",lambda r:r["creator_launches"]>3),("dev sold before entry",lambda r:r["dev_sold_before"])]
for k in (5,30):
    for name in ("tp20/300","hold60"):
        for label,fn in STRATA:
            d=desc([r for r in rows if r["k"]==k and r["exit"]==name and fn(r)])
            if d: report["strata"].append(dict(k=k,exit=name,stratum=label,**d))
lines=[f"window {HOURS:.0f}h: {len(tokens)} creations, {n_with_trades} with trades, collector creates={hb.get('creates')} trades={hb.get('trades')} reconnects={hb.get('reconnects')}"]
for g in best: lines.append(f"best: enter +{g['k']}s, {g['exit']}: mean {g['mean']:+.1f}% exBest {g['exbest']:+.1f}% win {g['win']}% (n={g['n']})")
base=[g for g in report["grid"] if g["k"]==5 and g["exit"]=="tp20/300"]
if base: lines.append(f"reference: enter +5s, tp20/300: mean {base[0]['mean']:+.1f}% median {base[0]['median']:+.1f}% win {base[0]['win']}% (n={base[0]['n']})")
pos=[s for s in report["strata"] if s["exbest"]>0]
lines.append("strata with positive ex-best: "+("; ".join(f"+{s['k']}s {s['exit']} {s['stratum']} {s['exbest']:+.1f}% (n={s['n']})" for s in pos[:6]) if pos else "none"))
# ---- second-mover thesis (wallet reputation + holder reconstruction), point in time
try:
    sys.path.insert(0, DIR)
    import genesis_wallets
    sm = genesis_wallets.run(con, [dict(t) for t in tokens])
    report["secondMover"] = sm
    lines += sm["lines"]
except Exception as e:
    lines.append(f"second-mover: failed: {e!r}")
report["lines"]=lines
json.dump(report,open(f"{DIR}/report.json","w"),indent=1)
print("\n".join(lines)); print("--- second-mover grid"); [print(f"  {g['rule']:48s} n={g['n']:5d} mean={g['mean']:+7.2f} exBest={g['exbest']:+7.2f} med={g['median']:+7.2f} win={g['win']:5.1f}% p90={g['p90']:+.1f}") for g in report.get("secondMover",{}).get("grid",[])[:12]]; print("--- grid (top 12 by exBest)")
for g in sorted(report["grid"],key=lambda g:-g["exbest"])[:12]: print(f"  +{g['k']:3d}s {g['exit']:9s} n={g['n']:5d} mean={g['mean']:+7.2f} exBest={g['exbest']:+7.2f} med={g['median']:+7.2f} win={g['win']:5.1f}% p90={g['p90']:+.1f}")
print("--- strata (top 10 by exBest)")
for s in sorted(report["strata"],key=lambda s:-s["exbest"])[:10]: print(f"  +{s['k']:3d}s {s['exit']:9s} {s['stratum']:32s} n={s['n']:5d} mean={s['mean']:+7.2f} exBest={s['exbest']:+7.2f} win={s['win']:5.1f}%")
if "--post" in sys.argv:
    env={}
    for line in open(os.path.expanduser("~/Projects/crowetrade/engine/.dev.vars.tokens")):
        if "=" in line: k,v=line.strip().split("=",1); env[k]=v
    r=subprocess.run(["curl","-s","-m","60","-X","POST","-A","CroweTrade-genesis/1.0 (curl)","-H","Authorization: Bearer "+env["ENGINE_ADMIN_TOKEN"],"-H","Content-Type: application/json","--data-binary","@"+f"{DIR}/report.json","https://crowetrade-engine.yellow-block-3adc.workers.dev/api/genesis"],capture_output=True,text=True)
    print("POST ->",r.stdout.strip()[:200])
