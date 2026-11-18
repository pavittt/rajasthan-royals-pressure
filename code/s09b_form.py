"""Step 9b. Form check: IPL 2025 and IPL 2026 plain numbers (all balls and RR pressure balls) for every listed player."""
import json, pandas as pd
import config as C
J = json.load(open(f"{C.OUT}/final_lists.json"))
B = {s: pd.read_parquet(f"{C.DATA}/{s}_balls.parquet") for s in ("bat", "bowl")}
def line(side, name, y):
    d = B[side]; key = "striker_name" if side == "bat" else "bowler_name"; d = d[(d[key] == name) & (d.year == y) & (d.fmt == "IPL")]
    p = d[d[f"RRP{C.RR_LEVER_Q}"]]
    if d.match_id.nunique() == 0: return dict(m=0, main="did not play", pres="none")
    if side == "bat":
        f, pf = d[d.faced == 1], p[p.faced == 1]
        return dict(m=d.match_id.nunique(), main=f"{int(d.runs.sum())} runs, SR {100*d.runs.sum()/max(1,len(f)):.0f}",
                    pres=f"SR {100*pf.runs.sum()/len(pf):.0f} on {len(pf)} balls, {int(pf.striker_out.sum())} out" if len(pf) else "none")
    l, pl = d[d.legal == 1], p[p.legal == 1]
    return dict(m=d.match_id.nunique(), main=f"{int(d.bowler_wkt.sum())} wkts, econ {6*d.bowler_runs.sum()/max(1,len(l)):.1f}",
                pres=f"econ {6*p.bowler_runs.sum()/len(pl):.1f} on {len(pl)} balls, {int(p.bowler_wkt.sum())} wkts" if len(pl) else "none")
rows = []
for side, lab in (("bat", "Batting"), ("bowl", "Bowling")):
    seen = set()
    for lst, L in (("Top 10", J[side]["official"]), ("Can buy", J[side]["can_buy"])):
        for r in L:
            a, b = line(side, r["player"], 2025), line(side, r["player"], 2026)
            rows.append(dict(side=lab, list=lst, rank=r.get("buy_rank", r["rank"]) if lst == "Can buy" else r["rank"], player=r["player"],
                             ipl25_matches=a["m"], ipl25=a["main"], ipl25_pressure=a["pres"], ipl26_matches=b["m"], ipl26=b["main"], ipl26_pressure=b["pres"]))
json.dump(rows, open(f"{C.OUT}/form_check.json", "w"))
print(pd.DataFrame(rows).to_string())
