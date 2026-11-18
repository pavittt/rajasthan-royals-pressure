"""Step 5. Official ratings: one locked method, identical for batters and bowlers.
Inputs : data/bat_balls.parquet, data/bowl_balls.parquet, data/xi.parquet
Outputs: out/ratings_bat_all.csv, out/ratings_bowl_all.csv (every eligible player),
         data/official_bat.csv, data/official_bowl.csv (top 10 before intervals are attached in step 6),
         out/eligibility_bat.csv, out/eligibility_bowl.csv, out/model_parameters.json"""
import json, numpy as np, pandas as pd
import config as C, lib

xi = pd.read_parquet(f"{C.DATA}/xi.parquet")
def team_2026(pid):
    x = xi[(xi.player_id == pid) & (xi.year == 2026) & xi.universe]
    ipl = x[x.fmt == "IPL"].team_name.mode(); intl = xi[(xi.player_id == pid) & (xi.fmt == "T20I")].team_name.mode()
    return (ipl.iat[0] if len(ipl) else ""), (intl.iat[0] if len(intl) else "")

def describe(side, d, res):
    """plain cricket numbers behind each rating (unweighted 2025-26 balls) + match-level consistency"""
    rows = []
    for pid in res.index:
        x = d[d.pid == pid]; p = x[x.pres]; o = x[~x.pres]
        if side == "bat":
            f = lambda z: dict(balls=len(z), SR=100 * z.runs.sum() / max(len(z), 1), outs=int(z.striker_out.sum()),
                               avg=z.runs.sum() / max(z.striker_out.sum(), 1))
        else:
            f = lambda z: dict(balls=int(z.legal.sum()), econ=6 * z.bowler_runs.sum() / max(z.legal.sum(), 1),
                               wkts=int(z.bowler_wkt.sum()), balls_per_wkt=z.legal.sum() / max(z.bowler_wkt.sum(), 1))
        fp, fo = f(p), f(o)
        r = {"pid": pid, **{f"pressure_{k}": v for k, v in fp.items()}, **{f"usual_{k}": v for k, v in fo.items()}}
        r["pressure_balls_ipl_share_%"] = 100 * (p.fmt == "IPL").mean()
        for y in (2025, 2026):
            py = p[p.year == y]; r[f"pressure_RV100_{y}"] = 100 * py.v.mean() if len(py) >= 20 else np.nan; r[f"pressure_balls_{y}"] = len(py)
        # consistency: matches with >= 6 pressure balls; share where his pressure value beat the league (RV>0)
        # and share where it beat his own usual per-ball level
        mu_usual = o.v.mean()
        pmx = p.groupby("match_id").v.agg(["sum", "size"]); pmx = pmx[pmx["size"] >= 6]
        r["pressure_matches_6plus"] = len(pmx)
        r["pct_matches_above_league"] = 100 * (pmx["sum"] > 0).mean() if len(pmx) else np.nan
        r["pct_matches_above_own_usual"] = 100 * (pmx["sum"] / pmx["size"] > mu_usual).mean() if len(pmx) else np.nan
        r["ipl_team_2026"], r["country"] = team_2026(pid)
        rows.append(r)
    return pd.DataFrame(rows).set_index("pid")

params = {}
for side in ("bat", "bowl"):
    res, aux = lib.rate(side, return_all=True)
    desc = describe(side, aux["d"], res)
    full = res.join(desc)
    full.to_csv(f"{C.OUT}/ratings_{side}_all.csv")
    rule = aux["rule"].copy(); rule.insert(0, "player", aux["names"].reindex(rule.index)); rule["eligible"] = rule.drop(columns="player").all(axis=1)
    rule.to_csv(f"{C.OUT}/eligibility_{side}.csv")
    full.head(10).to_csv(f"{C.DATA}/official_{side}.csv")
    params[side] = aux["info"]
    print(side, aux["info"]["eligible"], "eligible")
    cols = ["player", "ipl_team_2026", "country", "kP", "nP", "mP", "mU", "P", "D", "SCORE"]
    print(full[cols].head(15).round(2).to_string())
params["config"] = {"PRESSURE_Q": C.PRESSURE_Q, "WEIGHTS": {f"{a}_{b}": w for (a, b), w in C.WEIGHTS.items()},
                    "ELIG": C.ELIG, "SCORE": "P + D (runs per 100 balls vs league expectation)", "HYPER_YEARS": C.HYPER_YEARS}
json.dump(params, open(f"{C.OUT}/model_parameters.json", "w"), indent=1, default=float)
