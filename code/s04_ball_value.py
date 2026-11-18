"""Step 4. One ball-value measure for both sides: context-adjusted run value (RV), in runs.
  batter : RV = (runs off bat - expected) - wicket_cost * (out - expected outs)
  bowler : RV = (expected - runs conceded) + wicket_cost * (wicket - expected wickets)
Expected values come from a league table fitted on 2022-2026 IPL + top-nine T20I balls:
format x innings x over x wickets lost (4 bands) x required-rate band (chases), smoothed to its parent cell,
then rescaled per format AND YEAR (so 2025 and 2026 each have their own league level).
wicket_cost = runs-equivalent of a wicket implied by the win-probability model at that innings/over/wickets band.
Opponent adjustment: each ball also gets the opponent's usual shrunk quality added back (facing a bowler who
usually concedes 0.1 runs/ball below expectation is worth +0.1). Positive RV = better than the league.
Inputs : data/balls_wp.parquet, data/pressure_threshold.json  Outputs: data/bat_balls.parquet, data/bowl_balls.parquet,
out/wicket_cost.csv"""
import json, numpy as np, pandas as pd
import config as C

b = pd.read_parquet(f"{C.DATA}/balls_wp.parquet")
b = b[b.universe].copy()
thr = json.load(open(f"{C.DATA}/pressure_threshold.json"))
for q, v in thr.items(): b[f"P{q}"] = b.LI >= v
b["wkb"] = pd.cut(b.wk_before, [-1, 1, 3, 5, 10], labels=["0-1", "2-3", "4-5", "6+"]).astype(str)
b["rrb"] = np.where(b.innings == 1, "set", pd.cut(b.rrr, [-99, 8, 10, 12, 999], labels=["<8", "8-10", "10-12", "12+"]).astype(str))

# ---- wicket cost in runs, implied by the win-probability model:
# (WP after a dot - WP after a wicket) / (WP gained per run, from WP after a four vs after a dot) at that state,
# summarised as the median per innings x over x wickets-lost band (parent: innings x phase x band if < 50 balls).
L = b[b.legal].copy()
per_run = (L.wp_after_4 - L.wp_after_0) / 4
L["cost"] = ((L.wp_after_0 - L.wp_after_W) / per_run.where(per_run > 1e-4)).clip(0, 40)
L = L[L.cost.notna()]
cell = L.groupby(["innings", "over", "wkb"]).cost.agg(["median", "count"]).reset_index()
parent = L.groupby(["innings", "phase", "wkb"]).cost.median().rename("pmed").reset_index()
b = b.merge(cell, on=["innings", "over", "wkb"], how="left").merge(parent, on=["innings", "phase", "wkb"], how="left")
b["wcost"] = np.where(b["count"].fillna(0) >= 50, b["median"], b["pmed"])
b["wcost"] = b.wcost.fillna(L.cost.median()).clip(lower=0.5)
b = b.drop(columns=["median", "count", "pmed"])
re_ = b.groupby(["innings", "over", "wkb"]).wcost.first().unstack()
re_.to_csv(f"{C.OUT}/wicket_cost.csv")

def expect(d, ycol, k=200):
    """smoothed league expectation of ycol: cell -> parent (phase-level) -> grandparent, then per format-year scale"""
    d = d.copy()
    gp = d.groupby(["fmt", "innings", "phase"])[ycol].transform("mean")
    pg = d.groupby(["fmt", "innings", "phase", "wkb"])[ycol]
    par = (pg.transform("sum") + k * gp) / (pg.transform("count") + k)
    cg = d.groupby(["fmt", "innings", "over", "wkb", "rrb"])[ycol]
    e = (cg.transform("sum") + k * par) / (cg.transform("count") + k)
    scale = d.groupby(["fmt", "year"])[ycol].transform("sum") / e.groupby([d.fmt, d.year]).transform("sum")
    return e * scale

bat = b[b.faced].copy()
bat["out"] = bat.striker_out.astype(float); bat["r"] = bat.runs.astype(float)
bat["E_r"] = expect(bat, "r"); bat["E_o"] = expect(bat, "out")
bat["RV0"] = (bat.r - bat.E_r) - bat.wcost * (bat.out - bat.E_o)
bowl = b.copy()
bowl["w"] = bowl.bowler_wkt.astype(float); bowl["r"] = bowl.bowler_runs.astype(float)
bowl["E_r"] = expect(bowl, "r"); bowl["E_o"] = expect(bowl, "w")
bowl["RV0"] = (bowl.E_r - bowl.r) + bowl.wcost * (bowl.w - bowl.E_o)

def shrunk_mean(d, key):
    """empirical-Bayes shrunk mean RV per player over 2025-26 universe balls (prior mean 0 = league);
    between-player variance by marginal maximum likelihood on players with >= 120 balls"""
    from lib import eb_fit
    d = d[d.year.isin(C.SCORE_YEARS)]
    g = d.groupby(key).RV0.agg(["sum", "count"]); s2 = d.RV0.var(); m = g["sum"] / g["count"]
    big = g["count"] >= 120
    _, tau2 = eb_fit(m[big], s2 / g["count"][big])
    return (m * tau2 / (tau2 + s2 / g["count"])).to_dict()
q_bowl = shrunk_mean(bowl, "bowler_id"); q_bat = shrunk_mean(bat, "striker_id")
bat["opp_q"] = bat.bowler_id.map(q_bowl).fillna(0.0)      # bowler's usual quality (positive = tough bowler)
bowl["opp_q"] = bowl.striker_id.map(q_bat).fillna(0.0)    # batter's usual quality (positive = tough batter)
for x in (bat, bowl):   # scoring years only; centred within format-year so the league mean stays 0
    x.loc[~x.year.isin(C.SCORE_YEARS), "opp_q"] = 0.0
    sc = x.year.isin(C.SCORE_YEARS)
    x.loc[sc, "opp_q"] -= x[sc].groupby(["fmt", "year"]).opp_q.transform("mean")
    x["RV"] = x.RV0 + x.opp_q

cols = ["match_id", "innings", "innings_ball_num", "fmt", "year", "match_date", "batting_team_id", "bowling_team_id",
        "striker_id", "striker_name", "bowler_id", "bowler_name", "phase", "over", "LI", "wp_pre", "wpa_bat",
        "legal", "faced", "runs", "bowler_runs", "striker_out", "bowler_wkt", "wcost", "E_r", "E_o", "RV0", "opp_q", "RV",
        "is_four", "is_six", "rrr", "wk_before", "stage", "comp_name"] + [f"P{q}" for q in thr]
bat[cols].to_parquet(f"{C.DATA}/bat_balls.parquet"); bowl[cols].to_parquet(f"{C.DATA}/bowl_balls.parquet")
for nm, d in (("bat", bat), ("bowl", bowl)):
    print(nm, "opp_q sd per ball %.4f" % d.opp_q.std(), len(d), "mean RV0 %.4f RV %.4f sd %.3f" % (d.RV0.mean(), d.RV.mean(), d.RV.std()),
          d.groupby(["fmt", "year"]).RV0.mean().round(4).to_dict())
print(re_.round(1).iloc[::3].to_string())
