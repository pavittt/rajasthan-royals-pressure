"""Step 3. Pressure definition and evidence that it separates pressure from ordinary situations.
PRESSURE ball = Leverage Index >= the 75th percentile of LI across all 2022-2026 IPL + top-nine T20I balls.
Inputs : data/balls_wp.parquet   Outputs: data/pressure_threshold.json, out/pressure_vs_ordinary.csv/.md"""
import json, numpy as np, pandas as pd
import config as C

b = pd.read_parquet(f"{C.DATA}/balls_wp.parquet")
u = b[b.universe].copy()
thr = {str(q): float(u.LI.quantile(q)) for q in (0.65, 0.75, 0.85)}
json.dump(thr, open(f"{C.DATA}/pressure_threshold.json", "w"), indent=1)
u["pressure"] = u.LI >= thr[str(C.PRESSURE_Q)]
u["absWPA"] = u.wpa_bat.abs()
u["close"] = (u.wp_pre - 0.5).abs() < 0.2
u["wkts_in_hand"] = 10 - u.wk_before

def prof(d):
    f = d[d.faced]
    return pd.Series({
        "balls": len(d), "share_of_balls_%": 100 * len(d) / len(u),
        "share_of_all_win_prob_movement_%": 100 * d.absWPA.sum() / u.absWPA.sum(),
        "mean_leverage": d.LI.mean(), "mean_abs_win_prob_change_per_ball": d.absWPA.mean(),
        "win_prob_within_30_70_%": 100 * d.close.mean(),
        "powerplay_%": 100 * (d.phase == "powerplay").mean(), "middle_%": 100 * (d.phase == "middle").mean(),
        "death_%": 100 * (d.phase == "death").mean(), "second_innings_%": 100 * (d.innings == 2).mean(),
        "mean_wickets_in_hand": d.wkts_in_hand.mean(), "mean_balls_left": d.balls_left.mean(),
        "chase_mean_required_rate": d.loc[d.innings == 2, "rrr"].clip(upper=36).mean(),
        "runs_per_ball_off_bat": f.runs.mean(), "dot_%": 100 * (f.runs.eq(0) & ~f.striker_out).mean(),
        "boundary_%": 100 * (f.is_four | f.is_six).mean(), "wicket_%": 100 * d.wkt.mean(),
    })
tab = pd.DataFrame({"pressure": prof(u[u.pressure]), "ordinary": prof(u[~u.pressure])})
# within death overs only: pressure is not just "the death overs"
de = u[u.phase == "death"]
tab["death_overs_pressure"] = prof(de[de.pressure]); tab["death_overs_ordinary"] = prof(de[~de.pressure])
tab.round(3).to_csv(f"{C.OUT}/pressure_vs_ordinary.csv")

# what drives leverage: mean pressure share by situation
u["rrr_band"] = pd.cut(u.rrr, [-99, 6, 8, 10, 12, 99], labels=["<6", "6-8", "8-10", "10-12", "12+"])
u["wih_band"] = pd.cut(u.wkts_in_hand, [0, 3, 6, 10], labels=["1-3", "4-6", "7-10"])
chase = u[u.innings == 2].pivot_table(index="rrr_band", columns="wih_band", values="pressure", aggfunc="mean", observed=True)
byover = u.groupby(["innings", "over"]).pressure.mean().unstack(0)
with open(f"{C.OUT}/pressure_vs_ordinary.md", "w") as fh:
    fh.write(f"# Pressure vs ordinary balls (2022-2026, IPL + top-nine T20I, {len(u)} balls)\n\n")
    fh.write(f"Pressure ball = Leverage Index >= {thr[str(C.PRESSURE_Q)]:.3f} (75th percentile). Leverage Index = expected "
             "absolute change in win probability from the next ball, divided by the league average.\n\n")
    fh.write(tab.round(2).to_markdown() + "\n\n")
    fh.write("## Share of chase balls that are pressure balls, by required rate (rows) and wickets in hand (columns)\n\n")
    fh.write((100 * chase).round(0).to_markdown() + "\n\n")
    fh.write("## Share of balls that are pressure balls, by over and innings\n\n" + (100 * byover).round(0).to_markdown() + "\n")
print(tab.round(2).to_string()); print((100 * chase).round(0).to_string())
