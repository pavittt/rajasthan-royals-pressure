"""Step 9 (RR theory). Final lists and everything the dashboard and report show.
Inputs : out/ratings_*_all.csv, out/official_top10_*.csv, out/bootstrap_ranks_*.csv, out/rr_moments_kept.csv,
         out/sensitivity.csv, data/*_balls.parquet, inputs/squads2027.csv + pool2027.csv (hand-entered from news),
         /home/claude/work/grid_all.csv (matchup audit from the earlier stage, shown as background only)
Outputs: out/final_lists.json (single source for the page and the PDF), out/can_buy_top10_{bat,bowl}.csv,
         out/rr_squad_{bat,bowl}.csv"""
import json, numpy as np, pandas as pd
import config as C
sq = pd.read_csv(f"{C.INPUTS}/squads2027.csv"); pool = pd.read_csv(f"{C.INPUTS}/pool2027.csv")
rr_ids = set(sq[sq.team_name == "Rajasthan Royals"].player_id.dropna().astype(int))
ret = dict(zip(sq.player_id.dropna().astype(int), sq.team_name))
rel = dict(zip(pool.player_id, pool.released_by))
def status(pid):
    pid = int(pid)
    if pid in rr_ids: return "RR squad"
    if pid in ret: return f"Retained by {ret[pid]}"
    if pid in rel: return f"Released by {rel[pid]} (auction pool)"
    return "Not on any 2027 list (registration unknown)"
def clean(df):
    return json.loads(df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_json(orient="records"))
OUT = {}
for side in ("bat", "bowl"):
    R = pd.read_csv(f"{C.OUT}/ratings_{side}_all.csv"); B = pd.read_csv(f"{C.OUT}/bootstrap_ranks_{side}.csv").rename(columns={"Unnamed: 0": "pid"})
    R = R.merge(B[["pid", "rank_p05", "rank_p95", "p_top10"]], on="pid", how="left")
    R["status"] = R.pid.map(status); R["can_buy"] = ~R.status.str.startswith(("RR squad", "Retained"))
    if side == "bat":
        R["pressure_stat"], R["usual_stat"] = R.pressure_SR, R.usual_SR
        R["pressure_stat2"], R["usual_stat2"] = R.pressure_balls / R.pressure_outs.clip(lower=1), R.usual_balls / R.usual_outs.clip(lower=1)
    else:
        R["pressure_stat"], R["usual_stat"] = R.pressure_econ, R.usual_econ
        R["pressure_stat2"], R["usual_stat2"] = R.pressure_balls_per_wkt, R.usual_balls_per_wkt
    keep = ["rank", "player", "ipl_team_2026", "country", "status", "SCORE", "P", "D", "mP", "mU", "pressure_balls", "nP", "pressure_balls_ipl_share_%",
            "pressure_stat", "usual_stat", "pressure_stat2", "usual_stat2", "pressure_RV100_2025", "pressure_RV100_2026", "rank_p05", "rank_p95", "p_top10",
            "pct_matches_above_own_usual"]
    R = R.sort_values("SCORE", ascending=False)
    off = R.head(10)[keep].copy()
    buy = R[R.can_buy].head(10)[keep].copy(); buy.insert(0, "buy_rank", range(1, len(buy) + 1))
    rr = R[R.status == "RR squad"][keep].copy()
    buy.to_csv(f"{C.OUT}/can_buy_top10_{side}.csv", index=False); rr.to_csv(f"{C.OUT}/rr_squad_{side}.csv", index=False)
    OUT[side] = dict(official=clean(off.round(2)), can_buy=clean(buy.round(2)), rr=clean(rr.round(2)), eligible=len(R),
                     all=clean(R[keep].round(2)))
    d = pd.read_parquet(f"{C.DATA}/{side}_balls.parquet"); d = d[d.year.isin(C.SCORE_YEARS)]
    pres = d[f"RRP{C.RR_LEVER_Q}"]
    OUT[side]["pressure"] = dict(share_rr_moments=float(d.RR48.mean()), share_pressure=float(pres.mean()),
                                 swing_ratio_rr_moments=float(d[d.RR48].wpa_bat.abs().mean() / d[~d.RR48].wpa_bat.abs().mean()),
                                 swing_ratio_pressure=float(d[pres].wpa_bat.abs().mean() / d[~pres].wpa_bat.abs().mean()),
                                 share_of_swing=float(d[pres].wpa_bat.abs().sum() / d.wpa_bat.abs().sum()),
                                 close_share_pressure=float(d[pres].wp_pre.between(0.3, 0.7).mean()), close_share_other=float(d[~pres].wp_pre.between(0.3, 0.7).mean()))
# watch list: available players under the same method with a lower evidence bar (25 effective pressure balls, 12 IPL)
import lib
for side in ("bat", "bowl"):
    res = lib.rate(side, lib.default_cfg(ELIG_OVERRIDE=dict(min_eff_pressure=25, min_ipl_pressure=12)))[0]
    res["status"] = [status(p) for p in res.index]
    res = res[~res.status.str.startswith(("RR squad", "Retained"))].head(10).reset_index()
    res["evidence"] = np.where(res.nP >= C.ELIG["min_eff_pressure"], "full", "thin (25-40 effective pressure balls)")
    OUT[side]["watch"] = clean(res[["player", "status", "SCORE", "P", "D", "mP", "mU", "kP", "nP", "evidence"]].round(2))
M = pd.read_csv(f"{C.OUT}/rr_moments_kept.csv"); OUT["moments"] = clean(M.round(2))
S = pd.read_csv(f"{C.OUT}/sensitivity.csv")
summ = []
for side in ("bat", "bowl"):
    off = set(S[(S.side == side) & (S.scenario == "OFFICIAL")].player)
    for sc, g in S[S.side == side].groupby("scenario", sort=False):
        summ.append(dict(side=side, scenario=sc, overlap=len(off & set(g.player)), no1=g.sort_values("rank").player.iat[0]))
OUT["sensitivity"] = summ
G = pd.read_csv("/home/claude/work/grid_all.csv"); G = G[G.grid.isin(["matchup", "role_phase"])]
OUT["grid"] = clean(G[["grid", "side", "a", "b", "c", "rr_balls", "rr_raw_wins", "p_rr_below", "champ_wins", "luck_p", "rr_sr", "ch_sr", "lg_sr", "rr_econ", "ch_econ", "lg_econ",
                       "rr_bpd", "lg_bpd", "rr_bpw", "lg_bpw", "verdict", "fdr_ok"]].round(3))
rel = open(f"{C.OUT}/reliability.md").read(); OUT["reliability_md"] = rel
json.dump(OUT, open(f"{C.OUT}/final_lists.json", "w"))
for side in ("bat", "bowl"):
    print(side, "official:", [r["player"] for r in OUT[side]["official"]])
    print(side, "can buy:", [(r["player"], r["status"][:30]) for r in OUT[side]["can_buy"]])
    print(side, "watch:", [(r["player"], round(r["SCORE"], 1), r["evidence"][:4]) for r in OUT[side]["watch"]])
