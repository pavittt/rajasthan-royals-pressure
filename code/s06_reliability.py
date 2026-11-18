"""Step 6. Reliability of the FINAL score (not of an older definition).
 (a) split-half by match: all 2025-26 matches randomly split into two halves (100 repeats); every eligible player's
     SCORE recomputed on each half with the same locked hyper-parameters; correlation between halves.
 (b) 2025 vs 2026: SCORE from 2025 balls only vs from 2026 balls only.
 (c) bootstrap: matches resampled with replacement within format-year (1000 repeats) -> rank intervals, P(top 10).
Eligibility is held fixed at the official set. Same code for both sides.
Outputs: out/official_top10_bat.csv, out/official_top10_bowl.csv, out/reliability.md, out/bootstrap_ranks_{side}.csv"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import config as C, lib

rng = np.random.default_rng(C.SEED)
md = []
md.append("# Reliability of the final scores\n")
md.append("All numbers below recompute the FINAL score (P + D, locked method) on resampled or split data, holding the "
          "eligible set and the shrinkage parameters fixed.\n")
for side in ("bat", "bowl"):
    cfg = lib.default_cfg()
    res, aux = lib.rate(side, cfg, return_all=True)
    pm, var, hp = aux["pm"], aux["var"], aux["hp"]
    pids = list(res.index)
    mt = pm[["match_id"]].drop_duplicates().merge(aux["d"][["match_id", "fmt", "year"]].drop_duplicates(), on="match_id")
    mids = mt.match_id.values
    S = lib.sparse_stats(pm, pids, mids)
    full = lib.scores_from_counts(S, np.ones(len(mids)), var, hp, cfg)
    assert np.allclose(full["SCORE"], res.SCORE.values), "vectorised score must equal official score"

    # (a) split-half
    rows = []
    for _ in range(C.N_SPLIT):
        h = rng.random(len(mids)) < 0.5
        a = lib.scores_from_counts(S, h.astype(float), var, hp, cfg); b = lib.scores_from_counts(S, (~h).astype(float), var, hp, cfg)
        r = {}
        for k in ("SCORE", "P", "U", "D", "mP", "mD"):
            r[k] = spearmanr(a[k], b[k])[0]
        top_a = set(np.argsort(-a["SCORE"])[:10]); top_b = set(np.argsort(-b["SCORE"])[:10]); r["top10_overlap"] = len(top_a & top_b)
        rows.append(r)
    sh = pd.DataFrame(rows).mean()
    sb = lambda r: 2 * r / (1 + r) if r > -1 else np.nan

    # (b) 2025 vs 2026
    y25 = (mt.year.values == 2025).astype(float); y26 = (mt.year.values == 2026).astype(float)
    a = lib.scores_from_counts(S, y25, var, hp, cfg); b = lib.scores_from_counts(S, y26, var, hp, cfg)
    nP25 = np.asarray(S["nP"] @ y25).ravel(); nP26 = np.asarray(S["nP"] @ y26).ravel()
    both = (nP25 >= 30) & (nP26 >= 30)
    yr = {k: spearmanr(a[k][both], b[k][both])[0] for k in ("SCORE", "P", "U", "D")}

    # (c) bootstrap ranks
    groups = [np.where((mt.fmt.values == f) & (mt.year.values == y))[0] for f, y in mt[["fmt", "year"]].drop_duplicates().values]
    R = np.zeros((C.N_BOOT, len(pids)), dtype=int); SC = np.zeros((C.N_BOOT, len(pids))); top10_overlap = []
    off10 = set(range(10))
    for i in range(C.N_BOOT):
        c = np.zeros(len(mids))
        for g in groups: np.add.at(c, rng.choice(g, len(g), replace=True), 1)
        sc = lib.scores_from_counts(S, c, var, hp, cfg)["SCORE"]
        rk = np.empty(len(pids), int); rk[np.argsort(-sc)] = np.arange(1, len(pids) + 1); R[i] = rk; SC[i] = sc
        top10_overlap.append(len(off10 & set(np.where(rk <= 10)[0])))
    bt = pd.DataFrame({"player": res.player.values, "rank": res["rank"].values,
                       "rank_p05": np.percentile(R, 5, axis=0).astype(int), "rank_p50": np.median(R, axis=0).astype(int),
                       "rank_p95": np.percentile(R, 95, axis=0).astype(int),
                       "p_top3": (R <= 3).mean(0), "p_top10": (R <= 10).mean(0),
                       "score_p05": np.percentile(SC, 5, axis=0), "score_p95": np.percentile(SC, 95, axis=0)}, index=pids)
    bt.to_csv(f"{C.OUT}/bootstrap_ranks_{side}.csv")

    # official top 10 file
    off = res.join(pd.read_csv(f"{C.OUT}/ratings_{side}_all.csv", index_col=0).drop(columns=res.columns, errors="ignore"))
    off = off.join(bt[["rank_p05", "rank_p95", "p_top10"]]).head(10)
    off["SCORE_lo90"] = bt.loc[off.index, "score_p05"]
    off["SCORE_hi90"] = bt.loc[off.index, "score_p95"]
    if side == "bat":
        keep = ["rank", "player", "ipl_team_2026", "country", "pressure_balls", "pressure_SR", "usual_SR", "pressure_avg", "usual_avg",
                "pressure_balls_ipl_share_%", "nP", "mP", "mU", "mD", "P", "D", "SCORE", "SCORE_lo90", "SCORE_hi90",
                "rank_p05", "rank_p95", "p_top10", "pressure_RV100_2025", "pressure_RV100_2026", "pressure_matches_6plus",
                "pct_matches_above_league", "pct_matches_above_own_usual"]
    else:
        keep = ["rank", "player", "ipl_team_2026", "country", "pressure_balls", "pressure_econ", "usual_econ", "pressure_wkts",
                "pressure_balls_per_wkt", "usual_balls_per_wkt", "pressure_balls_ipl_share_%", "nP", "mP", "mU", "mD", "P", "D", "SCORE",
                "SCORE_lo90", "SCORE_hi90", "rank_p05", "rank_p95", "p_top10", "pressure_RV100_2025", "pressure_RV100_2026",
                "pressure_matches_6plus", "pct_matches_above_league", "pct_matches_above_own_usual"]
    ren = {"nP": "eff_pressure_balls", "mP": "raw_pressure_value_per100", "mU": "raw_usual_value_per100", "mD": "raw_pressure_minus_usual",
           "P": "P_shrunk", "D": "D_shrunk", "SCORE": "score", "SCORE_lo90": "score_lo90", "SCORE_hi90": "score_hi90"}
    off[keep].rename(columns=ren).round(2).to_csv(f"{C.OUT}/official_top10_{'bat' if side == 'bat' else 'bowl'}.csv", index=False)

    name = "Batting" if side == "bat" else "Bowling"
    md.append(f"\n## {name} ({len(pids)} eligible players)\n")
    md.append("| check | final SCORE | P (pressure level) | U (usual level) | D (pressure minus usual) |\n|---|---|---|---|---|")
    md.append(f"| split-half by match, Spearman (mean of {C.N_SPLIT}) | {sh.SCORE:.2f} | {sh.P:.2f} | {sh.U:.2f} | {sh.D:.2f} |")
    md.append(f"| ...Spearman-Brown full-length estimate | {sb(sh.SCORE):.2f} | {sb(sh.P):.2f} | {sb(sh.U):.2f} | {sb(sh.D):.2f} |")
    md.append(f"| 2025-only vs 2026-only, Spearman (n={both.sum()} with >=30 eff. pressure balls in both) | {yr['SCORE']:.2f} | {yr['P']:.2f} | {yr['U']:.2f} | {yr['D']:.2f} |")
    md.append(f"\nSplit-half top-10 overlap between halves: {sh.top10_overlap:.1f} of 10. Raw (unshrunk) pressure-minus-usual "
              f"split-half Spearman: {sh.mD:.2f}.\n")
    md.append(f"Bootstrap ({C.N_BOOT} match resamples): on average {np.mean(top10_overlap):.1f} of the official top 10 stay in the top 10 "
              f"(5th-95th pct: {np.percentile(top10_overlap, 5):.0f}-{np.percentile(top10_overlap, 95):.0f}).\n")
    md.append("| rank | player | 90% rank interval | P(top 3) | P(top 10) |\n|---|---|---|---|---|")
    for p, r in bt.head(15).iterrows():
        md.append(f"| {r['rank']} | {r.player} | {r.rank_p05}-{r.rank_p95} | {r.p_top3:.2f} | {r.p_top10:.2f} |")
    sure = bt[bt.p_top10 >= 0.5].shape[0]
    md.append(f"\nPlayers with P(top 10) >= 0.5: {sure}. Players with P(top 10) >= 0.8: {(bt.p_top10 >= 0.8).sum()}.\n")
    print(name, sh.round(2).to_dict(), yr, np.mean(top10_overlap))
    print(bt.head(15).to_string())

# (d) predictive validity on held-out seasons: does the score computed on earlier seasons predict a player's value
#     on PRESSURE balls in later seasons? (unweighted balls; players with >=40 pressure + 60 ordinary balls in the
#     training window and >=40 pressure balls in the test window). Compared with simple alternatives.
md.append("\n## Predictive validity on held-out seasons\n")
md.append("Correlation (Pearson) between a score computed on the training seasons and the player's actual value per "
          "pressure ball in the later test seasons. Official = P + D (univariate). This was used only to describe the "
          "method, not to choose players.\n")
md.append("| side | train -> test | players | official P+D | P only | usual level only | all balls | bivariate P+D |\n|---|---|---|---|---|---|---|---|")
for side in ("bat", "bowl"):
    cfg = lib.default_cfg()
    allb = lib.load_balls(side, cfg, years=C.HYPER_YEARS, weighted=False)
    for tr, te in (([2022, 2023, 2024], [2025, 2026]), ([2022, 2023], [2024]), ([2023, 2024], [2025]), ([2024, 2025], [2026])):
        dtr = allb[allb.year.isin(tr)]; dte = allb[allb.year.isin(te)]
        vtr = lib.ball_var(dtr)
        a = lib.summarise(lib.player_match(dtr)); b = lib.summarise(lib.player_match(dte))
        j = a.join(b[["mP", "nP"]], rsuffix="_te", how="inner"); j = j[(j.nP >= 40) & (j.nU >= 60) & (j.nP_te >= 40)]
        LW = lib.long_window_hyper(side, cfg)   # league prior variances (2022-26) - league-level, no player selection
        out = {}
        for nm, kw in (("off", {}), ("P", {"SCORE": "P"}), ("biv", {"MODEL": "bivariate"})):
            c2 = lib.default_cfg(**kw); hp = lib.fit_hyper(j, vtr, c2, LW)
            out[nm] = lib.score_table(j, vtr, hp, c2).SCORE
        allm = (j.sP + j.sU) / (j.nP + j.nU)
        r = lambda x: np.corrcoef(x, j.mP_te)[0, 1]
        md.append(f"| {side} | {tr[0]}-{tr[-1]} -> {"-".join(map(str, sorted({te[0], te[-1]})))} | {len(j)} | {r(out['off']):.2f} | {r(out['P']):.2f} | {r(j.mU):.2f} | {r(allm):.2f} | {r(out['biv']):.2f} |")
corr = {}
for side in ("bat", "bowl"):
    S_ = lib.long_window_hyper(side, lib.default_cfg())["S"]; corr[side] = S_[0, 1] / np.sqrt(S_[0, 0] * S_[1, 1])
md.append("\nAll correlations are low: next-season pressure performance is hard to predict for anyone. Scores that also "
          f"use ordinary balls (all balls, bivariate) predict slightly better in some splits, because a player's "
          f"true pressure level and usual level are highly correlated (bivariate prior correlation {corr['bat']:.2f} batting, {corr['bowl']:.2f} bowling). "
          "The official score still uses pressure balls only for the level term, because the brief asks who delivered under "
          "pressure, not who is best overall; the bivariate variant is reported in sensitivity.csv.\n")

md.append("\n## How to read this\n")
md.append("- U (usual level) is the most repeatable part; P (pressure level) is less repeatable because only a quarter of balls are "
          "pressure balls; D (pressure minus usual) is close to pure noise at 2025-26 sample sizes, which is why the locked "
          "shrinkage gives it little room to move a player.\n"
          "- Rank intervals are wide in the middle of each list: ranks are firm at the top and only roughly ordered below that.\n")
open(f"{C.OUT}/reliability.md", "w").write("\n".join(md))
