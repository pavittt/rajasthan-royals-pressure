"""Step 2. Out-of-fold win-probability model and Leverage Index (LI).
WP: gradient-boosted trees (LightGBM, monotone constraints), one model per innings, 5 folds split BY MATCH,
each ball scored by the model that never saw its match; Platt-calibrated out of fold.
LI of a ball = expected |change in batting side's win probability| over the league outcome distribution
(0,1,2,3,4,5,6,wicket) for that format/innings/phase. It depends only on the match state before the ball,
not on what actually happened, so it can define pressure for batter and bowler alike.
Inputs : data/balls.parquet   Outputs: data/balls_wp.parquet, out/wp_diagnostics.txt"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import config as C

b = pd.read_parquet(f"{C.DATA}/balls.parquet")
b = b[b.has_result].reset_index(drop=True)                       # no-results cannot train or be scored
b["fmt_code"] = b.fmt.map({"IPL": 0, "T20I": 1, "OTHER": 2})

# venue first-innings par, leave-one-match-out, shrunk to format mean (k = 8 matches)
fi = b[(b.innings == 1)].groupby("match_id").agg(score=("total_runs", "sum"), venue=("venue_id", "first"),
                                                  fmt=("fmt", "first"), alloc=("alloc", "first"))
fi = fi[fi.alloc == 120]
fmean = fi.groupby("fmt").score.mean(); vs = fi.groupby("venue").score.agg(["sum", "count"])
par = {}
for mid, ven, f in b[["match_id", "venue_id", "fmt"]].drop_duplicates().itertuples(index=False):
    s, n = vs.loc[ven] if ven in vs.index else (0.0, 0)
    if mid in fi.index: s -= fi.loc[mid, "score"]; n -= 1
    par[mid] = (s + 8 * fmean.get(f, fmean.mean())) / (n + 8)
b["venue_par"] = b.match_id.map(par)

def feats(inn, df, runs, wk, legal):
    left = np.maximum(df.alloc.values - legal, 0)
    if inn == 1:
        rr = runs / np.maximum(legal, 1) * 6
        return np.column_stack([legal, left, wk, runs, rr, runs + rr / 6 * left, df.alloc.values,
                                df.fmt_code.values, df.year.values, df.venue_par.values])
    need = df.target.values - runs
    return np.column_stack([left, wk, need, need / np.maximum(left, 1) * 6, df.target.values, df.alloc.values,
                            df.fmt_code.values, df.year.values])
MONO = {1: [0, 0, -1, 1, 0, 1, 0, 0, 0, -1], 2: [1, -1, -1, -1, 0, 0, 0, 0]}
P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.9,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=C.SEED, num_threads=2)
mids = b.match_id.unique(); perm = np.random.default_rng(C.SEED).permutation(len(mids))
b["fold"] = b.match_id.map(dict(zip(mids, perm % 5)))

models, cal = {}, {}
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
for inn in (1, 2):
    d = b[b.innings == inn]
    X = feats(inn, d, d.runs_before.values, d.wk_before.values, d.legal_before.values); y = d.bat_won.values
    oof = np.zeros(len(d))
    for f in range(5):
        tr = d.fold.values != f
        models[(inn, f)] = lgb.train({**P, "monotone_constraints": MONO[inn]}, lgb.Dataset(X[tr], y[tr]),
                                     num_boost_round=400 if inn == 2 else 250)
        oof[~tr] = models[(inn, f)].predict(X[~tr])
    cal[inn] = LogisticRegression(C=1e6).fit(logit(oof)[:, None], y)

def raw_pred(inn, df, runs, wk, legal):
    X = feats(inn, df, runs, wk, legal); p = np.zeros(len(df))
    for f in range(5):
        i = df.fold.values == f
        if i.any(): p[i] = cal[inn].predict_proba(logit(models[(inn, f)].predict(X[i]))[:, None])[:, 1]
    return p

def wp2(df, runs, wk, legal):
    p = raw_pred(2, df, runs, wk, legal)
    need = df.target.values - runs; left = np.maximum(df.alloc.values - legal, 0)
    p = np.where(need <= 0, 1.0, p); done = (wk >= 10) | (left <= 0)
    return np.where(done & (need > 1), 0.0, np.where(done & (need == 1), 0.5, p))

def wp(df, inn, runs, wk, legal):
    if inn == 2: return wp2(df, runs, wk, legal)
    p = raw_pred(1, df, runs, wk, legal)
    end = (wk >= 10) | (df.alloc.values - legal <= 0)
    if end.any():                                                   # innings over -> chase model at ball 0
        d2 = df[end].copy()
        d2["target"] = np.where(d2.revised_target.notna(), d2.revised_target, runs[end] + 1)
        d2["alloc"] = d2.second_inn_revised_overs.fillna(120).clip(upper=120)
        z = np.zeros(end.sum()); p[end] = 1 - wp2(d2, z, z, z)
    return p

leg = b[b.legal].copy(); leg["o"] = np.where(leg.wkt, "W", leg.total_runs.clip(upper=6).astype(str))
probs = leg.groupby(["innings", "fmt", "phase"]).o.value_counts(normalize=True)
out = []
for inn in (1, 2):
    d = b[b.innings == inn].copy()
    r0, w0, l0 = d.runs_before.values, d.wk_before.values, d.legal_before.values
    d["wp_pre"] = wp(d, inn, r0, w0, l0)
    d["wp_post"] = wp(d, inn, r0 + d.total_runs.values, w0 + d.wkt.values.astype(int), l0 + d.legal.values.astype(int))
    li = np.zeros(len(d)); tot = np.zeros(len(d))
    for o in ["0", "1", "2", "3", "4", "5", "6", "W"]:
        pr = np.array([probs.get((inn, f, ph, o), 0.0) for f, ph in zip(d.fmt, d.phase)])
        po = wp(d, inn, r0, w0 + 1, l0 + 1) if o == "W" else wp(d, inn, r0 + int(o), w0, l0 + 1)
        li += pr * np.abs(po - d.wp_pre.values); tot += pr
        if o in ("0", "4", "W"): d[f"wp_after_{o}"] = po          # kept to price a wicket in runs (step 4)
    d["li_raw"] = li / np.maximum(tot, 1e-9)
    out.append(d)
b = pd.concat(out).sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
b["LI"] = b.li_raw / b.loc[b.universe, "li_raw"].mean()
b["wpa_bat"] = b.wp_post - b.wp_pre
b.drop(columns=["fold"]).to_parquet(f"{C.DATA}/balls_wp.parquet")

with open(f"{C.OUT}/wp_diagnostics.txt", "w") as fh:
    for inn in (1, 2):
        d = b[b.innings == inn]
        fh.write(f"innings {inn}: out-of-fold AUC={roc_auc_score(d.bat_won, d.wp_pre):.3f} "
                 f"Brier={brier_score_loss(d.bat_won, d.wp_pre):.3f}\n")
        ct = d.groupby(pd.cut(d.wp_pre, np.linspace(0, 1, 11)), observed=True).bat_won.agg(["mean", "count"])
        ct.index = ct.index.astype(str); fh.write("calibration (predicted bin -> actual win rate)\n" + ct.round(3).to_string() + "\n\n")
print(open(f"{C.OUT}/wp_diagnostics.txt").read())
