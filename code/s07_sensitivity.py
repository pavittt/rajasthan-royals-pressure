"""Step 7. Pre-declared sensitivity runs (listed in config.SENSITIVITY before any ranking was looked at).
Each alternative re-runs the whole rating (eligibility included) with one change.
Outputs: out/sensitivity.csv (every scenario's top 10), out/sensitivity_summary.md"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import config as C, lib

rows, summ = [], []
for side in ("bat", "bowl"):
    off, _ = lib.rate(side)
    off10 = list(off.index[:10])
    for name, kw in [("OFFICIAL", {})] + list(C.SENSITIVITY.items()):
        res, info = lib.rate(side, lib.default_cfg(**kw))
        for p, r in res.head(10).iterrows():
            rows.append(dict(scenario=name, side=side, rank=int(r["rank"]), player=r.player, score=round(r.SCORE, 2),
                             in_official_top10=p in off10))
        common = off.index.intersection(res.index)
        rho = spearmanr(off.loc[common, "SCORE"], res.loc[common, "SCORE"])[0]
        summ.append(dict(side=side, scenario=name, eligible=len(res), top10_overlap=len(set(off10) & set(res.index[:10])),
                         same_no1=res.index[0] == off10[0], spearman_vs_official=round(rho, 3)))
    # how often each official top-10 player stays in the top 10 across the alternatives
S = pd.DataFrame(summ); R = pd.DataFrame(rows)
R.to_csv(f"{C.OUT}/sensitivity.csv", index=False)
md = ["# Sensitivity of the official top 10 (pre-declared alternatives)\n",
      "Each row re-runs the full pipeline with ONE change. Overlap = how many of the official top 10 are in that scenario's top 10.\n"]
for side in ("bat", "bowl"):
    s = S[S.side == side]
    md.append(f"\n## {'Batting' if side == 'bat' else 'Bowling'}\n")
    md.append(s.drop(columns="side").to_markdown(index=False))
    alt = R[(R.side == side) & (R.scenario != "OFFICIAL")]
    stay = alt[alt.in_official_top10].groupby("player").size().reindex(R[(R.side == side) & (R.scenario == "OFFICIAL")].player).fillna(0).astype(int)
    md.append(f"\nScenarios (of {alt.scenario.nunique()}) in which each official top-10 player stays in the top 10:\n")
    md.append(stay.rename("scenarios_in_top10").to_markdown())
    md.append(f"\nMedian overlap across alternatives: {s[s.scenario != 'OFFICIAL'].top10_overlap.median():.0f} of 10.\n")
open(f"{C.OUT}/sensitivity_summary.md", "w").write("\n".join(md))
print("\n".join(md))
