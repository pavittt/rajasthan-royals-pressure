"""Step 0 (RR theory). Rebuild the masks of RR's 48 pressure moments (from the earlier moment search, s38) on every ball
and export them as a key table, so the rest of this pipeline never execs old code again.
Output: data/rr_moment_members.parquet (match_id, innings, innings_ball_num, side, mid)."""
import pandas as pd, numpy as np
src = open("/home/claude/scripts/s38_all_moments.py").read()
ns = {}; exec(src[:src.index("OUT, CONTRIB, THEMES, THSC")], ns)
b, M = ns["b"], ns["M"]
rows = []
for x in M:
    m = np.asarray(x["mask"])
    k = b.loc[m, ["match_id", "innings", "innings_ball_num"]].copy(); k["side"] = "bat" if x["side"] == "batting" else "bowl"; k["mid"] = x["id"]
    rows.append(k)
R = pd.concat(rows); R.to_parquet("/home/claude/rr48/data/rr_moment_members.parquet")
print(R.groupby("side").mid.nunique(), len(R))
