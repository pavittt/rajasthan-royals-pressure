"""Step 12b. RR's 2027 moves and squad-v-buys by area, on the fixed score. Outputs: out/moves_check.json, out/squad_compare.json"""
import json, numpy as np, pandas as pd
import config as C
J = json.load(open(f"{C.OUT}/final_lists.json"))
tr = pd.read_csv("/home/claude/work/trades2027.csv"); pool = pd.read_csv(f"{C.INPUTS}/pool2027.csv")
print(tr.columns.tolist())
import lib
# moves are judged on the same score, rated WITHOUT the current-form rule so released / traded players with a thin 2026 still get a number
R = {s: lib.rate(s, lib.default_cfg(ELIG_OVERRIDE=dict(min_ipl_matches_2026=0, max_form_z_2026=-1e9)))[0].set_index("player") for s in ("bat", "bowl")}
moves = [(p, "Released by RR") for p in pool[pool.released_by == "Rajasthan Royals"].player_name]
for r in tr.to_dict("records"):
    if "Rajasthan" in str(r["from"]): moves.append((r["player"], "Traded out of RR"))
    if "Rajasthan" in str(r["to"]): moves.append((r["player"], "Traded into RR"))
rows = []
for p, mv in moves:
    for s, lab in (("bat", "Batting"), ("bowl", "Bowling")):
        if p in R[s].index:
            x = R[s].loc[p]; sc = float(x.SCORE)
            good = (sc < 0) if mv != "Traded into RR" else (sc > 0)
            rows.append(dict(player=p, move=mv, side=lab, score=round(sc, 2), score_2026=None, balls=int(round(float(x.nP))), verdict="Good move" if good else "Costly move"))
M = pd.DataFrame(rows); print(M.to_string())
json.dump({"rows": json.loads(M.to_json(orient="records")), "meta": [{"label": "s12b_moves.py", "value": open(__file__).read(), "type": "code", "lang": "python"}]}, open(f"{C.OUT}/moves_check.json", "w"))
# squad v buys by area: each pressure ball is assigned to its biggest-gap moment's area
mem = pd.read_parquet(f"{C.DATA}/rr_moment_members.parquet"); mk = pd.read_csv(f"{C.OUT}/rr_moments_kept.csv")
KEY = ["match_id", "innings", "innings_ball_num"]; sq = pd.read_csv(f"{C.INPUTS}/squads2027.csv"); rr_ids = set(sq[sq.team_name == "Rajasthan Royals"].player_id.dropna().astype(int))
buy = {s: {r["player"] for r in J[s]["can_buy"]} for s in ("bat", "bowl")}
out = []
for s in ("bat", "bowl"):
    g = mk[mk.side == s].set_index("id"); m = mem[mem.side == s].copy(); m["g"] = m.mid.map(g.gap_per_ball); m["theme"] = m.mid.map(g.theme)
    m = m.sort_values("g").drop_duplicates(KEY, keep="last")
    d = pd.read_parquet(f"{C.DATA}/{s}_balls.parquet"); d = d[d.year.isin(C.SCORE_YEARS) & d[f"RRP{C.RR_LEVER_Q}"]].merge(m[KEY + ["theme"]], on=KEY)
    pid = "striker_id" if s == "bat" else "bowler_id"; nm = "striker_name" if s == "bat" else "bowler_name"
    for th, q in d.groupby("theme"):
        p = q.groupby([pid, nm]).agg(n=("RV", "size"), v=("RV", "mean")).reset_index(); p = p[p.n >= 20]
        rr = p[p[pid].isin(rr_ids)]; b = p[p[nm].isin(buy[s])].sort_values("v", ascending=False)
        f = lambda x: ", ".join(f"{r[nm]} ({100 * r.v:+.0f})" for _, r in x.iterrows()) or "none with 20+ balls"
        out.append(dict(area=th, rr_wins_lost=round(float(g[g.theme == th].gap_per_ball.mean() * 100), 1), rr_above=f(rr[rr.v > 0]), rr_below=f(rr[rr.v <= 0]), best_buys=f(b[b.v > 0].head(4))))
S = pd.DataFrame(out); print(S.to_string())
json.dump({"rows": json.loads(S.to_json(orient="records")), "meta": [{"label": "s12b_moves.py", "value": open(__file__).read(), "type": "code", "lang": "python"}]}, open(f"{C.OUT}/squad_compare.json", "w"))
