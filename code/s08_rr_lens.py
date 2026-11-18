"""Step 8 (SECONDARY - not part of the official ranking). Rajasthan Royals lens.
 1. Where did RR lose value on PRESSURE balls in IPL 2025-26 vs the rest of the league and vs the champions
    (identified from the data: winners of the 2025 and 2026 finals)? 12 non-overlapping cells
    (batting/bowling x setting/chasing x powerplay/middle/death): each pressure ball is in exactly one cell, so the
    per-season run totals add up. Match-cluster bootstrap p-values, Benjamini-Hochberg FDR across the 12 cells, and a
    2025-vs-2026 sign check (a gap must point the same way in both seasons to be called persistent).
 2. Which players in the official rating pool are strong in RR's weakest cells, split by 2027 status.
    2027 status comes from inputs/squads2027.csv and inputs/pool2027.csv, HAND-ENTERED FROM NEWS REPORTS of the
    17-18 Nov 2026 retentions (not part of the supplied data).
Outputs: out/rr_gaps.csv, out/rr_lens.csv, out/rr_lens.md"""
import numpy as np, pandas as pd
import config as C, lib

rng = np.random.default_rng(C.SEED)
m = pd.read_parquet(f"{C.DATA}/matches.parquet")
team = dict(zip(m.home_team_id, m.home_name)) | dict(zip(m.away_team_id, m.away_name))
RR = [k for k, v in team.items() if v == "Rajasthan Royals"][0]
fin = m[(m.fmt == "IPL") & (m.stage == "Final") & m.year.isin(C.SCORE_YEARS)]
champ = {int(r.year): int(r.win_team_id) for r in fin.itertuples()}
Q = f"P{C.PRESSURE_Q}"

def team_cells(side):
    d = pd.read_parquet(f"{C.DATA}/{side}_balls.parquet")
    d = d[(d.fmt == "IPL") & d.year.isin(C.SCORE_YEARS) & d[Q]].copy()
    d["team"] = d.batting_team_id if side == "bat" else d.bowling_team_id
    lab = ("setting", "chasing") if side == "bat" else ("bowling first", "defending")
    d["cell"] = ("batting" if side == "bat" else "bowling") + " | " + np.where(d.innings == 1, lab[0], lab[1]) + " | " + d.phase
    d["is_champ"] = [champ.get(y) == t for y, t in zip(d.year, d.team)]
    return d[["match_id", "year", "team", "cell", "RV0", "is_champ"]]
d = pd.concat([team_cells("bat"), team_cells("bowl")])
d["grp"] = np.where(d.team == RR, "RR", "rest")

def gap_stats(x):
    """RR minus rest-of-league per 100 pressure balls, with match-cluster bootstrap"""
    rr, rest = x[x.grp == "RR"], x[x.grp == "rest"]
    ch = x[x.is_champ]
    g = lambda a: a.RV0.mean() * 100
    est = g(rr) - g(rest)
    a = rr.groupby("match_id").RV0.agg(["sum", "size"]); b = rest.groupby("match_id").RV0.agg(["sum", "size"])
    boots = []
    for _ in range(2000):
        ia = rng.integers(0, len(a), len(a)); ib = rng.integers(0, len(b), len(b))
        boots.append(100 * (a["sum"].values[ia].sum() / a["size"].values[ia].sum() - b["sum"].values[ib].sum() / b["size"].values[ib].sum()))
    boots = np.array(boots)
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    y = {yr: (g(rr[rr.year == yr]) - g(rest[rest.year == yr])) for yr in C.SCORE_YEARS}
    per_season = len(rr) / len(C.SCORE_YEARS)
    return pd.Series({"RR_pressure_balls": len(rr), "RR_per100": g(rr), "league_per100": g(rest), "champions_per100": g(ch),
                      "gap_vs_league_per100": est, "ci90_lo": np.percentile(boots, 5), "ci90_hi": np.percentile(boots, 95), "p": p,
                      "gap_2025": y[2025], "gap_2026": y[2026], "gap_vs_champions_per100": g(rr) - g(ch),
                      "runs_per_season_vs_league": est / 100 * per_season})
G = d.groupby("cell").apply(gap_stats, include_groups=False)
# Benjamini-Hochberg
o = G.p.sort_values(); k = len(o); q = (o * k / np.arange(1, k + 1))[::-1].cummin()[::-1].clip(upper=1)
G["q_fdr"] = q.reindex(G.index)
G["persistent"] = np.sign(G.gap_2025) == np.sign(G.gap_2026)
G = G.sort_values("runs_per_season_vs_league")
G.round(3).to_csv(f"{C.OUT}/rr_gaps.csv")
tot = G.runs_per_season_vs_league.sum()

# ---- player fits in RR's weakest cells (negative gap, persistent both seasons; FDR status reported, not required)
sq = pd.read_csv(f"{C.INPUTS}/squads2027.csv"); pool = pd.read_csv(f"{C.INPUTS}/pool2027.csv")
def status(pid):
    s = sq[sq.player_id == pid]
    if len(s): return ("RR retained" if (s.team_name == "Rajasthan Royals").any() else f"retained by {s.team_name.iat[0]}")
    p = pool[pool.player_id == pid]
    if len(p):
        note = p.note.iat[0] if isinstance(p.note.iat[0], str) else ""
        return f"released by {p.released_by.iat[0]} (auction pool)" + (f" - {note}" if note else "")
    return "not on any 2027 retained list (auction registration unknown)"

weak = G[(G.gap_vs_league_per100 < 0) & G.persistent].head(4)
rows = []
for cell, g in weak.iterrows():
    side = "bat" if cell.startswith("batting") else "bowl"
    inn = 1 if ("setting" in cell or "first" in cell) else 2; ph = cell.split(" | ")[2]
    res, aux = lib.rate(side, return_all=True)
    x = aux["d"]; x = x[x.pres & (x.innings == inn) & (x.phase == ph) & x.pid.isin(res.index)]
    t = x.groupby("pid").agg(s=("v", lambda v: (v * x.loc[v.index, "w"]).sum()), n=("w", "sum"), k=("v", "size"))
    t = t[t.k >= 12]
    # cell samples are small: shrink with the official pressure-level prior variance toward the cell's pool mean
    m_ = t.s / t.n; se = aux["var"]["P"] / t.n; t2 = aux["hp"]["P"][1]
    mu = np.sum(m_ / (t2 + se)) / np.sum(1 / (t2 + se))
    t["cell_value_per100"] = 100 * lib.eb_post(m_, se, mu, t2)[0]; t["cell_raw_per100"] = 100 * m_
    t = t.join(res[["player", "rank", "SCORE"]])
    t["status"] = [status(p) for p in t.index]
    t = t[~t.status.str.startswith("retained")]                 # only RR's own and buyable players
    t = t.sort_values(["cell_value_per100"], ascending=False)
    for pid, r in t.iterrows():
        rows.append(dict(gap_cell=cell, rr_gap_runs_per_season=round(g.runs_per_season_vs_league, 1), gap_q_fdr=round(g.q_fdr, 3),
                         player=r.player, status_2027=r.status, cell_pressure_balls=int(r.k),
                         cell_value_per100_raw=round(r.cell_raw_per100, 1), cell_value_per100_shrunk=round(r.cell_value_per100, 1),
                         official_rank=int(r["rank"]), official_score=round(r.SCORE, 2)))
L = pd.DataFrame(rows)

# ---- shortlist file: every player in the official pools who is RR's own or buyable, with weak-cell evidence
short = []
for side in ("bat", "bowl"):
    res, _ = lib.rate(side)
    for pid, r in res.iterrows():
        st = status(pid)
        if st.startswith("retained"): continue
        cells = L[(L.player == r.player) & L.gap_cell.str.startswith("batting" if side == "bat" else "bowling")]
        short.append(dict(side=side, player=r.player, status_2027=st, group="RR current squad" if st == "RR retained" else "available",
                          official_rank=int(r["rank"]), pool_size=len(res), score=round(r.SCORE, 2), P_pressure=round(r.P, 1),
                          D_vs_own_usual=round(r.D, 1), raw_pressure_per100=round(r.mP, 1), raw_usual_per100=round(r.mU, 1),
                          eff_pressure_balls=round(r.nP, 1),
                          weak_cell_evidence="; ".join(f"{c.gap_cell}: {c.cell_value_per100_shrunk} ({c.cell_pressure_balls} balls)"
                                                       for c in cells.itertuples())))
SH = pd.DataFrame(short).sort_values(["side", "group", "official_rank"]); SH.to_csv(f"{C.OUT}/rr_lens.csv", index=False)

md = ["# Rajasthan Royals lens (secondary analysis - not the official pressure ranking)\n",
      f"Champions used for comparison (from the data): " + ", ".join(f"{y}: {team[t]}" for y, t in champ.items()) + ".\n",
      "Gaps are RR minus the other nine teams, runs above league expectation per 100 PRESSURE balls, IPL 2025-26. "
      "Cells do not overlap, so the per-season run column adds up: total = "
      f"{tot:.1f} runs per season across all pressure balls.\n",
      G[["RR_pressure_balls", "RR_per100", "league_per100", "champions_per100", "gap_vs_league_per100", "ci90_lo", "ci90_hi",
         "q_fdr", "gap_2025", "gap_2026", "persistent", "runs_per_season_vs_league"]].round(2).to_markdown(),
      f"\nCells passing FDR q < 0.10: {(G.q_fdr < 0.10).sum()} of {len(G)}. Treat the gaps below as directions to look in, "
      "not proven weaknesses.\n",
      "\n## RR's own and buyable players in RR's weakest persistent cells (official rating pool, >= 12 balls in the cell)\n",
      L.to_markdown(index=False),
      "\n## Shortlist: available players in the official rating pools vs RR's current squad\n"]
for side in ("bat", "bowl"):
    x = SH[SH.side == side].drop(columns=["side", "weak_cell_evidence", "pool_size"])
    md.append(f"\n### {'Batting' if side == 'bat' else 'Bowling'}\n" + x.to_markdown(index=False))
md.append("\nNotes: 2027 status is typed from news reports (retention lists of 17-18 Nov 2026) and should be re-checked "
          "against the official auction list. Players released by RR are treated as buyable only because they are in the "
          "auction pool.\n")
open(f"{C.OUT}/rr_lens.md", "w").write("\n".join(md))
print("\n".join(md))
