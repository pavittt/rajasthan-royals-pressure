"""Step 0b (RR theory). Which of RR's 48 moments hold up, and the pressure flag used for rating.
OFFICIAL pressure uses ALL 48 moments, exactly as originally found (RRP*). The hold-up test below is reported for every moment
and used as a sensitivity run (RRH*): a moment HOLDS UP only if, on IPL balls (context-adjusted run value, data/*_balls.parquet):
  - RR were below the league in it in 2025 AND in 2026 (a gap that persists, not one bad season), and
  - the champions (RCB, IPL winners 2025 and 2026) were above the league in it over 2025-26.
RR gap of a kept moment = league minus RR, runs per ball, IPL 2025-26.
RR moment ball = a ball inside at least one kept moment (each ball counted ONCE, however many moments contain it).
Pressure ball (RRP) = an RR moment ball that is also live: leverage at or above the median of all 2022-26 balls
(RR's moments alone barely differ from ordinary balls in how much the result swings, 1.1x for batting; with the filter 2.4x / 3.1x).
Importance of a pressure ball = the largest RR gap among the kept moments containing it, scaled to average 1 over
2025-26 pressure balls and capped at 3, so the biggest RR leaks count most without one tiny cell dominating.
Outputs: data/bat_balls.parquet + data/bowl_balls.parquet gain columns RR48 (bool) and RRimp (float);
         out/rr_moments_kept.csv (every one of the 48 moments with its test results)."""
import pandas as pd, numpy as np
import config as C
RR, RCB = 3, 5
mem = pd.read_parquet(f"{C.DATA}/rr_moment_members.parquet")
lab = pd.read_csv("/home/claude/work/all_moments.csv").set_index("id")
KEY = ["match_id", "innings", "innings_ball_num"]
out = []
for side in ("bat", "bowl"):
    d = pd.read_parquet(f"{C.DATA}/{side}_balls.parquet")
    d = d.drop(columns=[c for c in ("RR48", "RRimp") if c in d])
    team = "batting_team_id" if side == "bat" else "bowling_team_id"
    if side == "bat": d_ok = d.faced
    else: d_ok = d.legal | (d.bowler_runs > 0)
    m = mem[mem.side == side].merge(d[KEY + [team, "fmt", "year", "RV"]][d_ok], on=KEY)
    ipl = m[(m.fmt == "IPL") & m.year.isin([2025, 2026])]
    rows = []
    for mid, q in ipl.groupby("mid"):
        r = {"side": side, "id": mid, "label": lab.label.get(mid, mid), "theme": lab.theme.get(mid, "")}
        for y in (2025, 2026):
            qy = q[q.year == y]; rr = qy[qy[team] == RR].RV; lg = qy[qy[team] != RR].RV
            r[f"rr_balls_{y}"] = len(rr); r[f"rr_minus_league_{y}"] = 100 * (rr.mean() - lg.mean()) if len(rr) else np.nan
        rr = q[q[team] == RR].RV; lg = q[q[team] != RR].RV; ch = q[q[team] == RCB].RV
        r["rr_per100"], r["league_per100"], r["champ_per100"] = 100 * rr.mean(), 100 * lg.mean(), 100 * ch.mean()
        r["rr_balls"], r["champ_balls"] = len(rr), len(ch)
        r["gap_per_ball"] = lg.mean() - rr.mean()
        r["kept"] = bool((r["rr_minus_league_2025"] < 0) and (r["rr_minus_league_2026"] < 0) and (ch.mean() > lg.mean()))
        r["why_not"] = "" if r["kept"] else ("RR not below the league in both seasons" if not ((r["rr_minus_league_2025"] < 0) and (r["rr_minus_league_2026"] < 0)) else "champions not above the league")
        rows.append(r)
    T = pd.DataFrame(rows); out.append(T)
    d = d.drop(columns=[c for c in d.columns if c.startswith(("RRP", "RRH", "RRimp", "RR48"))])
    for tag, ids in (("RRP", T.set_index("id").gap_per_ball), ("RRH", T[T.kept].set_index("id").gap_per_ball)):
        # RRP = all 48 moments (official, as originally defined); RRH = only moments that hold up in both seasons (sensitivity)
        g_m = ids.clip(lower=1e-6)
        mk = mem[(mem.side == side) & mem.mid.isin(g_m.index)].copy(); mk["g"] = mk.mid.map(g_m)
        g = mk.groupby(KEY).g.max().rename("gmax").reset_index()
        d = d.merge(g, on=KEY, how="left"); inm = d.gmax.notna()
        base = d.loc[inm & d.year.isin(C.SCORE_YEARS), "gmax"].mean()
        d[f"{tag}imp"] = (d.gmax / base).clip(upper=3.0).fillna(1.0)
        if tag == "RRP": d["RR48"] = inm; d["RRimp"] = d["RRPimp"]
        for qq in (0.4, 0.5, 0.6):
            d[f"{tag}{qq}"] = inm & (d.LI >= d.LI.quantile(qq))
        d = d.drop(columns="gmax")
    d.to_parquet(f"{C.DATA}/{side}_balls.parquet")
    s = d[d.year.isin(C.SCORE_YEARS)]
    print(side, "kept", int(T.kept.sum()), "of", len(T), "| pressure share of 2025-26 balls", round(s.RR48.mean(), 3))
pd.concat(out).sort_values(["side", "kept", "gap_per_ball"], ascending=[True, False, False]).round(3).to_csv(f"{C.OUT}/rr_moments_kept.csv", index=False)
