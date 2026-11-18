"""Step 11. Methodology report as PDF (out/report.html -> out/SupeRR_Round1_Report.pdf via headless Chrome)."""
import json, base64, io, subprocess, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config as C
J = json.load(open(f"{C.OUT}/final_lists.json")); M = pd.DataFrame(J["moments"]); M["holds"] = M["kept"].astype(bool); M["kept"] = True
PINK, BLUE, GOLD, INK, MUTED, LINE = "#d1176f", "#3a5fb0", "#b8860b", "#1c1a2b", "#625f74", "#e2dfea"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
def img(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=200, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
def moments_chart(side):
    k = M[(M.side == side) & M.kept].sort_values("gap_per_ball")
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(k) + 0.8))
    y = np.arange(len(k))
    for i, (_, r) in enumerate(k.iterrows()):
        v = [r.rr_per100, r.league_per100, r.champ_per100]; ax.plot([min(v), max(v)], [i, i], color=LINE, lw=3, zorder=1)
    ax.scatter(k.champ_per100, y, color=GOLD, s=26, zorder=3, label="Champions (RCB)")
    ax.scatter(k.league_per100, y, color=BLUE, s=26, zorder=3, label="League")
    ax.scatter(k.rr_per100, y, color=PINK, s=26, zorder=3, label="RR 2025-26")
    ax.set_yticks(y); ax.set_yticklabels([f"{r.id} {r.label}"[:62] for _, r in k.iterrows()])
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlabel("runs above league expectation per 100 balls (bowling: runs saved)")
    ax.grid(axis="x", color=LINE, lw=0.6); ax.set_axisbelow(True); ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    return img(fig)
def rank_chart(side):
    r = pd.DataFrame(J[side]["official"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); y = np.arange(len(r))
    ax.hlines(y, r.rank_p05, r.rank_p95, color=PINK, alpha=0.35, lw=5); ax.scatter(r["rank"], y, color=PINK, s=30, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{a}. {b}" for a, b in zip(r["rank"], r.player)]); ax.axvline(10, color=MUTED, ls="--", lw=0.8)
    ax.set_xlabel("rank: dot = official, bar = 90% range over 1,000 match resamples"); ax.grid(axis="x", color=LINE, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    return img(fig)
def tbl(rows, side, rank_key="rank", rng=True):
    bat = side == "bat"; st = "SR" if bat else "Econ"; st2 = "Balls/out" if bat else "Balls/wkt"
    h = f"<table><tr><th>#</th><th>Player</th><th>2027 status</th><th class=n>Score</th><th class=n>P</th><th class=n>D</th><th class=n>Pressure {st}</th><th class=n>Usual {st}</th><th class=n>Pressure {st2}</th><th class=n>Usual</th><th class=n>Pressure balls</th><th class=n>IPL %</th>{'<th class=n>Rank range</th>' if rng else ''}</tr>"
    fmt = (lambda v: f"{v:.0f}") if bat else (lambda v: f"{v:.1f}")
    for r in rows:
        s = r["status"].replace("Retained by ", "").replace(" (auction pool)", "").replace("Released by", "Released,")
        h += (f"<tr><td>{r[rank_key]}</td><td>{r['player']}</td><td>{s}</td><td class=n><b>{r['SCORE']:+.1f}</b></td><td class=n>{r['P']:+.1f}</td><td class=n>{r['D']:+.1f}</td>"
              f"<td class=n>{fmt(r['pressure_stat'])}</td><td class=n>{fmt(r['usual_stat'])}</td><td class=n>{r['pressure_stat2']:.0f}</td><td class=n>{r['usual_stat2']:.0f}</td>"
              f"<td class=n>{r['pressure_balls']}</td><td class=n>{r['pressure_balls_ipl_share_%']:.0f}</td>" + (f"<td class=n>{r['rank_p05']}-{r['rank_p95']}</td>" if rng else "") + "</tr>")
    return h + "</table>"
def mtable(side):
    k = M[M.side == side].sort_values("gap_per_ball", ascending=False)
    h = "<table><tr><th>Id</th><th>Moment</th><th class=n>RR</th><th class=n>League</th><th class=n>Champions</th><th class=n>RR v league 2025</th><th class=n>2026</th><th class=n>RR balls</th><th>Holds up both seasons</th></tr>"
    for _, r in k.iterrows():
        f = lambda v: "-" if pd.isna(v) else f"{v:+.1f}"
        h += f"<tr class='{'' if r.kept else 'drop'}'><td>{r.id}</td><td>{r.label}</td><td class=n>{f(r.rr_per100)}</td><td class=n>{f(r.league_per100)}</td><td class=n>{f(r.champ_per100)}</td><td class=n>{f(r.rr_minus_league_2025)}</td><td class=n>{f(r.rr_minus_league_2026)}</td><td class=n>{r.rr_balls}</td><td>{'Yes' if r.holds else 'No: ' + str(r.why_not)}</td></tr>"
    return h + "</table>"
S = pd.DataFrame(J["sensitivity"]); NAMES = {"rr_leverage_filter_top60pct": "Looser live-ball filter (top 60%)", "rr_leverage_filter_top40pct": "Stricter live-ball filter (top 40%)",
  "rr_no_importance_weight": "No extra weight for bigger RR gaps", "general_pressure_top25pct_leverage": "General pressure (top 25% leverage) instead of RR moments",
  "t20i_weight_0": "IPL only", "t20i_equal_ipl": "T20I counts as much as IPL", "years_equal": "2025 counts as much as 2026", "prior_var_x2": "Half the shrinkage",
  "prior_var_x0.5": "Twice the shrinkage", "prior_var_2025_26_only": "Shrinkage from 2025-26 only", "min_sample_60": "Minimum 60 effective pressure balls", "min_sample_25": "Minimum 25",
  "wicket_cost_x1.5": "Wickets worth 1.5x", "score_usual_plus_2D": "Usual + 2D", "bivariate_predictive": "Model borrowing usual form", "no_opp_adjust": "No opposition adjustment",
  "score_P_only": "Pressure level only", "score_D_only": "Pressure minus usual only", "score_P_plus_2D": "P + 2D"}
sens = "<table><tr><th>Alternative (one change each)</th><th class=n>Batting overlap</th><th class=n>Bowling overlap</th></tr>" + "".join(
    f"<tr><td>{NAMES.get(sc, sc)}</td><td class=n>{g[g.side == 'bat'].overlap.iat[0]}/10</td><td class=n>{g[g.side == 'bowl'].overlap.iat[0]}/10</td></tr>"
    for sc, g in S.groupby("scenario", sort=False) if sc != "OFFICIAL") + "</table>"
pb, pw = J["bat"]["pressure"], J["bowl"]["pressure"]
rel = open(f"{C.OUT}/reliability.md").read()
import re
def num(side, lab):
    m = re.search(rf"## {side}.*?{re.escape(lab)}[^|]*\|\s*([-\d.]+)", rel, re.S); return m.group(1)
boot = {s: re.search(rf"## {s}.*?on average ([\d.]+) of the official", rel, re.S).group(1) for s in ("Batting", "Bowling")}
kb, kw = int(M[(M.side == "bat") & M.holds].shape[0]), int(M[(M.side == "bowl") & M.holds].shape[0])
nm = lambda L: ", ".join(r["player"] for r in L[:4]) + " and " + L[4]["player"]
watch = ", ".join(f"{w['player']} ({w['SCORE']:+.1f}, thin evidence)" for w in J["bat"]["watch"] if w["evidence"] != "full")
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>SupeRR Selector S2 Round 1 Methodology</title><style>
@page{{size:A4;margin:16mm 14mm}} body{{font-family:'DejaVu Sans',Arial,sans-serif;color:{INK};font-size:9.6pt;line-height:1.45}}
h1{{font-size:21pt;margin:0 0 4px}} h2{{font-size:13.5pt;margin:16px 0 6px;color:{PINK};border-bottom:1px solid {LINE};padding-bottom:3px}} h3{{font-size:10.5pt;margin:10px 0 4px}}
p{{margin:4px 0}} ul{{margin:4px 0;padding-left:16px}} li{{margin:2px 0}}
table{{border-collapse:collapse;width:100%;font-size:7.8pt;margin:4px 0 8px;page-break-inside:auto}} tr{{page-break-inside:avoid}}
th{{text-align:left;color:{MUTED};border-bottom:1px solid #bbb;padding:3px 4px;font-weight:600}} td{{border-bottom:1px solid {LINE};padding:3px 4px}} .n{{text-align:right}}
tr.drop td{{color:#999}} .box{{border:1px solid {LINE};border-left:4px solid {PINK};padding:6px 10px;margin:6px 0;background:#faf8fb}}
img{{width:100%}} .sub{{color:{MUTED}}} .pb{{page-break-before:always}}
</style></head><body>
<h1>Who wins the moments Rajasthan Royals lose?</h1>
<p class=sub>SupeRR Selector Season 2, Round 1. Methodology report. Data: the supplied IPL and T20I ball-by-ball pack, 2022-2026. Players are scored on 2025-2026 only.</p>
<div class=box><b>Answer in one line.</b> Pressure is defined as RR's 48 moments, where the champions beat the league and RR fall short, on live balls. On those balls the best batters are {nm(J['bat']['official'])}, and the best bowlers are {nm(J['bowl']['official'])}. Each player is rated against the league and against his own usual level.</div>

<h2>1. Pressure definition</h2>
<p><b>Idea.</b> A title is decided in the situations where the best team pulls away from the rest. If RR lose those same situations, those are RR's pressure moments. Pressure here is built in three steps.</p>
<ul>
<li><b>Search.</b> 48 exact moments were found by searching every phase. Each moment is an over band plus up to two details: batting position, batter hand, bowler type (right- or left-arm pace, off-spin, left-arm finger spin, leg-spin, left-arm wrist spin), normal ball or wrong'un, wickets down, new batter, set partnership, match closeness, batting first or chasing, and a bowler just hit for 20+. The full grid of batting position x hand x bowler type x phase, and pressure situations x bowler type, was audited as well (614 cells) (shown on the dashboard). In every moment the champions beat the league and RR fell short.</li>
<li><b>All 48 moments are used.</b> A ball in a bigger RR gap gets more weight (see section 2). As a check, a hold-up test was run: RR below the league in both 2025 and 2026, and RCB above the league. {kb} batting and {kw} bowling moments pass it, and ranking on those alone is reported in section 5. Section 1.1 charts all 48; the appendix shows which hold up.</li>
<li><b>Live-ball filter.</b> Within the moments, a ball counts as pressure only if its leverage is at or above the median of all balls. Leverage is the expected change in win probability from the next delivery, scored by a win-probability model fitted out-of-fold by match. Each ball is counted once, however many moments contain it.</li>
</ul>
<p><b>It separates pressure from ordinary play.</b></p>
<ul>
<li>Batting pressure balls are {100*pb['share_pressure']:.0f}% of 2025-26 batting balls, and the result swings {pb['swing_ratio_pressure']:.1f}x as much on them as on other balls.</li>
<li>Bowling pressure balls are {100*pw['share_pressure']:.0f}% of bowling balls, carry {100*pw['share_of_swing']:.0f}% of all win-probability movement, and swing {pw['swing_ratio_pressure']:.1f}x as much as ordinary balls.</li>
<li>{100*pw['close_share_pressure']:.0f}% of bowling pressure balls come with the game between 30% and 70%, against {100*pw['close_share_other']:.0f}% of other balls.</li>
<li>Without the live-ball filter, RR's batting moments swing only {pb['swing_ratio_rr_moments']:.1f}x as much as other balls. That is why the filter is needed.</li>
</ul>
<h3>1.1 Batting moments (IPL 2025-26, runs above league expectation per 100 balls)</h3><img src="{moments_chart('bat')}">
<h3>Bowling moments</h3><img src="{moments_chart('bowl')}">

<h2 class=pb>2. Performance measurement</h2>
<ul>
<li><b>Ball value.</b> Each ball is valued in runs above what the league scores in the same context: format, year, innings, over, wickets down and required rate. A wicket is priced by its win-probability cost, and each ball is adjusted for the usual quality of the opposition. For batters, positive means runs added. For bowlers, positive means runs saved.</li>
<li><b>Baselines.</b> P is the player's value per pressure ball. U is his value per ordinary ball, which is his usual level. D = P - U is how much better or worse he is under pressure than usual.</li>
<li><b>Score.</b> SCORE = P + D, both shrunk by empirical Bayes toward the eligible-pool mean. The between-player spread is estimated on 2022-26 so that it is stable, but no pre-2025 ball enters a player's own numbers. A player who is elite under pressure is not punished for also being elite usually, and one who drops off under pressure loses points. The same formula and code are used for batters and bowlers.</li>
<li><b>Importance.</b> Within the pressure balls, a ball in a bigger RR gap counts more. The weight is scaled to average 1 and capped at 3, so the biggest leaks matter most and no tiny cell can dominate.</li>
</ul>

<h2>3. Fair comparisons</h2>
<ul>
<li><b>Recency and format.</b> One IPL 2026 ball counts as 1.0 ball of evidence, IPL 2025 as 0.7, a T20I between the top nine nations in 2026 as 0.5, and in 2025 as 0.35. These weights shrink the evidence count itself, so a T20I-heavy record is pulled harder toward average.</li>
<li><b>Eligibility (fixed data rules, no names removed by hand):</b>
  <ul>
  <li>played in 2026;</li>
  <li>at least 40 effective pressure balls, of which at least 20 are IPL;</li>
  <li>for bowlers, an average of at least 2 overs a match, so part-timers are out.</li>
  </ul>
  That leaves {J['bat']['eligible']} batters and {J['bowl']['eligible']} bowlers. Nothing is imputed: a player without enough IPL pressure balls is not ranked.</li>
<li><b>Small samples</b> are shrunk toward the mean in proportion to their evidence, and each rank is shown with its 90% bootstrap range.</li>
</ul>

<h2>4. Evidence for the top 10</h2>
<h3>Batters</h3>{tbl(J['bat']['official'], 'bat')}
<img src="{rank_chart('bat')}">
<h3>Bowlers</h3>{tbl(J['bowl']['official'], 'bowl')}
<img src="{rank_chart('bowl')}">
<p class=sub>P is the pressure level, D is pressure minus usual, and both are in runs per 100 balls. The plain columns are raw 2025-26 numbers on pressure balls and on ordinary balls. For bowlers D is small and similar for everyone: the data shows a bowler's pressure-minus-usual gap barely repeats, so it is shrunk hard and bowling ranks rest mainly on pressure level.</p>

<h2 class=pb>5. Reliability and limitations</h2>
<ul>
<li><b>Split-half by match</b> (100 random splits, final score): batting {num('Batting','split-half by match, Spearman')}, bowling {num('Bowling','split-half by match, Spearman')}. <b>2025 v 2026</b>: batting {num('Batting','2025-only vs 2026-only')} (few players qualify in both seasons), bowling {num('Bowling','2025-only vs 2026-only')}.</li>
<li><b>Bootstrap over matches:</b> on average {boot['Batting']} of the 10 official batters and {boot['Bowling']} of the 10 official bowlers stay in the top 10. The top three or four on each side are firm. The order lower down is not.</li>
<li><b>Pre-declared alternatives</b> (one setting changed at a time):</li></ul>{sens}
<ul>
<li>The definition is built from RR's own weak spots, so this is a ranking of who wins the moments RR lose. Replacing it with a general rule (top 25% leverage) changes most of the batting list. That alternative is reported above.</li>
<li>No single moment passes a strict false-discovery test on its own. The live-ball filter, shrinkage and the both-seasons check (reported above) guard against chance.</li>
<li>The data has no birth dates, so age is not modelled. Recent form is handled by the 2026 weighting and the must-have-played-2026 rule.</li>
<li>Win-probability wicket costs are modest (about 1-10 runs). With wickets worth 1.5x, the lists keep 8-9 of 10.</li>
</ul>

<h2>6. For Rajasthan Royals: the 2027 auction</h2>
<p>This is the same score, filtered to players RR could buy. 2027 squad and release status were typed from the 17-18 November 2026 retention news and are not part of the supplied data.</p>
<h3>Bowlers RR can buy</h3>{tbl(J['bowl']['can_buy'], 'bowl', 'buy_rank')}
<h3>Batters RR can buy</h3>{tbl(J['bat']['can_buy'], 'bat', 'buy_rank')}
<p>Only two batters RR could buy have enough evidence in these moments, and only Ajinkya Rahane is above average. With a lower bar the next ones are {watch}. RR's batting fix is therefore more likely to come from within the squad or a trade than from the auction.</p>
<h3>RR's own squad on the same score</h3>{tbl(J['bat']['rr'], 'bat', rng=False)}{tbl(J['bowl']['rr'], 'bowl', rng=False)}

<h2 class=pb>Appendix: all 48 moments (IPL 2025-26, runs per 100 balls v league expectation)</h2><h3>Batting</h3>{mtable('bat')}<h3>Bowling</h3>{mtable('bowl')}
<h2>7. Reproducibility</h2>
<p>All steps are numbered scripts that read their inputs and write their outputs. Run run_all.sh from the supplied data; every threshold sits in config.py and was fixed before the rankings were read. The interactive dashboard shows the same numbers, including the full matchup audit.</p>
</body></html>"""
open(f"{C.OUT}/report.html", "w").write(html)
subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={C.OUT}/SupeRR_Round1_Report.pdf", f"file://{C.OUT}/report.html"], check=True, capture_output=True, timeout=180)
print("pdf written")
