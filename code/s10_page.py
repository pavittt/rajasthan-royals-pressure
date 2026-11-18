"""Step 10. Build the self-contained dashboard page (out/dashboard.html) from out/final_lists.json."""
import json, config as C
J = json.load(open(f"{C.OUT}/final_lists.json"))
for s in ("bat", "bowl"): J[s].pop("all", None)
J.pop("reliability_md", None)
J["grid"] = [{k: g[k] for k in ("side", "a", "b", "c", "rr_balls", "rr_raw_wins", "p_rr_below", "rr_sr", "lg_sr", "rr_econ", "lg_econ", "verdict")} for g in J["grid"] if g["grid"] == "matchup"]
rel = open(f"{C.OUT}/reliability.md").read()
import re
def grab(side, label):
    m = re.search(rf"## {side}.*?{re.escape(label)}[^|]*\|\s*([-\d.]+)", rel, re.S); return float(m.group(1)) if m else None
J["rel"] = {s: dict(split=grab(n, "split-half by match, Spearman"), year=grab(n, "2025-only vs 2026-only"),
                    boot=float(re.search(rf"## {n}.*?on average ([\d.]+) of the official", rel, re.S).group(1)))
            for s, n in (("bat", "Batting"), ("bowl", "Bowling"))}
J["form"] = json.load(open(f"{C.OUT}/form_check.json"))
J["plan"] = json.load(open(f"{C.OUT}/auction_plan.json"))["rows"]
J["moves"] = json.load(open(f"{C.OUT}/moves_check.json"))["rows"]
J["cmp"] = json.load(open(f"{C.OUT}/squad_compare.json"))["rows"]
S = J["sensitivity"]; nalt = len({x["scenario"] for x in S}) - 1
ov = lambda sc, sd: next(x["overlap"] for x in S if x["scenario"] == sc and x["side"] == sd)
alts = [x for x in S if x["scenario"] != "OFFICIAL"]
lo = lambda sd: sorted(x["overlap"] for x in alts if x["side"] == sd)
med = lambda sd: lo(sd)[len(lo(sd)) // 2]
kb = sum(1 for m in J["moments"] if m["side"] == "bat" and m["kept"]); kw = sum(1 for m in J["moments"] if m["side"] == "bowl" and m["kept"])
J["sims"] = [
 ["Win probability", "Ball-by-ball win-probability model, fitted out-of-fold by match", "Every IPL and top-nine T20I ball, 2022-26", "Gives each ball's leverage (how much the result can swing) and the win cost of a wicket"],
 ["Moment search", "Narrow situations tested in every phase (over band plus up to two details)", "1,169 situations", "48 moments after removing duplicates, 24 batting and 24 bowling"],
 ["Champions luck test", "Random 'champions' drawn from each season, compared with the real champions", "1,000 draws per situation", "Kept only where the champions' edge had under a 10% chance of luck"],
 ["RR shortfall test", "RR's 2025-26 matches resampled", "1,000 resamples per situation", "Kept only where RR were below the league in at least 85% of resamples"],
 ["Matchup audit", "Batting role x hand x six bowler types x phase, and pressure situations x bowler type and x role, with false-discovery control and shrinkage of thin cells", "588 cells", "Confirms the moments; no unchecked matchup is left out"],
 ["Three-way splits", "Decision-tree search of RR's shortfalls, against shuffled results", "300 shuffles", "No three-way split beat luck, so moments stop at two details"],
 ["Hold-up check", "Moments where RR were below the league in both 2025 and 2026 and RCB above", "48 moments", f"{kb} batting and {kw} bowling moments hold up; ranking on them alone is an alternative below"],
 ["Pressure v ordinary", "Leverage of pressure balls against ordinary balls", "All 2025-26 balls", f"Pressure balls swing the result {J['bat']['pressure']['swing_ratio_pressure']:.1f}x (batting) and {J['bowl']['pressure']['swing_ratio_pressure']:.1f}x (bowling) as much"],
 ["Shrinkage", "Empirical-Bayes spread of P, U and D estimated on 2022-26", "All players with 60+ pressure and 60+ ordinary balls", "Small samples pulled toward the pool; no pre-2025 ball enters a player's own score"],
 ["Split-half", "Matches split at random into two halves, final score compared", "100 splits per side", f"Spearman {J['rel']['bat']['split']:.2f} batting, {J['rel']['bowl']['split']:.2f} bowling (low)"],
 ["Season to season", "2025-only ranking against 2026-only ranking", "Players with 30+ pressure balls in both", f"Spearman {J['rel']['bat']['year']:.2f} batting, {J['rel']['bowl']['year']:.2f} bowling"],
 ["Bootstrap", "Matches resampled, whole score recomputed", "1,000 resamples per side", f"{J['rel']['bat']['boot']:.1f} of 10 batters and {J['rel']['bowl']['boot']:.1f} of 10 bowlers stay in the top 10 on average"],
 ["Alternatives", "Pre-declared changes, one setting at a time, whole pipeline re-run", f"{nalt} alternatives per side", f"Median overlap with the official top 10: {med('bat')}/10 batting, {med('bowl')}/10 bowling; no current-form rule {ov('no_current_form_rule','bat')}/10 and {ov('no_current_form_rule','bowl')}/10"],
 ["Form check", "IPL 2025 and IPL 2026 plain numbers for every listed player", "All 20 picks and every buy", "Every pick played 5+ IPL matches in 2026 and is not below the league under pressure in 2026"],
]
html = open(f"{C.ROOT}/page_template.html").read().replace("/*DATA*/null", json.dumps(J, separators=(",", ":")))
open(f"{C.OUT}/dashboard.html", "w").write(html)
print("dashboard.html", len(html), J["rel"])
