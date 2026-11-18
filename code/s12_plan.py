"""Step 12. Auction plan (v9) on the fixed ranking: same six-slot structure, caps and rivals logic as the earlier plan;
only the player picks change. Output: out/auction_plan.json (dashboard dataset schema)."""
import json, numpy as np, pandas as pd
import config as C
J = json.load(open(f"{C.OUT}/final_lists.json")); R = pd.read_csv("/home/claude/work/rivals2027.csv")
ALL = {s: {r["player"]: r for r in J[s]["watch"] + J[s]["all"]} for s in ("bat", "bowl")}
AGE = {"Trent Boult": 37, "Deepak Chahar": 34, "Ajinkya Rahane": 38, "Allah Ghazanfar": 19, "Tushar Deshpande": 31, "Matheesha Pathirana": 24, "Mitchell Starc": 36, "Nitish Rana": 32}
OVS = {"Trent Boult", "Allah Ghazanfar", "Matheesha Pathirana", "Mitchell Starc", "Shimron Hetmyer"}
PLAN = [
 (1, "Overseas pacer for close games and overs 16-18", "Bowling: close games and overs 16-18, RR's biggest bowling leaks", 5.0, "compete_N3", [
   ("A", "Trent Boult", "bowl", "Highest-rated overseas pacer RR can buy; above the league in RR's moments over 2025-26 (22 IPL wickets in 2025). 2026 was weaker (5 matches, economy 11.6). Age 37: one-season buy.", "Price passes 5 cr"),
   ("B", "Matheesha Pathirana", "bowl", "Not ranked: only 1 IPL match in 2026. Strong in 2025 (13 wickets in close games and at the death). A fitness and form risk.", "Price passes 4 cr"),
   ("C", "Mitchell Starc", "bowl", "Not ranked: fails the 2026 form rule (economy 12.3 in RR's moments in 2026). Age 36.", "Price passes 3 cr")]),
 (2, "Indian new-ball pacer", "Bowling: new ball to right-handed openers", 4.0, "compete_N5", [
   ("A", "Deepak Chahar", "bowl", "Best bowler RR can buy on the final score; economy 8.7 in RR's moments in 2026 against 10.4 overall.", "Price passes 4 cr"),
   ("B", "Tushar Deshpande", "bowl", "Rated, but RR released him; economy 11.4 in RR's moments in 2026. Cheap fallback only.", "Price passes 1 cr")]),
 (3, "Overseas spinner", "Bowling: middle overs and close games", 4.0, "compete_N4", [
   ("A", "Allah Ghazanfar", "bowl", "Only spinner RR can buy who passes every rule; 15 IPL wickets in 2026. Age 19.", "Price passes 4 cr"),
   ("B", "Best uncapped spinner in the 3 December register", "bowl", "Too few balls in the data to rate uncapped players.", "")]),
 (4, "Indian batter for overs 7-10", "Batting: right-handers and No. 3-4 in overs 7-10", 2.0, "compete_N1", [
   ("A", "Ajinkya Rahane", "bat", "The only batter RR can buy who passes every rule (7th of 32 overall). Strike rate 183 in RR's moments in 2026. Age 38: a one-season buy.", "Price passes 2 cr"),
   ("B", "Nitish Rana", "bat", "Thin evidence (under 40 effective pressure balls).", "Price passes 1 cr")]),
 (5, "Third overseas place (only if a strong option appears)", "Batting: last two overs v right-arm pace", 4.0, "compete_N1", [
   ("A", "Keep the slot or trade", "bat", "No overseas batter RR can buy passes the rules (Hetmyer fails the 2026 form rule). Ferreira (RR) should finish.", "")]),
 (6, "Uncapped back-up", "Depth", 1.0, "compete_N4", [
   ("A", "Best uncapped player in the 3 December register", "bowl", "Too few balls in the data to rate uncapped players.", "")]),
]
rows = []
for slot, need, fixes, cap, comp, opts in PLAN:
    for opt, nm, side, why, sw in opts:
        r = ALL[side].get(nm); ovs = nm in OVS
        riv = R[(R[comp] >= 0.15) & (R.team != "Rajasthan Royals") & ((R.overseas_slots > 0) if ovs else True)].sort_values(comp, ascending=False).team.tolist()
        rows.append({"slot": slot, "need": need, "fixes": fixes, "slot_cap_cr": cap, "option": opt, "player": nm, "overseas": "Overseas" if ovs else "Indian",
                     "age": AGE.get(nm), "status": (r or {}).get("status", ""), "fix_score": None if not r else round(r["SCORE"], 2),
                     "recent_score": None if not r or r.get("pressure_RV100_2026") is None else round(r["pressure_RV100_2026"], 1),
                     "moment_balls": None if not r else r.get("pressure_balls", r.get("kP")), "stat": None if not r or r.get("pressure_stat") is None else round(r["pressure_stat"], 1),
                     "likely_rivals": ", ".join(riv[:3]) or "Few", "why": why, "switch_when": sw})
json.dump({"rows": rows, "meta": [{"label": "s12_plan.py", "value": open(__file__).read(), "type": "code", "lang": "python"}, {"label": "Input", "value": "final_lists.json, rivals2027.csv"}]}, open(f"{C.OUT}/auction_plan.json", "w"))
print(pd.DataFrame(rows)[["slot", "option", "player", "fix_score", "likely_rivals"]].to_string(index=False), sum(p[3] for p in PLAN))
