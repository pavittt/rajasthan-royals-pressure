"""Step 1. Read the supplied data pack, clean deliveries, rebuild the match state before every ball.
Inputs : RAW/ball_by_ball.csv, *matches.csv, *competitions.csv, *teams.csv, *playing_eleven.csv
Outputs: data/balls.parquet, data/matches.parquet, data/xi.parquet"""
import glob, os
import numpy as np, pandas as pd
import config as C

def raw(pat):
    f = glob.glob(os.path.join(C.RAW, pat)); assert len(f) == 1, (pat, f); return f[0]

b = pd.read_csv(raw("ball_by_ball.csv"))
m = pd.read_csv(raw("*matches.csv"))
c = pd.read_csv(raw("*competitions.csv"))
t = pd.read_csv(raw("*teams.csv"))
xi = pd.read_csv(raw("*playing_eleven.csv"))

m = m.merge(c[["comp_id", "comp_name", "match_format"]], on="comp_id")
m["fmt"] = np.where(m.comp_name.str.startswith("IPL"), "IPL", np.where(m.match_format == "T20I", "T20I", "OTHER"))
m["year"] = pd.to_datetime(m.match_date).dt.year
top9_ids = set(t[t.team_name.isin(C.TOP9)].team_id)
m["top9"] = m.home_team_id.isin(top9_ids) & m.away_team_id.isin(top9_ids)
# universe used for pressure threshold and player scoring: all IPL + top-nine-vs-top-nine T20I
m["universe"] = (m.fmt == "IPL") | ((m.fmt == "T20I") & m.top9)
m["has_result"] = m.win_team_id.notna() & (m.win_team_id > 0)
team_name = dict(zip(t.team_id, t.team_name))
m["home_name"] = m.home_team_id.map(team_name); m["away_name"] = m.away_team_id.map(team_name)

b = b[b.innings.isin([1, 2])].copy()                              # drop super overs
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
b["extras_type"] = b.extras_type.fillna("None")
b["is_wide"] = b.extras_type.eq("wide")
b["is_nb"] = b.extras_type.isin(["Overstep", "other"])           # 'other' rows are beamer / full-toss no-balls
b["legal"] = ~(b.is_wide | b.is_nb)
b["total_runs"] = b.runs + b.extras1_runs + b.extras_2_runs
b["bowler_runs"] = b.runs + np.where(b.is_wide | b.is_nb, b.extras1_runs + b.extras_2_runs, 0)
b["faced"] = ~b.is_wide
NOT_BOWLER = ["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"]
b["striker_out"] = b.is_wicket & (b.dismissed_player_id == b.striker_id) & ~b.dismissal_type.isin(["Retire Out"])
b["bowler_wkt"] = b.is_wicket & ~b.dismissal_type.isin(NOT_BOWLER)
b["wkt"] = b.is_wicket & ~b.dismissal_type.isin(["Retire Out"])

g = b.groupby(["match_id", "innings"])
b["runs_before"] = g.total_runs.cumsum() - b.total_runs
b["wk_before"] = g.wkt.cumsum() - b.wkt
b["legal_before"] = g.legal.cumsum() - b.legal

keep = ["match_id", "fmt", "year", "universe", "has_result", "venue_id", "win_team_id", "stage", "match_date",
        "first_inn_revised_overs", "second_inn_revised_overs", "revised_target", "comp_name"]
b = b.merge(m[keep], on="match_id")
b["alloc"] = np.where(b.innings == 1, b.first_inn_revised_overs, b.second_inn_revised_overs)
b["alloc"] = b.alloc.fillna(120).clip(upper=120)
first_tot = b[b.innings == 1].groupby("match_id").total_runs.sum()
b["target"] = np.where(b.revised_target.notna(), b.revised_target, b.match_id.map(first_tot) + 1)
b["bat_won"] = (b.batting_team_id == b.win_team_id).astype(int)
b["over"] = (b.legal_before // 6 + 1).clip(upper=20)
b["phase"] = pd.cut(b.legal_before, [-1, 35, 89, 200], labels=["powerplay", "middle", "death"]).astype(str)
b["balls_left"] = (b.alloc - b.legal_before).clip(lower=0)
b["need"] = np.where(b.innings == 2, b.target - b.runs_before, np.nan)
b["rrr"] = np.where(b.innings == 2, b.need / np.maximum(b.balls_left, 1) * 6, np.nan)

xi = xi[xi.is_played.astype(str).str.lower() == "true"].merge(m[["match_id", "fmt", "year", "universe"]], on="match_id")
xi["team_name"] = xi.team_id.map(team_name)

os.makedirs(C.DATA, exist_ok=True)
b.to_parquet(f"{C.DATA}/balls.parquet"); m.to_parquet(f"{C.DATA}/matches.parquet"); xi.to_parquet(f"{C.DATA}/xi.parquet")
print("balls", len(b), "matches", len(m), "universe matches", int(m.universe.sum()))
print(m[m.universe].groupby(["fmt", "year"]).size().to_string())
