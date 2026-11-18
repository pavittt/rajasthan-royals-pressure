"""All fixed choices for the pipeline, declared up front. Nothing below is tuned on the final rankings.
Every other script imports these values; change them here only."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("RAW_DIR", "/mnt/user-data/uploads")   # supplied data pack
DATA = os.path.join(ROOT, "data")                             # intermediate files
OUT = os.path.join(ROOT, "out")                               # deliverables
INPUTS = os.path.join(ROOT, "inputs")                         # hand-entered auction context (NOT supplied data)

SEED = 2027

# --- universe --------------------------------------------------------------------------------
# T20I matches count only when BOTH teams are among these nine nations (participant's choice).
TOP9 = ["India", "England", "West Indies", "Australia", "Sri Lanka", "South Africa",
        "Afghanistan", "New Zealand", "Pakistan"]
# Win-probability model + league expectation tables are fitted on 2022-2026 (stable baselines).
# Player scoring uses only 2025 and 2026.
SCORE_YEARS = [2025, 2026]

# --- evidence weights: one ball of this kind counts as w balls of evidence ------------------
# Applied to the ball COUNT used for shrinkage (effective balls), not only inside a mean,
# so T20I-heavy players genuinely carry less evidence and are pulled harder to the prior.
WEIGHTS = {("IPL", 2026): 1.00, ("IPL", 2025): 0.70,
           ("T20I", 2026): 0.50, ("T20I", 2025): 0.35}

# --- pressure definition ---------------------------------------------------------------------
# A ball is a PRESSURE ball when its Leverage Index (expected absolute change in win probability
# from the next delivery, from an out-of-fold win-probability model) is in the top quarter of all
# balls in the 2022-2026 IPL + top-nine T20I universe. Identical for batting and bowling.
PRESSURE_Q = 0.75            # used only by the general (non-RR) sensitivity run
PRESSURE = "RR"               # RR theory: pressure = RR's kept moments that are live (s00b)
RR_LEVER_Q = 0.5             # leverage filter quantile for RR moments
RR_IMPORTANCE = True         # weight pressure balls by RR's gap in that moment (mean 1, capped at 3)

# --- eligibility (fixed rules, all from the data) -------------------------------------------
ELIG = dict(
    active_2026=True,           # appeared in a 2026 IPL or top-nine T20I playing XI
    min_eff_pressure=40,        # >= 40 EFFECTIVE pressure balls (about seven overs), same number for both sides (RR moments are narrower than a top-25% rule)
    min_ipl_pressure=20,        # >= 20 of them from IPL (half): no T20I-only / never-IPL players
    bowl_balls_per_match=12,    # bowlers: >= 2 overs per match played in 2025-26 (no part-timers)
    min_ipl_matches_2026=5,     # current form: >= 5 IPL 2026 matches (T20Is alone do not prove 2026 IPL form)
    max_form_z_2026=-0.5,       # current form: out if his 2026 pressure value is more than half a standard error BELOW the league (z < -0.5)
)
# Shrinkage strength (between-player variance tau^2 of each component) is a LEAGUE parameter: estimated on
# 2022-2026 (all players with >= 60 pressure and 60 ordinary balls, unweighted) because two seasons are too few
# to estimate it stably. The prior MEAN is the 2025-26 eligible pool. No player's pre-2025 balls enter his score.
HYPER_YEARS = [2022, 2023, 2024, 2025, 2026]

# --- rating ------------------------------------------------------------------------------------
# SCORE = P + D, both empirical-Bayes shrunk, in runs per 100 balls versus league expectation:
#   P = value per pressure ball, from PRESSURE BALLS ONLY (absolute level when it matters; usual form cannot carry it)
#   D = P_raw - U_raw, pressure minus the player's OWN usual (ordinary-ball) level
# Each is shrunk toward the eligible-pool mean with its own league prior variance, so each moves a player exactly as
# far as its evidence supports. Same formula, same rules, for batters and bowlers.
OPP_ADJUST = True               # adjust each ball for the opponent's usual quality

# --- pre-declared sensitivity runs (07_sensitivity.py) ---------------------------------------
SENSITIVITY = {
    "rr_leverage_filter_top60pct": dict(RR_LEVER_Q=0.4),
    "rr_leverage_filter_top40pct": dict(RR_LEVER_Q=0.6),
    "rr_no_importance_weight": dict(RR_IMPORTANCE=False),
    "rr_only_moments_holding_up_both_seasons": dict(RR_SET="holdup"),
    "general_pressure_top25pct_leverage": dict(PRESSURE="LI", PRESSURE_Q=0.75),
    "t20i_weight_0": dict(T20I_MULT=0.0),
    "t20i_equal_ipl": dict(T20I_MULT="equal"),
    "years_equal": dict(YEARS_EQUAL=True),
    "prior_var_x2": dict(TAU_MULT=2.0),
    "prior_var_x0.5": dict(TAU_MULT=0.5),
    "prior_var_2025_26_only": dict(HYPER_FROM_SCORE_YEARS=True),
    "min_sample_60": dict(ELIG_OVERRIDE=dict(min_eff_pressure=60, min_ipl_pressure=30)),
    "min_sample_25": dict(ELIG_OVERRIDE=dict(min_eff_pressure=25, min_ipl_pressure=12)),
    "looser_form_rule_z_minus1": dict(ELIG_OVERRIDE=dict(max_form_z_2026=-1.0)),
    "no_current_form_rule": dict(ELIG_OVERRIDE=dict(min_ipl_matches_2026=0, max_form_z_2026=-1e9)),
    "wicket_cost_x1.5": dict(WCOST_MULT=1.5),
    "score_usual_plus_2D": dict(SCORE="U2D"),
    "bivariate_predictive": dict(MODEL="bivariate"),
    "no_opp_adjust": dict(OPP_ADJUST=False),
    "score_P_only": dict(SCORE="P"),
    "score_D_only": dict(SCORE="D"),
    "score_P_plus_2D": dict(SCORE="P2D"),
}
N_BOOT = 1000
N_SPLIT = 100
