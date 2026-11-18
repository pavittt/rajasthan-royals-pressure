"""Shared rating engine used by steps 5-8 (imported, never exec'd). Same code path for batters and bowlers."""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy import sparse
import config as C

PID = {"bat": "striker_id", "bowl": "bowler_id"}
PNAME = {"bat": "striker_name", "bowl": "bowler_name"}

def default_cfg(**kw):
    cfg = dict(PRESSURE_Q=C.PRESSURE_Q, T20I_MULT=None, YEARS_EQUAL=False, TAU_MULT=1.0, WCOST_MULT=1.0,
               OPP_ADJUST=C.OPP_ADJUST, SCORE="P+D", ELIG=dict(C.ELIG), WEIGHTS=dict(C.WEIGHTS),
               HYPER_FROM_SCORE_YEARS=False, MODEL="univariate",
               PRESSURE=C.PRESSURE, RR_LEVER_Q=C.RR_LEVER_Q, RR_IMPORTANCE=C.RR_IMPORTANCE, RR_SET="all")
    cfg.update({k: v for k, v in kw.items() if k != "ELIG_OVERRIDE"})
    if "ELIG_OVERRIDE" in kw: cfg["ELIG"].update(kw["ELIG_OVERRIDE"])
    return cfg

def weights(cfg):
    W = dict(cfg["WEIGHTS"])
    if cfg["YEARS_EQUAL"]:
        W = {("IPL", 2026): 1.0, ("IPL", 2025): 1.0, ("T20I", 2026): 0.5, ("T20I", 2025): 0.5}
    if cfg["T20I_MULT"] == "equal":
        W[("T20I", 2026)] = W[("IPL", 2026)]; W[("T20I", 2025)] = W[("IPL", 2025)]
    elif cfg["T20I_MULT"] is not None:
        W[("T20I", 2026)] = W[("IPL", 2026)] * cfg["T20I_MULT"]; W[("T20I", 2025)] = W[("IPL", 2025)] * cfg["T20I_MULT"]
    return W

def load_balls(side, cfg, years=None, weighted=True):
    d = pd.read_parquet(f"{C.DATA}/{side}_balls.parquet")
    d = d[d.year.isin(years or C.SCORE_YEARS)].copy()
    if weighted:
        W = weights(cfg); d["w"] = [W[(f, y)] for f, y in zip(d.fmt, d.year)]; d = d[d.w > 0]
    else:
        d["w"] = 1.0
    if cfg["PRESSURE"] == "RR":
        tag = "RRH" if cfg.get("RR_SET") == "holdup" else "RRP"
        d["pres"] = d[f"{tag}{cfg['RR_LEVER_Q']}"]
        if cfg["RR_IMPORTANCE"] and weighted:
            d["w"] = np.where(d.pres, d.w * d[f"{tag}imp"], d.w)
    else:
        d["pres"] = d[f"P{cfg['PRESSURE_Q']}"]
    d["v"] = d.RV if cfg["OPP_ADJUST"] else d.RV0
    if cfg["WCOST_MULT"] != 1.0:
        o = d.striker_out if side == "bat" else d.bowler_wkt
        sign = -1.0 if side == "bat" else 1.0
        d["v"] = d.v + sign * (cfg["WCOST_MULT"] - 1) * d.wcost * (o.astype(float) - d.E_o)
    d["pid"] = d[PID[side]]; d["pname"] = d[PNAME[side]]
    return d

def player_match(d):
    """per player-match sufficient statistics (weighted sums) on pressure (P) and ordinary (U) balls"""
    d = d.assign(wv=d.w * d.v, ipl=(d.fmt == "IPL"))
    g = d.groupby(["pid", "match_id", "pres"]).agg(s=("wv", "sum"), n=("w", "sum"), k=("v", "size")).unstack("pres", fill_value=0)
    g.columns = [f"{a}{'P' if p else 'U'}" for a, p in g.columns]
    for c in ["sP", "nP", "kP", "sU", "nU", "kU"]:
        if c not in g: g[c] = 0.0
    ipl = d[d.pres & d.ipl].groupby(["pid", "match_id"]).w.sum().rename("nP_ipl")
    return g.join(ipl).fillna({"nP_ipl": 0}).reset_index()

def ball_var(d):
    """pooled within-player variance of ball value, separately for pressure / ordinary balls"""
    out = {}
    for flag, key in ((True, "P"), (False, "U")):
        x = d[d.pres == flag]
        dev = x.v - x.groupby("pid").v.transform("mean")
        out[key] = float(np.average(dev ** 2, weights=x.w))
    return out

def eb_fit(m, se2, w=None):
    """normal-normal empirical Bayes: m_i ~ N(mu, tau2 + se2_i). Marginal maximum likelihood for (mu, tau2)."""
    m, se2 = np.asarray(m, float), np.asarray(se2, float)
    def nll(lt):
        t2 = np.exp(lt); v = t2 + se2; mu = np.sum(m / v) / np.sum(1 / v)
        return 0.5 * np.sum(np.log(v) + (m - mu) ** 2 / v)
    r = minimize_scalar(nll, bounds=(np.log(1e-8), np.log(10.0)), method="bounded")
    t2 = float(np.exp(r.x)); v = t2 + se2; mu = float(np.sum(m / v) / np.sum(1 / v))
    return mu, t2

def eb_post(m, se2, mu, t2):
    B = t2 / (t2 + se2)
    return mu + B * (m - mu), np.sqrt(B * se2), B

def activity(side):
    xi = pd.read_parquet(f"{C.DATA}/xi.parquet")
    xi = xi[xi.universe]
    active = set(xi[xi.year == 2026].player_id)
    mp = xi[xi.year.isin(C.SCORE_YEARS)].groupby("player_id").match_id.nunique()
    return active, mp

def summarise(pm):
    t = pm.groupby("pid")[["sP", "nP", "kP", "sU", "nU", "kU", "nP_ipl"]].sum()
    t["mP"] = t.sP / t.nP.where(t.nP > 0); t["mU"] = t.sU / t.nU.where(t.nU > 0)
    return t

def eligibility(side, d, pm, cfg):
    """fixed data rules -> eligible player ids"""
    E = cfg["ELIG"]; active, mp = activity(side)
    t = summarise(pm)
    rule = pd.DataFrame(index=t.index)
    rule["active_2026"] = t.index.isin(active)
    rule["eff_pressure_balls"] = t.nP >= E["min_eff_pressure"]
    rule["ipl_pressure_balls"] = t.nP_ipl >= E["min_ipl_pressure"]
    xi = pd.read_parquet(f"{C.DATA}/xi.parquet")
    m26 = xi[(xi.fmt == "IPL") & (xi.year == 2026)].groupby("player_id").match_id.nunique()
    rule["ipl_matches_2026"] = m26.reindex(t.index).fillna(0).values >= E.get("min_ipl_matches_2026", 0)
    p25 = d[d.pres]; sd_ball = float(p25.v.std())
    f26 = p25[p25.year == 2026].groupby("pid").v.agg(["mean", "size"])
    z26 = (f26["mean"] / (sd_ball / np.sqrt(f26["size"]))).reindex(t.index).fillna(0.0)
    rule["ipl_form_2026"] = (z26 >= E.get("max_form_z_2026", -1e9)).values
    if side == "bowl":
        legal = d[d.legal].groupby("pid").size()
        bpm = legal.reindex(t.index).fillna(0) / mp.reindex(t.index).fillna(1).clip(lower=1)
        rule["bowling_role"] = bpm >= E["bowl_balls_per_match"]
    el = rule.index[rule.all(axis=1)]
    return el, rule

# ---- bivariate empirical Bayes: each player has a true (pressure level P, usual level U), drawn from a
# bivariate normal across players; observed per-ball means carry binomial-like noise sigma^2 / effective balls.
def _A(S, se2P, se2U):
    a = S[0, 0] + se2P; d = S[1, 1] + se2U; b = np.full_like(a, S[0, 1]); det = a * d - b * b
    return a, b, d, det

def biv_fit(mP, mU, se2P, se2U):
    """marginal maximum likelihood for mu (2) and Sigma (tauP, tauU, rho)"""
    from scipy.optimize import minimize
    mP, mU, se2P, se2U = map(lambda x: np.asarray(x, float), (mP, mU, se2P, se2U))
    def unpack(th):
        tP, tU, r = np.exp(th[2]), np.exp(th[3]), np.tanh(th[4])
        return th[:2], np.array([[tP * tP, r * tP * tU], [r * tP * tU, tU * tU]])
    def nll(th):
        mu, S = unpack(th); a, b, d, det = _A(S, se2P, se2U)
        x, y = mP - mu[0], mU - mu[1]
        q = (d * x * x - 2 * b * x * y + a * y * y) / det
        return 0.5 * np.sum(np.log(det) + q)
    th0 = np.array([mP.mean(), mU.mean(), np.log(0.1), np.log(0.1), 0.5])
    r = minimize(nll, th0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-7, fatol=1e-9))
    mu, S = unpack(r.x)
    return mu, S

def long_window_hyper(side, cfg):
    """League shrinkage parameters, estimated on 2022-26, unweighted, players with >= 60 pressure and >= 60 ordinary
    balls: univariate between-player variances of P, U, D (official model) and the bivariate (P, U) covariance
    (alternative 'predictive' model)."""
    years = C.SCORE_YEARS if cfg["HYPER_FROM_SCORE_YEARS"] else C.HYPER_YEARS
    d = load_balls(side, cfg, years=years, weighted=False)
    var = ball_var(d); t = summarise(player_match(d)); t = t[(t.nP >= 60) & (t.nU >= 60)]
    tau2 = {"P": eb_fit(t.mP, var["P"] / t.nP)[1], "U": eb_fit(t.mU, var["U"] / t.nU)[1],
            "D": eb_fit(t.mP - t.mU, var["P"] / t.nP + var["U"] / t.nU)[1]}
    _, S = biv_fit(t.mP, t.mU, var["P"] / t.nP, var["U"] / t.nU)
    return dict(tau2=tau2, S=S, n=len(t))

def biv_post(mP, mU, se2P, se2U, mu, S):
    """posterior means and sds of P, U and D = P - U"""
    a, b, d, det = _A(S, se2P, se2U)
    x, y = mP - mu[0], mU - mu[1]
    ix, iy = (d * x - b * y) / det, (-b * x + a * y) / det          # (Sigma + S_i)^-1 r
    P = mu[0] + S[0, 0] * ix + S[0, 1] * iy
    U = mu[1] + S[1, 0] * ix + S[1, 1] * iy
    # posterior covariance V = Sigma - Sigma (Sigma+S_i)^-1 Sigma
    i00, i01, i11 = d / det, -b / det, a / det
    def q(u, v):  # u^T Sigma A^-1 Sigma v for unit vectors
        Su = S @ u; Sv = S @ v
        return Su[0] * (i00 * Sv[0] + i01 * Sv[1]) + Su[1] * (i01 * Sv[0] + i11 * Sv[1])
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    vPP = S[0, 0] - q(e1, e1); vUU = S[1, 1] - q(e2, e2); vPU = S[0, 1] - q(e1, e2)
    return P, U, P - U, np.sqrt(np.maximum(vPP, 0)), np.sqrt(np.maximum(vUU, 0)), np.sqrt(np.maximum(vPP + vUU - 2 * vPU, 0))

def fit_hyper(t, var, cfg, LW):
    """prior variances from the long window (scaled by TAU_MULT); prior means = precision-weighted means of the
    2025-26 eligible pool"""
    hp = {"model": cfg["MODEL"]}
    for k, m, se in (("P", t.mP, var["P"] / t.nP), ("U", t.mU, var["U"] / t.nU),
                     ("D", t.mP - t.mU, var["P"] / t.nP + var["U"] / t.nU)):
        t2 = LW["tau2"][k] * cfg["TAU_MULT"]; v = t2 + se
        hp[k] = (float(np.sum(m / v) / np.sum(1 / v)), float(t2))
    S = LW["S"] * cfg["TAU_MULT"]
    se2P, se2U = var["P"] / t.nP.values, var["U"] / t.nU.values
    a, b, d, det = _A(S, se2P, se2U)
    x, y = t.mP.values, t.mU.values
    Ia, Ib, Id = d / det, -b / det, a / det
    M = np.array([[Ia.sum(), Ib.sum()], [Ib.sum(), Id.sum()]])
    hp["mu"] = np.linalg.solve(M, np.array([(Ia * x + Ib * y).sum(), (Ib * x + Id * y).sum()])); hp["S"] = S
    return hp

def components(mP, mU, se2P, se2U, hp):
    """official (univariate): P from pressure balls only, U from ordinary balls only, D from both, each shrunk with
    its own league prior.  alternative (bivariate): joint posterior of (P, U), D = P - U."""
    if hp["model"] == "bivariate":
        P, U, D, sP, sU, sD = biv_post(mP, mU, se2P, se2U, hp["mu"], hp["S"])
        return P, U, D, sP, sU, sD
    P, sP, _ = eb_post(mP, se2P, *hp["P"]); U, sU, _ = eb_post(mU, se2U, *hp["U"])
    D, sD, _ = eb_post(mP - mU, se2P + se2U, *hp["D"])
    return P, U, D, sP, sU, sD

COMBINE = {"P+D": lambda P, U, D: P + D, "P": lambda P, U, D: P, "D": lambda P, U, D: D,
           "P2D": lambda P, U, D: P + 2 * D, "U2D": lambda P, U, D: U + 2 * D}

def score_table(t, var, hp, cfg):
    """t: per-player sums for eligible players -> components and SCORE (runs per 100 balls vs league)"""
    t = t.copy()
    t["mP"] = t.sP / t.nP; t["mU"] = t.sU / t.nU; t["mD"] = t.mP - t.mU
    P, U, D, sP, sU, sD = components(t.mP.values, t.mU.values, var["P"] / t.nP.values, var["U"] / t.nU.values, hp)
    t["P"], t["U"], t["D"], t["P_sd"], t["U_sd"], t["D_sd"] = P, U, D, sP, sU, sD
    t["SCORE"] = COMBINE[cfg["SCORE"]](t.P, t.U, t.D)
    for c in ["mP", "mU", "mD", "P", "P_sd", "U", "U_sd", "D", "D_sd", "SCORE"]: t[c] *= 100
    return t

def rate(side, cfg=None, return_all=False):
    cfg = cfg or default_cfg()
    d = load_balls(side, cfg); pm = player_match(d); var = ball_var(d)
    el, rule = eligibility(side, d, pm, cfg)
    LW = long_window_hyper(side, cfg)
    t = summarise(pm).loc[el]
    hp = fit_hyper(t, var, cfg, LW)
    res = score_table(t, var, hp, cfg).sort_values("SCORE", ascending=False)
    res["rank"] = np.arange(1, len(res) + 1)
    names = d.groupby("pid").pname.agg(lambda s: s.mode().iat[0])
    res.insert(0, "player", names.reindex(res.index))
    Sg = hp["S"]
    info = dict(model=cfg["MODEL"], ball_variance=var, long_window_players=LW["n"],
                prior_mean_per100={k: hp[k][0] * 100 for k in "PUD"},
                prior_sd_per100={k: np.sqrt(hp[k][1]) * 100 for k in "PUD"},
                reliability_at_100_pressure_balls=hp["P"][1] / (hp["P"][1] + var["P"] / 100),
                bivariate_corr_P_U=float(Sg[0, 1] / np.sqrt(Sg[0, 0] * Sg[1, 1])),
                eligible=len(el), rule_pass={c: int(rule[c].sum()) for c in rule})
    if return_all: return res, dict(d=d, pm=pm, var=var, hp=hp, info=info, rule=rule, names=names)
    return res, info

def sparse_stats(pm, pids, mids):
    """players x matches sparse matrices of the sufficient statistics (for bootstrap / split-half)"""
    pi = {p: i for i, p in enumerate(pids)}; mi = {m: j for j, m in enumerate(mids)}
    x = pm[pm.pid.isin(pi)]
    r = x.pid.map(pi).values; c = x.match_id.map(mi).values
    return {k: sparse.csr_matrix((x[k].values, (r, c)), shape=(len(pids), len(mids))) for k in ["sP", "nP", "sU", "nU"]}

def scores_from_counts(S, cvec, var, hp, cfg):
    """vectorised SCORE from sparse stats and a per-match multiplicity vector (bootstrap/split weights)"""
    sP, nP, sU, nU = (np.asarray(S[k] @ cvec).ravel() for k in ["sP", "nP", "sU", "nU"])
    nP = np.maximum(nP, 1e-9); nU = np.maximum(nU, 1e-9)
    mP, mU = sP / nP, sU / nU
    P, U, D, *_ = components(mP, mU, var["P"] / nP, var["U"] / nU, hp)
    sc = COMBINE[cfg["SCORE"]](P, U, D)
    return dict(SCORE=sc * 100, P=P * 100, U=U * 100, D=D * 100, mP=mP * 100, mD=(mP - mU) * 100)
