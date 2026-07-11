"""Vectorized, parameter-tunable twin of wc_model — for the calibration search ONLY.

WHY: wc_model.fit() brute-forces an (lambda_h, lambda_a) grid per match in pure Python (~0.4s each),
so one pass over the 3,504-match backtest corpus is ~25 min. A stage-2 calibration search re-runs the
corpus for every candidate parameter set, which is infeasible at that speed. KEY INSIGHT: the (lh,la)
candidate GRIDS are identical across matches — only the fit OBJECTIVE (de-vigged targets, O/U,
supremacy) changes per match. So we precompute the grid tensor ONCE per parameter set and vectorize
the per-match fit. Result: a full corpus pass drops from ~25 min to well under a second.

This is efficiency WITHOUT sacrifice: same Poisson PMF, same Dixon-Coles tau, same RHO, same grid
resolution (steps), same fit objective, same argmin/argmax tie-break ORDER as wc_model. The __main__
asserts bit-equivalence (identical evpick on every live WC fixture) so any drift is caught loudly.

Parameters exposed for stage-2 calibration (defaults = wc_model production values):
  rho, total_w, sup_w  — Dixon-Coles correlation + fit weights, currently hand-set, not data-fit.
"""
import math
import numpy as np
import wc_model as M

GRID_MAX = M.GRID_MAX            # 11
LAM_MAX = M.LAM_MAX              # 5.5
STEPS = np.array([i / 20 for i in range(2, int(LAM_MAX * 20) + 1)])  # identical to wc_model.fit
NG = GRID_MAX
NS = len(STEPS)
_N = np.arange(NG)

# Poisson PMF table: P[s, n] = pois(n, STEPS[s])  (vectorized, == math version within fp eps)
_logfact = np.array([math.lgamma(n + 1) for n in range(NG)])
_POIS = np.exp(-STEPS[:, None] + _N[None, :] * np.log(STEPS[:, None]) - _logfact[None, :])

# result masks over the (x, y) score grid
_X, _Y = np.meshgrid(_N, _N, indexing="ij")
_MH = (_X > _Y).astype(float)
_MD = (_X == _Y).astype(float)
_MA = (_X < _Y).astype(float)

# pool-scoring tensor PTS[hp, ap, H, A] under 12/5/2 (matches wc_model.pts exactly)
_R7 = np.arange(7)
def _pts(hp, ap, H, A):
    if hp == H and ap == A:
        return 12
    return (5 if (hp > ap) == (H > A) and (hp < ap) == (H < A) else 0) + (2 if (hp == H or ap == A) else 0)
_PTS = np.array([[[[_pts(hp, ap, H, A) for A in range(NG)] for H in range(NG)]
                  for ap in _R7] for hp in _R7], dtype=float)  # (7,7,11,11)


def build(rho=M.RHO):
    """Precompute (per rho) the normalized grid tensor + result-prob and shape tensors.
    Returns a dict of constants reused across all matches and all (total_w, sup_w) objectives."""
    # unnormalized grid G[i,j,x,y] = pois(x,STEPS[i]) * pois(y,STEPS[j])
    G = _POIS[:, None, :, None] * _POIS[None, :, None, :]
    G = np.array(G)  # writable copy
    si = STEPS[:, None]; sj = STEPS[None, :]
    # Dixon-Coles tau on the 4 low cells, clipped at 0 (== wc_model.tau + max(0,.))
    G[:, :, 0, 0] *= np.maximum(0.0, 1 - si * sj * rho)
    G[:, :, 0, 1] *= np.maximum(0.0, 1 + si * rho)          # tau(0,1)=1+lh*rho, lh=STEPS[i]
    G[:, :, 1, 0] *= np.maximum(0.0, 1 + sj * rho)          # tau(1,0)=1+la*rho, la=STEPS[j]
    G[:, :, 1, 1] *= np.maximum(0.0, 1 - rho)
    S = G.sum(axis=(2, 3))
    Gn = G / S[:, :, None, None]
    pw = (Gn * _MH).sum(axis=(2, 3))
    pd = (Gn * _MD).sum(axis=(2, 3))
    pl = (Gn * _MA).sum(axis=(2, 3))
    total = STEPS[:, None] + STEPS[None, :]
    sup = STEPS[:, None] - STEPS[None, :]
    return {"Gn": Gn, "pw": pw, "pd": pd, "pl": pl, "total": total, "sup": sup, "rho": rho}


def fit_idx(C, tw, td, tl, ou, tgt, total_w=M.TOTAL_W, sup_w=M.SUP_W):
    """Vectorized wc_model.fit: returns flat-argmin (i,j) index. Tie-break = first in (i-major,
    j-minor) order, identical to wc_model's strict-less-than loop + C-order flatten."""
    obj = (C["pw"] - tw) ** 2 + (C["pd"] - td) ** 2 + (C["pl"] - tl) ** 2 \
        + total_w * ((C["total"] - ou) / ou) ** 2
    if tgt is not None:
        obj = obj + sup_w * ((C["sup"] - tgt) / 2.0) ** 2
    k = int(np.argmin(obj))
    return divmod(k, NS)


def evpick(g, eps=M.EV_EPS):
    """Vectorized wc_model.evpick on a (11,11) grid g. Same within-class exact-prob tiebreak + the
    same DETERMINISTIC geometry resolution (fewest total goals, then lowest (hp,ap)) wc_model uses on
    residual EV/prob ties — so einsum-vs-scalar-sum ulp noise can never flip the pick vs wc_model
    (see M._EV_TOL/M._PROB_TOL). __main__ asserts bit-equivalence across every live fixture."""
    ev = np.einsum("haHA,HA->ha", _PTS, g)         # (7,7) expected points per candidate
    k = int(np.argmax(ev))                          # first max in (hp-major, ap-minor) order
    hp0, ap0 = divmod(k, 7)
    ev_cls = M.rcls(hp0, ap0)
    mx = float(ev[hp0, ap0])
    allowed = [(float(g[hp, ap]), (hp, ap))
               for hp in range(7) for ap in range(7)
               if ev[hp, ap] >= mx - max(eps, M._EV_TOL) and M.rcls(hp, ap) == ev_cls]
    pmax = max(p for p, _ in allowed)
    tied = [hpap for p, hpap in allowed if p >= pmax - M._PROB_TOL]
    return min(tied, key=lambda t: (t[0] + t[1], t))


def grid_for(C, i, j):
    return C["Gn"][i, j]


if __name__ == "__main__":
    # EQUIVALENCE GATE: fast_model must reproduce wc_model.evpick on every live WC fixture exactly.
    import json, os
    C = build()
    fixtures = json.load(open(os.path.expanduser("~/wc-pool/data/fixtures.json")))
    n = bad = 0
    max_grid_err = 0.0
    for f in fixtures:
        o = f.get("odds")
        if f["state"] != "pre" or not o or any(o.get(k) is None for k in ("hml", "dml", "aml")):
            continue
        n += 1
        # reference
        lh_ref, la_ref = M.fit(o)
        g_ref = M.grid(lh_ref, la_ref)
        pick_ref = M.evpick(g_ref)
        # fast
        tw, td, tl = M.devig(o)
        ou = o.get("ou") or 2.5
        sp = o.get("spread")
        tgt = (-float(sp)) if sp is not None else None
        i, j = fit_idx(C, tw, td, tl, ou, tgt)
        g_fast = grid_for(C, i, j)
        pick_fast = evpick(g_fast)
        # compare grid numerically + pick exactly
        g_ref_arr = np.array([[g_ref.get((x, y), 0.0) for y in range(NG)] for x in range(NG)])
        max_grid_err = max(max_grid_err, float(np.abs(g_ref_arr - g_fast).max()))
        if pick_ref != pick_fast or (STEPS[i], STEPS[j]) != (lh_ref, la_ref):
            bad += 1
            print(f"MISMATCH {f['home']} v {f['away']}: ref lam=({lh_ref},{la_ref}) pick={pick_ref} "
                  f"| fast lam=({STEPS[i]},{STEPS[j]}) pick={pick_fast}")
    print(f"checked {n} fixtures | mismatches: {bad} | max |grid diff| = {max_grid_err:.2e}")
    assert bad == 0, "fast_model diverged from wc_model"
    print("EQUIVALENCE OK — fast_model is a faithful vectorized twin of wc_model")
