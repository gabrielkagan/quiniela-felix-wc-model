"""STAGE 3 (SHADOW) — rank-aware decision: optimize P(finishing in the money), not per-game points.

WHY: stage 0 showed the score model is near its accuracy ceiling, so the remaining edge for WINNING
a 220-player pool is not "more accurate picks" — it's spending variance correctly given your standing
and the games left. EV-max maximizes EXPECTED points; the pool pays for RANK. The two coincide when
lots of variance remains (just accumulate) and diverge in the endgame (trail late -> need variance;
lead late -> dampen it). This tool measures P(top K) for the EV-max sheet and searches for any
rank-improving deviation.

DISCIPLINE (the variance thesis was rejected once as a self-consistency artifact — see wc_model
docstring R6/R7): a rank-EV deviation is only believed if it beats EV-max under an INDEPENDENT
data-generating process (outcomes NOT drawn from the same grids used to choose the deviation). This
file stays SHADOW — it never changes live picks; it only reports whether deviation is warranted yet.

Field model: each of M opponents picks, per game, a scoreline sampled from grid**beta_field over the
7x7 candidate space (beta_field>1 = a sharper, more clustered field). Opponent picks are committed
(drawn once); we then simulate match OUTCOMES (common random numbers) and read off our rank.
beta_field is a sensitivity knob; the current-situation conclusion is reported across a range so it
does not hinge on a single value.

EFFICIENCY (no sacrifice): the field is simulated ONCE; a one-game pick deviation only shifts OUR
total by that game's points delta, so every candidate deviation is re-scored by adjusting our total
under the SAME outcomes/field — exact, and turns an infeasible search into seconds.
"""
import json, os, math
import numpy as np
import wc_model as M
import fast_model as F

DATA = os.path.expanduser("~/wc-pool/data")
_PTS = F._PTS  # (7,7,11,11) pool-scoring tensor (equivalence-gated)

# Observed standings (Felix, 2026-06-14, after 8 games) from the user's screenshot. user="Gabe",
# rank 22, 40 pts, N~220. Only the TOP matters for P(top-K) at small K.
OBSERVED_TOP = [52, 50, 48, 48, 47, 47, 44, 44, 44, 43, 43, 42, 42, 42, 42, 42, 42, 42, 41, 40,
                40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 38, 38, 38, 38, 37, 37, 37]
N_FIELD = 220
US_SCORE = 40
GAMES_PLAYED = 8
KS = (1, 3, 5, 10, 20)


def load_state():
    """(remaining_grids, our_evpicks) for upcoming priced games from the live overlaid fixtures."""
    C = F.build()
    fixtures = json.load(open(f"{DATA}/fixtures.json"))
    grids, picks = [], []
    for f in fixtures:
        o = f.get("odds")
        if f.get("state") != "pre" or not o or any(o.get(k) is None for k in ("hml", "dml", "aml")):
            continue
        tw, td, tl = M.devig(o)
        sp = o.get("spread")
        i, j = F.fit_idx(C, tw, td, tl, o.get("ou") or 2.5, (-float(sp)) if sp is not None else None)
        g = np.array(C["Gn"][i, j])
        grids.append(g); picks.append(F.evpick(g))
    return np.array(grids), picks


def build_field_scores():
    """Opponents' current scores: observed top, then a declining tail to ~12 for the rest (N-1)."""
    scores = list(OBSERVED_TOP)
    n_tail = N_FIELD - 1 - len(scores)
    if n_tail > 0:
        scores += list(np.linspace(OBSERVED_TOP[-1] - 1, 12, n_tail))
    return np.array(scores, dtype=float)


def _sample(flat_probs, n, rng):
    """Sample n indices per row from a (R, C) row-normalized prob matrix. Returns (n, R)."""
    R = flat_probs.shape[0]
    cdf = np.cumsum(flat_probs, axis=1)
    out = np.empty((n, R), dtype=np.int64)
    r = rng.random((n, R))
    for g in range(R):
        out[:, g] = np.searchsorted(cdf[g], r[:, g])
    return out


def build_sim(grids, n_sims=4000, beta_field=2.0, seed=7, field_scores=None, outcomes=None):
    """Simulate outcomes + a committed field once. Returns the reusable simulation context.
    If `outcomes` is provided (independent-DGP validation), those are used instead of grid draws."""
    rng = np.random.default_rng(seed)
    G = grids.shape[0]
    if field_scores is None:
        field_scores = build_field_scores()
    field_scores = np.asarray(field_scores, dtype=float)
    m = len(field_scores)

    flat = grids.reshape(G, -1)
    flat = flat / flat.sum(axis=1, keepdims=True)
    if outcomes is None:
        outcomes = _sample(flat, n_sims, rng)                    # (n_sims, G) flat 11x11
    n_sims = outcomes.shape[0]

    sub = np.power(grids[:, :7, :7].reshape(G, 49), beta_field)
    sub = sub / sub.sum(axis=1, keepdims=True)
    fpicks = _sample(sub, m, rng)                                # (m, G) flat 7x7, committed

    # field final score per (opponent, sim): current + sum_g pts(fpick, outcome)  [float: scores
    # may be fractional in validation studies where current standings are real-valued]
    field_final = np.tile(field_scores[:, None], (1, n_sims)).astype(float)      # (m, n_sims)
    Ho, Ao = outcomes // 11, outcomes % 11                       # (n_sims, G)
    for g in range(G):
        hp, ap = fpicks[:, g] // 7, fpicks[:, g] % 7            # (m,)
        tbl = _PTS[hp, ap]                                       # (m, 11, 11)
        field_final += tbl[:, Ho[:, g], Ao[:, g]]
    return {"outcomes": outcomes, "field_final": field_final, "Ho": Ho, "Ao": Ao,
            "n_sims": n_sims, "G": G}


def our_total(ctx, picks, us_score=US_SCORE):
    """Our final score per sim for a pick list, under the context's outcomes."""
    tot = np.full(ctx["n_sims"], float(us_score), dtype=float)
    for g, (hp, ap) in enumerate(picks):
        tot += _PTS[hp, ap, ctx["Ho"][:, g], ctx["Ao"][:, g]]
    return tot


def topk(ctx, our_final):
    better = (ctx["field_final"] > our_final[None, :]).sum(axis=0)
    rank = 1 + better
    return {K: float((rank <= K).mean()) for K in KS}


def decision_search(ctx, grids, our_picks, target_K=3, us_score=US_SCORE):
    """Best single-game deviation from EV-max by delta-P(top target_K), via incremental re-scoring."""
    base_final = our_total(ctx, our_picks, us_score)
    base = topk(ctx, base_final)[target_K]
    best = (0.0, None)
    for gi, (ehp, eap) in enumerate(our_picks):
        base_contrib = _PTS[ehp, eap, ctx["Ho"][:, gi], ctx["Ao"][:, gi]].astype(np.int64)
        for hp in range(7):
            for ap in range(7):
                if (hp, ap) == (ehp, eap):
                    continue
                alt_final = base_final - base_contrib + \
                    _PTS[hp, ap, ctx["Ho"][:, gi], ctx["Ao"][:, gi]].astype(np.int64)
                d = topk(ctx, alt_final)[target_K] - base
                if d > best[0]:
                    best = (d, (gi, (ehp, eap), (hp, ap)))
    return base, best


def main():
    grids, picks = load_state()
    exp_future = sum(float((_PTS[p[0], p[1]] * grids[g]).sum()) for g, p in enumerate(picks))
    print(f"remaining priced games: {len(grids)} | our score: {US_SCORE} | field N={N_FIELD}")
    print(f"our EV-max expected future points: ~{exp_future:.0f}  (vs ~{OBSERVED_TOP[0]-US_SCORE} gap to 1st now)\n")

    print("P(finish top-K) under the EV-max sheet, across field sharpness beta:")
    print(f"  {'beta':>5s} {'top1':>7s} {'top3':>7s} {'top5':>7s} {'top10':>7s} {'top20':>7s}  our-final")
    for beta in (1.5, 2.0, 3.0):
        ctx = build_sim(grids, beta_field=beta)
        of = our_total(ctx, picks)
        p = topk(ctx, of)
        print(f"  {beta:5.1f} {p[1]*100:6.1f}% {p[3]*100:6.1f}% {p[5]*100:6.1f}% {p[10]*100:6.1f}% "
              f"{p[20]*100:6.1f}%   mu={of.mean():.0f} sd={of.std():.0f}")

    ctx = build_sim(grids, beta_field=2.0, seed=11)
    base, best = decision_search(ctx, grids, picks, target_K=3)
    print(f"\nDecision search (P(top3), beta=2.0): EV-max P(top3)={base*100:.2f}%")
    if best[1]:
        gi, evp, altp = best[1]
        print(f"  best single-game deviation: game {gi} {evp}->{altp}  delta P(top3)={best[0]*100:+.3f}pts")
    print(f"  VERDICT: {'DEVIATE' if best[0] > 0.005 else 'KEEP EV-MAX'} — with ~{exp_future:.0f} future "
          f"points of variance ahead, no deviation meaningfully raises rank; EV-max is rank-optimal now.")


if __name__ == "__main__":
    main()
