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
beta_field is a sensitivity knob; the P(top-K) table is reported across a beta range, while the
deviation verdict runs at the central beta=2.0 (its anti-noise bar is the $-floor + z-gate, not
beta robustness — re-check across betas by hand before acting on any DEVIATE).

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

# Observed standings (Felix). Loaded from data/standings.json when present — schema
# {"us": int, "top": [opponent scores, desc, EXCLUDING us], "n_field": int, "games_played": int} —
# so the endgame monitor never runs on a stale hardcoded leaderboard (adv 2026-07-11 MAJOR-3: the
# old June-14 literals, US_SCORE=40 after 8 games, were still live in July). The fallback literals
# below are the 2026-07-11 screenshot (us=540, rank 2). Only the TOP matters for P(top-K) at small K.
def _load_standings():
    try:
        d = json.load(open(f"{DATA}/standings.json"))
        return list(d["top"]), int(d["n_field"]), int(d["us"]), int(d["games_played"])
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        # LOUD fallback (adv 2026-07-11 R3 MAJOR-2): a silently-swallowed corrupt standings file
        # would re-open the stale-leaderboard class this loader exists to close.
        print(f"*** WARNING: data/standings.json unusable ({e.__class__.__name__}: {e}) — falling "
              f"back to the HARDCODED 2026-07-11 snapshot (us=540 / leader 547). Any verdict below "
              f"runs on that snapshot; fix the file if the pool has moved on. ***")
        return [547, 524, 518, 517, 504, 501, 494, 494, 494], 220, 540, 98

OBSERVED_TOP, N_FIELD, US_SCORE, GAMES_PLAYED = _load_standings()
KS = (1, 3, 5, 10, 20)
# Committed opponent picks are drawn ONCE per seed, so single-seed P(top-K) numbers are dominated
# by field-draw noise exactly in the endgame regime this file exists for (adv 2026-07-11 MAJOR-3:
# base P(top1) swung 12%->42% across seeds). The P(top-K) REPORT table therefore averages over
# SEEDS below and prints the spread. The DEVIATION decision rule is separate — see the PRIZES/DEV_*
# block (draw-averaged mean over DEV_DRAWS, $-floor + z-gate; per-draw consistency deliberately NOT
# required).
SEEDS = (7, 11, 23, 42, 101)


def load_state():
    """(remaining_grids, our_picks) for upcoming priced games from the live overlaid fixtures.
    Picks use the SAME rule as daily_run: ko_pick (ET-corrected recorded basis + shootout free-roll)
    for KNOCKOUT-bucketed games, plain evpick otherwise — so the monitor evaluates the sheet we
    actually submit, not a group-stage approximation (adv 2026-07-11 MAJOR-3 sub-finding)."""
    import daily_run as DR
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
        if DR.bucket(f["date"])[1].startswith("KNOCKOUT"):
            gd = {(x, y): float(g[x, y]) for x in range(g.shape[0]) for y in range(g.shape[1])}
            lh, la = float(F.STEPS[i]), float(F.STEPS[j])
            pk, _so = DR.ko_pick(gd, o, lh, la)
            # simulate KO outcomes on the same RECORDED-score basis Felix grades (90' non-draws
            # stand; draws extended through ET). Mass beyond the 11x11 sim grid (<1e-6) is clamped
            # into the edge cell. The +5 shootout side-bet is NOT simulated (none of our picks and
            # only the field's rare draw picks carry it — a small, symmetric-enough undercount).
            R, _pp = DR.ko_recorded_grid(gd, lh, la)
            g = np.zeros_like(g)
            for (H, A), p in R.items():
                g[min(H, g.shape[0] - 1), min(A, g.shape[1] - 1)] += p
        else:
            pk = F.evpick(g)
        grids.append(g); picks.append(tuple(pk))
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


# Full-pot payout, 1st..5th (the group-stage pot is already settled; only this one is still in play).
PRIZES = (450.0, 250.0, 150.0, 75.0, 50.0)
# A deviation is believed only when its DRAW-AVERAGED prize-EV gain clears DEV_MIN_GAIN dollars
# (2.5% of the $200 1st-vs-2nd spread — below that, field-model error dominates) AND the mean is
# statistically separated from zero (one-sided z > DEV_Z over DEV_DRAWS independent field draws).
# WHY this aggregation (adv 2026-07-11 R2 MAJOR): a fixed small-K objective (P(top3)) saturates near
# 100% for a high-ranked player and can never fire; and any per-draw consistency requirement (argmax
# unanimity, or a hard positive-in-k-of-n sign bar) misreads genuine posterior uncertainty about the
# LEADER'S committed pick as sim noise — near the endgame a candidate's delta is often bimodal
# (large + when the leader's sampled pick matches EV-max, large − when it doesn't), so per-draw signs
# legitimately flip while the posterior MEAN, the actual decision quantity, is well-defined.
# Averaging each CANDIDATE's delta across many draws estimates that mean; DEV_Z guards against
# firing on draw-sampling error without ever being structurally unable to fire. NOTE: the z is
# computed on the max over ~144 correlated candidates with the SAME draws that selected it (no
# multiplicity correction), so it is NOT a calibrated 5% false-fire rate — the $-floor
# DEV_MIN_GAIN is the load-bearing anti-noise bar (measured: z alone breached 1.645 in 3/8
# independent null blocks; the $5 floor blocked all of them).
DEV_MIN_GAIN = 5.0
DEV_DRAWS = tuple(range(1000, 1025))    # SELECT stage: field draws the best candidate is chosen on
# CONFIRM stage (adv 2026-07-11 R3 MAJOR-1): the selected candidate is the max over ~144 correlated
# candidates evaluated on the SAME draws that chose it, so its in-sample mean is winner's-curse
# inflated (measured: +$1.0 mean / +$3.4 max inflation on null blocks; ~10-20% spurious-fire rate in
# a reconstructed tight final, where per-draw deltas are heavy-tailed, SD $16-27). The $-floor +
# z-gate is therefore applied to a fresh re-estimate of the ONE selected candidate on disjoint
# draws — a pre-registered single test, which the z is actually sized for. Measured twice (adv
# R3/R4 experiments, different harnesses): the split killed every observed false fire (0/24 null
# blocks post-fix vs ~10-20%/run pre-fix) while the genuine near-clone true positive passed both
# stages in both harnesses (R3: select +$16.0/confirm +$14.9; R4 variant: +$34.4/+$53.4).
DEV_CONFIRM_DRAWS = tuple(range(5000, 5025))
DEV_Z = 1.645                           # one-sided 95% (per stage)


def prize_pay(ctx, our_final):
    """Per-sim prize $ for our final scores under the pool payout (ties broken in our favor —
    slightly optimistic at boundaries, defensible on the exact-count tiebreak; see topk note)."""
    better = (ctx["field_final"] > our_final[None, :]).sum(axis=0)
    pay = np.zeros(ctx["n_sims"])
    for r, amt in enumerate(PRIZES):
        pay[better == r] = amt
    return pay


def deviation_table(ctx, our_picks, us_score=US_SCORE):
    """(base prize-EV$, {(game, alt_scoreline): delta prize-EV$}) for EVERY single-game deviation
    from the EV-max sheet, under ONE committed field draw (incremental re-scoring, exact)."""
    base_final = our_total(ctx, our_picks, us_score)
    base_ev = float(prize_pay(ctx, base_final).mean())
    out = {}
    for gi, (ehp, eap) in enumerate(our_picks):
        bc = _PTS[ehp, eap, ctx["Ho"][:, gi], ctx["Ao"][:, gi]]
        for hp in range(7):
            for ap in range(7):
                if (hp, ap) == (ehp, eap):
                    continue
                alt_final = base_final - bc + _PTS[hp, ap, ctx["Ho"][:, gi], ctx["Ao"][:, gi]]
                out[(gi, (hp, ap))] = float(prize_pay(ctx, alt_final).mean()) - base_ev
    return base_ev, out


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


def _count_unpriced():
    """Upcoming scoreable games with no usable 3-way line yet (unresolved bracket placeholders)."""
    try:
        fixtures = json.load(open(f"{DATA}/fixtures.json"))
    except (FileNotFoundError, ValueError):
        return 0
    return sum(1 for f in fixtures
               if f.get("state") == "pre" and (not f.get("odds") or
                  any((f.get("odds") or {}).get(k) is None for k in ("hml", "dml", "aml"))))


def main():
    # standings freshness gate (adv 2026-07-11 R3 MAJOR-2): the gap to the leader is THE decision
    # variable, and every played game requires a manual standings.json update. Refuse to compute a
    # verdict when the snapshot's games_played disagrees with the fixtures' played count (stale
    # standings OR stale fixtures — either way the verdict would be built on the wrong gap).
    try:
        _fx = json.load(open(f"{DATA}/fixtures.json"))
        n_post = len({f.get("id") for f in _fx if f.get("state") == "post"})
    except (FileNotFoundError, ValueError):
        n_post = None
    if n_post is not None and n_post != GAMES_PLAYED:
        print(f"*** ABORT: standings.json says games_played={GAMES_PLAYED} but fixtures.json shows "
              f"{n_post} played games. Update data/standings.json from the live app (us / top / "
              f"n_field / games_played) — or refresh fixtures — and re-run. No verdict on a stale "
              f"gap. ***")
        return
    grids, picks = load_state()
    if len(picks) == 0:   # transiently reachable between a round finishing and the next being priced
        print("no priced upcoming games — nothing to monitor (re-run after the next odds pull).")
        return
    exp_future = sum(float((_PTS[p[0], p[1]] * grids[g]).sum()) for g, p in enumerate(picks))
    print(f"remaining priced games: {len(grids)} | our score: {US_SCORE} | field N={N_FIELD}")
    print(f"our expected future points: ~{exp_future:.0f}  (vs ~{OBSERVED_TOP[0]-US_SCORE} gap to 1st now)\n")

    print(f"P(finish top-K) under the EV-max sheet (mean +/- spread over {len(SEEDS)} field draws):")
    print(f"  {'beta':>5s} {'top1':>12s} {'top3':>12s} {'top5':>12s} {'top10':>7s}")
    for beta in (1.5, 2.0, 3.0):
        acc = {K: [] for K in KS}
        for s in SEEDS:
            ctx = build_sim(grids, beta_field=beta, seed=s)
            p = topk(ctx, our_total(ctx, picks))
            for K in KS:
                acc[K].append(p[K])
        def ms(K):
            a = np.array(acc[K]); return f"{a.mean()*100:5.1f}±{a.std()*100:4.1f}%"
        print(f"  {beta:5.1f} {ms(1):>12s} {ms(3):>12s} {ms(5):>12s} {np.mean(acc[10])*100:6.1f}%")

    # deviation search on the PRIZE-WEIGHTED objective: draw-averaged per-candidate delta EV$ with
    # a z-separation criterion (see PRIZES/DEV_* comment for why not fixed-K / consistency bars).
    tables, bases = [], []
    for s in DEV_DRAWS:
        ctx = build_sim(grids, beta_field=2.0, seed=s)
        b, t = deviation_table(ctx, picks)
        bases.append(b); tables.append(t)
    agg = {k: np.array([t[k] for t in tables]) for k in tables[0]}
    mean_d = {k: float(v.mean()) for k, v in agg.items()}
    best_k = max(mean_d, key=mean_d.get)
    bv = agg[best_k]
    se = float(bv.std(ddof=1) / np.sqrt(len(bv)))
    npos = int((bv > 0).sum())
    gi, alt = best_k
    sel_z = mean_d[best_k] / se if se else float("inf")
    sel_pass = mean_d[best_k] > DEV_MIN_GAIN and sel_z > DEV_Z
    print(f"\nDeviation search (prize-weighted EV$, beta=2.0, {len(DEV_DRAWS)} select draws): "
          f"EV-max sheet EV$={np.mean(bases):.2f}")
    print(f"  selected candidate: game {gi} {picks[gi]}->{alt}  mean delta ${mean_d[best_k]:+.2f} "
          f"± SE {se:.2f} (z={sel_z:.1f}; positive in {npos}/{len(DEV_DRAWS)} draws)")
    # CONFIRM stage: re-estimate the ONE selected candidate on fresh disjoint draws (see the
    # DEV_CONFIRM_DRAWS comment — kills winner's-curse false fires; a real edge passes both).
    conf = []
    for s in DEV_CONFIRM_DRAWS:
        ctx = build_sim(grids, beta_field=2.0, seed=s)
        base_final = our_total(ctx, picks)
        bc = _PTS[picks[gi][0], picks[gi][1], ctx["Ho"][:, gi], ctx["Ao"][:, gi]]
        alt_final = base_final - bc + _PTS[alt[0], alt[1], ctx["Ho"][:, gi], ctx["Ao"][:, gi]]
        conf.append(float(prize_pay(ctx, alt_final).mean() - prize_pay(ctx, base_final).mean()))
    conf = np.array(conf)
    cm = float(conf.mean())
    cse = float(conf.std(ddof=1) / np.sqrt(len(conf)))
    cz = cm / cse if cse else float("inf")
    conf_pass = cm > DEV_MIN_GAIN and cz > DEV_Z
    print(f"  confirm ({len(DEV_CONFIRM_DRAWS)} fresh draws): mean ${cm:+.2f} ± SE {cse:.2f} "
          f"(z={cz:.1f})")
    deviate = sel_pass and conf_pass
    # horizon guard: with unpriced scoreable games still ahead (SF2/3rd-place/final placeholders),
    # the sim treats the priced games as ALL remaining variance, inflating deviation deltas — a
    # DEVIATE verdict is only decision-grade when every remaining game is priced.
    n_unpriced = _count_unpriced()
    if deviate and n_unpriced:
        print(f"  VERDICT: KEEP EV-MAX (deviation ${cm:+.2f} is HORIZON-TRUNCATED — "
              f"{n_unpriced} unpriced game(s) of variance remain; re-run when all games are priced)")
    else:
        print(f"  VERDICT: {'DEVIATE' if deviate else 'KEEP EV-MAX'} — "
              + (f"select ${mean_d[best_k]:+.2f} AND fresh-draw confirm ${cm:+.2f} (z={cz:.1f}) "
                 f"both clear the ${DEV_MIN_GAIN:.0f} + z>{DEV_Z} bar, and the full horizon is priced."
                 if deviate else
                 f"{'select' if not sel_pass else 'fresh-draw confirm'} stage fails the "
                 f"${DEV_MIN_GAIN:.0f} + z>{DEV_Z} bar "
                 f"(select ${mean_d[best_k]:+.2f} z={sel_z:.1f}; confirm ${cm:+.2f} z={cz:.1f})."))


if __name__ == "__main__":
    main()
