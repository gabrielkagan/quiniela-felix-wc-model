"""STAGE 3 validation (REBUILT) — does a rank-aware variance deviation EVER beat EV-max on the real
objective, under an honest model of the pool? If not, the everyday skill can recommend EV-max with
confidence. This replaces an earlier version whose TEST D produced a 0.00% "collapse" that an
adversarial review showed was a CODING ARTIFACT (we had placed ourselves strictly below the whole
field, so any rival copying our gamble was tautologically ahead). Fixes applied:

  1. OUR POSITION is a field percentile (trailing / mid / leading), never the strict minimum.
  2. RIVALS ARE DIVERSE: skill-heterogeneous (each samples picks from grid**beta_i), and "chasers"
     gamble on their OWN sampled upsets (grid**0.5), NOT identical clones of our one best gamble.
  3. REALISTIC FIELD (M large) and current-score spread; the "endgame" is encoded by how many games
     remain (W) relative to that spread.
  4. PAIRED CONFIDENCE INTERVALS: rank-EV vs EV-max are scored on the SAME windows/field/outcomes,
     and we report the paired delta in P(top K) with a 95% CI so a result inside Monte-Carlo noise is
     never reported as a finding.
  5. INDEPENDENT DGP preserved: deviations are CHOSEN on model grids, SCORED on ACTUAL results.

Decision rule under test: deviate from EV-max only if the grid-estimated dP(topK) exceeds tau (a
noise gate); pick the single-game deviation maximizing grid dP(topK) for the SAME K being evaluated.
"""
import csv, glob, os, math
import numpy as np
import wc_model as M
import fast_model as F
import stage3 as S3

BT = os.path.expanduser("~/wc-pool/data/backtest")
_PTS = F._PTS


def d2a(d):
    d = float(d)
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def load_corpus(C):
    out = []
    for path in sorted(glob.glob(os.path.join(BT, "*.csv"))):
        for r in csv.DictReader(open(path, encoding="latin-1")):
            try:
                H, A = int(r["FTHG"]), int(r["FTAG"])
                psh, psd, psa = float(r["PSH"]), float(r["PSD"]), float(r["PSA"])
                if min(psh, psd, psa) <= 1.0:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            try:
                sp = float(r["AHh"]) if r.get("AHh") not in ("", None) else None
            except ValueError:
                sp = None
            o = {"hml": d2a(psh), "dml": d2a(psd), "aml": d2a(psa), "ou": 2.5, "spread": sp}
            tw, td, tl = M.devig(o)
            i, j = F.fit_idx(C, tw, td, tl, 2.5, (-sp) if sp is not None else None)
            out.append((np.array(C["Gn"][i, j]), (H, A)))
    return out


def _opp_future(grids, actual, betas_group, chase_mask, rng):
    """Vectorized opponent future points on ACTUAL outcomes. Non-chasers sample from grid**2.0,
    chasers from grid**0.5 (gamble on upsets) — each opponent samples INDEPENDENTLY (diverse, not
    clones). betas_group is unused placeholder kept for clarity; group betas are fixed (2.0 / 0.5)."""
    W = grids.shape[0]
    Mf = chase_mask.shape[0]
    fut = np.zeros(Mf)
    for g in range(W):
        flat = grids[g, :7, :7].reshape(49)
        cdf_s = np.cumsum(np.power(flat, 2.0)); cdf_s /= cdf_s[-1]   # sharp (typical players)
        cdf_c = np.cumsum(np.power(flat, 0.5)); cdf_c /= cdf_c[-1]   # chasers (upset-seeking)
        u = rng.random(Mf)
        ks = np.where(chase_mask, np.searchsorted(cdf_c, u), np.searchsorted(cdf_s, u))
        hp, ap = ks // 7, ks % 7
        h, a = actual[g]
        fut += _PTS[hp, ap, h, a]
    return fut


def choose_deviation(grids, evpicks, K, us_score, field_scores, tau, n_sims, seed):
    """Choose the single-game deviation maximizing grid dP(topK); return picks (EV-max if edge<=tau)."""
    ctx = S3.build_sim(grids, n_sims=n_sims, beta_field=2.0, seed=seed, field_scores=field_scores)
    base, best = S3.decision_search(ctx, grids, evpicks, target_K=K, us_score=us_score)
    if best[1] and best[0] > tau:
        gi, _, alt = best[1]
        picks = list(evpicks); picks[gi] = alt
        return picks, True
    return list(evpicks), False


def study(corpus, gap, W, K, rho_chase=0.3, M_field=120, n_windows=2000, tau=0.0,
          sigma_cur=8.0, seed=0, n_sims=1500, M_decision=40):
    """Paired P(topK): EV-max vs disciplined rank-EV deviation, on ACTUAL outcomes, realistic field.
    We start `gap` points behind the top-K cutoff (the bubble where the gamble decision is live);
    gap reachable in W games (max 12W). Decision uses a subsampled field; eval uses the full field.
    Independent DGP: decide on grids, score on actual results."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(corpus))
    ev_hits, dev_hits, diffs, n_dev = 0, 0, [], 0
    for w in range(n_windows):
        sel = rng.choice(idx, size=W, replace=False)
        grids = np.array([corpus[k][0] for k in sel])
        actual = [corpus[k][1] for k in sel]
        evpicks = [F.evpick(g) for g in grids]

        field_cur = rng.normal(0.0, sigma_cur, M_field)         # heterogeneous current standings
        cutoff = float(np.quantile(field_cur, 1 - K / M_field))  # score of the K-th best opponent
        us_cur = cutoff - gap                                    # we sit `gap` behind the cutoff (bubble)
        # Chasers are the BUBBLE COMPETITORS — rivals just below the cutoff, within reach of top-K in
        # the W games left, who would rationally gamble to climb in. These are the ones whose gamble
        # CROWDS ours (the R7 collapse mechanism); bottom-of-table rivals can't reach the cutoff, so
        # making them chase is inert (R2 MAJOR). reachable = max points gainable in W games.
        reachable = 12 * W
        chase_mask = (field_cur < cutoff) & (field_cur > cutoff - reachable) & (rng.random(M_field) < rho_chase)
        opp_future = field_cur + _opp_future(grids, actual, None, chase_mask, rng)

        dec_field = field_cur[rng.choice(M_field, M_decision, replace=False)].tolist()
        devpicks, used = choose_deviation(grids, evpicks, K, us_cur, dec_field, tau, n_sims,
                                          seed=10_000 + w)
        n_dev += used
        ev_final = us_cur + sum(_PTS[p[0], p[1], h, a] for p, (h, a) in zip(evpicks, actual))
        dev_final = us_cur + sum(_PTS[p[0], p[1], h, a] for p, (h, a) in zip(devpicks, actual))
        ev_top = int(1 + int((opp_future > ev_final).sum()) <= K)
        dev_top = int(1 + int((opp_future > dev_final).sum()) <= K)
        ev_hits += ev_top; dev_hits += dev_top; diffs.append(dev_top - ev_top)
    n = n_windows
    md = float(np.mean(diffs)); se = float(np.std(diffs)) / math.sqrt(n)
    return {"ev": ev_hits / n, "dev": dev_hits / n, "delta": md, "ci": (md - 1.96 * se, md + 1.96 * se),
            "pct_dev": n_dev / n}


def main():
    C = F.build()
    corpus = load_corpus(C)
    print(f"corpus: {len(corpus)} matches | field M=120, diversified rivals, rho_chase=0.3, paired 95% CI")
    print("Scenario: we sit `gap` pts behind the top-K cutoff (the bubble) with W games left.")
    print("delta = P(topK) rank-EV minus EV-max, on ACTUAL outcomes. CI excluding 0 = real signal.\n")
    print(f"{'gap':>4s} {'W':>3s} {'K':>2s} {'%dev':>6s} {'EV-max':>8s} {'rank-EV':>8s} {'delta':>8s} {'95% CI':>18s}")
    for W in (2, 5):
        for gap in (0, 4, 8):
            for K in (1, 3):
                r = study(corpus, gap, W, K, seed=1, n_windows=5000)
                lo, hi = r["ci"]
                sig = "  <-- SIGNAL" if (lo > 0 or hi < 0) else ""
                print(f"{gap:>4d} {W:>3d} {K:>2d} {r['pct_dev']*100:5.0f}% {r['ev']*100:7.2f}% "
                      f"{r['dev']*100:7.2f}% {r['delta']*100:+7.2f}% [{lo*100:+6.2f},{hi*100:+6.2f}]{sig}")

    # rho sweep at the contested bubble (gap=0, W=2, K=1): confirm bubble-competitor chasing is LIVE
    # (rho now changes the field) and that EV-max still wins as rivals crowd the gamble (the R7 test).
    print("\nrho-sweep at gap=0, W=2, K=1 (bubble competitors chasing) — does EV-max survive crowding?")
    print(f"  {'rho':>5s} {'EV-max':>8s} {'rank-EV':>8s} {'delta':>8s} {'95% CI':>18s}")
    for rho in (0.0, 0.2, 0.5, 0.8):
        r = study(corpus, 0, 2, 1, rho_chase=rho, seed=1, n_windows=5000)
        lo, hi = r["ci"]
        print(f"  {rho:5.1f} {r['ev']*100:7.2f}% {r['dev']*100:7.2f}% {r['delta']*100:+7.2f}% "
              f"[{lo*100:+6.2f},{hi*100:+6.2f}]")


if __name__ == "__main__":
    main()
