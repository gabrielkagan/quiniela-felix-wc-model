"""STAGE 2 — calibrate scoreline DISPERSION against history (train/test, no overfitting).

Stage 0 measured the headroom: the fitted Poisson grid is OVER-CONFIDENT on its high-probability
scorelines (predicts ~22% where reality is ~11%) — real football is over-dispersed. Result-level
calibration is already perfect, so we fix ONLY the within-result-class score shape and we must NOT
disturb the result probabilities.

THE FIX (minimal, principled): temperature beta on the grid, then renormalize WITHIN each result
class (H / D / A) back to the market's de-vigged result probabilities.
  g'(x,y) = g(x,y)^beta, rescaled per class so  sum_{x>y} g' = P_home,  sum_{x=y} g' = P_draw,
            sum_{x<y} g' = P_away   (all exactly preserved).
  beta = 1 -> identity;  beta < 1 -> flatter within-class shape (less over-confident on modal scores).

DISCIPLINE: beta is fit on TRAIN seasons only (smooth proper score = exact log-loss), then judged on
a held-out TEST season by the REAL objective (pool points) + calibration. It ships only if it beats
beta=1 out-of-sample. This is the same ship bar stage 1/3 must clear.
"""
import csv, glob, math, os, statistics
import numpy as np
import wc_model as M
import fast_model as F

BT = os.path.expanduser("~/wc-pool/data/backtest")
NG = M.GRID_MAX
_X, _Y = np.meshgrid(np.arange(NG), np.arange(NG), indexing="ij")
MH = (_X > _Y); MD = (_X == _Y); MA = (_X < _Y)
R7 = range(7)


def d2a(d):
    d = float(d)
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def load_rows_by_season():
    """Returns {season: [rows]} — season is the football-data file prefix (e.g. '2324')."""
    out = {}
    for path in sorted(glob.glob(os.path.join(BT, "*.csv"))):
        season = os.path.basename(path).split("_")[0]
        with open(path, encoding="latin-1") as fh:
            for r in csv.DictReader(fh):
                try:
                    H, A = int(r["FTHG"]), int(r["FTAG"])
                    psh, psd, psa = float(r["PSH"]), float(r["PSD"]), float(r["PSA"])
                    if min(psh, psd, psa) <= 1.0:
                        continue
                except (KeyError, ValueError, TypeError):
                    continue
                try:
                    spread = float(r["AHh"]) if r.get("AHh") not in ("", None) else None
                except ValueError:
                    spread = None
                out.setdefault(season, []).append({
                    "H": H, "A": A,
                    "o": {"hml": d2a(psh), "dml": d2a(psd), "aml": d2a(psa),
                          "ou": 2.5, "spread": spread}})
    return out


def targets(o):
    tw, td, tl = M.devig(o)
    ou = o.get("ou") or 2.5
    sp = o.get("spread")
    return tw, td, tl, ou, ((-float(sp)) if sp is not None else None)


def recalibrate(g, pw, pd, pl, beta):
    """Temperature-flatten then renormalize within each result class to (pw,pd,pl). beta=1 -> g."""
    if beta == 1.0:
        return g
    u = np.power(g, beta)
    out = np.zeros_like(g)
    for mask, target in ((MH, pw), (MD, pd), (MA, pl)):
        s = (u * mask).sum()
        if s > 0:
            out += u * mask * (target / s)
    return out


def precompute(rows, C):
    """Fit each match once; cache (grid, result-probs, actual). Reused across all beta."""
    cache = []
    for r in rows:
        i, j = F.fit_idx(C, *targets(r["o"]))
        cache.append((C["Gn"][i, j], float(C["pw"][i, j]), float(C["pd"][i, j]),
                      float(C["pl"][i, j]), r["H"], r["A"]))
    return cache


def evaluate(cache, beta):
    """Return (pts/game, exact%, mean exact-log-loss) for a given beta over a cached fold."""
    n = len(cache); tot = 0; exact = 0; ll = 0.0
    for g0, pw, pd, pl, H, A in cache:
        g = recalibrate(g0, pw, pd, pl, beta)
        hp, ap = F.evpick(g)
        tot += M.pts(hp, ap, H, A)
        exact += (hp, ap) == (H, A)
        pe = float(g[H, A]) if H < NG and A < NG else 0.0
        ll += -math.log(max(pe, 1e-12))
    return tot / n, 100 * exact / n, ll / n


def main():
    seasons = load_rows_by_season()
    keys = sorted(seasons)
    print(f"seasons: " + ", ".join(f"{k}({len(seasons[k])})" for k in keys))
    C = F.build()

    # train = earliest season(s), test = latest — strict temporal holdout (no leakage)
    train_keys, test_keys = keys[:-1], keys[-1:]
    train = [r for k in train_keys for r in seasons[k]]
    test = [r for k in test_keys for r in seasons[k]]
    print(f"train {train_keys} n={len(train)}  |  test {test_keys} n={len(test)}\n")

    tr = precompute(train, C)
    te = precompute(test, C)

    betas = [round(0.5 + 0.05 * k, 2) for k in range(0, 19)]  # 0.50 .. 1.40
    print(f"{'beta':>5s} | {'TRAIN pts':>9s} {'exact%':>7s} {'logloss':>8s} | {'TEST pts':>9s} {'exact%':>7s} {'logloss':>8s}")
    print("-" * 70)
    best_ll_beta, best_ll = 1.0, 1e9
    for b in betas:
        trp, tre, trl = evaluate(tr, b)
        tep, tee, tel = evaluate(te, b)
        if trl < best_ll:
            best_ll, best_ll_beta = trl, b
        mark = "  <- train-logloss-opt" if False else ""
        print(f"{b:5.2f} | {trp:9.4f} {tre:7.2f} {trl:8.4f} | {tep:9.4f} {tee:7.2f} {tel:8.4f}")

    # Decision: beta chosen on TRAIN log-loss, judged on TEST pool points vs beta=1
    base_p, base_e, _ = evaluate(te, 1.0)
    cal_p, cal_e, _ = evaluate(te, best_ll_beta)
    diffs = []
    for g0, pw, pd, pl, H, A in te:
        b1 = F.evpick(g0)
        b2 = F.evpick(recalibrate(g0, pw, pd, pl, best_ll_beta))
        diffs.append(M.pts(*b2, H, A) - M.pts(*b1, H, A))
    md = statistics.mean(diffs); se = statistics.pstdev(diffs) / math.sqrt(len(diffs))
    print(f"\nbeta* (train log-loss optimum) = {best_ll_beta}")
    print(f"TEST: beta=1 {base_p:.4f} pts/game ({base_e:.2f}% exact)  ->  beta*={best_ll_beta} {cal_p:.4f} ({cal_e:.2f}%)")
    print(f"TEST pick delta: {md:+.4f} pts/game  95% CI [{md-1.96*se:+.4f}, {md+1.96*se:+.4f}]"
          f"  ({md*64:+.2f} over a 64-game WC)")
    print("SHIP" if md - 1.96 * se > 0 else "DO NOT SHIP (calibration helps probabilities but not "
          "out-of-sample picks at significance) — keep beta=1, revisit on international data")


if __name__ == "__main__":
    main()
