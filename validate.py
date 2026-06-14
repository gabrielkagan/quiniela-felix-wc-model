"""STAGE 0 — validation harness (backtest the model on the REAL 12/5/2 objective).

First-principles purpose: before we add ANY data/sophistication, establish a measured baseline —
does the current market-anchored Dixon-Coles + EV-max pick rule actually beat naive rules on the
pool's scoring, and is its scoreline distribution calibrated? Every later stage (multi-book inputs,
dispersion calibration, rank-EV decision) must beat THIS harness's numbers out-of-sample to ship.

Corpus: football-data.co.uk closing odds (Pinnacle 3-way PSH/PSD/PSA + O/U 2.5 P>2.5/P<2.5 +
Asian-handicap AHh) joined to actual full-time scores (FTHG/FTAG). League-agnostic on purpose —
the model is market-anchored, so any league with a sharp closing line tests the SAME machinery
(de-vig -> grid -> evpick) we run on the World Cup.

Reuses wc_model.{fit,grid,evpick,pts,devig,rcls} verbatim — zero reimplementation, so a harness
"pass" is a statement about the production code, not a parallel copy of it.
"""
import csv, functools, glob, math, os, statistics, sys
import wc_model as M

# Speed: pois() is called with L only on the fit's 0.05 lambda-grid (~110 values) x n in 0..10,
# so ~1200 distinct calls. Caching it leaves results identical and makes fit() ~10x faster.
M.pois = functools.lru_cache(maxsize=None)(M.pois)

BT = os.path.expanduser("~/wc-pool/data/backtest")
RANGE = range(7)  # candidate scoreline space, matches evpick()

def d2a(d):
    """decimal odds -> american moneyline (wc_model.dec is the inverse)."""
    d = float(d)
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)

def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(BT, "*.csv"))):
        with open(path, encoding="latin-1") as fh:
            for r in csv.DictReader(fh):
                try:
                    H, A = int(r["FTHG"]), int(r["FTAG"])
                    psh, psd, psa = float(r["PSH"]), float(r["PSD"]), float(r["PSA"])
                    if min(psh, psd, psa) <= 1.0:
                        continue
                except (KeyError, ValueError, TypeError):
                    continue
                ah = r.get("AHh", "")
                try:
                    spread = float(ah) if ah not in ("", None) else None
                except ValueError:
                    spread = None
                rows.append({"H": H, "A": A,
                             "o": {"hml": d2a(psh), "dml": d2a(psd), "aml": d2a(psa),
                                   "ou": 2.5, "spread": spread}})
    return rows

def fav_score(o):
    """naive: pick the de-vigged favorite result, scored 1-0 / 0-1 / 1-1."""
    h, d, a = M.devig(o)
    if h >= d and h >= a:
        return (1, 0)
    if a >= d and a >= h:
        return (0, 1)
    return (1, 1)

def modal(g):
    return max(((hp, ap) for hp in RANGE for ap in RANGE), key=lambda s: g.get(s, 0.0))

def rps_ordered(p, actual):
    """Ranked Probability Score for ordered (H,D,A) — proper score for ordinal result."""
    cp = ca = c = 0.0
    for pi, ai in zip(p, actual):
        cp += pi; ca += ai; c += (cp - ca) ** 2
    return c / 2.0

def main():
    rows = load_rows()
    n = len(rows)
    print(f"corpus: {n} matches with Pinnacle closing odds + actual scores (FULL, no subsample)\n")

    methods = {
        "model (evpick EV-max)": lambda g, o: M.evpick(g),
        "grid modal (argmax P)": lambda g, o: modal(g),
        "fav 1-0 / 0-1 / 1-1":   lambda g, o: fav_score(o),
        "always 1-1":            lambda g, o: (1, 1),
        "always 1-0":            lambda g, o: (1, 0),
    }
    per_game = {k: [] for k in methods}
    exact_hits = {k: 0 for k in methods}

    rps_model, rps_market, exact_brier = [], [], []
    rel = {}  # 5%-wide predicted-prob buckets -> [sum_pred, n_scorelines, n_hits]

    for row in rows:
        o, H, A = row["o"], row["H"], row["A"]
        g = M.grid(*M.fit(o))                      # fit + grid ONCE per match; reused everywhere
        for name, fn in methods.items():
            hp, ap = fn(g, o)
            per_game[name].append(M.pts(hp, ap, H, A))
            if (hp, ap) == (H, A):
                exact_hits[name] += 1
        # result-prob calibration: model grid vs raw de-vig market (both proper-scored)
        actual = (1, 0, 0) if H > A else ((0, 1, 0) if H == A else (0, 0, 1))
        rps_model.append(rps_ordered(M.outcome(g), actual))
        rps_market.append(rps_ordered(M.devig(o), actual))
        # scoreline calibration: Brier on P(exact actual) + full reliability over every candidate score
        exact_brier.append((1 - g.get((H, A), 0.0)) ** 2)
        for hp in RANGE:
            for ap in RANGE:
                p = g.get((hp, ap), 0.0)
                bb = rel.setdefault(min(9, int(p * 20)), [0.0, 0, 0])
                bb[0] += p; bb[1] += 1; bb[2] += 1 if (hp, ap) == (H, A) else 0

    print(f"{'method':26s} {'tot pts':>9s} {'pts/game':>9s} {'exact%':>8s}")
    print("-" * 56)
    for k in sorted(methods, key=lambda k: -statistics.mean(per_game[k])):
        tot = sum(per_game[k])
        print(f"{k:26s} {tot:9.0f} {tot/n:9.4f} {100*exact_hits[k]/n:7.2f}%")

    best_base = max((k for k in methods if not k.startswith("model")),
                    key=lambda k: statistics.mean(per_game[k]))
    diffs = [per_game["model (evpick EV-max)"][i] - per_game[best_base][i] for i in range(n)]
    md = statistics.mean(diffs)
    se = statistics.pstdev(diffs) / math.sqrt(n)
    print(f"\nmodel - [{best_base}]: {md:+.4f} pts/game  95% CI [{md-1.96*se:+.4f}, {md+1.96*se:+.4f}]")
    print(f"  over a 64-game World Cup that is {md*64:+.2f} pts vs the best naive rule.")

    print(f"\ncalibration:")
    print(f"  result RPS  model grid {statistics.mean(rps_model):.4f}  vs raw market {statistics.mean(rps_market):.4f}  (lower=better)")
    print(f"  scoreline Brier on P(exact)  {statistics.mean(exact_brier):.4f}")
    print(f"  exact-score reliability (predicted P vs empirical frequency, 5%-wide buckets):")
    print(f"    {'pred~':>7s} {'empirical':>10s} {'n_scorelines':>14s}")
    for bkt in sorted(rel):
        s, cnt, hit = rel[bkt]
        if cnt < 50:
            continue
        print(f"    {s/cnt:7.3f} {hit/cnt:10.3f} {cnt:14d}")

if __name__ == "__main__":
    main()
