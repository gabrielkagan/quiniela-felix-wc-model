"""Overlay the-odds-api multi-book CONSENSUS 3-way onto ESPN fixtures (stage 1, production).

The model fits to a de-vigged 3-way line; a Pinnacle-weighted consensus of ~48 books is strictly
sharper than the single DraftKings line ESPN provides. We override the 3-way moneylines (the
dominant signal) AND the O/U total: `ou` anchors the fit's lambda_total directly, and ESPN's flat
2.5 was measured 0.2-0.35 goals below the fetched multi-book totals curve on 2026-07-11 (Norway-
England: curve-implied ~2.8 vs ESPN 2.5), enough to flip the EV-max scoreline within the winning
result class (adv MAJOR). market_total() inverts the curve to the line where P(over)=0.5. ESPN's
spread is kept as-is (small weight in the fit; no measured drift).

SAFETY: this never removes a game's odds. A game is overlaid only on a confident name match; any
unmatched game keeps its ESPN DraftKings line (fallback). The whole step is best-effort — daily_run
wraps it so a key/quota/network failure leaves the ESPN-based sheet intact.
"""
import json, os, unicodedata

# the-odds-api team name -> ESPN-normalized name (after accent strip + lowercasing)
_ALIAS = {"turkey": "turkiye", "czech republic": "czechia", "dr congo": "congo dr",
          "bosnia and herzegovina": "bosnia herzegovina", "usa": "united states"}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "and").replace(".", "").replace("-", " ").strip()
    return _ALIAS.get(s, s)


def prob_to_american(p):
    """Fair probability -> american moneyline. wc_model.devig re-normalizes, so feeding fair odds
    that already sum to 1 across H/D/A recovers exactly the consensus probabilities."""
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    dec = 1.0 / p
    return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)


# |dP(over)/d(line)| near WC-typical totals (~2.5), measured off the fetched curves (e.g. Norway-
# England 2.5->3.0: 0.558->0.448 = 0.22/goal). Used ONLY when the curve quotes a single line, and the
# extrapolation is clamped to +/-0.5 goals of a quoted line so a sparse curve can't drag ou far.
_TOT_SLOPE = 0.22


def market_total(curve):
    """Market-implied main total: the line where the totals curve crosses P(over)=0.5, linearly
    interpolated between quoted lines. If 0.5 isn't bracketed (curve entirely over/under, or a
    single quoted line), extrapolate from the nearest quoted line with the curve's own slope
    (generic _TOT_SLOPE when only one point), clamped to +/-0.5 goals. None when no curve."""
    pts = sorted((float(l), float(p)) for l, p in (curve or {}).items())
    if not pts:
        return None
    for (l1, p1), (l2, p2) in zip(pts, pts[1:]):
        if p1 >= 0.5 >= p2 and p1 > p2:
            return l1 + (l2 - l1) * (p1 - 0.5) / (p1 - p2)
    if len(pts) >= 2:
        # no interior crossing: take the slope at the end nearest the crossing
        (l1, p1), (l2, p2) = (pts[-2], pts[-1]) if pts[-1][1] > 0.5 else (pts[0], pts[1])
        slope = (p1 - p2) / (l2 - l1)
        if slope <= 0:
            slope = _TOT_SLOPE  # non-monotone curve — fall back to the generic slope
    else:
        slope = _TOT_SLOPE
    l, p = min(pts, key=lambda x: abs(x[1] - 0.5))
    return max(l - 0.5, min(l + 0.5, l + (p - 0.5) / slope))


def overlay(fixtures, odds_api):
    """Return (new_fixtures, stats). Overlays consensus 3-way + market total onto matched PRE games."""
    games = odds_api.get("games", []) if isinstance(odds_api, dict) else []
    cons = {(norm(g["home"]), norm(g["away"])): g
            for g in games if g.get("consensus_3way")}
    overlaid = fallback = 0
    for f in fixtures:
        if f.get("state") != "pre":
            continue
        o = f.get("odds")
        if not o or any(o.get(k) is None for k in ("hml", "dml", "aml")):
            continue  # KO placeholder / no line — leave as-is (already skipped by the model)
        g = cons.get((norm(f["home"]), norm(f["away"])))
        if g is None:
            fallback += 1
            continue
        c = g["consensus_3way"]
        o["hml"] = prob_to_american(c["home"])
        o["dml"] = prob_to_american(c["draw"])
        o["aml"] = prob_to_american(c["away"])
        mt = market_total(g.get("totals_curve") or g.get("pinnacle_totals"))
        if mt is not None:
            o["ou"] = round(mt, 2)
        o["provider"] = "consensus(odds-api)"
        overlaid += 1
    return fixtures, {"overlaid": overlaid, "fallback": fallback}


def apply_to_files(data_dir):
    """Best-effort file overlay used by daily_run --pull. Raises only on truly unexpected errors."""
    fpath = os.path.join(data_dir, "fixtures.json")
    opath = os.path.join(data_dir, "odds_api.json")
    fixtures = json.load(open(fpath))
    odds_api = json.load(open(opath))
    fixtures, stats = overlay(fixtures, odds_api)
    tmp = fpath + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(fixtures, fh)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, fpath)
    return stats


if __name__ == "__main__":
    print(apply_to_files(os.path.expanduser("~/wc-pool/data")))
