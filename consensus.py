"""Overlay the-odds-api multi-book CONSENSUS 3-way onto ESPN fixtures (stage 1, production).

The model fits to a de-vigged 3-way line; a Pinnacle-weighted consensus of ~48 books is strictly
sharper than the single DraftKings line ESPN provides. We override ONLY the 3-way moneylines (the
dominant signal) and keep ESPN's O/U + spread, so the change is minimal and easy to reason about.

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


def overlay(fixtures, odds_api):
    """Return (new_fixtures, stats). Overlays consensus 3-way onto matched PRE games' odds."""
    games = odds_api.get("games", []) if isinstance(odds_api, dict) else []
    cons = {(norm(g["home"]), norm(g["away"])): g["consensus_3way"]
            for g in games if g.get("consensus_3way")}
    overlaid = fallback = 0
    for f in fixtures:
        if f.get("state") != "pre":
            continue
        o = f.get("odds")
        if not o or any(o.get(k) is None for k in ("hml", "dml", "aml")):
            continue  # KO placeholder / no line — leave as-is (already skipped by the model)
        c = cons.get((norm(f["home"]), norm(f["away"])))
        if c is None:
            fallback += 1
            continue
        o["hml"] = prob_to_american(c["home"])
        o["dml"] = prob_to_american(c["draw"])
        o["aml"] = prob_to_american(c["away"])
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
