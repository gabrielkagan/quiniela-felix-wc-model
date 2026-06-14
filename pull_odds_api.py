"""STAGE 1 — richer market inputs from the-odds-api (multi-book consensus + multi-line totals).

First-principles motivation: our pick quality is bounded by how well we identify the *scoreline*
distribution, and we currently feed the fit only ~4 numbers from ONE book (DraftKings 3-way + a
single O/U + one handicap). the-odds-api gives ~48 books incl. Pinnacle (the sharpest) and Pinnacle
posts a full totals CURVE (1.75 ... 4.25). Two upgrades, both strictly more information:

  1. CONSENSUS de-vig: de-vig each book's 3-way independently, then Pinnacle-weight the average.
     De-vigging per-book-then-averaging is correct (averaging raw vig-laden odds double-counts margin);
     Pinnacle-weighting leans on the sharpest line without discarding the wisdom of the crowd.
  2. TOTALS CURVE: the set of (line, P(over)) points pins the goal-total marginal — and its DISPERSION
     — instead of forcing a single-point match. This is the raw material stage 2 calibrates against.

This module ONLY fetches + structures data (writes data/odds_api.json). It does not change the model.
The model A/B (does consensus+curve beat single-book?) is decided in validate.py, never shipped blind.
"""
import json, os, urllib.request, urllib.parse

SPORT = "soccer_fifa_world_cup"
WINNER_SPORT = "soccer_fifa_world_cup_winner"
BASE = "https://api.the-odds-api.com/v4"
OUT = os.path.expanduser("~/wc-pool/data/odds_api.json")
# Pinnacle is the sharpest book — weight it above the field without ignoring the crowd.
PINNACLE_WEIGHT = 3.0
DEFAULT_WEIGHT = 1.0


def _key():
    env = os.path.expanduser("~/wc-pool/.env")
    for line in open(env):
        if line.startswith("ODDS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("ODDS_API_KEY missing from ~/wc-pool/.env")


def _get(path, **params):
    params["apiKey"] = _key()
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    r = urllib.request.urlopen(url, timeout=30)
    remaining = r.headers.get("x-requests-remaining")
    return json.load(r), remaining


def _devig_3way(h, d, a):
    """decimal 3-way -> de-vigged (P_home, P_draw, P_away). Proportional (basic) de-vig."""
    rh, rd, ra = 1.0 / h, 1.0 / d, 1.0 / a
    s = rh + rd + ra
    return rh / s, rd / s, ra / s


def _book_weight(key):
    return PINNACLE_WEIGHT if key == "pinnacle" else DEFAULT_WEIGHT


def _consensus_3way(bookmakers):
    """Weighted average of per-book de-vigged 3-way probs. Returns (ph,pd,pa,n_books) or None."""
    num = [0.0, 0.0, 0.0]
    wsum = 0.0
    n = 0
    for b in bookmakers:
        h = d = a = None
        home_name, away_name = b.get("_home"), b.get("_away")
        for m in b.get("markets", []):
            if m["key"] != "h2h":
                continue
            for o in m["outcomes"]:
                if o["name"] == "Draw":
                    d = o["price"]
                elif o["name"] == home_name:
                    h = o["price"]
                elif o["name"] == away_name:
                    a = o["price"]
        if h and d and a:
            ph, pd, pa = _devig_3way(h, d, a)
            w = _book_weight(b["key"])
            num[0] += w * ph; num[1] += w * pd; num[2] += w * pa
            wsum += w; n += 1
    if wsum == 0:
        return None
    return num[0] / wsum, num[1] / wsum, num[2] / wsum, n


def _totals_curve(bookmakers):
    """Aggregate a consensus totals curve: {line: P(over)} de-vigged + Pinnacle-weighted.
    Also returns the Pinnacle-only curve separately (sharpest single source)."""
    agg = {}        # line -> [num_p_over, wsum]
    pinn = {}       # line -> P(over) from pinnacle only
    for b in bookmakers:
        for m in b.get("markets", []):
            if m["key"] != "totals":
                continue
            over = under = line = None
            for o in m["outcomes"]:
                if o["name"] == "Over":
                    over = o["price"]; line = o.get("point")
                elif o["name"] == "Under":
                    under = o["price"]
            if over and under and line is not None:
                p_over = (1.0 / over) / (1.0 / over + 1.0 / under)  # de-vig the 2-way
                w = _book_weight(b["key"])
                cell = agg.setdefault(line, [0.0, 0.0]); cell[0] += w * p_over; cell[1] += w
                if b["key"] == "pinnacle":
                    pinn[line] = p_over
    curve = {ln: v[0] / v[1] for ln, v in agg.items() if v[1] > 0}
    return curve, pinn


def _consensus_spread(bookmakers):
    """Consensus Asian-handicap home line (supremacy proxy): median of per-book home points."""
    pts = []
    for b in bookmakers:
        for m in b.get("markets", []):
            if m["key"] != "spreads":
                continue
            for o in m["outcomes"]:
                if o["name"] == b.get("_home") and o.get("point") is not None:
                    pts.append(o["point"])
    if not pts:
        return None
    pts.sort()
    return pts[len(pts) // 2]


def fetch(regions="us,uk,eu"):
    games, remaining = _get(f"sports/{SPORT}/odds", regions=regions,
                            markets="h2h,totals,spreads", oddsFormat="decimal")
    out = []
    for g in games:
        home, away = g["home_team"], g["away_team"]
        # tag each bookmaker with the game's home/away so the consensus helpers can map by name
        for b in g.get("bookmakers", []):
            b["_home"] = home; b["_away"] = away
        cons = _consensus_3way(g.get("bookmakers", []))
        if cons is None:
            continue
        ph, pd, pa, nb = cons
        curve, pinn = _totals_curve(g.get("bookmakers", []))
        out.append({
            "home": home, "away": away, "commence": g["commence_time"],
            "n_books": nb,
            "consensus_3way": {"home": ph, "draw": pd, "away": pa},
            "totals_curve": {str(k): v for k, v in sorted(curve.items())},
            "pinnacle_totals": {str(k): v for k, v in sorted(pinn.items())},
            "consensus_spread": _consensus_spread(g.get("bookmakers", [])),
        })
    return out, remaining


def fetch_winner(regions="us,uk,eu"):
    """Tournament-winner market -> de-vigged title probabilities (for the bracket sim).
    Pinnacle doesn't post WC outrights here; the Betfair EXCHANGE is the sharpest available source,
    so de-vig each book's full 48-team field then weight the exchange above sportsbooks."""
    data, remaining = _get(f"sports/{WINNER_SPORT}/odds", regions=regions,
                           markets="outrights", oddsFormat="decimal")
    num = {}
    wsum = 0.0
    books = []
    for ev in data:
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m["key"] != "outrights":
                    continue
                raw = {o["name"]: 1.0 / o["price"] for o in m["outcomes"] if o.get("price")}
                s = sum(raw.values())
                if s <= 0:
                    continue
                w = 3.0 if b["key"].startswith("betfair_ex") else 1.0  # exchange = sharpest
                for k, v in raw.items():
                    num[k] = num.get(k, 0.0) + w * (v / s)  # de-vig per book, then weight
                wsum += w
                books.append(b["key"])
    probs = {k: v / wsum for k, v in num.items()} if wsum else {}
    return probs, remaining, books


def main():
    games, rem = fetch()
    winner, rem2, wbooks = fetch_winner()
    payload = {"games": games, "winner_probs": winner}
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, OUT)
    avg_curve = sum(len(g["totals_curve"]) for g in games) / max(1, len(games))
    print(f"wrote {len(games)} games -> {OUT} | requests remaining: {rem2 or rem}")
    print(f"  avg totals-curve points/game: {avg_curve:.1f} | winner-market teams: {len(winner)}")
    if games:
        g = games[0]
        print(f"  sample: {g['home']} vs {g['away']}  consensus "
              f"H={g['consensus_3way']['home']:.3f} D={g['consensus_3way']['draw']:.3f} "
              f"A={g['consensus_3way']['away']:.3f}  ({g['n_books']} books)")
        print(f"          totals curve: {g['totals_curve']}")


if __name__ == "__main__":
    main()
