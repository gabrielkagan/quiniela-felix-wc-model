"""Bracket simulator for the 2026 World Cup champion / runner-up.

WHY: champion and runner-up must come from OPPOSITE halves of the bracket — they can only meet in
the final. Spain (Group H) and France (Group I) both feed Half 1 if they WIN their groups, so they
collide in the semifinal, NOT the final; picking both is impossible. This sim respects the real
bracket so the recommended (champion, runner-up) pair is always achievable.

METHOD: Monte-Carlo. Group stage simulated with Elo-derived Poisson goals (points/GD/GF standings);
knockouts with the Elo win-expectancy. Over N sims we record (champion, final-loser) and report the
champion = most-likely title winner, and runner-up = most-likely FINAL OPPONENT of that champion
(conditioning on the champion automatically yields an opposite-half, bracket-consistent runner-up).

SCOPE/CAVEATS: group WINNER and RUNNER-UP slotting (which fixes the contenders' halves) is EXACT,
verified against the FIFA R32 match tree (matches 73-102) and the projected Spain-France /
Argentina-Portugal semifinals. The best-third-place SLOTTING is APPROXIMATED (8 best thirds filled
into the 8 third-slots in match order) — thirds are weak and this only changes which group winner a
third faces, with negligible effect on title/runner-up odds. Groups use Elo, not the daily market
odds (the daily PICKS still use market odds; this is a separate tournament projection).
"""
import json, os, math, random, unicodedata
from collections import Counter

DATA = os.path.expanduser("~/wc-pool/data")
ELO = json.load(open(f"{DATA}/elo.json"))

# Official group letters: contenders (C,E,F,G,H,I,J,K,L) confirmed via ESPN; hosts A/B/D
# (Mexico/Canada/USA) are longshots and do not affect the title/runner-up output.
GROUPS = {
 "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
 "B": ["Canada", "Bosnia", "Qatar", "Switzerland"],
 "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
 "D": ["USA", "Paraguay", "Australia", "Turkey"],
 "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
 "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
 "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
 "H": ["Spain", "Uruguay", "Cape Verde", "Saudi Arabia"],
 "I": ["France", "Senegal", "Iraq", "Norway"],
 "J": ["Argentina", "Algeria", "Austria", "Jordan"],
 "K": ["Portugal", "Colombia", "DR Congo", "Uzbekistan"],
 "L": ["England", "Croatia", "Ghana", "Panama"],
}
DEFAULT_ELO = 1500
# _RATINGS is the live strength table the sim reads. Default = raw Elo; run() swaps in ratings
# RE-RATED to the betting market's title distribution when data/odds_api.json is present (Elo runs
# hotter than the market AND disagrees on ordering — e.g. the market rates France/England above our
# Elo and Argentina below it, so a simple flatten is insufficient; we re-rate to match the market).
_RATINGS = dict(ELO)
def elo(t): return _RATINGS.get(t, DEFAULT_ELO)

# R32 (matches 73-88): each slot is ('W',group) winner, ('R',group) runner-up, ('T',i) i-th third.
R32 = [
 (("R","A"),("R","B")), (("W","E"),("T",0)), (("W","F"),("R","C")), (("W","C"),("R","F")),
 (("W","I"),("T",1)),   (("R","E"),("R","I")),(("W","A"),("T",2)), (("W","L"),("T",3)),
 (("W","D"),("T",4)),   (("W","G"),("T",5)), (("R","K"),("R","L")),(("W","H"),("R","J")),
 (("W","B"),("T",6)),   (("W","J"),("R","H")),(("W","K"),("T",7)), (("R","D"),("R","G")),
]
R16 = [(1,4),(0,2),(3,5),(6,7),(10,11),(8,9),(13,15),(12,14)]  # pairs of R32 indices
QF  = [(0,1),(4,5),(2,3),(6,7)]                                 # pairs of R16 indices
SF  = [(0,1),(2,3)]                                             # pairs of QF indices
# Half 1 (SF[0]) = winners D,E,F,G,H,I ; Half 2 (SF[1]) = winners A,B,C,J,K,L (runner-up halves flip)

def spois(L):
    Lk = math.exp(-L); k = 0; p = 1.0
    while True:
        k += 1; p *= random.random()
        if p <= Lk: return k - 1

def group_goals(a, b, T=2.6, C=220):
    sup = (elo(a) - elo(b)) / C
    la = max(0.15, (T + sup) / 2); lb = max(0.15, (T - sup) / 2)
    return spois(la), spois(lb)

def sim_group(teams, known=None):
    known = known or {}      # {frozenset({a,b}): (goals_a_keyed_by_team)} actual results to lock in
    pts = {t: 0 for t in teams}; gf = {t: 0 for t in teams}; ga = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            res = known.get(frozenset((a, b)))
            if res is not None:                 # use the ACTUAL played result, don't re-simulate
                x, y = res[a], res[b]
            else:
                x, y = group_goals(a, b)
            gf[a] += x; ga[a] += y; gf[b] += y; ga[b] += x
            if x > y: pts[a] += 3
            elif x < y: pts[b] += 3
            else: pts[a] += 1; pts[b] += 1
    rank = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t], random.random()), reverse=True)
    return rank, pts, gf, ga

# map every team to its group letter, and load actual completed group results from fixtures.json
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}
# ESPN display-name variants -> our GROUPS names (same mapping the rest of the engine uses)
NORM = {"Türkiye": "Turkey", "Curaçao": "Curacao", "Bosnia-Herzegovina": "Bosnia",
        "United States": "USA", "Côte d'Ivoire": "Ivory Coast", "Cabo Verde": "Cape Verde",
        "Korea Republic": "South Korea", "Congo DR": "DR Congo"}
def _nm(n): return NORM.get(n, n)
def load_known_results(path=None):
    path = path or f"{DATA}/fixtures.json"
    known = {}
    try:
        fixtures = json.load(open(path))
    except Exception:
        return known
    for f in fixtures:
        if f.get("state") != "post":
            continue
        h, a = _nm(f.get("home")), _nm(f.get("away"))
        # only lock in genuine GROUP games (both teams in the same group); skip KO / unknowns
        if h in TEAM_GROUP and a in TEAM_GROUP and TEAM_GROUP[h] == TEAM_GROUP[a]:
            try:
                hs, as_ = int(f["hs"]), int(f["as"])
            except (TypeError, ValueError):
                continue
            known[frozenset((h, a))] = {h: hs, a: as_}
    return known

def load_ko_eliminated(path=None, market_teams=None):
    """Teams knocked OUT in already-played KNOCKOUT games (different-group post fixtures).

    Without this the champion sim re-simulates the whole knockout from group standings and lets
    teams that ALREADY lost a KO game (e.g. Netherlands, Germany) keep their raw Elo and 'win' sims,
    leaking title mass to eliminated teams. `ko()` forces any eliminated team to lose wherever it
    appears, which is equivalent to folding in the actual advancers.

    Decided games -> loser is the lower score. Penalty-shootout games (regulation draw) are resolved
    from the fixture's authoritative `win` slot ("home"/"away") — the same field daily_run trusts for
    the +5 scoring — so a longshot shootout survivor can never be wrongly eliminated. Only if that slot
    is missing do we fall back to the winner market (the eliminated team is the one ABSENT from it)."""
    path = path or f"{DATA}/fixtures.json"
    market_teams = market_teams if market_teams is not None else set(market_title_probs())
    elim = set()
    try:
        fixtures = json.load(open(path))
    except Exception:
        return elim
    for f in fixtures:
        if f.get("state") != "post":
            continue
        h, a = _nm(f.get("home")), _nm(f.get("away"))
        # KNOCKOUT game = both teams known AND from DIFFERENT groups (same-group => group stage)
        if not (h in TEAM_GROUP and a in TEAM_GROUP and TEAM_GROUP[h] != TEAM_GROUP[a]):
            continue
        try:
            hs, as_ = int(f["hs"]), int(f["as"])
        except (TypeError, ValueError):
            continue
        if hs > as_:
            elim.add(a)
        elif as_ > hs:
            elim.add(h)
        else:                                   # regulation draw -> penalty shootout
            win = f.get("win")                  # authoritative advancer slot (as daily_run's +5 uses)
            if win == "home":   elim.add(a)
            elif win == "away": elim.add(h)
            elif market_teams:                  # fallback only if win slot missing: survivor stays priced
                if h not in market_teams: elim.add(h)
                if a not in market_teams: elim.add(a)
    return elim

def ko(a, b, elim=None):
    # fold in already-played KO results: an eliminated team always loses wherever it's paired
    if elim:
        ea, eb = a in elim, b in elim
        if ea != eb:
            return b if ea else a
    pa = 1.0 / (1.0 + 10 ** (-(elo(a) - elo(b)) / 400.0))
    return a if random.random() < pa else b

def sim_tournament(known=None, elim=None):
    winners, runners, thirds = {}, {}, []
    for g, teams in GROUPS.items():
        rank, pts, gf, ga = sim_group(teams, known)
        winners[g] = rank[0]; runners[g] = rank[1]
        thirds.append((rank[2], pts[rank[2]], gf[rank[2]] - ga[rank[2]], gf[rank[2]]))
    # 8 best third-placed teams (by pts, GD, GF) — APPROXIMATE slotting into the 8 third-slots
    top8 = [t[0] for t in sorted(thirds, key=lambda x: (x[1], x[2], x[3], random.random()), reverse=True)[:8]]
    def fill(slot):
        kind, key = slot
        return winners[key] if kind == "W" else runners[key] if kind == "R" else top8[key]
    r32w = [ko(fill(s1), fill(s2), elim) for s1, s2 in R32]
    r16w = [ko(r32w[i], r32w[j], elim) for i, j in R16]
    qfw  = [ko(r16w[i], r16w[j], elim) for i, j in QF]
    sfw  = [ko(qfw[i], qfw[j], elim) for i, j in SF]
    champ = ko(sfw[0], sfw[1], elim)
    runner = sfw[1] if champ == sfw[0] else sfw[0]
    return champ, runner

# ---- market re-rating: anchor the sim's title distribution to the betting market ----
_BY_LOWER = {unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower(): t
             for ts in GROUPS.values() for t in ts}
# Canonical-name aliases for EVERY spelling the winner market might use (both the odds-api forms seen
# today AND the ESPN-style variants) -> GROUPS key. Hardened beyond the 2 live cases so a provider
# spelling switch can never SILENTLY drop a team from the market re-rating (the R1 Bosnia bug class).
_CANON_ALIAS = {
    "czech republic": "Czechia", "bosnia and herzegovina": "Bosnia", "bosnia": "Bosnia",
    "turkiye": "Turkey", "cabo verde": "Cape Verde", "korea republic": "South Korea",
    "south korea": "South Korea", "congo dr": "DR Congo", "dr congo": "DR Congo",
    "united states": "USA", "usa": "USA", "ivory coast": "Ivory Coast",
    "cote divoire": "Ivory Coast", "curacao": "Curacao",
}
def _canon(name):
    # mirror consensus.norm (& -> and, strip '.' '-' ') so odds-api names like "Bosnia & Herzegovina"
    # / "Czech Republic" / "Côte d'Ivoire" canonicalize consistently with the picks side.
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "and").replace(".", "").replace("-", " ").replace("'", "").strip()
    return _CANON_ALIAS.get(s) or _BY_LOWER.get(s) or name

def market_title_probs(path=None):
    """{GROUPS-name: title_prob} from data/odds_api.json winner market (renormalized over the field)."""
    path = path or f"{DATA}/odds_api.json"
    try:
        wp = json.load(open(path)).get("winner_probs", {})
    except Exception:
        return {}
    out = {}
    for name, p in wp.items():
        c = _canon(name)
        if c in TEAM_GROUP:
            out[c] = out.get(c, 0.0) + p
    s = sum(out.values())
    return {t: p / s for t, p in out.items()} if s > 0 else {}

def calibrate_to_market(market, known, elim=None, iters=40, n=12000, seed=20260613, K=24.0, max_delta=220.0):
    """Iteratively nudge team ratings so the sim's title probabilities match the market's.
    R_t += K * log(market_t / sim_t): teams the sim under-rates vs the market get boosted, and
    vice-versa. Preserves the EXACT bracket structure; only the strengths are re-anchored.

    Stability (title prob is a steep function of rating through the bracket, so naive steps diverge):
      - SMALL gain K with many iters (gradient descent, not a one-shot jump),
      - per-step move clipped to +/-K so a near-zero sim prob can't launch a longshot,
      - total drift from Elo clipped to +/-max_delta so re-rating refines, never fabricates.

    Convergence (iters=40, n=12000, K=24): the prior (12, 6000, 18) was SYSTEMATICALLY under-converged
    — 12 gradient steps could not close a ~2pt market gap on the top two, leaving a stable ~1.5% bias
    that pinned the WRONG champion (Argentina robustly #1 across seeds) despite France leading the
    title market 20.2% vs 18.1% on 2026-06-29. The richer schedule restores the market ordering
    robustly across calibration seeds (France #1 in all of 5 tested), at ~30s vs ~5s — acceptable for
    a once-daily run."""
    global _RATINGS
    R = dict(ELO)
    for it in range(iters):
        _RATINGS = R
        random.seed(seed + it)
        champs = Counter()
        for _ in range(n):
            c, _r = sim_tournament(known, elim); champs[c] += 1
        for t, mp in market.items():
            sp = champs.get(t, 0) / n
            step = K * math.log(max(mp, 1e-4) / max(sp, 1e-4))
            step = max(-K, min(K, step))                          # clip per-step move
            base = ELO.get(t, DEFAULT_ELO)
            R[t] = max(base - max_delta, min(base + max_delta, R.get(t, base) + step))
    return R

def run(n=20000, seed=20260613, known=None, use_market=True):
    global _RATINGS
    if known is None:
        known = load_known_results()
    market = market_title_probs() if use_market else {}
    elim = load_ko_eliminated(market_teams=set(market))  # fold in already-played KO results
    if market:
        _RATINGS = calibrate_to_market(market, known, elim)
        anchor = "market-anchored (odds-api title odds)"
    else:
        _RATINGS = dict(ELO)
        anchor = "Elo projection"
    random.seed(seed)
    champs = Counter(); finals = Counter()  # finals[(champ,runner)]
    for _ in range(n):
        c, r = sim_tournament(known, elim); champs[c] += 1; finals[(c, r)] += 1
    champ = champs.most_common(1)[0][0]
    # runner-up = most common FINAL OPPONENT of the chosen champion (opposite-half by construction)
    opp = Counter({r: cnt for (c, r), cnt in finals.items() if c == champ})
    runner = opp.most_common(1)[0][0]
    return {"n": n, "champion": champ, "runner_up": runner, "anchor": anchor,
            "champ_prob": champs[champ] / n,
            "runner_prob_given_champ": opp[runner] / max(1, sum(opp.values())),
            "top_champions": [(t, round(c / n, 3)) for t, c in champs.most_common(6)],
            "top_runners_given_champ": [(t, round(c / max(1, sum(opp.values())), 3)) for t, c in opp.most_common(5)]}

if __name__ == "__main__":
    res = run()
    print(f"CHAMPION:   {res['champion']}  (wins title {res['champ_prob']*100:.1f}% of sims)")
    print(f"RUNNER-UP:  {res['runner_up']}  (most common final opponent of {res['champion']}: "
          f"{res['runner_prob_given_champ']*100:.0f}% of {res['champion']} titles)")
    print("\nTop title contenders:")
    for t, p in res["top_champions"]: print(f"  {t:<12} {p*100:4.1f}%")
    print(f"\nMost likely runner-up (final opponent) GIVEN {res['champion']} wins:")
    for t, p in res["top_runners_given_champ"]: print(f"  {t:<12} {p*100:4.1f}%")
