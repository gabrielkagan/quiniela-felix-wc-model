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
import json, os, math, random
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
def elo(t): return ELO.get(t, DEFAULT_ELO)

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

def ko(a, b):
    pa = 1.0 / (1.0 + 10 ** (-(elo(a) - elo(b)) / 400.0))
    return a if random.random() < pa else b

def sim_tournament(known=None):
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
    r32w = [ko(fill(s1), fill(s2)) for s1, s2 in R32]
    r16w = [ko(r32w[i], r32w[j]) for i, j in R16]
    qfw  = [ko(r16w[i], r16w[j]) for i, j in QF]
    sfw  = [ko(qfw[i], qfw[j]) for i, j in SF]
    champ = ko(sfw[0], sfw[1])
    runner = sfw[1] if champ == sfw[0] else sfw[0]
    return champ, runner

def run(n=20000, seed=20260613, known=None):
    random.seed(seed)
    if known is None:
        known = load_known_results()
    champs = Counter(); finals = Counter()  # finals[(champ,runner)]
    for _ in range(n):
        c, r = sim_tournament(known); champs[c] += 1; finals[(c, r)] += 1
    champ = champs.most_common(1)[0][0]
    # runner-up = most common FINAL OPPONENT of the chosen champion (opposite-half by construction)
    opp = Counter({r: cnt for (c, r), cnt in finals.items() if c == champ})
    runner = opp.most_common(1)[0][0]
    return {"n": n, "champion": champ, "runner_up": runner,
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
