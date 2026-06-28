"""Pull fresh FIFA World Cup fixtures + DraftKings 3-way odds from ESPN's public APIs.
Writes ~/wc-pool/data/fixtures.json. No keys required. Idempotent — safe to run daily.
"""
import urllib.request, json, time, os

# Full tournament window (group stage + knockouts) so KO fixtures auto-pull. June 11 -> July 19.
DATES = ([f"202606{d:02d}" for d in range(11, 31)] + [f"202607{d:02d}" for d in range(1, 20)])
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={}"
ODDS = ("https://sports.core.api.espn.com/v2/sports/soccer/leagues/fifa.world/events/"
        "{eid}/competitions/{eid}/odds")
OUT = os.path.expanduser("~/wc-pool/data/fixtures.json")

def get(url):
    try:
        return json.load(urllib.request.urlopen(url, timeout=25))
    except Exception:
        return None

def g(d, *ks, default=None):
    for k in ks:
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d if d is not None else default

def pick_provider(items, want="DraftKings"):
    """Prefer the named book; fall back to first available. Pinning avoids a silent book swap."""
    for it in items:
        if it and g(it, "provider", "name") == want:
            return it
    return next((it for it in items if it), None)

def main():
    fixtures = []
    failures = 0
    for dt in DATES:
        sb = get(SCOREBOARD.format(dt))
        if sb is None:
            failures += 1            # a failed date query -> do NOT trust a partial result
            continue
        for e in (sb.get("events") or []):
            comps = e.get("competitions") or [{}]
            c = comps[0]
            cs = c.get("competitors") or []
            if len(cs) != 2:
                continue
            home = next((x for x in cs if x.get("homeAway") == "home"), cs[0])
            away = next((x for x in cs if x.get("homeAway") == "away"), cs[1])
            # winner side (competitor `winner` flag). For a knockout decided on penalties ESPN records
            # a level regulation/ET score (hs==as) yet still flags one competitor winner -> this lets
            # daily_run score the Felix +5 shootout pick. None for draws/unfinished group games.
            win = "home" if home.get("winner") else ("away" if away.get("winner") else None)
            eid = e.get("id")
            state = g(e, "status", "type", "state")
            odds = None
            # only pull a market line for PRE games (in-play/post lines encode the live score)
            if state == "pre":
                core = get(ODDS.format(eid=eid))
                o = pick_provider((core or {}).get("items") or [])
                if o:
                    odds = {"spread": o.get("spread"), "ou": o.get("overUnder"),
                            "hml": g(o, "homeTeamOdds", "moneyLine"),
                            "aml": g(o, "awayTeamOdds", "moneyLine"),
                            "dml": g(o, "drawOdds", "moneyLine"),
                            "provider": g(o, "provider", "name")}
            fixtures.append({"id": eid, "date": e.get("date"), "state": state,
                             "desc": g(e, "status", "type", "description"),
                             "home": g(home, "team", "displayName"),
                             "away": g(away, "team", "displayName"),
                             "hs": home.get("score"), "as": away.get("score"),
                             "win": win, "odds": odds})
            time.sleep(0.05)
        time.sleep(0.1)

    # dedup by event id (ESPN can list a late-night game under two consecutive date queries; an
    # undeduped dup would double-count a played game's score downstream and freeze the abort-guard).
    # When the same id appears twice, keep the MOST-INFORMATIVE row: a played 'post' result beats a
    # pre row (R12 midnight-crossing case), and among non-post a PRICED row beats an unpriced one
    # (R11 masking case). This makes dedup deterministic and never discards the better signal.
    def _rank(f):
        return (1 if f.get("state") == "post" else 0,
                1 if (f.get("odds") or {}).get("hml") is not None else 0)
    def dedup_by_id(rows):
        out = {}
        for f in rows:
            k = f.get("id")
            if k not in out or _rank(f) > _rank(out[k]):
                out[k] = f
        return list(out.values())
    fixtures = dedup_by_id(fixtures)

    # completeness guard: never overwrite a good file with a degraded pull
    prior = []
    if os.path.exists(OUT):
        try: prior = json.load(open(OUT))
        except Exception: prior = []
    # dedup PRIOR too (same priced-preferring rule) before any count/coverage comparison — else a
    # duplicate id ever present in the old file (104 unique vs 105 rows) makes `len(fixtures) <
    # len(prior)` abort EVERY run, freezing the data forever with no self-heal (R10 CRITICAL).
    prior = dedup_by_id(prior)
    # Abort on real data loss. Count alone is fooled by same-count-wrong-data, so also require:
    #  (a) no prior game that HAD a real odds line disappeared, and (b) odds coverage didn't regress.
    # A prior game WITHOUT odds is a KO placeholder whose ID legitimately changes at bracket cutover,
    # so we only treat the loss of a *priced* game as data loss (avoids false aborts on bracket draws).
    new_by_id = {f.get("id"): f for f in fixtures}
    new_ids = set(new_by_id)
    lost_priced = [f for f in prior if (f.get("odds") or {}).get("hml") is not None and f.get("id") not in new_ids]
    # PER-ID coverage check (not an aggregate sum, which a newly-priced game could mask): a game that
    # is STILL pre in the new pull but LOST its odds line is a real regression. A pre->post transition
    # legitimately drops odds and is excluded (it's no longer in still_pre) — that was the R3 freeze.
    still_pre = {f["id"] for f in fixtures if f["state"] == "pre"}
    regressed = [f["id"] for f in prior
                 if (f.get("odds") or {}).get("hml") is not None and f.get("id") in still_pre
                 and (new_by_id.get(f["id"], {}).get("odds") or {}).get("hml") is None]
    if len(fixtures) < max(len(prior), 1) or lost_priced or regressed:
        print(f"PULL ABORTED — {failures} failed date(s); count {len(fixtures)}/{len(prior)}, "
              f"lost_priced={len(lost_priced)}, odds-regressed={len(regressed)}. Keeping {OUT} unchanged.")
        return prior
    if failures:
        priced = sum(1 for f in fixtures if (f.get("odds") or {}).get("hml") is not None)
        print(f"(note: {failures} date query/queries failed but data held — count {len(fixtures)}, priced {priced})")
    # atomic write: fsync'd temp file + rename, with guaranteed cleanup so a crash leaves no .tmp turd
    os.makedirs(os.path.dirname(OUT), exist_ok=True)   # fresh-machine safety
    tmp = OUT + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(fixtures, fh)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, OUT)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    post = sum(1 for f in fixtures if f["state"] == "post")
    pre = sum(1 for f in fixtures if f["state"] == "pre" and f["odds"] and f["odds"].get("hml"))
    print(f"pulled {len(fixtures)} fixtures -> {OUT} | played={post} | upcoming-with-odds={pre}")
    return fixtures

if __name__ == "__main__":
    main()
