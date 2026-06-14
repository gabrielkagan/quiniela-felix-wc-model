"""Daily re-optimize: refresh data -> score played games -> regenerate EV-max picks ->
diff vs prior sheet -> print report. Run `python3 ~/wc-pool/daily_run.py` (add --pull to refetch).

Pick rule and all modeling live in wc_model.py (the adversarially-converged engine).
"""
import importlib.util, json, os, sys
from datetime import datetime, timedelta

HOME = os.path.expanduser("~/wc-pool")
DATA = f"{HOME}/data"
spec = importlib.util.spec_from_file_location("wc_model", f"{HOME}/wc_model.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

ES = {'South Korea':'Corea','Czechia':'Chequia','South Africa':'Sudáfrica','Switzerland':'Suiza',
 'Qatar':'Catar','United States':'EE.UU.','Brazil':'Brasil','Morocco':'Marruecos','Haiti':'Haití',
 'Scotland':'Escocia','Germany':'Alemania','Curaçao':'Curazao','Ivory Coast':'C.Marfil',
 'Netherlands':'P.Bajos','Japan':'Japón','Sweden':'Suecia','Tunisia':'Túnez','Spain':'España',
 'Cape Verde':'Cabo Verde','Saudi Arabia':'Arabia S.','Belgium':'Bélgica','Egypt':'Egipto',
 'Iran':'Irán','New Zealand':'N.Zelanda','France':'Francia','Iraq':'Irak','Norway':'Noruega',
 'Algeria':'Argelia','Jordan':'Jordania','Congo DR':'RD Congo','Uzbekistan':'Uzbekistán',
 'England':'Inglaterra','Croatia':'Croacia','Panama':'Panamá','Mexico':'México','Canada':'Canadá',
 'Türkiye':'Turquía','Ecuador':'Ecuador','Paraguay':'Paraguay','Australia':'Australia',
 'Uruguay':'Uruguay','Senegal':'Senegal','Austria':'Austria','Argentina':'Argentina',
 'Portugal':'Portugal','Colombia':'Colombia','Ghana':'Ghana','Bosnia-Herzegovina':'Bosnia'}
def sp(n): return ES.get(n, n)
def bucket(d):
    """Group-stage matchday (1/2/3) for Jun<=27; otherwise a per-DATE knockout bucket so KO games are
    NEVER mislabeled under a group header. Returns (sort_key, display_label)."""
    dt = datetime.fromisoformat(d.replace('Z', '+00:00')) - timedelta(hours=5)
    if dt.month == 6 and dt.day <= 27:
        n = 1 if dt.day <= 17 else (2 if dt.day <= 23 else 3)
        return ((0, n), f"FECHA {n}")
    iso = dt.strftime('%Y-%m-%d')
    return ((1, iso), f"KNOCKOUT {iso}")

def main(pull=False):
    if pull:
        import pull_data; pull_data.main()
    try:
        fixtures = json.load(open(f"{DATA}/fixtures.json"))
    except (FileNotFoundError, ValueError):
        print(f"*** ERROR: {DATA}/fixtures.json missing or corrupt. Run with --pull first. ***")
        return
    # dedup by id at the CONSUMER too (belt-and-suspenders): a no-pull run on a stale/edited file with
    # a duplicate id would otherwise double-count a played game's score. Last-wins.
    _seen = {}
    for f in fixtures:
        _seen[f.get("id")] = f
    fixtures = list(_seen.values())
    # cumulative ledger keyed by stable fixture ID -> [home, away, pick]; NEVER drop a pick (so games
    # that move to 'played' stay scoreable). Keyed on ID, NOT (home,away), because ESPN renames teams
    # between pulls (Türkiye/Turkey, Czechia/Czech Republic) which would silently drop picks.
    ledger = {}
    if os.path.exists(f"{DATA}/picks_prev.json"):
        try:
            rows = json.load(open(f"{DATA}/picks_prev.json"))
        except ValueError:
            print("*** WARNING: picks_prev.json corrupt — starting a fresh ledger (history lost). ***")
            rows = []
        for x in rows:
            # tolerate structurally-malformed rows (short/foreign) without crashing the run
            if isinstance(x, (list, tuple)) and len(x) >= 4 and isinstance(x[3], (list, tuple)) and len(x[3]) >= 2:
                ledger[x[0]] = [x[1], x[2], list(x[3])]
    prev_by_id = {pid: tuple(v[2]) for pid, v in ledger.items()}

    # 1) score played games against our prior picks (join on fixture ID)
    played, score, exact = [], 0, 0
    for f in fixtures:
        if f["state"] != "post":
            continue
        pk = prev_by_id.get(f["id"])
        hs, as_ = str(f["hs"]), str(f["as"])
        if pk and hs.lstrip("-").isdigit() and as_.lstrip("-").isdigit():  # need real integer scores
            pts = M.pts(pk[0], pk[1], int(hs), int(as_))
            score += pts; exact += (pts == 12)
            played.append((sp(f["home"]), pk, (f["hs"], f["as"]), sp(f["away"]), pts))

    # 2) regenerate EV-max picks for all upcoming games + diff. Bucket by matchday/KO-date dynamically
    #    (group-stage -> FECHA 1/2/3; knockouts -> per-date KNOCKOUT headers, never mislabeled).
    rows_by_bucket, labels, changes, n_picked = {}, {}, [], 0
    def add(d, row):
        key, lbl = bucket(d); labels[key] = lbl
        rows_by_bucket.setdefault(key, []).append(row)
    for f in fixtures:
        o = f.get("odds")
        if f["state"] != "pre":   # played OR in-play -> locked; never re-pick (live odds encode the score)
            mark = "PLAYED" if f["state"] == "post" else "LIVE — locked"
            add(f["date"], (sp(f["home"]), f["hs"], f["as"], sp(f["away"]), mark))
            continue
        if not o or any(o.get(k) is None for k in ("hml", "dml", "aml")):
            continue   # incomplete 3-way line -> can't fit; skip (don't crash the run)
        lh, la = M.fit(o); pk = M.evpick(M.grid(lh, la)); n_picked += 1
        ledger[f["id"]] = [f["home"], f["away"], list(pk)]   # update; keeps played-game picks intact
        op = prev_by_id.get(f["id"])
        tag = ""
        if op and tuple(pk) != op:
            tag = f"CHANGED (was {op[0]}-{op[1]})"
            changes.append((sp(f["home"]), op, pk, sp(f["away"])))
        add(f["date"], (sp(f["home"]), pk[0], pk[1], sp(f["away"]), tag))

    # loud cutover guard: zero fresh picks usually means stale data or a KO-round date gap
    future_unscored = sum(1 for f in fixtures if f["state"] == "pre")
    if n_picked == 0:
        print("\n*** WARNING: 0 fresh picks generated. Either all games are played/live, or the\n"
              "    DATES window in pull_data.py needs extending for the knockout rounds, or the\n"
              "    pull is stale. Do NOT present an empty sheet — investigate. ***\n")
    elif n_picked < future_unscored:
        print(f"\n*** NOTE: {future_unscored - n_picked} upcoming game(s) had no usable odds line "
              f"(skipped). ***\n")

    # 3) report
    print("=" * 60)
    print("PLAYED GAMES — our picks vs actual:")
    for h, pk, ac, a, pts in played:
        print(f"  {h:>11} {pk[0]}-{pk[1]} | actual {ac[0]}-{ac[1]} {a:<11} -> {pts} pts")
    print(f"  TOTAL on scored games: {score} pts, {exact} exact\n")
    print(f"PICK CHANGES from fresh odds: {len(changes)}")
    for h, op, pk, a in changes:
        print(f"  {h} {op[0]}-{op[1]} -> {pk[0]}-{pk[1]} {a}")
    print("\nFULL SHEET (PLAYED = locked):")
    for key in sorted(rows_by_bucket):
        print(f"\n  --- {labels[key]} ---")
        for h, x1, x2, a, tag in rows_by_bucket[key]:
            mark = " ✓" if tag == "PLAYED" else (f"   <- {tag}" if tag else "")
            print(f"    {h:>11} {x1}-{x2} {a:<11}{mark}")

    # 3b) CHAMPION / RUNNER-UP via the bracket simulator (folds in actual results; runner-up is the
    #     champion's most-likely FINAL opponent -> always from the OPPOSITE half, so the pair is achievable).
    try:
        import bracket
        br = bracket.run()
        print("\n" + "=" * 60)
        print("CHAMPION / SUBCAMPEÓN (bracket-consistent, Elo projection):")
        print(f"  🏆 Campeón:    {br['champion']}  ({br['champ_prob']*100:.0f}% of sims)")
        print(f"  🥈 Subcampeón: {br['runner_up']}  (top final opponent of {br['champion']}, opposite half)")
        print("  Top title contenders: " + ", ".join(f"{t} {p*100:.0f}%" for t, p in br["top_champions"][:5]))
    except Exception as e:
        print(f"\n[bracket sim unavailable: {e}]")

    # 4) persist the FULL cumulative ledger atomically (never drops a pick; a crash mid-write must not
    #    corrupt the file — a corrupt ledger would lose all scored history).
    out = [[pid, h, a, pk] for pid, (h, a, pk) in ledger.items()]
    dst = f"{DATA}/picks_prev.json"; tmp = dst + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(out, fh); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"\n[saved {len(out)} picks (cumulative ledger) to {DATA}/picks_prev.json]")

if __name__ == "__main__":
    sys.path.insert(0, HOME)
    main(pull=("--pull" in sys.argv))
