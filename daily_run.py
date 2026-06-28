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

# KO objective extends the regulation EV-max pick (evpick) with the Felix +5 penalty-shootout free-roll,
# which is unlocked ONLY by predicting a DRAW. So total E[pts] for a KO scoreline = regulation E[pts]
# + (for a draw) P(reach pens)·P(SO pick correct)·5. We proxy P(reach pens) = draw_mass·KO_PEN_KAPPA
# (the share of 90-min draws that survive extra time to penalties; ~half historically) and
# P(SO correct) = 0.5 (a shootout is ~a coinflip; the small favorite edge is ignored, conservatively).
# The single resulting flip-to-draw is robustness-checked stable across KO_PEN_KAPPA ∈ [0.30, 0.50].
# HARD GUARDRAIL: never punt a clear market favorite (de-vig win prob ≥ 0.5) to a draw.
# evpick (wc_model) stays the pure-regulation, validation-gated engine; this layer only adds the KO +5.
KO_PEN_KAPPA = 0.50
def ko_pick(g, o):
    """Knockout pick. Returns (scoreline, so_slot|None): so_slot is 'home'/'away' when we predict a draw
    (the stronger de-vig side is the +5 shootout pick — a SLOT, never a name, so ESPN renames can't drift
    it), else None. A draw is chosen over the best non-draw only when its regulation EV plus the shootout
    free-roll wins AND no clear favorite would be punted."""
    base = M.evpick(g)                                   # validated regulation EV-max scoreline
    tw, _td, tl = M.devig(o)
    so_slot = "home" if tw >= tl else "away"
    if base[0] == base[1]:                               # regulation EV-max is already a draw -> +5 unlocked
        return base, so_slot
    if max(tw, tl) >= 0.5:                               # clear favorite -> never punt to a draw
        return base, None
    reg = {(hp, ap): sum(p * M.pts(hp, ap, H, A) for (H, A), p in g.items())
           for hp in range(7) for ap in range(7)}
    best_nd = max(v for k, v in reg.items() if k[0] != k[1])   # == reg[base] here (base is non-draw)
    draw_mass = sum(p for (H, A), p in g.items() if H == A)
    freeroll = draw_mass * KO_PEN_KAPPA * 0.5 * 5
    # best draw scoreline by regulation EV; exact-prob then geometry tiebreak within the draw class
    draws = [(reg[k], g.get(k, 0.0), k) for k in reg if k[0] == k[1]]
    bd_ev = max(c[0] for c in draws)
    bd = min((c for c in draws if c[0] >= bd_ev - 1e-9),
             key=lambda c: (-c[1], c[2][0] + c[2][1], c[2]))[2]
    if reg[bd] + freeroll > best_nd:                     # draw + shootout free-roll beats the best non-draw
        return bd, so_slot
    return base, None

def main(pull=False):
    if pull:
        import pull_data; pull_data.main()
        # Stage 1: overlay multi-book CONSENSUS 3-way (sharper than ESPN's single DraftKings line)
        # onto the freshly-pulled fixtures. Best-effort: any failure (no key, quota, network) leaves
        # the ESPN-based sheet fully intact, so the daily run can never be broken by the odds-api.
        try:
            import pull_odds_api; pull_odds_api.main()
            import consensus; cstats = consensus.apply_to_files(DATA)
            print(f"[consensus overlay: {cstats['overlaid']} games sharpened via odds-api, "
                  f"{cstats['fallback']} kept ESPN fallback]")
        except Exception as e:
            print(f"[consensus overlay skipped ({e}) — using ESPN DraftKings odds]")
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
            H, A = int(hs), int(as_)
            base = M.pts(pk[0], pk[1], H, A)
            # Felix KO shootout +5: only when WE predicted a DRAW (pk carries a stored shootout pick
            # pk[2]) AND the game actually went to penalties — i.e. a knockout recorded as a level
            # regulation/ET score (H==A; a KO can't truly end drawn). Scored against the positively
            # confirmed actual shootout winner (competitor winner flag captured as f["win"]). Group
            # draws never carry pk[2]; regulation-decided KO games (H!=A) get no shootout score.
            so_pts, so_note = 0, ""
            if len(pk) >= 3 and H == A:
                # pk[2] and f["win"] are both SLOTS ("home"/"away") — compared directly so an ESPN team
                # rename between the pre-pull (when the pick was stored) and the post-pull (the result)
                # can never cause a false mismatch (the very drift the ID-keyed ledger defends against).
                win = f.get("win")
                so_pts = M.pts_shootout(pk[2], win)
                pk_team = sp(f["home"] if pk[2] == "home" else f["away"])
                win_team = sp(f["home"]) if win == "home" else (sp(f["away"]) if win == "away" else "??")
                so_note = f"  [SO pick {pk_team} vs winner {win_team} -> +{so_pts}]"
            pts = base + so_pts
            score += pts; exact += (base == 12)   # exact = regulation scoreline hit, not the +5 bonus
            played.append((sp(f["home"]), pk, (f["hs"], f["as"]), sp(f["away"]), pts, so_note))

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
            add(f["date"], (sp(f["home"]), f["hs"], f["as"], sp(f["away"]), mark, None))
            continue
        if not o or any(o.get(k) is None for k in ("hml", "dml", "aml")):
            continue   # incomplete 3-way line -> can't fit; skip (don't crash the run)
        g = M.grid(*M.fit(o)); n_picked += 1
        # Group stage: pure regulation EV-max (evpick). Knockout: ko_pick adds the Felix +5 shootout
        # free-roll, which can flip a coin-flip game to a draw (+SO winner) — see ko_pick docstring.
        _, lbl = bucket(f["date"])
        if lbl.startswith("KNOCKOUT"):
            pk, so_pick = ko_pick(g, o)
        else:
            pk, so_pick = M.evpick(g), None
        entry = [int(pk[0]), int(pk[1])] + ([so_pick] if so_pick else [])
        ledger[f["id"]] = [f["home"], f["away"], entry]   # update; keeps played-game picks intact
        op = prev_by_id.get(f["id"])
        tag = ""
        if op and tuple(pk) != tuple(op[:2]):   # compare scoreline only (op may carry a 3rd SO element)
            tag = f"CHANGED (was {op[0]}-{op[1]})"
            changes.append((sp(f["home"]), op, pk, sp(f["away"])))
        so_disp = sp(f["home"] if so_pick == "home" else f["away"]) if so_pick else None
        add(f["date"], (sp(f["home"]), pk[0], pk[1], sp(f["away"]), tag, so_disp))

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
    for h, pk, ac, a, pts, so_note in played:
        print(f"  {h:>11} {pk[0]}-{pk[1]} | actual {ac[0]}-{ac[1]} {a:<11} -> {pts} pts{so_note}")
    print(f"  TOTAL on scored games: {score} pts, {exact} exact\n")
    print(f"PICK CHANGES from fresh odds: {len(changes)}")
    for h, op, pk, a in changes:
        print(f"  {h} {op[0]}-{op[1]} -> {pk[0]}-{pk[1]} {a}")
    print("\nFULL SHEET (PLAYED = locked):")
    for key in sorted(rows_by_bucket):
        print(f"\n  --- {labels[key]} ---")
        for h, x1, x2, a, tag, so in rows_by_bucket[key]:
            mark = " ✓" if tag == "PLAYED" else (f"   <- {tag}" if tag else "")
            so_s = f"   [+5 SO winner: {so}]" if so else ""
            print(f"    {h:>11} {x1}-{x2} {a:<11}{mark}{so_s}")

    # 3b) CHAMPION / RUNNER-UP via the bracket simulator (folds in actual results; runner-up is the
    #     champion's most-likely FINAL opponent -> always from the OPPOSITE half, so the pair is achievable).
    try:
        import bracket
        br = bracket.run()
        print("\n" + "=" * 60)
        print(f"CHAMPION / SUBCAMPEÓN (bracket-consistent, {br.get('anchor', 'Elo projection')}):")
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
