# Adversarial review context — WC pool daily re-optimize

## What this is
A market-anchored Dixon-Coles model that picks group-stage scorelines to MAXIMIZE EXPECTED POINTS
under the Quiniela Felix pool scoring. The user is chasing from mid-table and wants to win.

## STAGE 1 (SHIPPED 2026-06-14) — read before reviewing
Picks are now fit to a **multi-book CONSENSUS 3-way** (`consensus.py` overlays the-odds-api Pinnacle-
weighted de-vig onto ESPN fixtures in `daily_run --pull`; ESPN DraftKings is the fallback). The
bracket sim is **re-rated to the betting market's title odds** (`bracket.calibrate_to_market`), not raw
Elo. **Reviewers MUST NOT run `daily_run.py --pull` or `pull_odds_api.py`** — they spend paid the-odds-
api credits. Run `daily_run.py` (no --pull), `fast_model.py` (equivalence twin), `validate.py` instead.

## KO RECORDED-BASIS + TOTALS OVERLAY (SHIPPED 2026-07-11) — read before reviewing
`consensus.py` ALSO overrides `ou` from the multi-book totals curve (`market_total()`: the
P(over)=0.5 crossing), so overlaid pre games carry `provider: consensus(odds-api)` with a consensus
total, NOT ESPN's flat 2.5. Knockout picks come from `daily_run.ko_pick(g, o, lh, la)`, which is
EV-max over the ET-corrected **RECORDED-final-score distribution** (`ko_recorded_grid`: 90' draws
extended with extra-time goals at lambda/3 a side; Felix grades the recorded score — verified by the
live standings reconcile) plus the +5 shootout free-roll (p_pens·0.5·5) on draw picks only.
`stage3.py` (shadow endgame rank monitor, never changes live picks) reads live standings from
`data/standings.json` (loud fallback + games_played-vs-fixtures ABORT) and searches single-game
deviations on a prize-weighted EV$ objective — two-stage: best candidate SELECTED on DEV_DRAWS,
then CONFIRMED on disjoint DEV_CONFIRM_DRAWS, both stages $-floor + z-gated, plus a horizon guard
(no verdict while unpriced games remain); its P(top-K) table averages over SEEDS.

## Scoring rule (the objective function)
- Exact score (both teams' goals) = **12 pts**
- Correct result only (winner or draw) = **5 pts**
- Correct goal count of exactly ONE team = **2 pts**
- 12 is a flat bonus for exact (not additive). Tiebreaker = most exact scores.

## Files (review ALL of these — the orchestrator + puller + bracket are edit-sensitive, not just the model)
- `~/wc-pool/wc_model.py` — the per-match engine (de-vig, Dixon-Coles fit, evpick = EV-max under 12/5/2).
- `~/wc-pool/consensus.py` — overlays multi-book consensus 3-way onto fixtures (accent+alias join, ESPN fallback).
- `~/wc-pool/pull_odds_api.py` — fetches the-odds-api consensus + WC-winner title market (PAID; don't run in review).
- `~/wc-pool/fast_model.py` — numpy equivalence twin of wc_model (run it to confirm 0 mismatches after model edits).
- `~/wc-pool/daily_run.py` — orchestrator: scores played games (id-keyed ledger), regenerates picks, diffs, buckets by FECHA/KNOCKOUT, prints the bracket champion/runner-up.
- `~/wc-pool/pull_data.py` — fetches fixtures + DraftKings 3-way odds; atomic write with abort-guard.
- `~/wc-pool/bracket.py` — Monte-Carlo tournament sim for CHAMPION / RUNNER-UP. Champion+runner-up MUST
  come from OPPOSITE bracket halves (they can only meet in the final). Spain(H)+France(I) both feed Half 1
  if they WIN their groups (a group RUNNER-UP goes to the opposite half from its winner). Runner-up is
  computed as the champion's most-likely FINAL opponent (opposite-half by construction). Folds in actual
  played group results from fixtures.json. Group sim = Elo Poisson goals; KO = Elo win-expectancy.
  EXACT: winner/runner-up slotting (verified vs the FIFA R32 match tree 73-102 + projected Spain-France /
  Argentina-Portugal semifinals). APPROX: best-third-place slotting (8 thirds into 8 slots in match order
  — thirds are weak, negligible effect on title odds). Elo title probs run HOTTER than the market (Elo is
  more decisive) — the PICK (champion + opposite-half runner-up) is what's used and is robust to that.
- `~/wc-pool/stage3.py` — SHADOW endgame rank monitor (prize-weighted deviation search; never changes live picks).
- `~/wc-pool/data/elo.json` — World Football Elo ratings per team (bracket-sim strength input).
- `~/wc-pool/data/fixtures.json` — fixtures: id, date, state(pre/in/post), home, away, hs/as, odds{hml,dml,aml,spread,ou,provider}.
- `~/wc-pool/data/standings.json` — live pool leaderboard snapshot consumed by stage3 (us/top/n_field/games_played).

## Method (already converged through a multi-round adversarial review — see wc_model.py docstring)
1. De-vig DraftKings 3-way moneylines -> true P(home/draw/away).
2. Fit (lambda_home, lambda_away) under Dixon-Coles (RHO=-0.13) to the de-vigged 3-way + over/under + spread.
3. evpick() = expected-points-maximizing scoreline under 12/5/2, with a within-result-class
   exact-prob tiebreak only (free; never punts a clear favorite). Variance-chasing was REJECTED.

## What the DAILY review must verify (find CRITICAL/MAJOR only)
1. DATA INTEGRITY of today's fresh pull: every pre game has a real, complete 3-way line (consensus
   provider after overlay, or ESPN DraftKings fallback); played games correctly marked 'post' and
   excluded from picks; no None/stale/duplicate; team↔odds join correct.
2. SCORING: played-game points computed correctly under 12/5/2 (pts()).
3. PICK SANITY: regenerate all picks; flag ANY pick whose predicted result has <25% de-vigged
   market probability, any punt of a clear favorite (devig >= 0.5), or any indefensible scoreline.
4. EV-OPTIMALITY: spot-check that evpick == brute-force argmax-E[pts] per game (tiny within-class
   tiebreak sacrifice only).
5. If wc_model.py was EDITED this run: full correctness re-check (pts, de-vig, Dixon-Coles tau+RHO
   sign, fit spread sign, grid normalization, determinism, range sufficiency).
6. DOC DRIFT: any claim in code/output not supported by the data.

Run actual Python to verify. End each review with exactly: "ROUND VERDICT: N CRITICAL, M MAJOR".
Convergence = 2 consecutive rounds of 0 CRITICAL / 0 MAJOR.
