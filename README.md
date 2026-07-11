# Quiniela Felix — World Cup Score Model

A market-anchored model that picks World Cup scorelines to **maximize expected points** under the
Quiniela Felix pool scoring (**12** exact score / **5** correct result / **2** one team's goal count).

The core idea: the betting market is the best available prior. We de-vig sharp closing odds into true
outcome probabilities, fit a Dixon–Coles scoreline distribution to them, and pick the scoreline that
maximizes expected pool points — never opinion, never chasing upsets.

## Quick start

```bash
# 1. Python 3 (stdlib only for the daily run; no install needed)
# 2. Get a free odds API key from https://the-odds-api.com  (500 req/mo is plenty)
echo 'ODDS_API_KEY=your_key_here' >> .env && chmod 600 .env

# 3. Pull fresh fixtures + odds, score played games, regenerate picks
python3 daily_run.py --pull
```

`.env` is gitignored — your key never leaves your machine.

## What's here

| File | Role |
|---|---|
| `wc_model.py` | The engine: de-vig → Dixon–Coles fit → expected-points-maximizing pick (`evpick`). |
| `daily_run.py` | Orchestrator: scores played games, regenerates picks, prints the sheet by fecha. |
| `pull_data.py` | Fetches fixtures + odds (ESPN public API) with an abort-guard on degraded pulls. |
| `bracket.py` | Monte-Carlo champion / runner-up (opposite-half consistent). |
| `validate.py` | **Stage 0** backtest harness — scores the model vs naive rules on ~3,500 historical matches. |
| `data/elo.json` | World Football Elo ratings (bracket-sim input). |

## How the pick rule works

For each match we compute a full scoreline probability grid, then pick the score `(h, a)` that
maximizes `E[points] = Σ P(H,A) · pts(h, a, H, A)`. Ties are broken toward the higher-probability
scoreline **only within the expected-points-maximizing result class**, so a clear favorite is never
punted to a draw or upset. Variance-chasing (deliberately picking upsets to leapfrog the field) was
tested and rejected — it loses to expected-points maximization under any honest simulation.

## Validation (`validate.py`)

Before anything new ships, it has to beat the current model on the **real 12/5/2 objective**
out-of-sample. The harness backtests against football-data.co.uk closing odds + actual scores:

```bash
bash scripts/fetch_backtest.sh   # one-time: download the public corpus
python3 validate.py              # model vs naive baselines + calibration report
```

## Roadmap

The model is being made more data-rich and robust in validated stages — each stage must clear the
`validate.py` bar before it touches live picks:

- **Stage 0** — backtest harness + baseline (this file). ✅
- **Stage 1** — sharper inputs: Pinnacle-weighted multi-book consensus + multi-line totals.
- **Stage 2** — calibrate scoreline dispersion/correlation against historical data.
- **Stage 3** — rank-aware decision (optimize probability of finishing in the money, not just points)
  — stays in shadow until it beats expected-points maximization under an independent simulator.
