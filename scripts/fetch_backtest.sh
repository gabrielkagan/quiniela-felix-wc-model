#!/usr/bin/env bash
# Fetch the football-data.co.uk backtest corpus used by validate.py.
# Public data (closing Pinnacle 3-way + O/U 2.5 + Asian handicap + actual scores). Re-runnable.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/backtest"
mkdir -p "$DEST"
ok=0; fail=0
for season in 2324 2425; do
  for lg in E0 D1 I1 SP1 F1; do
    url="https://www.football-data.co.uk/mmz4281/${season}/${lg}.csv"
    if curl -fsS -o "${DEST}/${season}_${lg}.csv" "$url"; then ok=$((ok+1)); else fail=$((fail+1)); echo "FAIL $url" >&2; fi
  done
done
echo "fetched ok=$ok fail=$fail -> $DEST"
