# Momentum Factor Scanner

Ranks S&P 500 / 400 / 600 / 1500 constituents by volatility-adjusted momentum.
Built to run unattended every Monday morning ahead of a weekly rebalance.

## The score

```
                 P[t-21] / P[t-252] - 1          <- 12-month return, skipping the last month
score  =  ------------------------------------
           stdev(daily returns, 126d) x sqrt(252)   <- annualized 6-month realized volatility
```

The numerator is the price return from twelve months ago **up to one month ago**.
Skipping the most recent month is deliberate: stocks tend to mean-revert over
short horizons, so the last month's move is noise that works against the
twelve-month trend.

The denominator rewards a trend that arrived smoothly over one that arrived in
a few violent jumps. Annualizing multiplies every score by the same constant, so
it changes the printed numbers but never the ordering.

### What v1 got wrong

The original `app.py` computed the numerator as `ret_252 - ret_21` — subtracting
a one-month return from a twelve-month return. That is not the same as the
return between those two points; you have to divide the growth factors, not
subtract the percentages.

The error is small for a calm stock and severe for exactly the high-flyers that
land at the top of a momentum ranking:

| 12mo | 1mo | v1 `ret_252 - ret_21` | Correct | Error |
|--:|--:|--:|--:|--:|
| +10% | +2% | 0.080 | 0.078 | 0.2 pts |
| +80% | +10% | 0.700 | 0.636 | 6.4 pts |
| +200% | +50% | 1.500 | 1.000 | **50 pts** |
| +60% | −20% | 0.800 | 1.000 | **−20 pts** |

It is also directional, not just imprecise. A name that spiked last month keeps
too much credit; a name that dipped is over-penalized. That is the short-term
reversal exposure the skip-month exists to remove, so the old formula partly
defeated its own purpose — at the top of the list, where it matters most.

Three other v1 bugs are fixed here:

- Only `BRK.B` was remapped to Yahoo's `BRK-B` format. `BF.B`, `MOG.A` and
  `CWEN.A` returned empty and were dropped with no error, so they never once
  appeared in a ranking. All share classes are now converted.
- A zero-variance ticker (halted, or stale-priced) divided to `+inf` and sorted
  straight to rank 1. It now yields `NaN` and drops out.
- Position weights were `score / score.sum()`, which is `NaN` when every
  selected name has negative momentum. It now returns zero weights.

## Running it

```bash
./.venv/bin/python momentum_scan.py --universe sp1500 --top 30
```

There is no volatility ceiling by default. `--max-vol` exists but is off unless
you ask for it — acquisition targets are handled by the screen below, which is
the problem a ceiling was previously standing in for.

| Flag | Purpose |
|---|---|
| `--universe` | `sp500`, `sp400`, `sp600`, `sp1500`, or `file:/path/to.csv` (needs a `Symbol` column) |
| `--max-vol` | Drop names above this **annualized** volatility, in percent |
| `--min-vol` | Drop names below it |
| `--exclude` | Comma-separated tickers to drop, e.g. `--exclude TSLA,SMCI` |
| `--include` | Comma-separated tickers to add to the universe |
| `--exclude-file` | Standing exclusion lists, default `exclusions.txt acquisitions.txt` |
| `--deal-vol` `--deal-ratio` | Thresholds for the acquisition screen below, default 20% and 0.60 |
| `--lookback` `--skip` `--vol-window` | Trading-day windows, default 252 / 21 / 126 |
| `--cache-ttl` | Hours before prices are refetched, default 20. `0` forces a fresh download |
| `--offline-universe` | Skip the Wikipedia refresh, use saved lists |
| `--top` | Rows to print. The CSV always holds the full ranking |

Add / subtract tickers either per-run with `--exclude` and `--include`, or
permanently by editing `exclusions.txt`. The scheduled task reads that file, so
edits there apply to every future Monday without touching the task.

## The acquisition problem

A stock under a pending cash acquisition trades pinned to the agreed takeout
price. Its realized volatility collapses toward zero while the trailing return
stays large — and since the score divides one by the other, a shrinking
denominator inflates it. These names float to rank 1 despite having no trend
left to ride and upside capped at the deal price.

The scan can't know about deals on its own, so it screens for the *signature*:
volatility that is unusually low in absolute terms, or that has collapsed
relative to its own trailing level (`recent_vol_pct / ann_vol_pct`). Flagged
names are printed under `CHECK FOR PENDING ACQUISITIONS`.

That is a screen, not a verdict — a genuinely stable low-beta stock trips it
too, and a freshly announced or all-stock deal may not trip it at all. The
Monday scheduled task resolves it by web-searching each flagged name, removing
only those with a confirmed announced deal, and recording them in
`acquisitions.txt` so later runs skip them without re-checking. It also prunes
that file when a deal closes or breaks.

Use the **absolute** volatility number, not the ratio, when judging the screen
output. On 2026-08-14 all four flagged names were confirmed deals, including two
whose recent/trailing ratio was above 1 — a rising ratio is not evidence against
a deal.

To screen by volatility alone without the search step, `--min-vol 15` removes
most deal-pinned names bluntly.

## Research basis

**The numerator is the canonical academic construction.** Momentum was
established by Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling
Losers*, Journal of Finance. Skipping the most recent month is the standard
convention: the Fama–French momentum factor (UMD, as published in Ken French's
data library) is built on the cumulative return from t−12 to t−2. The reason is
short-horizon reversal, documented in Jegadeesh (1990) and Lehmann (1990) — the
last month's move tends to work *against* the twelve-month trend. Asness,
Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*, use the same 12−2
definition across asset classes.

One deliberate departure: the literature uses calendar months, this uses fixed
trading-day offsets (252 / 21). That is a common practitioner approximation, but
it does not align to month boundaries the way published factor data does.

**The denominator is a practitioner variant, not the academic factor.** UMD does
not risk-adjust at all — it sorts on raw 12−2 return. Volatility scaling has
strong support at the *strategy* level: Barroso & Santa-Clara (2015), *Momentum
Has Its Moments*, and Daniel & Moskowitz (2016), *Momentum Crashes*, both scale
the whole momentum portfolio by its realized volatility to control crash risk.
But that is scaling one portfolio-level series, which is a different operation
from dividing each stock's score by its own volatility.

The closest institutional precedent for per-stock risk adjustment is the MSCI
Momentum Indexes, which divide price momentum (excluding the most recent month)
by the annualized standard deviation of *weekly* returns over the trailing
*three years*. This implementation matches that in form but not in
parameterization: 126 daily returns is a far shorter and more reactive window,
which is what allows a one-month volatility collapse to dominate the ranking.

**Known inconsistency.** The numerator ends at t−21 but the volatility window
ends at t, so the denominator still contains the month the numerator excludes.
The skip is therefore only half applied. See the note in `momentum_core.py`.

## Data handling

Index membership is scraped fresh from Wikipedia on each run and written to
`universe/`; if Wikipedia is unreachable the saved copy is used and its age is
reported. Prices come from Yahoo via `yfinance` with `auto_adjust=True`, so
closes are split- and dividend-adjusted.

To stay under Yahoo's rate limits a full 1500-name run is split into batches of
50 with a 1–2s pause between them, three retries with exponential backoff, and a
20-hour on-disk cache in `.cache/`. A cold run takes ~2.5 minutes; a repeat run
the same day sends no requests at all.

If the market is open when the scan runs, Yahoo returns a live partial bar for
the session in progress. That bar is dropped, so a 7am and a 10am Monday run
produce the same answer — built on Friday's close.

## Running it on request (no Mac needed)

There is no schedule. The scan runs only when asked, through the
`Momentum scan` GitHub Action (`.github/workflows/scan.yml`, `workflow_dispatch`).
Ask Claude in the Life project to run it, or press "Run workflow" in the
Actions tab. The Action runs the tests, scans the S&P 1500 with a fresh
download, writes `results/latest.md` plus the dated CSV, and commits both.

Acquisition check: the run prints `CHECK FOR PENDING ACQUISITIONS`. Claude
web-searches each flagged name and adds confirmed deals to `acquisitions.txt`
(then re-runs if the top 10 changed), which replaces the old Monday task.

## Verifying the math

```bash
./.venv/bin/python test_momentum_core.py
```

Nine checks, no pytest needed. They prove the skip-month return is exact to
1e-12 on an analytic price path, demonstrate both directions of the v1 error,
confirm annualizing volatility leaves ranks untouched, and cover the zero-variance,
all-negative-weights, volatility-filter and short-history edge cases.

## Layout

| File | Role |
|---|---|
| `momentum_core.py` | The math. Pure functions, no I/O — this is the part to trust and test |
| `momentum_scan.py` | Fetching, caching, filtering, reporting, CLI |
| `test_momentum_core.py` | Checks on the math |
| `exclusions.txt` | Your standing "never show me these" list |
| `acquisitions.txt` | Confirmed pending deals, maintained by Claude when a run flags one |
| `app.py` | Legacy Streamlit UI. Still uses the old, incorrect formula — kept only for reference |
