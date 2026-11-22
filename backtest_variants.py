#!/usr/bin/env python3
"""Backtest of momentum ranking and sizing variants.

Replicates the live process: signal from Friday's close, rebalance Monday,
hold the top 10 for a week, weights from the score. Crosses four ranking
formulas with three sizing rules and reports net-of-cost results.

Rankings
  raw 12-1        cumulative simple return t-252 -> t-21, no risk adjustment
                  (the academic UMD-style sort)
  ret/vol(t)      simple 12-1 return / 126d ann vol ending at t   <- live today
  ret/vol(t-21)   same, but the vol window ends at t-21 (fully aligned skip)
  logdrift/vol    annualized log drift over t-252..t-21 / 126d ann vol at t
                  (puts numerator and denominator on the same footing)

Sizings
  equal   1/N across the top 10
  prop    proportional to positive score (the live rule)
  cap20   proportional, but no position above 20%

Honesty notes, also printed in the report:
  * Universe = TODAY'S S&P 1500 members. Survivorship bias inflates every
    number, and not evenly: past blowups are missing (flatters high-vol
    variants) and past completed acquisitions are missing (understates the
    deal-pinning penalty that ret/vol(t) suffers live).
  * No acquisition screen is applied historically - it cannot be reproduced.
  * Costs: 10bp per dollar traded, Sharpe uses rf=0.

Usage
  python backtest_variants.py --quick     # smoke test on the short cache
  python backtest_variants.py             # full run, downloads ~9y of prices
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import momentum_core as mc
import momentum_scan as ms

HERE = Path(__file__).resolve().parent
LOOK, SKIP, VOLW = mc.DEFAULT_LOOKBACK, mc.DEFAULT_SKIP, mc.DEFAULT_VOL_WINDOW


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def build_scores(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    simple = close.shift(SKIP) / close.shift(LOOK) - 1.0
    lg = np.log(close)
    drift = (lg.shift(SKIP) - lg.shift(LOOK)) * (252.0 / (LOOK - SKIP))

    daily = close.pct_change(fill_method=None)
    vol = daily.rolling(VOLW, min_periods=VOLW).std() * np.sqrt(252)
    vol = vol.where(vol > 0)
    vol_lag = vol.shift(SKIP)

    return {
        "raw 12-1": simple,
        "ret/vol(t)": simple / vol,
        "ret/vol(t-21)": simple / vol_lag,
        "logdrift/vol": drift / vol,
    }


# --------------------------------------------------------------------------
# Sizing rules
# --------------------------------------------------------------------------

def cap_weights(scores: pd.Series, cap: float = 0.20) -> pd.Series:
    w = mc.weight_by_score(scores)
    for _ in range(60):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        rest = w[~over]
        if rest.sum() <= 0:
            break
        w[~over] = rest + excess * rest / rest.sum()
    return w


SIZINGS = {
    "equal": lambda s: pd.Series(1.0 / len(s), index=s.index),
    "prop": mc.weight_by_score,
    "cap20": cap_weights,
}


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def rebalance_dates(idx: pd.DatetimeIndex, freq: str, first: str) -> list[pd.Timestamp]:
    """First trading day of each week (usually Monday) or month."""
    entries = idx.to_series().groupby(idx.to_period(freq)).first()
    return [d for d in entries if d >= pd.Timestamp(first)]


def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    sizing_fn,
    entries: list[pd.Timestamp],
    top_n: int = 10,
    cost_per_dollar: float = 0.0010,
) -> pd.DataFrame:
    """Weekly loop. Signal is read the trading day BEFORE each entry date,
    so a Monday rebalance uses Friday's close - no look-ahead."""
    idx = close.index
    out_dates, gross_l, net_l, traded_l = [], [], [], []
    w_prev = pd.Series(dtype=float)
    px_prev = None

    for k in range(len(entries) - 1):
        e0, e1 = entries[k], entries[k + 1]
        pos = idx.get_indexer([e0])[0]
        if pos <= 0:
            continue
        sig = scores.iloc[pos - 1]
        px0, px1 = close.loc[e0], close.loc[e1]

        elig = sig.notna() & px0.notna()
        top = sig[elig].nlargest(top_n)
        if len(top) < top_n:
            continue  # still in warm-up

        w = sizing_fn(top)
        name_ret = (px1[w.index] / px0[w.index] - 1.0).fillna(0.0)
        gross = float((w * name_ret).sum())

        # Dollars traded = |new weights - previous weights drifted to today|
        if len(w_prev):
            drifted = (w_prev * (px0[w_prev.index] / px_prev[w_prev.index])).fillna(0.0)
            drifted = drifted / drifted.sum() if drifted.sum() > 0 else drifted
        else:
            drifted = pd.Series(dtype=float)
        both = w.index.union(drifted.index)
        traded = float(
            (w.reindex(both, fill_value=0) - drifted.reindex(both, fill_value=0)).abs().sum()
        )

        out_dates.append(e1)
        gross_l.append(gross)
        net_l.append(gross - traded * cost_per_dollar)
        traded_l.append(traded)
        w_prev, px_prev = w, px0

    return pd.DataFrame(
        {"gross": gross_l, "net": net_l, "traded": traded_l},
        index=pd.DatetimeIndex(out_dates, name="date"),
    )


def summarize(bt: pd.DataFrame, periods_per_year: float) -> dict:
    net = bt["net"]
    years = (net.index[-1] - net.index[0]).days / 365.25
    total = float((1 + net).prod())
    cum = (1 + net).cumprod()
    gross_total = float((1 + bt["gross"]).prod())
    return {
        "CAGR%": (total ** (1 / years) - 1) * 100,
        "Vol%": net.std() * np.sqrt(periods_per_year) * 100,
        "Sharpe": net.mean() / net.std() * np.sqrt(periods_per_year) if net.std() > 0 else np.nan,
        "MaxDD%": float((cum / cum.cummax() - 1).min()) * 100,
        "Turn%/reb": float(np.mean(bt["traded"])) / 2 * 100,
        "Cost%/yr": ((gross_total ** (1 / years)) - (total ** (1 / years))) * 100,
        "$1->": total,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="smoke test on the existing short price cache")
    p.add_argument("--data-start", default="2017-01-01")
    p.add_argument("--trade-start", default="2018-04-01",
                   help="first rebalance (needs lookback+vol warm-up after data-start)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=10.0,
                   help="cost per dollar traded, in basis points")
    p.add_argument("--offline-universe", action="store_true")
    args = p.parse_args(argv)

    t0 = time.time()
    cost = args.cost_bps / 10_000

    if args.quick:
        cache = HERE / ".cache" / "prices_sp1500.csv"
        close = pd.read_csv(cache, index_col=0, parse_dates=True)
        ms.log(f"quick mode: {close.shape[0]} days x {close.shape[1]} tickers from scan cache")
        trade_start = str(close.index[0].date())
    else:
        uni = ms.load_universe("sp1500", allow_network=not args.offline_universe)
        symbols = sorted({mc.normalize_ticker(s) for s in uni["Symbol"].astype(str)})
        end = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        close = ms.download_prices(symbols, args.data_start, str(end),
                                   cache_key="bt_sp1500", cache_ttl_hours=72.0)
        trade_start = args.trade_start

    close = ms.drop_incomplete_session(close).ffill(limit=5)
    scores = build_scores(close)
    entries = rebalance_dates(close.index, "W", trade_start)
    ms.log(f"{len(entries)} weekly rebalance dates, "
           f"{entries[0].date()} -> {entries[-1].date()}")

    # ---- benchmark ----
    spy = ms.download_prices(["SPY"], args.data_start if not args.quick
                             else str(close.index[0].date()),
                             str(close.index[-1].date() + timedelta(days=1)),
                             cache_key="bt_spy", cache_ttl_hours=72.0)
    spy = spy.ffill(limit=5).reindex(close.index).ffill(limit=5)

    # ---- main grid ----
    rows, per_year_rows, results = [], [], {}
    for rname, panel in scores.items():
        for sname, fn in SIZINGS.items():
            bt = run_backtest(close, panel, fn, entries, args.top, cost)
            if bt.empty:
                ms.log(f"{rname} | {sname}: no complete periods (warm-up too long?)")
                continue
            results[(rname, sname)] = bt
            rows.append({"ranking": rname, "sizing": sname, **summarize(bt, 52)})

    if not rows:
        print("No variant produced results - not enough history.")
        return 1

    summary = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

    # SPY over the same span. Anchor on the rebalance date immediately before
    # the first completed period - warm-up can push that well past entries[0],
    # and anchoring there would book the whole gap as one "week".
    some_bt = next(iter(results.values()))
    spy_px = spy.iloc[:, 0]
    prev_entry = max(d for d in entries if d < some_bt.index[0])
    spy_ret = spy_px.reindex([prev_entry] + list(some_bt.index)).pct_change(
        fill_method=None).iloc[1:]
    spy_bt = pd.DataFrame({"gross": spy_ret, "net": spy_ret, "traded": 0.0})
    summary = pd.concat([summary, pd.DataFrame(
        [{"ranking": "SPY buy&hold", "sizing": "-", **summarize(spy_bt, 52)}])],
        ignore_index=True)

    print("\n=== Variant grid, net of costs "
          f"({args.cost_bps:g}bp per dollar traded), weekly Monday rebalance, "
          f"top {args.top} ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---- per-year, prop sizing (the live rule) ----
    print("\n=== Net return by calendar year, 'prop' sizing ===")
    py = {}
    for rname in scores:
        key = (rname, "prop")
        if key in results:
            r = results[key]["net"]
            py[rname] = (1 + r).groupby(r.index.year).prod() - 1
    py["SPY"] = (1 + spy_bt["net"]).groupby(spy_bt.index.year).prod() - 1
    print((pd.DataFrame(py) * 100).to_string(float_format=lambda x: f"{x:+,.1f}"))

    if not args.quick:
        # ---- sensitivity: drop the six currently deal-pinned names ----
        deals = ms.read_symbol_file(HERE / "acquisitions.txt")
        have = [t for t in deals if t in close.columns]
        if have:
            close_x = close.drop(columns=have)
            sc_x = build_scores(close_x)
            print(f"\n=== Sensitivity: excluding today's pending deals "
                  f"({', '.join(have)}), 'prop' sizing ===")
            xr = []
            for rname, panel in sc_x.items():
                bt = run_backtest(close_x, panel, SIZINGS["prop"], entries,
                                  args.top, cost)
                if not bt.empty:
                    xr.append({"ranking": rname, **summarize(bt, 52)})
            print(pd.DataFrame(xr).to_string(index=False,
                                             float_format=lambda x: f"{x:,.2f}"))

        # ---- sensitivity: monthly rebalance ----
        m_entries = rebalance_dates(close.index, "M", trade_start)
        print("\n=== Sensitivity: monthly rebalance, 'prop' sizing ===")
        mr = []
        for rname, panel in scores.items():
            bt = run_backtest(close, panel, SIZINGS["prop"], m_entries,
                              args.top, cost)
            if not bt.empty:
                mr.append({"ranking": rname, **summarize(bt, 12)})
        print(pd.DataFrame(mr).to_string(index=False,
                                         float_format=lambda x: f"{x:,.2f}"))

    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    stamp = close.index[-1].date()
    summary.to_csv(outdir / f"backtest_summary_{stamp}.csv", index=False)
    print(f"\nSummary CSV: {outdir / f'backtest_summary_{stamp}.csv'}")

    print("\nCaveats: universe is TODAY'S S&P 1500 (survivorship bias inflates "
          "all rows; missing blowups flatter high-vol variants; missing "
          "completed acquisitions understate the deal-pinning penalty in "
          "ret/vol(t)). No acquisition screen applied historically. rf=0.")
    ms.log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
