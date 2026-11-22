"""Momentum factor math.

Everything in this module is pure: it takes a DataFrame of adjusted closing
prices (dates as rows, tickers as columns) and returns numbers. No network,
no files, no globals - so the math can be tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Trading-day approximations: 12 months, 1 month, 6 months.
DEFAULT_LOOKBACK = 252
DEFAULT_SKIP = 21
DEFAULT_VOL_WINDOW = 126


def normalize_ticker(symbol: str) -> str:
    """Convert an index-listing symbol to Yahoo Finance's format.

    Share classes are written BRK.B / BF.B / MOG.A on Wikipedia and by S&P,
    but Yahoo expects BRK-B / BF-B / MOG-A. Passing the dotted form returns an
    empty frame rather than an error, so an unmapped ticker disappears from the
    results silently.
    """
    return symbol.strip().upper().replace(".", "-")


def skip_month_return(
    close: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
) -> pd.DataFrame:
    """Price return from `lookback` days ago up to `skip` days ago.

        P[t - skip] / P[t - lookback] - 1

    This is *not* the same as (return over lookback) - (return over skip).
    Subtracting two simple returns is only a first-order approximation, and it
    leaves a residual positive exposure to the most recent month - the exact
    exposure that skipping the month exists to remove.
    """
    if lookback <= 0 or skip < 0:
        raise ValueError("lookback must be positive and skip non-negative")
    if skip >= lookback:
        raise ValueError(f"skip ({skip}) must be smaller than lookback ({lookback})")
    return close.shift(skip) / close.shift(lookback) - 1.0


def realized_volatility(
    close: pd.DataFrame,
    window: int = DEFAULT_VOL_WINDOW,
    annualize: bool = True,
) -> pd.DataFrame:
    """Standard deviation of daily returns over a trailing window.

    Requires a full window of observations, so a freshly listed name gets NaN
    rather than a volatility estimate built from a handful of days.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    # fill_method=None: a gap must stay a gap. Padding the last price forward
    # inserts artificial zero-return days, which understates volatility for
    # exactly the thinly traded names that should look risky.
    daily = close.pct_change(fill_method=None)
    vol = daily.rolling(window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def momentum_score(
    close: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    vol_window: int = DEFAULT_VOL_WINDOW,
    annualize_vol: bool = True,
) -> pd.DataFrame:
    """Volatility-adjusted momentum: skip-month return divided by realized vol.

    Annualizing the denominator multiplies every score by a constant
    sqrt(252), so it changes the printed values but never the ranking.

    Known inconsistency, inherited from the original implementation: the
    numerator stops at t-skip, but the volatility window ends at t. The
    denominator therefore still contains the month the numerator excludes, so
    the skip is only half applied. A stock whose volatility collapses in that
    final month - a pending cash acquisition is the usual cause - gets a
    shrinking denominator from data the signal claims to ignore. Measured on
    2026-08-14 S&P 1500 data, aligning the volatility window to end at t-skip
    moved one confirmed acquisition target from rank 2 to rank 498.

    Changing this alters the signal, so it is left as-is rather than silently
    corrected. See the Research basis section of README.md.
    """
    mom = skip_month_return(close, lookback, skip)
    vol = realized_volatility(close, vol_window, annualize_vol)
    # A zero-variance column (a halted or stale-priced name) would divide to
    # infinity and sort straight to the top.
    return mom / vol.where(vol > 0)


def build_table(
    close: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    vol_window: int = DEFAULT_VOL_WINDOW,
    max_annual_vol: float | None = None,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Rank every ticker by momentum score on a single date.

    `max_annual_vol` is a fraction, not a percent: 0.40 keeps names whose
    annualized volatility is under 40%. Filtering happens before ranking, so
    ranks are always 1..N with no gaps.

    Returns the date used and the ranked table.
    """
    scores = momentum_score(close, lookback, skip, vol_window, annualize_vol=True)
    scored_rows = scores.dropna(how="all")
    if scored_rows.empty:
        raise ValueError(
            "No date has a momentum score. Need at least "
            f"{max(lookback, vol_window) + 1} rows of history; got {len(close)}."
        )

    date = pd.Timestamp(as_of) if as_of is not None else scored_rows.index.max()
    if date not in scores.index:
        raise KeyError(f"{date.date()} is not a trading day in the price history")

    mom = skip_month_return(close, lookback, skip).loc[date]
    vol = realized_volatility(close, vol_window, annualize=True).loc[date]
    safe_vol = vol.where(vol > 0)

    # Every input to the score, so the arithmetic can be checked by hand:
    #   return_12_1 = price_skip / price_start - 1
    #   ann_vol     = daily_vol * sqrt(252)
    #   score       = return_12_1 / ann_vol
    price_start = close.shift(lookback).loc[date]
    price_skip = close.shift(skip).loc[date]

    # A short-window volatility next to the headline one. A ratio well below 1
    # means volatility has recently collapsed, which is the signature of a
    # stock pinned to a pending cash acquisition price: the trailing return
    # stays large while the denominator shrinks, inflating the score.
    recent_vol = realized_volatility(close, max(2, skip), annualize=True).loc[date]

    table = pd.DataFrame(
        {
            "score": scores.loc[date],
            "price_start": price_start.round(2),
            "price_skip": price_skip.round(2),
            "return_12_1_pct": (mom * 100).round(2),
            "daily_vol_pct": (vol / np.sqrt(TRADING_DAYS_PER_YEAR) * 100).round(3),
            "ann_vol_pct": (vol * 100).round(2),
            "recent_vol_pct": (recent_vol * 100).round(2),
            "vol_ratio": (recent_vol / safe_vol).round(3),
            "last_price": close.loc[date],
        }
    ).dropna(subset=["score"])

    if max_annual_vol is not None:
        table = table[table["ann_vol_pct"] <= max_annual_vol * 100]

    table = table.sort_values("score", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.index.name = "ticker"
    return date, table


def weight_by_score(scores: pd.Series) -> pd.Series:
    """Allocate capital proportionally to positive momentum score.

    Negative scores get zero weight. If nothing is positive the result is all
    zeros rather than NaN, which is what naive score/score.sum() produces.
    """
    positive = scores.clip(lower=0)
    total = positive.sum()
    if total <= 0:
        return pd.Series(0.0, index=scores.index)
    return positive / total
