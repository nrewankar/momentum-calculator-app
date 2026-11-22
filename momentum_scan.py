#!/usr/bin/env python3
"""Weekly momentum scan.

Pulls a fresh index membership list, downloads adjusted closes, ranks every
name by volatility-adjusted skip-month momentum, and prints the result.

    python momentum_scan.py --universe sp1500 --top 25 --max-vol 40

The math lives in momentum_core.py and is covered by test_momentum_core.py.
"""

from __future__ import annotations

import argparse
import io
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import momentum_core as mc

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
UNIVERSE_DIR = HERE / "universe"

WIKI_SOURCES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
COMPOSITES = {"sp1500": ["sp500", "sp400", "sp600"]}
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Yahoo tolerates bursts poorly. Small batches with a pause between them keep a
# 1500-name scan under the rate limit.
CHUNK_SIZE = 50
PAUSE_RANGE = (1.0, 2.0)
MAX_ATTEMPTS = 3


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Index membership
# --------------------------------------------------------------------------

def _scrape_index(key: str) -> pd.DataFrame:
    import requests

    resp = requests.get(WIKI_SOURCES[key], headers={"User-Agent": BROWSER_UA}, timeout=30)
    resp.raise_for_status()

    for table in pd.read_html(io.StringIO(resp.text)):
        col = next((c for c in ("Symbol", "Ticker", "Ticker symbol") if c in table.columns), None)
        if col is None or len(table) < 50:
            continue
        out = table.rename(columns={col: "Symbol", "Company": "Security"})
        keep = [c for c in ("Symbol", "Security", "GICS Sector") if c in out.columns]
        return out[keep].dropna(subset=["Symbol"])
    raise ValueError(f"no constituent table found on the {key} page")


def load_universe(name: str, allow_network: bool = True) -> pd.DataFrame:
    """Fetch index members, preferring a live scrape and falling back to disk.

    A successful scrape is written back to universe/<name>.csv, so the saved
    copy stays current and a later offline run still gets a recent list.
    """
    name = name.lower()

    if name.startswith("file:"):
        path = Path(name[5:]).expanduser()
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
        if "Symbol" not in df.columns:
            raise ValueError(f"{path} has no 'Symbol' column")
        log(f"universe: {len(df)} symbols from {path}")
        return df

    if name in COMPOSITES:
        parts = [load_universe(p, allow_network) for p in COMPOSITES[name]]
        merged = pd.concat(parts, ignore_index=True).drop_duplicates("Symbol")
        log(f"universe: {name} -> {len(merged)} unique symbols")
        return merged

    if name not in WIKI_SOURCES:
        raise ValueError(f"unknown universe {name!r}")

    UNIVERSE_DIR.mkdir(exist_ok=True)
    cached = UNIVERSE_DIR / f"{name}.csv"

    if allow_network:
        try:
            df = _scrape_index(name)
            df.to_csv(cached, index=False)
            log(f"universe: {name} -> {len(df)} symbols (refreshed from Wikipedia)")
            return df
        except Exception as exc:
            log(f"universe: {name} live fetch failed ({type(exc).__name__}: {exc})")

    for path in (cached, HERE / f"stocks_{name}_current.csv"):
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
            age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
            log(f"universe: {name} -> {len(df)} symbols from {path.name} "
                f"({age.days}d old)")
            return df
    raise RuntimeError(f"could not load universe {name!r} from network or disk")


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------

def _download_chunk(symbols: list[str], start: str, end: str) -> pd.DataFrame | None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            data = yf.download(
                symbols, start=start, end=end,
                auto_adjust=True,      # split- and dividend-adjusted closes
                actions=False, progress=False, threads=True,
            )
            if data is None or data.empty:
                raise ValueError("empty response")
            if isinstance(data.columns, pd.MultiIndex):
                return data["Close"]
            return data[["Close"]].rename(columns={"Close": symbols[0]})
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                log(f"  chunk failed after {MAX_ATTEMPTS} attempts: {exc}")
                return None
            backoff = 2 ** attempt * 2
            log(f"  retry {attempt}/{MAX_ATTEMPTS - 1} in {backoff}s ({exc})")
            time.sleep(backoff)
    return None


def download_prices(
    symbols: list[str], start: str, end: str,
    cache_key: str, cache_ttl_hours: float = 20.0,
) -> pd.DataFrame:
    """Download adjusted closes, reusing a recent on-disk copy when possible."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"prices_{cache_key}.csv"

    if cache_ttl_hours > 0 and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < cache_ttl_hours:
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            missing = set(symbols) - set(cached.columns)
            # The cache must also reach back to the requested start. A file
            # saved for a shorter window has the right columns but truncated
            # history, and returning it would silently shrink every lookback.
            short = cached.empty or (
                cached.index[0] > pd.Timestamp(start) + pd.Timedelta(days=10))
            if not missing and not short:
                log(f"prices: reusing cache ({age_h:.1f}h old, "
                    f"{cached.shape[1]} tickers) - no requests sent")
                return cached.loc[pd.Timestamp(start):, symbols]
            log(f"prices: cache missing {len(missing)} tickers, refetching"
                if missing else
                f"prices: cache starts {cached.index[0].date()}, "
                f"need {start} - refetching")

    chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
    log(f"prices: downloading {len(symbols)} tickers in {len(chunks)} batches")

    frames, failed = [], 0
    for i, chunk in enumerate(chunks, 1):
        got = _download_chunk(chunk, start, end)
        if got is None:
            failed += len(chunk)
        else:
            # Hold a column for every symbol the batch asked for. A delisted
            # ticker then lives in the cache as an empty column instead of
            # looking absent, which would make the cache seem incomplete and
            # trigger a full re-download every run. Symbols from a *failed*
            # batch stay out, so a transient error is retried next time.
            frames.append(got.reindex(columns=chunk))
        log(f"  batch {i}/{len(chunks)}")
        if i < len(chunks):
            time.sleep(random.uniform(*PAUSE_RANGE))

    if not frames:
        raise RuntimeError("every batch failed - Yahoo is unreachable or blocking")
    if failed:
        log(f"prices: {failed} tickers in failed batches")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.to_csv(cache_file)
    log(f"prices: {prices.shape[0]} days x {prices.shape[1]} tickers, cached")
    return prices


def drop_incomplete_session(prices: pd.DataFrame) -> pd.DataFrame:
    """Remove today's bar if the US market is still open.

    Yahoo reports a live, partial bar for the session in progress. Keeping it
    would make the scan's output depend on the minute it happened to run, so a
    Monday 7am scan and a Monday 10am scan would disagree. Dropping it means
    the signal is always built on completed sessions.
    """
    if prices.empty:
        return prices
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if prices.index[-1].date() == now_et.date() and now_et.hour < 16:
        log(f"prices: dropping partial bar for {prices.index[-1].date()} "
            "(market still open)")
        return prices.iloc[:-1]
    return prices


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _fmt(value, pattern: str) -> str:
    return "-" if pd.isna(value) else pattern.format(value)


def to_markdown(table: pd.DataFrame, limit: int) -> str:
    """Rank, ticker, score, then every variable the score is built from.

    Reading across: price_skip / price_start - 1 is the 12-1 return, daily
    sigma x sqrt(252) is the annualized vol, and the first divided by the
    second is the score.
    """
    view = table.head(limit).reset_index()
    rows = [
        "| # | Ticker | Score | P(t-252) | P(t-21) | 12-1 Ret | Daily s | Ann Vol | Max day | Jump share | FIP ID |",
        "|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in view.itertuples(index=False):
        rows.append(
            f"| {r.rank} | {r.ticker} | {r.score:.2f} | ${r.price_start:,.2f} | "
            f"${r.price_skip:,.2f} | {r.return_12_1_pct:+.1f}% | "
            f"{r.daily_vol_pct:.2f}% | {r.ann_vol_pct:.1f}% | "
            f"{_fmt(r.max_day_pct, '{:+.1f}%')} | {_fmt(r.jump_share, '{:.2f}')} | {_fmt(r.fip_id, '{:+.3f}')} |"
        )
    return "\n".join(rows)


def flag_pinned_names(table: pd.DataFrame, max_vol: float, max_ratio: float) -> pd.DataFrame:
    """Rows whose volatility is unusually low or has recently collapsed.

    This is a screen, not a verdict - a genuinely stable low-beta name will
    trip it too. It exists to give a short list worth checking for pending
    M&A rather than to decide anything on its own.
    """
    return table[(table["ann_vol_pct"] <= max_vol) | (table["vol_ratio"] <= max_ratio)]


def jump_diagnostics(close: pd.DataFrame, date: pd.Timestamp, lookback: int, skip: int) -> pd.DataFrame:
    """How the 12-1 move was made: steadily, or in one jump.

    Momentum works best when a trend builds from many small moves (gradual
    information diffusion); a return made in one or two event days (a bid,
    an FDA ruling, a settlement, a squeeze) tends not to continue.
    Da, Gurun & Warachka (2014), "Frog in the Pan", Review of Financial Studies.

      max_day_pct   biggest single-day gain inside the t-252..t-21 window
      jump_share    that day's log return / the window's total log return
      fip_id        sign(total) x (%down days - %up days). More negative is
                    smoother. Near zero or positive means the move was lumpy.

    Report-only: none of this changes the score or the ranking.
    """
    pos = close.index.get_loc(date)
    window = close.iloc[pos - lookback: pos - skip + 1]
    logret = np.log(window / window.shift(1)).iloc[1:]
    total = logret.sum()
    up = (logret > 0).mean()
    down = (logret < 0).mean()
    biggest = logret.max()
    share = (biggest / total).where(total > 0)
    return pd.DataFrame({
        "max_day_pct": ((np.exp(biggest) - 1) * 100).round(1),
        "jump_share": share.round(2),
        "fip_id": (np.sign(total) * (down - up)).round(3),
    })


def flag_spiky_names(table: pd.DataFrame, max_share: float, max_id: float) -> pd.DataFrame:
    """Rows whose trend came mostly from one day, or that moved in lumps."""
    return table[(table["jump_share"] >= max_share) | (table["fip_id"] >= max_id)]


def read_symbol_file(path: Path) -> list[str]:
    """One symbol per line; blank lines and # comments ignored."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(mc.normalize_ticker(line))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", default="sp1500",
                   help="sp500 | sp400 | sp600 | sp1500 | file:/path/to.csv")
    p.add_argument("--top", type=int, default=20, help="rows to print and research (default 20)")
    p.add_argument("--hold", type=int, default=10, help="names to hold; prints their weights (default 10)")
    p.add_argument("--jump-share", type=float, default=0.30,
                   help="flag names whose biggest day is at least this share of the 12-1 log return")
    p.add_argument("--jump-id", type=float, default=0.0,
                   help="flag names whose frog-in-the-pan ID is at or above this (lumpy trend)")
    p.add_argument("--max-vol", type=float, default=None, metavar="PCT",
                   help="drop names above this annualized volatility, e.g. 40")
    p.add_argument("--min-vol", type=float, default=None, metavar="PCT")
    p.add_argument("--exclude", default="", help="comma-separated tickers to drop")
    p.add_argument("--include", default="", help="comma-separated tickers to add")
    p.add_argument("--exclude-file", nargs="*",
                   default=["exclusions.txt", "acquisitions.txt", "event_exclusions.txt"],
                   help="files of tickers to always drop (one per line)")
    p.add_argument("--deal-vol", type=float, default=20.0, metavar="PCT",
                   help="flag top names under this annualized vol as possibly deal-pinned")
    p.add_argument("--deal-ratio", type=float, default=0.60,
                   help="flag names whose recent/trailing vol ratio is under this")
    p.add_argument("--lookback", type=int, default=mc.DEFAULT_LOOKBACK)
    p.add_argument("--skip", type=int, default=mc.DEFAULT_SKIP)
    p.add_argument("--vol-window", type=int, default=mc.DEFAULT_VOL_WINDOW)
    p.add_argument("--cache-ttl", type=float, default=20.0,
                   help="hours before the price cache is refetched (0 disables)")
    p.add_argument("--offline-universe", action="store_true",
                   help="skip the Wikipedia refresh and use saved lists")
    p.add_argument("--out", default="results", help="directory for the CSV")
    args = p.parse_args(argv)

    started = time.time()

    # A file: universe carries a full path; keep only a filename-safe stem.
    label = Path(args.universe[5:]).stem if args.universe.startswith("file:") else args.universe
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)

    universe = load_universe(args.universe, allow_network=not args.offline_universe)
    symbols = {mc.normalize_ticker(s) for s in universe["Symbol"].astype(str)}

    manual_adds = [mc.normalize_ticker(s) for s in args.include.split(",") if s.strip()]
    symbols.update(manual_adds)

    dropped = {mc.normalize_ticker(s) for s in args.exclude.split(",") if s.strip()}
    for fname in args.exclude_file:
        dropped.update(read_symbol_file(HERE / fname))
    if dropped:
        hits = sorted(symbols & dropped)
        symbols -= dropped
        log(f"excluded {len(hits)} tickers: {', '.join(hits) if hits else '(none matched)'}")
    if manual_adds:
        log(f"added {len(manual_adds)} tickers: {', '.join(manual_adds)}")

    ordered = sorted(symbols)

    # Enough history for the lookback plus the volatility window, with slack
    # for holidays. Downloading more just slows the run down.
    days_needed = max(args.lookback, args.vol_window) + args.skip + 40
    start = (datetime.now(timezone.utc) - timedelta(days=int(days_needed * 1.5))).date()
    end = (datetime.now(timezone.utc) + timedelta(days=1)).date()

    prices = download_prices(
        ordered, str(start), str(end),
        cache_key=label,
        cache_ttl_hours=args.cache_ttl,
    )

    prices = drop_incomplete_session(prices)
    # A few missing prints shouldn't disqualify a name, but a long gap should.
    prices = prices.ffill(limit=5)

    # Rank the whole universe first, then filter. Doing it in this order keeps
    # "couldn't be scored" and "scored but screened out" as separate counts -
    # otherwise a volatility filter looks like a data outage.
    as_of, ranked_all = mc.build_table(
        prices,
        lookback=args.lookback, skip=args.skip, vol_window=args.vol_window,
    )
    no_data = sorted(set(ordered) - set(ranked_all.index))

    table = ranked_all
    if args.max_vol is not None:
        table = table[table["ann_vol_pct"] <= args.max_vol]
    if args.min_vol is not None:
        table = table[table["ann_vol_pct"] >= args.min_vol]
    table = table.copy()
    table["rank"] = range(1, len(table) + 1)
    table = table.join(jump_diagnostics(prices, as_of, args.lookback, args.skip))
    screened_out = len(ranked_all) - len(table)

    outdir = HERE / args.out
    outdir.mkdir(exist_ok=True)
    csv_path = outdir / f"momentum_{label}_{as_of.date()}.csv"
    table.to_csv(csv_path)

    bounds = []
    if args.max_vol is not None:
        bounds.append(f"vol <= {args.max_vol:g}%")
    if args.min_vol is not None:
        bounds.append(f"vol >= {args.min_vol:g}%")
    filt = (", " + " and ".join(bounds)) if bounds else ""

    print(f"# Momentum scan - {label.upper()} - as of {as_of.date()}\n")
    print(f"{len(table)} tickers ranked from a {len(ordered)}-name universe "
          f"({args.lookback}d lookback, {args.skip}d skip, "
          f"{args.vol_window}d vol{filt}).\n")
    print(to_markdown(table, args.top))
    print(f"\nFull results: {csv_path}")

    pinned = flag_pinned_names(table.head(args.top), args.deal_vol, args.deal_ratio)
    if len(pinned):
        print(f"\nCHECK FOR PENDING ACQUISITIONS - {len(pinned)} of the top "
              f"{args.top} have unusually low or collapsing volatility, which is "
              "what a stock pinned to a cash takeout price looks like:")
        for ticker, r in pinned.iterrows():
            print(f"  {ticker:6} rank {int(r['rank']):3d} | ann vol "
                  f"{r['ann_vol_pct']:5.1f}% | last-21d vol {r['recent_vol_pct']:5.1f}% "
                  f"| ratio {r['vol_ratio']:.2f}")
        print("  These are candidates to verify, not confirmed deals.")

    spiky = flag_spiky_names(table.head(args.top), args.jump_share, args.jump_id)
    if len(spiky):
        print(f"\nCHECK FOR EVENT SPIKES - {len(spiky)} of the top {args.top} made much of "
              "their 12-1 move in one day or in lumps, which momentum research says "
              "tends not to continue:")
        for ticker, r in spiky.iterrows():
            print(f"  {ticker:6} rank {int(r['rank']):3d} | biggest day {r['max_day_pct']:+6.1f}% "
                  f"| share of move {r['jump_share']:.2f} | FIP ID {r['fip_id']:+.3f}")
        print("  Research the cause. Exclude only if the move was a one-off event.")

    held = table.head(args.hold)
    weights = mc.weight_by_score(held["score"])
    print(f"\n## Hold the top {args.hold} (weights proportional to score)\n")
    print("| # | Ticker | Weight | Last price |")
    print("|--:|:--|--:|--:|")
    for ticker, r in held.iterrows():
        print(f"| {int(r['rank'])} | {ticker} | {weights[ticker] * 100:.1f}% | ${r['last_price']:,.2f} |")

    print(f"\nUniverse: {len(ranked_all)} scored, {screened_out} screened out by "
          f"the volatility filter, {len(no_data)} without usable price history.")
    if no_data:
        preview = ", ".join(no_data[:12]) + ("..." if len(no_data) > 12 else "")
        print(f"No usable price history: {preview}")
    log(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
