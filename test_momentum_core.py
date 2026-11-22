"""Checks on the momentum math. Run directly: python test_momentum_core.py

No pytest required - each check raises AssertionError on failure and main()
prints a report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import momentum_core as mc


def _frame(values, name="TEST") -> pd.DataFrame:
    """Wrap a price array in a DataFrame with business-day dates."""
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({name: np.asarray(values, dtype=float)}, index=idx)


def test_skip_month_return_is_exact():
    """On a clean exponential path the 11-month return is analytic."""
    n, daily = 300, 1.001
    close = _frame([100.0 * daily**i for i in range(n)])

    got = mc.skip_month_return(close, lookback=252, skip=21).iloc[-1, 0]
    expected = daily ** (252 - 21) - 1.0  # compounding from t-252 to t-21

    assert abs(got - expected) < 1e-12, f"{got} != {expected}"
    return f"11-month return = {got:.6f} (exact to 1e-12)"


def test_naive_subtraction_overstates_recent_winners():
    """The v1 formula (r252 - r21) inflates a name that just spiked.

    Path: 100 -> 150 over the first 278 days, then 150 -> 450 in the final
    month. The true skip-month return stops at 150, so the tripling must be
    fully excluded. Subtracting simple returns does not exclude it.
    """
    n, jump_at = 300, 278
    ramp = [100.0 * (1.5 ** (i / jump_at)) for i in range(jump_at + 1)]
    spike = [150.0 * (3.0 ** ((i - jump_at) / (n - 1 - jump_at)))
             for i in range(jump_at + 1, n)]
    close = _frame(ramp + spike)

    px = close.iloc[:, 0]
    p_lookback, p_skip, p_now = px.iloc[-253], px.iloc[-22], px.iloc[-1]

    correct = mc.skip_month_return(close, 252, 21).iloc[-1, 0]
    naive = (p_now / p_lookback - 1) - (p_now / p_skip - 1)

    assert abs(correct - (p_skip / p_lookback - 1)) < 1e-12
    assert naive > correct + 0.5, "expected the naive formula to overstate badly"
    return (f"correct = {correct:+.3f} vs v1 naive = {naive:+.3f} "
            f"-> v1 overstates by {(naive - correct) * 100:.1f} points")


def test_naive_subtraction_understates_recent_losers():
    """Mirror case: a name that dipped last month is over-penalised by v1."""
    n, dip_at = 300, 278
    ramp = [100.0 * (2.0 ** (i / dip_at)) for i in range(dip_at + 1)]
    dip = [200.0 * (0.7 ** ((i - dip_at) / (n - 1 - dip_at)))
           for i in range(dip_at + 1, n)]
    close = _frame(ramp + dip)

    px = close.iloc[:, 0]
    p_lookback, p_skip, p_now = px.iloc[-253], px.iloc[-22], px.iloc[-1]

    correct = mc.skip_month_return(close, 252, 21).iloc[-1, 0]
    naive = (p_now / p_lookback - 1) - (p_now / p_skip - 1)

    assert naive < correct, "expected the naive formula to understate"
    return (f"correct = {correct:+.3f} vs v1 naive = {naive:+.3f} "
            f"-> v1 understates by {(correct - naive) * 100:.1f} points")


def test_annualizing_vol_preserves_ranking():
    """sqrt(252) is a constant, so it may move scores but never their order."""
    rng = np.random.default_rng(7)
    n, tickers = 400, [f"T{i}" for i in range(25)]
    paths = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=(n, len(tickers))), axis=0))
    close = pd.DataFrame(paths, columns=tickers,
                         index=pd.bdate_range("2020-01-01", periods=n))

    daily_rank = mc.momentum_score(close, annualize_vol=False).iloc[-1].rank()
    annual_rank = mc.momentum_score(close, annualize_vol=True).iloc[-1].rank()

    assert daily_rank.equals(annual_rank)
    ratio = (mc.momentum_score(close, annualize_vol=False).iloc[-1]
             / mc.momentum_score(close, annualize_vol=True).iloc[-1]).dropna()
    assert np.allclose(ratio, np.sqrt(252))
    return f"ranks identical across {len(tickers)} tickers; scores differ by exactly sqrt(252)"


def test_flat_price_does_not_rank_first():
    """A halted / stale-priced name has zero variance and must not divide to inf."""
    close = _frame([100.0] * 300, name="FLAT")
    score = mc.momentum_score(close).iloc[-1, 0]
    assert not np.isinf(score), "zero-volatility name produced an infinite score"
    assert np.isnan(score)
    return "zero-variance ticker yields NaN, not +inf"


def test_vol_filter_reranks_without_gaps():
    rng = np.random.default_rng(11)
    n, tickers = 400, [f"T{i}" for i in range(30)]
    vols = np.linspace(0.005, 0.05, len(tickers))
    steps = rng.normal(0.0008, 1.0, size=(n, len(tickers))) * vols
    close = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), columns=tickers,
                         index=pd.bdate_range("2020-01-01", periods=n))

    _, unfiltered = mc.build_table(close)
    _, filtered = mc.build_table(close, max_annual_vol=0.40)

    assert len(filtered) < len(unfiltered), "filter removed nothing"
    assert filtered["ann_vol_pct"].max() <= 40.0
    assert list(filtered["rank"]) == list(range(1, len(filtered) + 1))
    return (f"{len(unfiltered)} -> {len(filtered)} tickers under 40% vol; "
            "ranks contiguous 1..N")


def test_weights_handle_all_negative():
    scores = pd.Series({"A": -0.5, "B": -1.2, "C": -0.1})
    w = mc.weight_by_score(scores)
    assert not w.isna().any(), "all-negative scores produced NaN weights"
    assert w.sum() == 0.0

    mixed = pd.Series({"A": 2.0, "B": -1.0, "C": 6.0})
    wm = mc.weight_by_score(mixed)
    assert abs(wm.sum() - 1.0) < 1e-12 and wm["B"] == 0.0
    return "all-negative -> zero weights (v1 produced NaN); mixed -> sums to 1.0"


def test_ticker_normalization():
    cases = {"BRK.B": "BRK-B", "BF.B": "BF-B", "MOG.A": "MOG-A",
             "CWEN.A": "CWEN-A", "aapl": "AAPL", " MSFT ": "MSFT"}
    for raw, want in cases.items():
        assert mc.normalize_ticker(raw) == want, f"{raw} -> {mc.normalize_ticker(raw)}"
    return f"{len(cases)} symbols normalized for Yahoo"


def test_short_history_raises():
    close = _frame([100.0 + i for i in range(60)])
    try:
        mc.build_table(close)
    except ValueError:
        return "insufficient history raises ValueError instead of returning junk"
    raise AssertionError("expected ValueError on 60 rows of history")


def main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    print(f"Running {len(checks)} checks on the momentum math\n")
    for fn in checks:
        label = fn.__name__.removeprefix("test_").replace("_", " ")
        try:
            print(f"  PASS  {label}\n          {fn()}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {label}\n          {exc}")
    print(f"\n{len(checks) - failures}/{len(checks)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
