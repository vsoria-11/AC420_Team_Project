# [AI-GENERATED]
"""Simple tests on synthetic data (no internet needed). Run with:  pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import get_signal, run_capm, walk_forward          # noqa: E402
from outlook import (alignment_verdict, blume_beta, run_forecast, score_headlines,   # noqa: E402
                     simulate_prices)


def make_df(beta=1.3, alpha=0.0, n=200, noise=0.01, seed=0):
    """Fake monthly returns where we KNOW the true alpha and beta."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2005-01-31", periods=n, freq="ME")
    market_excess = rng.normal(0.007, 0.04, n)
    rf = np.full(n, 0.03 / 12)
    stock_excess = alpha + beta * market_excess + rng.normal(0, noise, n)
    return pd.DataFrame({"stock": stock_excess + rf, "market": market_excess + rf, "rf": rf,
                         "stock_excess": stock_excess, "market_excess": market_excess}, index=index)


def test_capm_recovers_beta():
    capm = run_capm(make_df(beta=1.3), "Monthly")
    assert abs(capm["beta"] - 1.3) < 0.1
    assert capm["r2"] > 0.9


def test_signals():
    assert get_signal(run_capm(make_df(alpha=0.01), "Monthly"), 0.10, 0.30)[0] == "BUY"
    assert get_signal(run_capm(make_df(alpha=-0.01), "Monthly"), 0.10, 0.30)[0] == "SELL"
    assert get_signal(run_capm(make_df(alpha=0.0, seed=3), "Monthly"), 0.10, 0.30)[0] == "HOLD"


def test_low_r2_blocks_buy():
    capm = run_capm(make_df(beta=0.1, alpha=0.01, noise=0.08), "Monthly")
    assert capm["r2"] < 0.30
    assert get_signal(capm, 0.10, 0.30)[0] != "BUY"


def test_blume_beta():
    assert abs(blume_beta(2.0) - 1.67) < 1e-9
    assert abs(blume_beta(1.0) - 1.0) < 1e-9


def test_simulation_average_matches_chosen_return():
    vols = np.full(252, 0.01)
    paths = simulate_prices(100.0, 0.10, vols, n_sims=20000)
    assert paths.shape == (252, 20000)
    assert abs(paths[-1].mean() / 100 - 1.10) < 0.01


def test_forecast_runs():
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 1000)))
    capm = run_capm(make_df(), "Monthly")
    result = run_forecast(prices, capm, 126)
    assert result["view"] in {"Favorable", "Mixed", "Unfavorable"}
    assert 0 <= result["prob_beat_rf"] <= 1
    assert run_forecast(prices.iloc[:100], capm, 126) is None       # too little history


def test_news_scoring():
    good = [{"title": "Company beats earnings and stock surges to record high", "publisher": "A"}] * 3
    bad = [{"title": "Company faces fraud probe as shares plunge", "publisher": "B"}] * 3
    assert score_headlines(good)["flag"] == "Positive"
    assert score_headlines(bad)["flag"] == "Negative"
    assert score_headlines(good[:2])["flag"] == "Insufficient news"


def test_alignment_verdict():
    kind, text = alignment_verdict("BUY", "Favorable", "Negative")
    assert kind == "error" and "Conflict" in text
    assert alignment_verdict("BUY", "Favorable", "Positive")[0] == "success"
    assert alignment_verdict("BUY", "Unfavorable", "Neutral")[0] == "warning"
    assert alignment_verdict("HOLD", "Mixed", "Negative", strong_negative=True)[0] == "warning"


def test_walk_forward():
    rows = walk_forward(make_df(n=200), window=60, horizon=6, alpha_level=0.10, r2_min=0.30)
    assert len(rows) > 10
    assert set(rows[0]) == {"date", "signal", "beta", "stock_return", "forward_alpha"}
    assert rows[0]["signal"] in {"BUY", "SELL", "HOLD"}
