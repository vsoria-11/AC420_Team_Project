import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.capm import recommend, residual_tests, rolling_beta, run_capm  # noqa: E402


def make_df(beta=1.3, alpha=0.0, n=120, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    mkt_ex = rng.normal(0.007, 0.04, n)
    rf = np.full(n, 0.03 / 12)
    stock_ex = alpha + beta * mkt_ex + rng.normal(0, noise, n)
    return pd.DataFrame(
        {"stock": stock_ex + rf, "market": mkt_ex + rf, "rf": rf, "stock_excess": stock_ex, "market_excess": mkt_ex},
        index=idx,
    )


def test_beta_recovery():
    res = run_capm(make_df(beta=1.3), "TEST", "Monthly")
    assert abs(res.beta - 1.3) < 0.1
    assert res.r2 > 0.9


def test_buy_sell_hold():
    assert recommend(run_capm(make_df(alpha=0.01), "T", "Monthly")).label == "BUY"
    assert recommend(run_capm(make_df(alpha=-0.01), "T", "Monthly")).label == "SELL"
    assert recommend(run_capm(make_df(alpha=0.0, seed=3), "T", "Monthly")).label == "HOLD"


def test_low_r2_is_hold():
    res = run_capm(make_df(beta=0.1, alpha=0.01, noise=0.08), "T", "Monthly")
    assert res.r2 < 0.3
    assert recommend(res).label in {"HOLD", "SELL"}


def test_diagnostics_run():
    res = run_capm(make_df(), "T", "Monthly")
    assert len(residual_tests(res)) == 3
    assert len(rolling_beta(res.data, 24)) > 0
