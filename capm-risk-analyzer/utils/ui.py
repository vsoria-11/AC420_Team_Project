"""Shared sidebar controls (settings persist across pages via st.session_state)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import streamlit as st

from .data import MARKET_TICKER, PERIODS_PER_YEAR


@dataclass
class Settings:
    ticker: str
    start: dt.date
    end: dt.date
    freq: str
    alpha_level: float
    r2_min: float


def _sync(name: str) -> None:
    st.session_state[name] = st.session_state[f"w_{name}"]


def sidebar_settings(show_ticker: bool = True) -> Settings:
    today = dt.date.today()
    defaults = {
        "ticker": "AAPL",
        "dates": (today - dt.timedelta(days=5 * 365), today),
        "freq": "Monthly",
        "alpha_level": 0.10,
        "r2_min": 0.30,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    sb = st.sidebar
    sb.header("Model settings")

    if show_ticker:
        sb.text_input(
            "Stock ticker",
            value=st.session_state["ticker"],
            key="w_ticker",
            on_change=_sync,
            args=("ticker",),
            help="Any Yahoo Finance symbol, e.g. AAPL, MSFT, TSLA, JPM.",
        )

    sb.slider(
        "Date range",
        min_value=dt.date(1990, 1, 1),
        max_value=today,
        value=st.session_state["dates"],
        format="YYYY-MM-DD",
        key="w_dates",
        on_change=_sync,
        args=("dates",),
        help="Estimation window for the CAPM regression.",
    )
    freqs = list(PERIODS_PER_YEAR)
    sb.selectbox(
        "Return frequency",
        freqs,
        index=freqs.index(st.session_state["freq"]),
        key="w_freq",
        on_change=_sync,
        args=("freq",),
        help="Monthly returns over ~5 years is the textbook beta estimate (60 observations).",
    )
    sb.select_slider(
        "Significance level for alpha",
        options=[0.01, 0.05, 0.10],
        value=st.session_state["alpha_level"],
        key="w_alpha_level",
        on_change=_sync,
        args=("alpha_level",),
        help="Alpha must have a p-value below this level to count as statistically significant.",
    )
    sb.slider(
        "Minimum R² for a reliable signal",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["r2_min"]),
        step=0.05,
        key="w_r2_min",
        on_change=_sync,
        args=("r2_min",),
        help="BUY signals require R² at or above this threshold.",
    )
    sb.caption(f"Benchmark: S&P 500 ({MARKET_TICKER}) · Risk-free proxy: 13-week T-bill (^IRX)")

    start, end = st.session_state["dates"]
    return Settings(
        ticker=st.session_state["ticker"].strip().upper(),
        start=start,
        end=end,
        freq=st.session_state["freq"],
        alpha_level=float(st.session_state["alpha_level"]),
        r2_min=float(st.session_state["r2_min"]),
    )


def banner(label: str, text: str) -> None:
    {"BUY": st.success, "SELL": st.error, "HOLD": st.warning}[label](text)


DISCLAIMER = (
    "Educational project for AC 420. Outputs are statistical estimates from historical data "
    "and are not investment advice."
)
