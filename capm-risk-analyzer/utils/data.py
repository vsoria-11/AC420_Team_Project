"""Data pipeline: yfinance downloads -> aligned excess-return panel."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

MARKET_TICKER = "^GSPC"  # S&P 500 benchmark
RF_TICKER = "^IRX"  # 13-week T-bill yield (quoted in annualized percent)
PERIODS_PER_YEAR = {"Daily": 252, "Weekly": 52, "Monthly": 12}
_RESAMPLE_RULE = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}
MIN_OBS = 24  # minimum number of return observations for a usable regression


class DataError(Exception):
    """Raised when market data cannot be retrieved or is unusable."""


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closes for `tickers` plus the market index and risk-free proxy."""
    symbols = list(dict.fromkeys([*tickers, MARKET_TICKER, RF_TICKER]))
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive
    try:
        raw = yf.download(
            symbols, start=start, end=end_excl, auto_adjust=True, progress=False, threads=True
        )
    except Exception as exc:  # network / API failures
        raise DataError(f"Could not reach Yahoo Finance: {exc}") from exc
    if raw is None or raw.empty:
        raise DataError("Yahoo Finance returned no data. Check the ticker symbol and date range.")

    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    close = close.copy()
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.index = close.index.normalize()
    return close.sort_index()


def build_returns(close: pd.DataFrame, ticker: str, freq: str) -> pd.DataFrame:
    """Return a DataFrame of stock/market/rf period returns and excess returns."""
    if ticker == MARKET_TICKER:
        raise DataError("Pick a stock other than the market index itself (beta would be trivially 1).")
    if ticker not in close.columns or close[ticker].dropna().empty:
        raise DataError(f"No price data found for '{ticker}'. Check the symbol and date range.")
    if MARKET_TICKER not in close.columns or close[MARKET_TICKER].dropna().empty:
        raise DataError("Could not download S&P 500 (^GSPC) data for this date range.")

    px = close[[ticker, MARKET_TICKER]].dropna()
    if RF_TICKER in close.columns:
        rf = close[RF_TICKER].ffill().reindex(px.index, method="ffill").bfill().fillna(0.0) / 100.0
    else:  # fall back to a zero risk-free rate
        rf = pd.Series(0.0, index=px.index)

    ppy = PERIODS_PER_YEAR[freq]
    rule = _RESAMPLE_RULE[freq]
    if rule:
        px = px.resample(rule).last().dropna()
        rf = rf.resample(rule).last().reindex(px.index).ffill()

    rets = px.pct_change()
    out = pd.DataFrame(
        {
            "stock": rets[ticker],
            "market": rets[MARKET_TICKER],
            "rf": rf.shift(1) / ppy,  # yield known at the start of each period
        }
    ).dropna()
    out["stock_excess"] = out["stock"] - out["rf"]
    out["market_excess"] = out["market"] - out["rf"]

    if len(out) < MIN_OBS:
        raise DataError(
            f"Only {len(out)} {freq.lower()} observations for '{ticker}' "
            f"(need at least {MIN_OBS}). Widen the date range or use a higher frequency."
        )
    return out


def get_returns(ticker: str, start, end, freq: str) -> pd.DataFrame:
    close = load_prices((ticker,), str(start), str(end))
    return build_returns(close, ticker, freq)


@st.cache_data(ttl=300, show_spinner=False)
def get_live_quote(ticker: str) -> tuple[float, float] | None:
    """Return (last_price, previous_close) or None if unavailable."""
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info["last_price"]), float(info["previous_close"])
    except Exception:
        return None
