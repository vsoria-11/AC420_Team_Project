# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed a function,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Core helpers used by every page:
sidebar settings -> download data -> returns -> CAPM regression -> Buy/Sell/Hold signal -> backtest."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

MARKET = "^GSPC"        # S&P 500 = our proxy for "the market"
RISK_FREE = "^IRX"      # 13-week T-bill yield (quoted in % per year) = our risk-free rate
PERIODS = {"Daily": 252, "Weekly": 52, "Monthly": 12}   # periods per year, used to annualize
MIN_OBS = 24            # fewer observations than this makes a regression meaningless

SIGNAL_COLORS = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04"}
DISCLAIMER = "Educational project for AC 420. Not investment advice."


# [AI-GENERATED]
def sidebar_settings(show_ticker=True, show_window=True):
    """Draw the sidebar widgets and return the user's choices as a dictionary.

    Streamlit forgets widget values when you switch pages. Setting each key to itself
    at the top of every page is the simple trick that keeps the values."""
    today = date.today()
    defaults = {
        "ticker": "AAPL",
        "dates": (today - timedelta(days=5 * 365), today),
        "freq": "Monthly",
        "alpha_level": 0.10,
        "r2_min": 0.30,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
        else:
            st.session_state[key] = st.session_state[key]

    st.sidebar.header("Model settings")
    if show_ticker:
        st.sidebar.text_input("Stock ticker", key="ticker", help="Any Yahoo Finance symbol, e.g. AAPL, MSFT, JPM")
    if show_window:
        st.sidebar.slider("Date range", min_value=date(1990, 1, 1), max_value=today,
                          key="dates", format="YYYY-MM-DD", help="Estimation window for the regression")
        st.sidebar.selectbox("Return frequency", ["Daily", "Weekly", "Monthly"], key="freq",
                             help="Monthly returns over about 5 years is the textbook way to estimate beta")
    st.sidebar.select_slider("Significance level for alpha", options=[0.01, 0.05, 0.10], key="alpha_level",
                             help="Alpha counts as 'real' only if its p-value is below this level")
    st.sidebar.slider("Minimum R² for a reliable signal", min_value=0.0, max_value=1.0, step=0.05, key="r2_min",
                      help="A BUY needs R² at or above this value")
    st.sidebar.caption("Benchmark: S&P 500 (^GSPC) | Risk-free: 13-week T-bill (^IRX)")

    start, end = st.session_state["dates"]
    return {
        "ticker": st.session_state["ticker"].strip().upper(),
        "start": start,
        "end": end,
        "freq": st.session_state["freq"],
        "alpha_level": st.session_state["alpha_level"],
        "r2_min": st.session_state["r2_min"],
    }


# [AI-GENERATED]
@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end):
    """Download daily adjusted closing prices for the tickers plus the market and the T-bill yield.
    st.cache_data remembers the result for an hour so the app does not re-download on every click."""
    symbols = list(dict.fromkeys(list(tickers) + [MARKET, RISK_FREE]))
    end_plus_one = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end date is exclusive
    try:
        data = yf.download(symbols, start=start, end=end_plus_one, auto_adjust=True, progress=False)
    except Exception as error:
        raise ValueError("Could not reach Yahoo Finance: " + str(error))
    if data is None or data.empty:
        raise ValueError("Yahoo Finance returned no data. Check the ticker and the date range.")

    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(symbols[0])
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    return prices.sort_index()


# [AI-GENERATED]
def make_returns(prices, ticker, freq):
    """Turn prices into returns and EXCESS returns (return minus the risk-free rate).
    CAPM is a regression of the stock's excess return on the market's excess return."""
    if ticker == MARKET:
        raise ValueError("Pick a stock, not the S&P 500 itself (its beta would always be 1).")
    if ticker not in prices.columns or prices[ticker].dropna().empty:
        raise ValueError("No price data found for '" + ticker + "'. Check the symbol and date range.")

    px = prices[[ticker, MARKET]].dropna()
    # ^IRX is quoted in percent per year, so divide by 100. Fill gaps with the last known value.
    rf_yield = prices[RISK_FREE].ffill().reindex(px.index, method="ffill").bfill().fillna(0.0) / 100

    periods = PERIODS[freq]
    if freq == "Weekly":
        px = px.resample("W-FRI").last().dropna()
        rf_yield = rf_yield.resample("W-FRI").last().reindex(px.index).ffill()
    if freq == "Monthly":
        px = px.resample("ME").last().dropna()
        rf_yield = rf_yield.resample("ME").last().reindex(px.index).ffill()

    returns = px.pct_change()
    df = pd.DataFrame({
        "stock": returns[ticker],
        "market": returns[MARKET],
        "rf": rf_yield.shift(1) / periods,   # yearly yield turned into a per-period rate
    }).dropna()
    df["stock_excess"] = df["stock"] - df["rf"]
    df["market_excess"] = df["market"] - df["rf"]

    if len(df) < MIN_OBS:
        raise ValueError("Only " + str(len(df)) + " observations for " + ticker + " (need " + str(MIN_OBS)
                         + "). Use a longer date range or a higher frequency.")
    return df


# [AI-GENERATED]
def run_capm(df, freq):
    """Fit the CAPM regression:  stock excess return = alpha + beta * market excess return + error."""
    periods = PERIODS[freq]
    X = sm.add_constant(df["market_excess"])          # the constant is what gives us alpha
    model = sm.OLS(df["stock_excess"], X).fit()

    alpha = float(model.params["const"])
    beta = float(model.params["market_excess"])
    rf_annual = float(df["rf"].mean() * periods)
    market_premium = float(df["market_excess"].mean() * periods)   # average yearly market excess return
    avg_excess_annual = float(df["stock_excess"].mean() * periods)

    return {
        "model": model,                                  # kept so the Diagnostics page can show the full table
        "alpha": alpha,                                  # per period
        "alpha_annual": alpha * periods,
        "alpha_p": float(model.pvalues["const"]),
        "beta": beta,
        "beta_se": float(model.bse["market_excess"]),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "n_obs": int(model.nobs),
        "rf_annual": rf_annual,
        "market_premium": market_premium,
        "avg_excess_annual": avg_excess_annual,
        "expected_return": rf_annual + beta * market_premium,   # the CAPM formula: Rf + beta * (Rm - Rf)
        "realized_return": rf_annual + avg_excess_annual,
        "volatility": float(df["stock"].std() * np.sqrt(periods)),
        "market_return": rf_annual + market_premium,
        "market_volatility": float(df["market"].std() * np.sqrt(periods)),
    }


# [AI-GENERATED]
def get_signal(capm, alpha_level, r2_min):
    """The decision matrix. Returns ("BUY" / "SELL" / "HOLD", list of reasons).
    This is the ONLY place the Buy/Sell/Hold label is decided. Forecasts and news never change it."""
    significant = capm["alpha_p"] < alpha_level       # is alpha statistically different from zero?
    reliable = capm["r2"] >= r2_min                   # does the market explain enough of the stock?
    alpha_text = f"{capm['alpha_annual']:+.2%} per year"
    p_text = f"p = {capm['alpha_p']:.3f}"

    # SELL rule 1: alpha is significantly negative
    if significant and capm["alpha"] < 0:
        reasons = ["Alpha is negative (" + alpha_text + ") and statistically significant (" + p_text + ")."]
        if not reliable:
            reasons.append(f"Caution: R² is only {capm['r2']:.2f}, so the CAPM fit is weak.")
        return "SELL", reasons

    # SELL rule 2: very high risk (beta above 1.5) with no excess return to show for it
    if capm["beta"] > 1.5 and capm["avg_excess_annual"] <= 0:
        return "SELL", [f"Beta is {capm['beta']:.2f} (very high risk) but the average excess return is "
                        f"{capm['avg_excess_annual']:+.2%} per year (zero or negative)."]

    # BUY: alpha is significantly positive AND the model is reliable
    if significant and capm["alpha"] > 0 and reliable:
        return "BUY", ["Alpha is positive (" + alpha_text + ") and statistically significant (" + p_text + ").",
                       f"R² is {capm['r2']:.2f}, high enough to trust the estimate."]

    # Everything else is HOLD
    reasons = []
    if not significant:
        reasons.append("Alpha (" + alpha_text + ") is not statistically different from zero (" + p_text
                       + "), so the stock earns roughly what its risk predicts.")
    if significant and capm["alpha"] > 0 and not reliable:
        reasons.append("Alpha is positive and significant, but the model is not reliable enough to act on.")
    if not reliable:
        reasons.append(f"R² is only {capm['r2']:.2f}: the market explains little of this stock's movement, "
                       "so beta is not very informative.")
    return "HOLD", reasons


# [AI-GENERATED]
def beta_label(beta):
    if beta < 0:
        return "Negative beta"
    if beta > 1.5:
        return "Very high beta"
    if beta > 1:
        return "High beta (aggressive)"
    return "Low beta (defensive)"


# [AI-GENERATED]
def r2_label(r2):
    if r2 >= 0.5:
        return "High reliability"
    if r2 >= 0.25:
        return "Moderate reliability"
    return "Low reliability"


# [AI-GENERATED]
@st.cache_data(ttl=300, show_spinner=False)
def get_live_quote(ticker):
    """Latest price and previous close from Yahoo Finance, or None if unavailable."""
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info["last_price"]), float(info["previous_close"])
    except Exception:
        return None


# [AI-GENERATED]
def walk_forward(df, window, horizon, alpha_level, r2_min):
    """Out-of-sample test of the CAPM signal on one stock (df = MONTHLY returns).

    Step 1: fit CAPM on the last `window` months and issue BUY/SELL/HOLD.
    Step 2: look at the NEXT `horizon` months, which the model has never seen.
    Step 3: forward alpha = actual stock return - what CAPM says it should have earned
            (risk-free + beta * what the market actually did).
    Then jump ahead `horizon` months and repeat, so the test periods never overlap."""
    rows = []
    for end_pos in range(window, len(df) - horizon + 1, horizon):
        train = df.iloc[end_pos - window:end_pos]
        capm = run_capm(train, "Monthly")
        signal, _ = get_signal(capm, alpha_level, r2_min)

        future = df.iloc[end_pos:end_pos + horizon]
        stock_return = (1 + future["stock"]).prod() - 1
        market_return = (1 + future["market"]).prod() - 1
        rf_return = (1 + future["rf"]).prod() - 1
        capm_expected = rf_return + capm["beta"] * (market_return - rf_return)

        rows.append({
            "date": df.index[end_pos - 1],
            "signal": signal,
            "beta": capm["beta"],
            "stock_return": float(stock_return),
            "forward_alpha": float(stock_return - capm_expected),
        })
    return rows


# ---------------------------------------------------------------------------
# [TEAM-WRITTEN] space: put code your team writes yourselves below this line.
# ---------------------------------------------------------------------------
