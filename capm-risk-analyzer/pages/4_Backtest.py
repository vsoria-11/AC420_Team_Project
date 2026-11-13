# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed it,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Page 4: walk-forward backtest of the CAPM Buy/Sell/Hold signal."""

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from helpers import DISCLAIMER, MARKET, download_prices, make_returns, sidebar_settings, walk_forward

st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")
settings = sidebar_settings(show_ticker=False, show_window=False)   # only alpha level and R² threshold apply here

st.title("🧪 Backtest: did the signal work out-of-sample?")
st.markdown(
    "The CAPM signal is estimated on **past** data. This page tests it on data the model has **never seen**:\n"
    "1. Fit CAPM on a trailing window of months and issue BUY / SELL / HOLD.\n"
    "2. Look at the next 3, 6 or 12 months.\n"
    "3. **Forward alpha** = actual return minus the return CAPM said the stock should earn "
    "(risk-free rate + beta x what the market actually did).\n"
    "4. Jump ahead and repeat, so test periods never overlap.")

DEFAULTS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "KO", "PFE", "WMT", "CAT", "BA"]
chosen = st.multiselect("Stocks to test", DEFAULTS + ["GOOGL", "AMZN", "NVDA", "BAC", "CVX", "PG", "DIS", "HD", "T"],
                        default=DEFAULTS)
extra = st.text_input("Add other tickers (comma-separated)", placeholder="e.g. COST, AMD")
dates = st.slider("Backtest period", min_value=date(1990, 1, 1), max_value=date.today(),
                  value=(date(2000, 1, 1), date.today()), format="YYYY-MM-DD")
window = st.select_slider("Training window (months)", options=[36, 48, 60, 72, 84], value=60)
st.caption(f"Signal thresholds come from the sidebar: alpha significance {settings['alpha_level']:.2f}, "
           f"minimum R² {settings['r2_min']:.2f}.")

tickers = set(chosen)
for text in extra.split(","):
    if text.strip() != "":
        tickers.add(text.strip().upper())
tickers.discard(MARKET)
tickers = sorted(tickers)

HORIZONS = [3, 6, 12]

# Results are stored in session_state so they stay on screen when a widget changes.
if st.button("Run backtest", type="primary"):
    if len(tickers) == 0:
        st.warning("Select at least one stock.")
        st.stop()
    try:
        with st.spinner("Downloading prices and running the walk-forward test..."):
            prices = download_prices(tuple(tickers), str(dates[0]), str(dates[1]))
            all_rows = []
            for ticker in tickers:
                try:
                    df = make_returns(prices, ticker, "Monthly")
                except ValueError as error:
                    st.warning(ticker + ": " + str(error))
                    continue
                for horizon in HORIZONS:
                    for row in walk_forward(df, window, horizon, settings["alpha_level"], settings["r2_min"]):
                        row["ticker"] = ticker
                        row["horizon"] = horizon
                        all_rows.append(row)
            st.session_state["backtest_rows"] = all_rows
    except ValueError as error:
        st.error(str(error))
        st.stop()

rows = st.session_state.get("backtest_rows")
if rows is None:
    st.info("Choose your stocks and settings, then click **Run backtest**.")
    st.stop()
if len(rows) == 0:
    st.warning("Not enough history to run the test. Use a longer backtest period or a shorter training window.")
    st.stop()

results = pd.DataFrame(rows)

# ---- Summary table: one row per horizon and signal -----------------------------------------------
summary = []
for horizon in HORIZONS:
    for signal in ["BUY", "HOLD", "SELL"]:
        part = results[(results["horizon"] == horizon) & (results["signal"] == signal)]
        if len(part) == 0:
            continue
        # Hit rate: a BUY is a hit if it beat CAPM, a SELL is a hit if it lagged CAPM. HOLD has no direction.
        if signal == "BUY":
            hit_rate = (part["forward_alpha"] > 0).mean() * 100
        elif signal == "SELL":
            hit_rate = (part["forward_alpha"] < 0).mean() * 100
        else:
            hit_rate = None
        summary.append({"Horizon (months)": horizon, "Signal": signal, "Tests": len(part),
                        "Hit rate (%)": hit_rate,
                        "Avg forward alpha (%)": part["forward_alpha"].mean() * 100,
                        "Avg stock return (%)": part["stock_return"].mean() * 100})
summary = pd.DataFrame(summary)

st.subheader("Results by signal")
st.dataframe(summary, hide_index=True, column_config={
    "Hit rate (%)": st.column_config.NumberColumn(format="%.1f"),
    "Avg forward alpha (%)": st.column_config.NumberColumn(format="%.2f"),
    "Avg stock return (%)": st.column_config.NumberColumn(format="%.2f"),
})
fig = px.bar(summary, x="Horizon (months)", y="Avg forward alpha (%)", color="Signal", barmode="group",
             color_discrete_map={"BUY": "#16a34a", "HOLD": "#ca8a04", "SELL": "#dc2626"},
             title="Average forward alpha by signal (a working signal: BUY above zero, SELL below zero)")
fig.add_hline(y=0, line_color="gray")
st.plotly_chart(fig)

# ---- Simple long-short result: buy the BUYs, short the SELLs -------------------------------------
st.subheader("Simple long-short result")
long_short = []
for horizon in HORIZONS:
    buys = results[(results["horizon"] == horizon) & (results["signal"] == "BUY")]
    sells = results[(results["horizon"] == horizon) & (results["signal"] == "SELL")]
    if len(buys) > 0 and len(sells) > 0:
        long_short.append({"Horizon (months)": horizon,
                           "Avg BUY return (%)": buys["stock_return"].mean() * 100,
                           "Avg SELL return (%)": sells["stock_return"].mean() * 100,
                           "Long-short spread (%)": (buys["stock_return"].mean() - sells["stock_return"].mean()) * 100})
if long_short:
    st.dataframe(pd.DataFrame(long_short), hide_index=True)
    st.caption("Spread = average return of BUY stocks minus average return of SELL stocks over the same horizon. "
               "Positive means the signal ranked stocks in the right order.")
else:
    st.info("Need both BUY and SELL signals in the same horizon to compute a long-short spread.")

# ---- Honest caveats ------------------------------------------------------------------------------
with st.expander("How to read these results (please read)", expanded=True):
    st.markdown(
        "- **Results may be mixed.** Alpha is noisy and only weakly persistent, so do not be surprised if BUY does not "
        "clearly beat SELL. A tested and honest result is more useful than an untested confident one.\n"
        "- **Small samples.** Few tests per signal means the averages are unreliable. Treat them as illustrative.\n"
        "- **Survivorship bias.** You pick stocks that exist today, which flatters the results.\n"
        "- **Only the CAPM signal is backtested.** The news layer cannot be, because yfinance does not provide "
        "historical headlines. The forecast layer is not backtested either.\n"
        "- **No trading costs or taxes** are included.")

st.caption(DISCLAIMER)
