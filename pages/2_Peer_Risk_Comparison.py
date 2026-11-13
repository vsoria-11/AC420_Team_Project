# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed it,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Page 2: compare several stocks against the S&P 500."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import (DISCLAIMER, MARKET, SIGNAL_COLORS, download_prices, get_signal,
                     make_returns, run_capm, sidebar_settings)

st.set_page_config(page_title="Peer Risk Comparison", page_icon="📊", layout="wide")
settings = sidebar_settings(show_ticker=False)

st.title("📊 Peer Risk Comparison")
st.markdown("Compare several stocks against the S&P 500 on market risk (beta), model fit (R²) and alpha.")

POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ",
           "PFE", "UNH", "WMT", "KO", "PEP", "PG", "DIS", "NFLX", "HD", "CAT", "BA", "T"]
chosen = st.multiselect("Select stocks", POPULAR, default=["AAPL", "JPM", "XOM", "TSLA", "JNJ"])
extra = st.text_input("Add other tickers (comma-separated)", placeholder="e.g. COST, AMD")

tickers = set(chosen)
for text in extra.split(","):
    if text.strip() != "":
        tickers.add(text.strip().upper())
tickers.discard(MARKET)
tickers = sorted(tickers)

if len(tickers) == 0:
    st.info("Select at least one stock.")
    st.stop()

# ---- Run CAPM for every stock -----------------------------------------------------------------
try:
    prices = download_prices(tuple(tickers), str(settings["start"]), str(settings["end"]))
except ValueError as error:
    st.error(str(error))
    st.stop()

rows = []
for ticker in tickers:
    try:
        df = make_returns(prices, ticker, settings["freq"])
    except ValueError as error:
        st.warning(ticker + ": " + str(error))
        continue
    capm = run_capm(df, settings["freq"])
    signal, _ = get_signal(capm, settings["alpha_level"], settings["r2_min"])
    rows.append({"Ticker": ticker, "Beta": capm["beta"], "R²": capm["r2"],
                 "Alpha (%/yr)": capm["alpha_annual"] * 100, "Alpha p-value": capm["alpha_p"],
                 "CAPM expected return (%)": capm["expected_return"] * 100,
                 "Realized return (%)": capm["realized_return"] * 100,
                 "Volatility (%)": capm["volatility"] * 100, "Signal": signal,
                 "rf": capm["rf_annual"], "premium": capm["market_premium"],
                 "mkt_return": capm["market_return"], "mkt_vol": capm["market_volatility"]})
if len(rows) == 0:
    st.stop()

table = pd.DataFrame(rows)
rf = table["rf"].mean()
premium = table["premium"].mean()
colors = [SIGNAL_COLORS[s] for s in table["Signal"]]

st.subheader("Summary table")
show = table.drop(columns=["rf", "premium", "mkt_return", "mkt_vol"])
st.dataframe(show, hide_index=True, column_config={
    "Beta": st.column_config.NumberColumn(format="%.2f"),
    "R²": st.column_config.NumberColumn(format="%.2f"),
    "Alpha (%/yr)": st.column_config.NumberColumn(format="%.2f"),
    "Alpha p-value": st.column_config.NumberColumn(format="%.3f"),
    "CAPM expected return (%)": st.column_config.NumberColumn(format="%.2f"),
    "Realized return (%)": st.column_config.NumberColumn(format="%.2f"),
    "Volatility (%)": st.column_config.NumberColumn(format="%.2f"),
})
st.download_button("Download table (CSV)", show.to_csv(index=False), "capm_peer_comparison.csv", "text/csv")

tab1, tab2, tab3 = st.tabs(["Beta & R²", "Security Market Line", "Capital Market Line"])

with tab1:
    names = list(table["Ticker"]) + ["S&P 500"]               # the market itself has beta = 1 and R² = 1
    beta_colors = ["#ef4444" if b > 1 else "#3b82f6" for b in table["Beta"]] + ["gray"]
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=names, y=list(table["Beta"]) + [1.0], marker_color=beta_colors))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Beta (red = above 1, high market risk)")
        st.plotly_chart(fig)
    with col2:
        fig = go.Figure(go.Bar(x=names, y=list(table["R²"]) + [1.0], marker_color=beta_colors))
        fig.update_layout(title="R² (how well the market explains the stock)")
        st.plotly_chart(fig)

with tab2:
    # The Security Market Line is the CAPM formula drawn as a line: return = Rf + beta x premium
    betas = [0, max(1.5, table["Beta"].max() * 1.1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas, y=[(rf + b * premium) * 100 for b in betas], mode="lines",
                             name="Security Market Line", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=table["Beta"], y=table["Realized return (%)"], mode="markers+text",
                             text=table["Ticker"], textposition="top center", name="Realized return",
                             marker=dict(size=13, color=colors)))
    fig.update_layout(title="Beta vs return (a dot above the line = positive alpha; color = signal)",
                      xaxis_title="Beta", yaxis_title="Annualized return (%)")
    st.plotly_chart(fig)

with tab3:
    # The Capital Market Line starts at the risk-free rate and passes through the market portfolio
    mkt_return = table["mkt_return"].mean()
    mkt_vol = table["mkt_vol"].mean()
    slope = (mkt_return - rf) / mkt_vol
    vols = [0, max(table["Volatility (%)"].max(), mkt_vol * 100) * 1.15]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=[(rf + slope * v / 100) * 100 for v in vols], mode="lines",
                             name="Capital Market Line", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=table["Volatility (%)"], y=table["Realized return (%)"], mode="markers+text",
                             text=table["Ticker"], textposition="top center", name="Stocks",
                             marker=dict(size=13, color=colors)))
    fig.add_trace(go.Scatter(x=[mkt_vol * 100], y=[mkt_return * 100], mode="markers+text", text=["S&P 500"],
                             textposition="bottom right", name="Market", marker=dict(size=14, symbol="star")))
    fig.update_layout(title="Total volatility vs return (single stocks usually sit below the line)",
                      xaxis_title="Annualized volatility (%)", yaxis_title="Annualized return (%)")
    st.plotly_chart(fig)

st.caption(DISCLAIMER)
