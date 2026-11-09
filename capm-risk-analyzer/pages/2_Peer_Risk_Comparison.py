"""Page 2: Peer risk comparison dashboard."""
import pandas as pd
import streamlit as st

from utils.capm import recommend, run_capm, to_row
from utils.charts import beta_r2_bars, cml_chart, sml_chart
from utils.data import MARKET_TICKER, DataError, build_returns, load_prices
from utils.ui import DISCLAIMER, sidebar_settings

st.set_page_config(page_title="Peer Risk Comparison", page_icon="📊", layout="wide")
s = sidebar_settings(show_ticker=False)

st.title("📊 Peer Risk Comparison")
st.markdown("Compare several stocks against the S&P 500 on systematic risk, model fit, and CAPM alpha.")

POPULAR = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ",
    "PFE", "UNH", "WMT", "KO", "PEP", "PG", "DIS", "NFLX", "HD", "CAT", "BA", "T",
]
chosen = st.multiselect("Select stocks to compare", POPULAR, default=["AAPL", "JPM", "XOM", "TSLA", "JNJ"])
extra = st.text_input("Add other tickers (comma-separated)", placeholder="e.g. COST, AMD")
tickers = sorted({t.strip().upper() for t in [*chosen, *extra.split(",")] if t.strip()} - {MARKET_TICKER})

if not tickers:
    st.info("Select at least one stock to compare.")
    st.stop()

try:
    with st.spinner("Downloading data and fitting CAPM models..."):
        close = load_prices(tuple(tickers), str(s.start), str(s.end))
except DataError as exc:
    st.error(str(exc))
    st.stop()

rows, skipped = [], []
for t in tickers:
    try:
        res = run_capm(build_returns(close, t, s.freq), t, s.freq)
    except DataError as exc:
        skipped.append(f"**{t}**: {exc}")
        continue
    rows.append(to_row(res, recommend(res, s.alpha_level, s.r2_min)))

for msg in skipped:
    st.warning(msg)
if not rows:
    st.stop()

table = pd.DataFrame(rows)
rf, mrp = table["rf_annual"].mean(), table["mrp_annual"].mean()

st.subheader("Summary table")
show = pd.DataFrame(
    {
        "Ticker": table["ticker"],
        "Beta": table["beta"],
        "R²": table["r2"],
        "Alpha (%/yr)": table["alpha_annual"] * 100,
        "Alpha p-value": table["alpha_p"],
        "CAPM expected return (%)": table["expected_return"] * 100,
        "Realized return (%)": table["realized_return"] * 100,
        "Volatility (%)": table["vol_annual"] * 100,
        "Signal": table["recommendation"],
    }
)
st.dataframe(
    show,
    hide_index=True,
    column_config={
        "Beta": st.column_config.NumberColumn(format="%.2f"),
        "R²": st.column_config.NumberColumn(format="%.2f"),
        "Alpha (%/yr)": st.column_config.NumberColumn(format="%.2f"),
        "Alpha p-value": st.column_config.NumberColumn(format="%.3f"),
        "CAPM expected return (%)": st.column_config.NumberColumn(format="%.2f"),
        "Realized return (%)": st.column_config.NumberColumn(format="%.2f"),
        "Volatility (%)": st.column_config.NumberColumn(format="%.2f"),
    },
)
st.download_button("Download table (CSV)", show.to_csv(index=False), "capm_peer_comparison.csv", "text/csv")

tab1, tab2, tab3 = st.tabs(["Beta & R²", "Security Market Line", "Capital Market Line"])
with tab1:
    st.plotly_chart(beta_r2_bars(table))
    st.caption("Red bars: β > 1 (high systematic risk). Blue: β < 1. Dashed line: market β = 1. "
               "Low R² means the market explains little of the stock's movement, so its beta is less informative.")
with tab2:
    st.plotly_chart(sml_chart(table, rf, mrp))
    st.caption("The SML plots CAPM-expected return against beta. Stocks above the line earned positive alpha over the "
               "window; stocks below it earned negative alpha. Marker colors show the Buy/Sell/Hold signal.")
with tab3:
    st.plotly_chart(cml_chart(table, rf, table["mkt_return"].mean(), table["mkt_vol"].mean()))
    st.caption("The CML uses total volatility on the x-axis. It applies to efficient portfolios, so single stocks "
               "normally plot below it.")

st.caption(DISCLAIMER)
