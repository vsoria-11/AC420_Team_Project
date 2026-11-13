# [AI-GENERATED]  (change to [AI-GENERATED, TEAM-REVIEWED] once your team has reviewed a function,
#                  or to [TEAM-WRITTEN] for code you rewrite yourselves)
"""Forward-looking helpers: GARCH volatility, Monte Carlo simulation, analyst view,
news sentiment, and the alignment verdict. NOTHING here changes the Buy/Sell/Hold label."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from arch import arch_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TRADING_DAYS = 252
MIN_HEADLINES = 3          # fewer headlines than this is not enough to say anything
POSITIVE_CUTOFF = 0.15     # average VADER score at or above this = Positive
NEGATIVE_CUTOFF = -0.15    # at or below this = Negative
STRONG_NEGATIVE = -0.35    # at or below this = strongly negative

# VADER is built for social media, so we teach it a few words that mean more in finance headlines.
FINANCE_WORDS = {
    "upgrade": 2.0, "upgrades": 2.0, "downgrade": -2.0, "downgrades": -2.0,
    "beat": 1.5, "beats": 1.5, "miss": -1.5, "misses": -1.5,
    "surge": 2.0, "surges": 2.0, "soar": 2.0, "soars": 2.0, "rally": 1.5,
    "plunge": -2.5, "plunges": -2.5, "tumble": -2.0, "tumbles": -2.0, "slump": -2.0,
    "lawsuit": -1.5, "probe": -1.5, "fraud": -3.0, "bankruptcy": -3.0,
    "layoffs": -1.5, "recall": -1.5, "outperform": 2.0, "underperform": -2.0, "record": 1.0,
}


# [AI-GENERATED]
def blume_beta(raw_beta):
    """Betas drift back toward 1 over time, so adjust the raw beta for forecasting."""
    return 0.67 * raw_beta + 0.33 * 1.0


# [AI-GENERATED]
def rolling_beta_check(df, freq):
    """Re-estimate beta on rolling windows to see whether it is stable over time."""
    window = {"Daily": 126, "Weekly": 52, "Monthly": 36}[freq]
    if len(df) < window + 12:
        return {"message": "Not enough data for a rolling-beta check.", "stable": None}
    covariance = df["stock_excess"].rolling(window).cov(df["market_excess"])
    variance = df["market_excess"].rolling(window).var()
    rolling = (covariance / variance).dropna()
    low, high = float(rolling.min()), float(rolling.max())
    if high - low > 1.0:
        return {"message": f"Beta is unstable: it ranged from {low:.2f} to {high:.2f}, so treat it with caution.",
                "stable": False}
    return {"message": f"Beta is fairly stable: it ranged from {low:.2f} to {high:.2f}.", "stable": True}


# [AI-GENERATED]
def garch_daily_vols(daily_returns, n_days):
    """GARCH(1,1) forecasts tomorrow's volatility from recent shocks and recent volatility,
    so it captures 'calm periods and stormy periods'. Returns one volatility per future day."""
    try:
        model = arch_model(daily_returns * 100, mean="Constant", vol="GARCH", p=1, q=1)  # x100 helps the optimizer
        fit = model.fit(disp="off")
        variance = fit.forecast(horizon=n_days).variance.iloc[-1].values   # percent squared, per day
        return np.sqrt(variance) / 100, "GARCH(1,1) forecast"
    except Exception:
        return np.full(n_days, daily_returns.std()), "GARCH did not fit, so historical volatility was used"


# [AI-GENERATED]
def simulate_prices(start_price, annual_return, daily_vols, n_sims=5000):
    """Monte Carlo: simulate thousands of possible price paths.
    Each day: log return = drift - half the variance + volatility * random shock.
    Subtracting half the variance makes the AVERAGE outcome match the annual return we chose."""
    n_days = len(daily_vols)
    daily_drift = np.log(1 + max(annual_return, -0.99)) / TRADING_DAYS
    rng = np.random.default_rng(42)                      # fixed seed so results do not change on every click
    shocks = rng.standard_normal((n_days, n_sims))
    vols = daily_vols.reshape(-1, 1)
    log_returns = daily_drift - 0.5 * vols ** 2 + vols * shocks
    return start_price * np.exp(np.cumsum(log_returns, axis=0))   # shape: (days, simulations)


# [AI-GENERATED]
def run_forecast(prices, capm, n_days):
    """Build the forward view for one stock. `prices` = daily closes (about the last 5 years)."""
    prices = prices.dropna()
    daily_returns = prices.pct_change().dropna().iloc[-1260:]
    if len(daily_returns) < 250:
        return None                                      # under about a year of data: skip the forecast

    start_price = float(prices.iloc[-1])
    adj_beta = blume_beta(capm["beta"])
    # Scenario 1: CAPM drift only (no alpha). Used for the fan chart, risk-free odds and Value-at-Risk.
    forward_return = capm["rf_annual"] + adj_beta * capm["market_premium"]
    # Scenario 2: CAPM drift PLUS the historical alpha. "What if the past alpha continues?"
    alpha_return = forward_return + capm["alpha_annual"]

    vols, vol_note = garch_daily_vols(daily_returns, n_days)
    paths = simulate_prices(start_price, forward_return, vols)
    alpha_paths = simulate_prices(start_price, alpha_return, vols)

    years = n_days / TRADING_DAYS
    rf_target = (1 + capm["rf_annual"]) ** years - 1       # what cash would earn over the horizon
    capm_target = (1 + forward_return) ** years - 1        # what CAPM says the stock should earn
    final = paths[-1] / start_price - 1
    alpha_final = alpha_paths[-1] / start_price - 1
    prob_beat_rf = float(np.mean(final > rf_target))

    # Forecast view: a simple label used by the alignment verdict
    if prob_beat_rf >= 0.60:
        view = "Favorable"
    elif prob_beat_rf < 0.50:
        view = "Unfavorable"
    else:
        view = "Mixed"

    return {
        "view": view,
        "adj_beta": adj_beta,
        "forward_return": forward_return,
        "start_price": start_price,
        "paths": paths,
        "vol_note": vol_note,
        "forecast_vol": float(np.sqrt(np.mean(vols ** 2) * TRADING_DAYS)),
        "prob_beat_rf": prob_beat_rf,
        "prob_beat_capm_if_alpha": float(np.mean(alpha_final > capm_target)),
        "var5": float(np.percentile(final, 5)),          # return at the 5th percentile = 5% Value-at-Risk
        "median": float(np.median(final)),
    }


# [AI-GENERATED]
def fan_chart(paths, start_price, n_days, ticker):
    """Plot the simulated paths as bands: 5-95%, 25-75% and the median."""
    months = np.arange(n_days + 1) / 21

    def band(percentile):
        return np.r_[start_price, np.percentile(paths, percentile, axis=1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=band(95), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=band(5), line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(37,99,235,0.15)", name="5%-95% of outcomes"))
    fig.add_trace(go.Scatter(x=months, y=band(75), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=band(25), line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(37,99,235,0.30)", name="25%-75% of outcomes"))
    fig.add_trace(go.Scatter(x=months, y=band(50), line=dict(color="#1d4ed8", width=3), name="Median path"))
    fig.update_layout(title=f"{ticker}: simulated price range ({n_days // 21} months)",
                      xaxis_title="Months from today", yaxis_title="Price ($)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=60))
    return fig


# [AI-GENERATED]
@st.cache_data(ttl=3600, show_spinner=False)
def get_analyst_view(ticker):
    """Analyst mean price target from Yahoo Finance, turned into an implied return. None if missing."""
    try:
        info = yf.Ticker(ticker).info
        target = info.get("targetMeanPrice")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not target or not price:
            return None
        return {"target": float(target), "price": float(price),
                "implied_return": float(target) / float(price) - 1,
                "analysts": info.get("numberOfAnalystOpinions")}
    except Exception:
        return None


# [AI-GENERATED]
@st.cache_data(ttl=900, show_spinner=False)
def get_headlines(ticker):
    """Recent headlines from yfinance. Yahoo has changed the format over time, so handle both."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    headlines = []
    for item in items:
        content = item.get("content", item)
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider")
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else content.get("publisher", "")
        headlines.append({"title": title, "publisher": publisher})
    return headlines


# [AI-GENERATED]
def score_headlines(headlines):
    """Score each headline with VADER (-1 = very negative, +1 = very positive) and average them."""
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(FINANCE_WORDS)               # add our finance words to VADER's dictionary
    rows = []
    for headline in headlines:
        score = analyzer.polarity_scores(headline["title"])["compound"]
        rows.append({"Headline": headline["title"], "Publisher": headline["publisher"], "Score": round(score, 2)})

    count = len(rows)
    average = sum(row["Score"] for row in rows) / count if count else 0.0
    if count < MIN_HEADLINES:
        flag = "Insufficient news"
    elif average >= POSITIVE_CUTOFF:
        flag = "Positive"
    elif average <= NEGATIVE_CUTOFF:
        flag = "Negative"
    else:
        flag = "Neutral"
    return {"flag": flag, "average": average, "count": count, "rows": rows,
            "strong_negative": count >= MIN_HEADLINES and average <= STRONG_NEGATIVE}


# [AI-GENERATED]
def alignment_verdict(signal, forecast_view, news_flag, strong_negative=False):
    """Compare the three layers and return (message type, explanation).
    Advisory only: this text never changes the BUY/SELL/HOLD label."""
    if signal == "BUY":
        if news_flag == "Negative":
            return "error", ("Conflict: the numbers say BUY, but recent headlines are negative. The history looks strong "
                             "while current news is a warning sign, so investigate before buying. An investor who already "
                             "holds the stock may consider selling; one who does not may hold off buying.")
        if forecast_view == "Unfavorable":
            return "warning", ("Caution: the numbers say BUY, but the forward simulation gives the stock less than a 50% "
                               "chance of beating the risk-free rate, so past alpha may not persist.")
        if forecast_view == "Favorable" and news_flag == "Positive":
            return "success", ("Strong alignment: the historical signal, the forward simulation and recent news all point "
                               "the same way (BUY, favorable forecast, positive news).")
        return "info", ("Partly confirmed: the BUY signal is not contradicted, but it is not fully backed up either "
                        f"(forecast: {forecast_view.lower()}, news: {news_flag.lower()}).")

    if signal == "SELL":
        if news_flag == "Positive" and forecast_view == "Favorable":
            return "warning", ("Conflict: the numbers say SELL, but the forecast and recent news are positive. Check "
                               "whether the weakness is temporary before acting.")
        if news_flag == "Negative" or forecast_view == "Unfavorable":
            return "error", ("Alignment: the SELL signal is backed up by the forward view or recent news "
                             f"(forecast: {forecast_view.lower()}, news: {news_flag.lower()}).")
        return "info", ("The SELL signal is not contradicted, but the forecast and news do not clearly confirm it "
                        f"(forecast: {forecast_view.lower()}, news: {news_flag.lower()}).")

    # HOLD
    if strong_negative:
        return "warning", ("Watch: there is no statistical edge (HOLD), but recent headlines are strongly negative. "
                           "Keep an eye on this stock.")
    return "info", ("No statistical edge (HOLD). "
                    f"Forecast: {forecast_view.lower()}. News: {news_flag.lower()}.")


# ---------------------------------------------------------------------------
# [TEAM-WRITTEN] space: put code your team writes yourselves below this line.
# ---------------------------------------------------------------------------
