# baby_aladdin_app.py
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import date, timedelta
import math
import requests  # New: For NewsAPI

# ---------------------------
# PAGE SETUP & THEME CSS
# ---------------------------
st.set_page_config(page_title="Black - Wealth Mastery", layout="wide")

# Apple-inspired matte black CSS (updated with button styling)
st.markdown(
    """
    <style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;700&display=swap');

    /* Layout */
    .stApp {
        background: #070707; /* Matte black background */
        color: #ffffff; /* White text */
        font-family: 'SF Pro Display', -apple-system, 'Helvetica Neue', Arial, sans-serif;
        margin: 0;
        overflow-x: hidden;
    }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* Hero Section */
    .hero {
        text-align: center;
        padding: 60px 20px;
        max-width: 1200px;
        margin: 40px auto;
        background: linear-gradient(180deg, #0b0b0b 0%, #1a1a1a 100%); /* Dark gradient */
        border-radius: 18px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        opacity: 0;
        animation: fadeIn 1s ease-out forwards;
    }
    .title {
        font-size: 56px;
        font-weight: 700;
        color: #ffffff; /* White title */
        margin-bottom: 20px;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    .subtitle {
        font-size: 24px;
        font-weight: 400;
        color: #e6eef6; /* Light white-gray */
        max-width: 600px;
        margin: 0 auto 30px;
        line-height: 1.4;
    }

    /* Apple-Style Buttons (targets Streamlit buttons) */
    button[data-testid="stButton"] {
        background: #0071e3 !important; /* Apple blue */
        color: #ffffff !important; /* White text */
        border-radius: 980px !important; /* Apple-style rounded corners */
        padding: 16px 32px !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        border: none !important;
        cursor: pointer !important;
        transition: background 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        margin: 5px !important;
        display: inline-block !important;
    }
    button[data-testid="stButton"]:hover {
        background: #005bb5 !important; /* Darker blue on hover */
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.4) !important;
    }
    button[data-testid="stButton"]:focus {
        outline: none !important;
    }

    /* Tip Card */
    .tip-card {
        background: #1a1a1a; /* Dark matte background */
        color: #e6eef6; /* Light white text */
        border-radius: 18px;
        padding: 20px;
        max-width: 1200px;
        margin: 20px auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        opacity: 0;
        animation: fadeIn 1s ease-out 0.5s forwards;
    }
    .small {
        font-size: 14px;
        color: #b0c4de; /* Subtle light gray-white */
        text-align: center;
    }

    /* Cards and Panels */
    .card {
        background: linear-gradient(180deg, #0b0b0b, #111);
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #1a1a1a;
        box-shadow: 0 6px 30px rgba(0, 0, 0, 0.7);
        color: #ffffff; /* White text for cards */
    }
    .interpret-positive {
        background: #083825;
        color: #bfffe3;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #0a5b3e;
    }
    .interpret-negative {
        background: #3a0b0b;
        color: #ffd6d6;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #601313;
    }
    .interpret-neutral {
        background: #0b133b;
        color: #dbe9ff;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #0a2b5b;
    }
    .metric { background: transparent; }
    a { color: #9bdcff; }
    .back { color: #b0c4de; font-weight: 600; text-decoration: none; }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .title {
            font-size: 36px;
        }
        .subtitle {
            font-size: 18px;
        }
        button[data-testid="stButton"] {
            padding: 12px 24px !important;
            font-size: 16px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers: financial functions, UI helpers
# ---------------------------
POS_WORDS = {"good", "gain", "gains", "beat", "beats", "upgrade", "positive", "record", "profit", "growth", "surge", "rise", "up", "outperform"}
NEG_WORDS = {"drop", "decline", "loss", "losses", "miss", "missed", "downgrade", "negative", "falls", "fall", "slump", "plunge", "criticism", "lawsuit", "warn", "underperform"}

def simple_sentiment(text: str) -> float:
    if not text:
        return 0.0
    txt = text.lower()
    score = 0
    for w in POS_WORDS:
        if w in txt:
            score += 1
    for w in NEG_WORDS:
        if w in txt:
            score -= 1
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 5.0))

def fetch_stock(ticker: str, start: date, end: date):
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end)
    info = {}
    try:
        info = t.info
    except Exception:
        info = {}
    return df, info, t

def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0).fillna(0)
    down = -1 * delta.clip(upper=0).fillna(0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def monte_carlo_sim(current_price: float, returns: pd.Series, days: int, simulations: int = 1000, shock_pct: float = 0.0, vol_multiplier: float = 1.0):
    mu = returns.mean()
    sigma = returns.std() * vol_multiplier
    sims = np.zeros((simulations, days + 1))
    sims[:, 0] = current_price * (1 + shock_pct)
    for d in range(1, days + 1):
        rand = np.random.normal(mu, sigma, simulations)
        sims[:, d] = sims[:, d - 1] * (1 + rand)
    return sims

def summarize_simulation(sims: np.ndarray):
    mean_path = sims.mean(axis=0)
    median_path = np.median(sims, axis=0)
    p10 = np.percentile(sims, 10, axis=0)
    p90 = np.percentile(sims, 90, axis=0)
    return {"mean": mean_path, "median": median_path, "p10": p10, "p90": p90}

def show_insight(text: str, level: str = "neutral"):
    if level == "positive":
        st.markdown(f"<div class='interpret-positive'>{text}</div>", unsafe_allow_html=True)
    elif level == "negative":
        st.markdown(f"<div class='interpret-negative'>{text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='interpret-neutral'>{text}</div>", unsafe_allow_html=True)

def generate_stock_insights(info, df, sims_by_horizon, shock_pct, vol_multiplier, simulations):
    """Return list of (title, text, level)"""
    insights = []
    if df.empty:
        return insights
    curr = float(df['Close'].iloc[-1])
    returns = df['Close'].pct_change().dropna()
    vol = returns.std() * math.sqrt(252) if not returns.empty else 0.0
    insights.append(("Volatility (annualized)", f"{vol:.2%}", "neutral"))

    # Momentum
    if len(df) >= 50:
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        if df['Close'].iloc[-1] > ma50:
            insights.append(("Momentum", "Short-term trend positive (price > 50-day MA).", "positive"))
        else:
            insights.append(("Momentum", "Short-term trend weak (price < 50-day MA).", "negative"))

    # Predictions
    for label, sims in sims_by_horizon.items():
        stats = summarize_simulation(sims)
        mean_target = stats['mean'][-1]
        pct = (mean_target - curr) / curr * 100
        level = "neutral"
        if pct > 12 and vol < 0.6:
            level = "positive"
        if pct < -8 and vol > 0.4:
            level = "negative"
        insights.append((f"Prediction ({label})", f"Mean {mean_target:.2f} ({pct:.2f}%), 10/90: [{stats['p10'][-1]:.2f},{stats['p90'][-1]:.2f}]", level))

    # Scenario notes
    if shock_pct < 0:
        insights.append(("Scenario Shock", f"Immediate bearish shock {shock_pct*100:.0f}% applied — forecasts shifted lower.", "negative"))
    elif shock_pct > 0:
        insights.append(("Scenario Shock", f"Bullish shock +{shock_pct*100:.0f}% applied — forecasts shifted higher.", "positive"))

    if vol_multiplier > 1.0:
        insights.append(("Volatility Multiplier", f"Volatility x{vol_multiplier:.2f} — wider forecast bands.", "neutral"))
    else:
        insights.append(("Volatility Multiplier", f"Volatility x{vol_multiplier:.2f} — narrower forecast bands.", "neutral"))

    if simulations >= 3000:
        insights.append(("Simulations", f"{simulations} runs — predictions more stable.", "positive"))
    elif simulations <= 500:
        insights.append(("Simulations", f"{simulations} runs — predictions less stable.", "negative"))

    # Heuristic recommendation
    pos = sum(1 for _, _, l in insights if l == "positive")
    neg = sum(1 for _, _, l in insights if l == "negative")
    rec = "HOLD"
    if pos > neg + 1:
        rec = "BUY"
    elif neg > pos + 1:
        rec = "SELL"
    insights.append(("Recommendation", f"Heuristic recommendation: {rec}. Combine with your research.", "neutral"))
    return insights

# New Helper: Fetch news from NewsAPI
def fetch_news(query: str, api_key: str = "decaf9729323488b87a9c8bab3ae7250"):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": api_key
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"NewsAPI error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"News fetch failed: {e}")
        return []

# ---------------------------
# NAVIGATION (Home / Stock / Portfolio)
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def go_stock():
    st.session_state.page = "stock"

def go_portfolio():
    st.session_state.page = "portfolio"

# ---------------------------
# COMMON SIDEBAR CONTROLS
# ---------------------------
# Date range
st.sidebar.header("Date range & scenario")
default_end = date.today()
default_start = default_end - timedelta(days=365)
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", default_end)
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date")

# Stress testing
st.sidebar.header("Scenario & Stress Testing")
shock_pct = st.sidebar.slider("Immediate market shock (%)", -50, 50, 0, 1) / 100.0
vol_multiplier = st.sidebar.slider("Volatility multiplier", 0.2, 3.0, 1.0, 0.1)
simulations = int(st.sidebar.number_input("Simulation runs", min_value=200, max_value=5000, value=1000, step=100))

# ---------------------------
# HOME PAGE
# ---------------------------
if st.session_state.page == "home":
    # Hero Title and Subtitle
    st.markdown(
        """
        <div class='hero'>
            <div class='title'>⚫Black</div>
            <div class='subtitle'>AI-Powered Wealth Mastery<br>Unleash Insights with Unmatched Precision</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Buttons using native st.button (styled via CSS)
    col1, col2, col3 = st.columns([1, 0.5, 1])  # Balanced columns for buttons
    with col1:
        if st.button("📈 Stock Analysis", key="home_stock", help="Deep-dive into single stock (predictions, charts, news)"):
            go_stock()
            st.rerun()
    with col2:
        st.empty()  # Spacer column
    with col3:
        if st.button("📊 Portfolio Analysis", key="home_portfolio", help="Analyze up to 5 stocks together (corr, frontier, portfolio forecasts)"):
            go_portfolio()
            st.rerun()

    # Tip Card
    st.markdown(
        """
        <div class='tip-card'>
            <div class='small'>Adjust scenarios or timelines in the left panel — insights evolve seamlessly. No signup required.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# STOCK PAGE
# ---------------------------
elif st.session_state.page == "stock":
    # top-left back
    cols = st.columns([0.5, 1, 4])
    with cols[0]:
        if st.button("⬅", key="back_stock"):
            go_home()
            st.rerun()
    with cols[1]:
        st.write("")  # spacer
    with cols[2]:
        st.markdown("<div style='text-align:right' class='small'>Matte black • Live insights</div>", unsafe_allow_html=True)

    st.markdown("<h2 style='margin-top:6px'>📈 Stock Analysis</h2>", unsafe_allow_html=True)
    ticker = st.text_input("Ticker (e.g. AAPL, TCS.NS)", value="AAPL").upper()

    # fetch data
    df, info, ticker_obj = fetch_stock(ticker, start_date, end_date)
    if df.empty:
        st.warning("No price data for this ticker & date range. Check ticker or dates.")
    else:
        # Display Company Full Name
        company_name = info.get("longName") or info.get("shortName") or "Unknown Company"
        st.markdown(f"<div class='subtitle'>Company: {company_name}</div>", unsafe_allow_html=True)

        # Top metrics
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;gap:18px'>", unsafe_allow_html=True)
        try:
            current_price = float(info.get("currentPrice", df['Close'].iloc[-1]))
        except Exception:
            current_price = float(df['Close'].iloc[-1])
        prev_close = info.get("previousClose", None)
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Current Price", f"${current_price:.2f}")
        h2.metric("Previous Close", f"${prev_close:.2f}" if prev_close else "N/A")
        h3.metric("52W High", f"${info.get('fiftyTwoWeekHigh','N/A')}")
        h4.metric("52W Low", f"${info.get('fiftyTwoWeekLow','N/A')}")
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Price Chart (candlestick + MAs)
        st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
        st.markdown("<h3>Interactive Price Chart</h3>", unsafe_allow_html=True)
        ma_short = st.selectbox("Short MA window", [5, 10, 20, 50], index=2, key="ma_short")
        ma_long = st.selectbox("Long MA window", [50, 100, 200], index=0, key="ma_long")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        short_ma_series = df['Close'].rolling(ma_short).mean()
        long_ma_series = df['Close'].rolling(ma_long).mean()
        fig.add_trace(go.Scatter(x=df.index, y=short_ma_series, name=f"{ma_short} MA"))
        fig.add_trace(go.Scatter(x=df.index, y=long_ma_series, name=f"{ma_long} MA"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#070707", plot_bgcolor="#070707", height=520)
        st.plotly_chart(fig, use_container_width=True)
        # Dynamic Interpretation for Selected MA cross
        if len(df) >= ma_long:
            short_ma_val = short_ma_series.iloc[-1]
            long_ma_val = long_ma_series.iloc[-1]
            if short_ma_val > long_ma_val:
                show_insight(f"{ma_short}-day MA ({short_ma_val:.2f}) is above {ma_long}-day MA ({long_ma_val:.2f}) — bullish crossover signal.", "positive")
            else:
                show_insight(f"{ma_short}-day MA ({short_ma_val:.2f}) is below {ma_long}-day MA ({long_ma_val:.2f}) — bearish signal, monitor closely.", "negative")
        else:
            show_insight(f"Not enough history for {ma_short}/{ma_long} MA interpretation.", "neutral")
        st.markdown("</div>", unsafe_allow_html=True)

        # RSI
        st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
        st.markdown("<h3>RSI (14)</h3>", unsafe_allow_html=True)
        df['RSI'] = rsi(df['Close'], 14)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig_rsi.update_layout(template="plotly_dark", paper_bgcolor="#070707", plot_bgcolor="#070707", height=260, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)
        latest_rsi = df['RSI'].iloc[-1]
        if latest_rsi < 30:
            show_insight(f"RSI = {latest_rsi:.2f}. Stock appears OVERSOLD — potential rebound.", "positive")
        elif latest_rsi > 70:
            show_insight(f"RSI = {latest_rsi:.2f}. Stock appears OVERBOUGHT — risk of correction.", "negative")
        else:
            show_insight(f"RSI = {latest_rsi:.2f}. Momentum is neutral.", "neutral")
        st.markdown("</div>", unsafe_allow_html=True)

        # New: Volume Chart (additional chart)
        st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
        st.markdown("<h3>Trading Volume with MA</h3>", unsafe_allow_html=True)
        vol_ma = st.selectbox("Volume MA window", [5, 10, 20], index=1, key="vol_ma")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
        fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volume'].rolling(vol_ma).mean(), name=f"{vol_ma} MA", line=dict(color='orange')))
        fig_vol.update_layout(template="plotly_dark", paper_bgcolor="#070707", plot_bgcolor="#070707", height=300)
        st.plotly_chart(fig_vol, use_container_width=True)
        latest_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].rolling(vol_ma).mean().iloc[-1]
        if latest_vol > avg_vol * 1.5:
            show_insight(f"Recent volume spike ({latest_vol:,.0f} vs {vol_ma}-day avg {avg_vol:,.0f}) — high interest or volatility.", "positive" if latest_rsi > 50 else "negative")
        else:
            show_insight(f"Volume stable at {latest_vol:,.0f}.", "neutral")
        st.markdown("</div>", unsafe_allow_html=True)

        # Multi-horizon Monte Carlo
        st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
        st.markdown("<h3>Multi-horizon Predictions (Monte Carlo)</h3>", unsafe_allow_html=True)
        horizons = {"30D": 30, "150D": 150, "6M (180D)": 180, "12M (365D)": 365}
        sims_by_horizon = {}
        for label, days in horizons.items():
            sims = monte_carlo_sim(current_price, df['Close'].pct_change().dropna(), days,
                                   simulations=simulations, shock_pct=shock_pct, vol_multiplier=vol_multiplier)
            sims_by_horizon[label] = sims

        # Summary table
        rows = []
        for label, sims in sims_by_horizon.items():
            ssum = summarize_simulation(sims)
            mean_target = ssum['mean'][-1]
            p10 = ssum['p10'][-1]
            p90 = ssum['p90'][-1]
            pct_mean = (mean_target - current_price) / current_price * 100
            rows.append({"Horizon": label, "Mean Target": f"{mean_target:.2f}", "P10": f"{p10:.2f}", "P90": f"{p90:.2f}", "Mean %": f"{pct_mean:.2f}%"})
        st.table(pd.DataFrame(rows).set_index("Horizon"))

        # Interpret predictions
        insights = generate_stock_insights(info, df, sims_by_horizon, shock_pct, vol_multiplier, simulations)
        st.subheader("Interpretation & Recommendation")
        for title, text, level in insights:
            show_insight(f"**{title}:** {text}", level)

        # Plot selected horizon
        sel_h = st.selectbox("Show forecast horizon on chart", list(horizons.keys()), index=1, key="sel_h")
        sims = sims_by_horizon[sel_h]
        date_index = pd.date_range(end_date, periods=sims.shape[1])
        figf = go.Figure()
        # sample paths
        for i in range(min(200, sims.shape[0])):
            figf.add_trace(go.Scatter(x=date_index, y=sims[i], mode='lines', opacity=0.03, showlegend=False))
        ssum = summarize_simulation(sims)
        figf.add_trace(go.Scatter(x=date_index, y=ssum['mean'], mode='lines', name='Mean Forecast', line=dict(width=2)))
        figf.add_trace(go.Scatter(x=date_index, y=ssum['p90'], mode='lines', name='P90', line=dict(width=1), opacity=0.6))
        figf.add_trace(go.Scatter(x=date_index, y=ssum['p10'], mode='lines', name='P10', line=dict(width=1), opacity=0.6))
        figf.update_layout(template="plotly_dark", height=480, paper_bgcolor="#070707", plot_bgcolor="#070707")
        st.plotly_chart(figf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # News & Sentiment (using NewsAPI)
        st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
        st.markdown("<h3>News & Sentiment</h3>", unsafe_allow_html=True)
        news_items = fetch_news(query=company_name or ticker)
        if not news_items:
            st.info("No news articles found. Try a different ticker or check API status.")
        else:
            scores = []
            for item in news_items:
                title = item.get('title', '')
                description = item.get('description', '')
                url = item.get('url', '')
                text = f"{title} {description}"
                score = simple_sentiment(text)
                scores.append(score)
                lvl = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
                show_insight(f"**{title}** — {description}  \n<a href='{url}'>Read more</a>", lvl)
            agg_score = np.mean(scores) if scores else 0
            st.write(f"Aggregate sentiment score: {agg_score:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Backtest playback
        st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
        st.markdown("<h3>Backtest / Scenario Playback</h3>", unsafe_allow_html=True)
        bt_start = st.date_input("Backtest start", value=start_date, key="bt_start")
        bt_end = st.date_input("Backtest end", value=end_date if end_date <= date.today() else date.today(), key="bt_end")
        if bt_start < bt_end:
            df_bt, _, _ = fetch_stock(ticker, bt_start, bt_end)
            if not df_bt.empty:
                st.write(df_bt['Close'].pct_change().describe().to_frame().T)
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], mode='lines', name='Close'))
                fig_bt.update_layout(template="plotly_dark", height=320, paper_bgcolor="#070707", plot_bgcolor="#070707")
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.warning("No historical data for the backtest window.")
        else:
            st.info("Pick a valid backtest window.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Company summary
        st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
        st.markdown("<h3>Company Summary</h3>", unsafe_allow_html=True)
        if info:
            st.write(info.get("longBusinessSummary", "No summary available."))
        else:
            st.write("No company info available via yfinance.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# PORTFOLIO PAGE (up to 5)
# ---------------------------
else:
    # st.session_state.page == "portfolio"
    cols = st.columns([0.5, 1, 4])
    with cols[0]:
        if st.button("⬅", key="back_port"):
            go_home()
            st.rerun()
    with cols[2]:
        st.markdown("<div style='text-align:right' class='small'>Portfolio Mode • Up to 5 tickers</div>", unsafe_allow_html=True)

    st.markdown("<h2>📊 Portfolio Analysis</h2>", unsafe_allow_html=True)
    tickers_raw = st.text_input("Enter up to 5 tickers (comma separated)", value="AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip() != ""]
    if len(tickers) == 0:
        st.warning("Add at least one ticker.")
    else:
        if len(tickers) > 5:
            st.warning("Trim to first 5 tickers for this demo.")
            tickers = tickers[:5]

        data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
        if data.empty:
            st.warning("No data for selected tickers/dates.")
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Price table (recent)</h3>", unsafe_allow_html=True)
            recent_data = data.tail(5)  # Last 5 days
            st.dataframe(recent_data)
            # New: Interpretations for recent price table
            recent_returns = recent_data.pct_change().mean() * 100  # Avg daily % change over window
            best_recent = recent_data.iloc[-1].idxmax()
            worst_recent = recent_data.iloc[-1].idxmin()
            show_insight(f"Recent 5-day avg return: {recent_returns.mean():.2f}%. Best: {best_recent} at {recent_data[best_recent].iloc[-1]:.2f}.", "positive" if recent_returns.mean() > 0 else "negative")
            show_insight(f"Worst recent performer: {worst_recent} — monitor for recovery or exit.", "negative" if recent_data.pct_change().mean()[worst_recent] < 0 else "neutral")
            st.markdown("</div>", unsafe_allow_html=True)

            # Comparative chart
            st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
            st.markdown("<h3>Comparative Price Chart</h3>", unsafe_allow_html=True)
            figp = go.Figure()
            for t in tickers:
                figp.add_trace(go.Scatter(x=data.index, y=data[t], mode='lines', name=t))
            figp.update_layout(template="plotly_dark", height=420, paper_bgcolor="#070707", plot_bgcolor="#070707")
            st.plotly_chart(figp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Normalized performance + interpretation
            st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
            st.markdown("<h3>Normalized Performance (base = first date)</h3>", unsafe_allow_html=True)
            norm = data / data.iloc[0] * 100
            fign = go.Figure()
            for t in tickers:
                fign.add_trace(go.Scatter(x=norm.index, y=norm[t], mode='lines', name=t))
            fign.update_layout(template="plotly_dark", height=300, paper_bgcolor="#070707", plot_bgcolor="#070707")
            st.plotly_chart(fign, use_container_width=True)

            last_perf = norm.iloc[-1]
            best = last_perf.idxmax()
            worst = last_perf.idxmin()
            best_pct = last_perf.max() - 100
            worst_pct = last_perf.min() - 100
            show_insight(f"Best performer: {best} ({best_pct:.2f}%) • Worst: {worst} ({worst_pct:.2f}%).", "neutral")
            if best_pct > 10:
                show_insight(f"{best} is up {best_pct:.2f}% since start — strong relative performance.", "positive")
            if worst_pct < -10:
                show_insight(f"{worst} is down {worst_pct:.2f}% — consider rebalancing.", "negative")
            st.markdown("</div>", unsafe_allow_html=True)

            # New: Cumulative Returns Chart (additional chart)
            st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
            st.markdown("<h3>Cumulative Returns</h3>", unsafe_allow_html=True)
            cum_returns = (data.pct_change().fillna(0) + 1).cumprod() * 100
            fig_cum = go.Figure()
            for t in tickers:
                fig_cum.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[t], mode='lines', name=t))
            fig_cum.update_layout(template="plotly_dark", height=300, paper_bgcolor="#070707", plot_bgcolor="#070707")
            st.plotly_chart(fig_cum, use_container_width=True)
            total_cum = cum_returns.iloc[-1].mean() - 100
            show_insight(f"Portfolio cumulative return: {total_cum:.2f}% over period.", "positive" if total_cum > 0 else "negative")
            st.markdown("</div>", unsafe_allow_html=True)

            # Correlation heatmap and interpretation
            st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
            st.markdown("<h3>Correlation Heatmap</h3>", unsafe_allow_html=True)
            corr = data.pct_change().corr()
            figc = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis'))
            figc.update_layout(template="plotly_dark", height=380, paper_bgcolor="#070707", plot_bgcolor="#070707")
            st.plotly_chart(figc, use_container_width=True)
            avg_corr = corr.mean().mean()
            if avg_corr < 0.4:
                show_insight(f"Average pairwise correlation: {avg_corr:.2f}. Portfolio is well diversified.", "positive")
            elif avg_corr < 0.7:
                show_insight(f"Average pairwise correlation: {avg_corr:.2f}. Moderately correlated — ok but check edge cases.", "neutral")
            else:
                show_insight(f"Average pairwise correlation: {avg_corr:.2f}. Highly correlated — diversify across sectors/asset classes.", "negative")
            st.markdown("</div>", unsafe_allow_html=True)

            # Efficient frontier (simulated)
            st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
            st.markdown("<h3>Efficient Frontier (simulated portfolios)</h3>", unsafe_allow_html=True)
            returns = data.pct_change().dropna()
            n_port = 2000
            rets = []
            vols = []
            for _ in range(n_port):
                w = np.random.random(len(tickers))
                w /= w.sum()
                port_ret = np.sum(returns.mean() * w) * 252
                port_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
                rets.append(port_ret)
                vols.append(port_vol)
            ef_fig = go.Figure(data=go.Scatter(x=vols, y=rets, mode='markers', marker=dict(color=rets, colorscale='Viridis', showscale=True)))
            ef_fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#070707", plot_bgcolor="#070707", xaxis_title="Volatility", yaxis_title="Expected Return")
            st.plotly_chart(ef_fig, use_container_width=True)
            show_insight("Efficient frontier shows risk-return combinations from randomly generated portfolios.", "neutral")
            st.markdown("</div>", unsafe_allow_html=True)

            # Portfolio Monte Carlo aggregated forecasts
            st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
            st.markdown("<h3>Portfolio Forecast (aggregate Monte Carlo)</h3>", unsafe_allow_html=True)
            port_horizons = {"30D": 30, "150D": 150, "6M (180D)": 180, "12M (365D)": 365}
            equal_w = st.checkbox("Assume equal weights", value=True, key="equal_w")
            if equal_w:
                weights = np.array([1 / len(tickers)] * len(tickers))
            else:
                w_str = st.text_input("Comma weights summing to 1", value=",".join([str(round(1/len(tickers), 2)) for _ in tickers]))
                try:
                    weights = np.array([float(x.strip()) for x in w_str.split(",")])
                    if len(weights) != len(tickers) or abs(weights.sum() - 1.0) > 1e-6:
                        show_insight("Invalid weights — falling back to equal weights.", "negative")
                        weights = np.array([1 / len(tickers)] * len(tickers))
                except Exception:
                    show_insight("Weights parse error — falling back to equal weights.", "negative")
                    weights = np.array([1 / len(tickers)] * len(tickers))

            portfolio_summaries = []
            sims_port_by_horizon = {}
            for label, days in port_horizons.items():
                comps = []
                for t in tickers:
                    curr_price = data[t].iloc[-1]
                    sims_t = monte_carlo_sim(curr_price, data[t].pct_change().dropna(), days, simulations=max(200, int(simulations/2)), shock_pct=shock_pct, vol_multiplier=vol_multiplier)
                    comps.append(sims_t)
                comps = np.array(comps)  # (n_tickers, sim, days+1)
                if comps.ndim != 3:
                    st.error("Simulation error for portfolio.")
                    break
                sims_weighted = np.tensordot(weights, comps, axes=(0, 0))  # (sim, days+1)
                sims_port_by_horizon[label] = sims_weighted
                ssum = summarize_simulation(sims_weighted)
                portfolio_summaries.append({"Horizon": label, "MeanTarget": f"{ssum['mean'][-1]:.2f}", "P10": f"{ssum['p10'][-1]:.2f}", "P90": f"{ssum['p90'][-1]:.2f}"})
            st.table(pd.DataFrame(portfolio_summaries).set_index("Horizon"))
            show_insight("Portfolio forecasts shown above. Use scenario sliders for shock/volatility adjustments.", "neutral")
            st.markdown("</div>", unsafe_allow_html=True)

            # Portfolio suggestions
            st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
            st.markdown("<h3>Portfolio Suggestions (heuristic)</h3>", unsafe_allow_html=True)
            if avg_corr > 0.6:
                show_insight("Portfolio is fairly correlated — consider adding non-correlated assets or sectors.", "negative")
            else:
                show_insight("Portfolio diversification looks decent based on selected tickers.", "positive")
            st.markdown("</div>", unsafe_allow_html=True)

            # Backtest presets
            st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
            st.markdown("<h3>Backtesting: Preset events</h3>", unsafe_allow_html=True)
            bt_choice = st.selectbox("Preset", ["Custom Range", "COVID Dip (Feb 2020 - Jun 2020)", "2008 Financial Crisis (Sep 2008 - Mar 2009)"])
            if bt_choice == "Custom Range":
                bt_s = st.date_input("Start (backtest)", value=start_date, key="pb_s")
                bt_e = st.date_input("End (backtest)", value=end_date, key="pb_e")
            elif bt_choice.startswith("COVID"):
                bt_s, bt_e = date(2020,2,1), date(2020,6,30)
                st.write(f"Simulating: {bt_s} to {bt_e}")
            else:
                bt_s, bt_e = date(2008,9,1), date(2009,3,31)
                st.write(f"Simulating: {bt_s} to {bt_e}")

            if bt_s < bt_e:
                data_bt = yf.download(tickers, start=bt_s, end=bt_e)['Close'].dropna()
                if not data_bt.empty:
                    fig_bt = go.Figure()
                    for t in tickers:
                        fig_bt.add_trace(go.Scatter(x=data_bt.index, y=data_bt[t] / data_bt[t].iloc[0] * 100, mode='lines', name=t))
                    fig_bt.update_layout(template="plotly_dark", height=380, paper_bgcolor="#070707", plot_bgcolor="#070707")
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    # New: Interpretation for backtest chart
                    bt_returns = (data_bt.iloc[-1] / data_bt.iloc[0] - 1) * 100
                    port_return = bt_returns.mean()
                    if bt_choice.startswith("COVID"):
                        if port_return > 0:
                            show_insight(f"Portfolio recovered {port_return:.2f}% during COVID dip — resilient performance.", "positive")
                        else:
                            show_insight(f"Portfolio down {port_return:.2f}% in COVID period — high impact from market crash.", "negative")
                    elif bt_choice.startswith("2008"):
                        if port_return > -20:
                            show_insight(f"Portfolio lost {port_return:.2f}% in 2008 crisis — better than market averages, defensive.", "neutral")
                        else:
                            show_insight(f"Severe drawdown {port_return:.2f}% in 2008 — vulnerability exposed.", "negative")
                    else:
                        show_insight(f"Custom period return: {port_return:.2f}% — assess based on benchmarks.", "neutral")
                else:
                    st.warning("No data for preset backtest window.")
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='color:#b0c4de;padding:8px'>Built with ❤️ — Black. Replace heuristics with ML & robust NLP when ready.</div>", unsafe_allow_html=True)