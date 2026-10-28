import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="NSE Terminal", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp {background:#0b0b0b;color:#f5f5f5}
.block-container {padding-top:1.4rem;max-width:1400px}
[data-testid="stMetric"] {background:#171717;border:1px solid #303030;padding:14px;border-radius:12px}
div.stButton>button {border-radius:9px;font-weight:600}
</style>
""", unsafe_allow_html=True)

st.title("📈 NSE Terminal")
st.caption("Stock analysis • technical indicators • future growth and downside-risk scenarios")
st.info("Educational tool only. Forecasts are estimates based on historical data, not guaranteed predictions or financial advice.")

def ticker_for(symbol):
    symbol = symbol.strip().upper()
    if symbol.startswith("^") or symbol.endswith((".NS", ".BO")):
        return symbol
    return symbol + ".NS"

@st.cache_data(ttl=600, show_spinner=False)
def load_history(symbol, period):
    return yf.Ticker(ticker_for(symbol)).history(period=period, auto_adjust=True)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def macd(series):
    line = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line-signal

def adx(data, period=14):
    high, low, close = data["High"], data["Low"], data["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def safe_num(value, digits=2):
    try:
        if pd.isna(value) or not np.isfinite(float(value)):
            return None
        return round(float(value), digits)
    except Exception:
        return None

with st.sidebar:
    st.header("Stock selection")
    symbol = st.text_input("NSE symbol", "RELIANCE").strip().upper()
    period = st.selectbox("Historical data", ["1y", "2y", "5y", "10y"], index=1)
    st.caption("Examples: RELIANCE, TCS, INFY, HDFCBANK, SBIN")
    st.divider()
    st.subheader("Forecast settings")
    forecast_days = st.select_slider("Forecast horizon (trading days)", options=[7, 30, 90, 252], value=30)
    simulations = st.select_slider("Simulation paths", options=[500, 1000, 2000], value=1000)

if not symbol:
    st.warning("Enter an NSE symbol in the sidebar.")
    st.stop()

with st.spinner(f"Loading {symbol} market data..."):
    try:
        hist = load_history(symbol, period)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

if hist is None or hist.empty or "Close" not in hist:
    st.error("No price data found. Check the NSE symbol and try again.")
    st.stop()

hist = hist.dropna(subset=["Close"]).copy()
close = hist["Close"]
if len(close) < 60:
    st.error("Not enough historical data for a useful analysis.")
    st.stop()

price = float(close.iloc[-1])
daily_returns = close.pct_change().dropna()
rsi_series = rsi(close)
macd_line, macd_signal, macd_hist = macd(close)
hist["RSI"] = rsi_series
hist["EMA 20"] = close.ewm(span=20, adjust=False).mean()
hist["EMA 50"] = close.ewm(span=50, adjust=False).mean()
hist["SMA 50"] = close.rolling(50).mean()
hist["SMA 200"] = close.rolling(200).mean()
hist["MACD"] = macd_line
hist["MACD Signal"] = macd_signal
hist["ADX"] = adx(hist)
middle = close.rolling(20).mean()
std = close.rolling(20).std()
hist["BB Upper"] = middle + 2*std
hist["BB Lower"] = middle - 2*std

tab_analysis, tab_forecast, tab_risk, tab_news = st.tabs([
    "📊 Analysis", "🔮 Growth & Loss Forecast", "⚖️ Risk", "📰 News"
])

with tab_analysis:
    st.subheader(f"{symbol} — Stock Analysis")
    cols = st.columns(4)
    cols[0].metric("Latest available price", f"₹{price:,.2f}")
    cols[1].metric("RSI (14)", f"{rsi_series.iloc[-1]:.2f}" if pd.notna(rsi_series.iloc[-1]) else "N/A")
    cols[2].metric("MACD histogram", f"{macd_hist.iloc[-1]:.3f}" if pd.notna(macd_hist.iloc[-1]) else "N/A")
    cols[3].metric("ADX (14)", f"{hist['ADX'].iloc[-1]:.2f}" if pd.notna(hist["ADX"].iloc[-1]) else "N/A")

    last = hist.iloc[-1]
    bullish_votes = sum([
        pd.notna(last["SMA 50"]) and pd.notna(last["SMA 200"]) and last["SMA 50"] > last["SMA 200"],
        pd.notna(macd_hist.iloc[-1]) and macd_hist.iloc[-1] > 0,
        pd.notna(rsi_series.iloc[-1]) and 50 < rsi_series.iloc[-1] < 70,
        pd.notna(last["EMA 20"]) and price > last["EMA 20"],
    ])
    if bullish_votes >= 3:
        signal, color = "Bullish indicators", "🟢"
    elif bullish_votes <= 1:
        signal, color = "Bearish indicators", "🔴"
    else:
        signal, color = "Mixed / neutral indicators", "🟡"
    st.markdown(f"### {color} Current technical picture: **{signal}**")
    st.caption("This is an indicator summary, not a standalone buy/sell recommendation.")

    fig = go.Figure()
    view = hist.tail(252)
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], name="Price"))
    for col, name in [("EMA 20", "EMA 20"), ("SMA 50", "SMA 50"), ("SMA 200", "SMA 200"), ("BB Upper", "Bollinger upper"), ("BB Lower", "Bollinger lower")]:
        if view[col].notna().any():
            fig.add_trace(go.Scatter(x=view.index, y=view[col], mode="lines", name=name))
    fig.update_layout(template="plotly_dark", height=560, xaxis_rangeslider_visible=False, title=f"{symbol} — last 12 months")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Indicator readings")
        indicator_df = pd.DataFrame({
            "Indicator": ["RSI (14)", "MACD", "MACD Signal", "EMA 20", "SMA 50", "SMA 200", "ADX (14)", "Bollinger upper", "Bollinger lower"],
            "Latest value": [safe_num(rsi_series.iloc[-1]), safe_num(macd_line.iloc[-1]), safe_num(macd_signal.iloc[-1]), safe_num(last["EMA 20"]), safe_num(last["SMA 50"]), safe_num(last["SMA 200"]), safe_num(last["ADX"]), safe_num(last["BB Upper"]), safe_num(last["BB Lower"])]
        })
        st.dataframe(indicator_df, hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Historical price data")
        st.dataframe(hist.tail(10).reset_index(), hide_index=True, use_container_width=True)

with tab_forecast:
    st.subheader("Potential future growth and loss")
    st.write(f"Forecast horizon: **{forecast_days} trading days** (approximately {round(forecast_days/21)} months).")
    log_returns = np.log(close / close.shift(1)).dropna()
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    rng = np.random.default_rng(42)
    # GBM-style scenario simulation; not a trained or validated predictive model.
    shocks = rng.normal(mu - 0.5*sigma**2, sigma, size=(simulations, forecast_days))
    paths = price * np.exp(np.cumsum(shocks, axis=1))
    final_prices = paths[:, -1]
    prob_up = float((final_prices > price).mean()*100)
    prob_down = float((final_prices < price).mean()*100)
    q05, q25, q50, q75, q95 = np.percentile(final_prices, [5, 25, 50, 75, 95])
    upside = (q95/price - 1)*100
    downside = (q05/price - 1)*100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current price", f"₹{price:,.2f}")
    m2.metric("Simulated probability above current", f"{prob_up:.1f}%")
    m3.metric("Simulated probability below current", f"{prob_down:.1f}%")
    m4.metric("Median scenario", f"₹{q50:,.2f}")

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Pessimistic scenario (5th percentile)", f"₹{q05:,.2f}", f"{(q05/price-1)*100:+.1f}%")
    sc2.metric("Middle scenario (50th percentile)", f"₹{q50:,.2f}", f"{(q50/price-1)*100:+.1f}%")
    sc3.metric("Optimistic scenario (95th percentile)", f"₹{q95:,.2f}", f"{(q95/price-1)*100:+.1f}%")

    st.markdown("#### Simulated price range over time")
    day_idx = np.arange(1, forecast_days+1)
    q05_path, q50_path, q95_path = np.percentile(paths, [5, 50, 95], axis=0)
    chart_df = pd.DataFrame({"Trading day": day_idx, "5th percentile": q05_path, "Median": q50_path, "95th percentile": q95_path})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["Trading day"], y=chart_df["95th percentile"], mode="lines", name="Optimistic range (95th percentile)"))
    fig.add_trace(go.Scatter(x=chart_df["Trading day"], y=chart_df["Median"], mode="lines", name="Median scenario"))
    fig.add_trace(go.Scatter(x=chart_df["Trading day"], y=chart_df["5th percentile"], mode="lines", name="Pessimistic range (5th percentile)", fill="tonexty"))
    fig.update_layout(template="plotly_dark", height=450, xaxis_title="Trading days ahead", yaxis_title="Simulated price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### What the scenarios mean")
    st.write(f"- **Potential upside scenario:** {upside:+.1f}% relative to today's latest available price.")
    st.write(f"- **Potential downside scenario:** {downside:+.1f}% relative to today's latest available price.")
    st.write("- These percentages describe simulated outcomes, not guaranteed limits on gains or losses.")
    st.warning("This is a historical-return Monte Carlo simulation, not a trained AI model. It assumes recent return behaviour continues and may miss market shocks, news, results, and policy changes. The simulated probabilities are not calibrated real-world probabilities.")

with tab_risk:
    st.subheader("Historical risk measures")
    ann_vol = float(daily_returns.std()*np.sqrt(252)*100)
    years = len(daily_returns)/252
    ann_return = ((price/float(close.iloc[0]))**(1/years)-1)*100 if years > 0 and close.iloc[0] > 0 else np.nan
    drawdown = close/close.cummax()-1
    max_dd = float(drawdown.min()*100)
    daily_var = float(np.percentile(daily_returns, 5)*100)
    downside_returns = daily_returns[daily_returns < 0]
    downside_dev = float(downside_returns.std()*np.sqrt(252)*100) if len(downside_returns) > 1 else np.nan
    risk_free = 6.5
    sharpe = (ann_return-risk_free)/ann_vol if ann_vol > 0 else np.nan
    risk_level = "Low" if ann_vol < 20 else "Moderate" if ann_vol < 35 else "High"
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Risk level (historical volatility)", risk_level)
    r2.metric("Annualized volatility", f"{ann_vol:.2f}%")
    r3.metric("Annualized historical return", f"{ann_return:.2f}%")
    r4.metric("Maximum drawdown", f"{max_dd:.2f}%")
    r5, r6, r7 = st.columns(3)
    r5.metric("Sharpe ratio (approx.)", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
    r6.metric("Daily VaR (95%)", f"{daily_var:.2f}%")
    r7.metric("Annualized downside deviation", f"{downside_dev:.2f}%" if np.isfinite(downside_dev) else "N/A")
    dd_df = pd.DataFrame({"Date": drawdown.index, "Drawdown (%)": drawdown.values*100})
    fig = px.area(dd_df.tail(252), x="Date", y="Drawdown (%)", title="Drawdown over the last 12 months", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Risk metrics are calculated from available historical prices; past performance does not predict future returns.")

with tab_news:
    st.subheader(f"Latest available news: {symbol}")
    if st.button("Load stock news"):
        try:
            news_items = yf.Ticker(ticker_for(symbol)).news or []
            if not news_items:
                st.info("No news items were returned by the data provider.")
            for item in news_items[:15]:
                content = item.get("content", item)
                title = content.get("title", "Untitled")
                publisher = (content.get("provider") or {}).get("displayName") or content.get("publisher") or "Unknown publisher"
                url = (content.get("canonicalUrl") or {}).get("url") or content.get("link")
                published = content.get("pubDate") or ""
                with st.expander(title):
                    st.write(f"Publisher: {publisher}")
                    if published:
                        st.write(f"Published: {published}")
                    summary = content.get("summary")
                    if summary:
                        st.write(summary)
                    if url:
                        st.link_button("Open article", url)
        except Exception as exc:
            st.error(f"Could not load news: {exc}")

st.divider()
st.caption(f"Last price observation from provider: {hist.index[-1].date()} • Data source: Yahoo Finance via yfinance • Not financial advice.")
