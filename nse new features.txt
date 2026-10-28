"""
NSE Stock Analysis API v2 — FastAPI + yfinance (100% free, no new installs)
================================================================================
Original endpoints (unchanged):  /  /analyze/{symbol}  /chart/{symbol}  /watchlist
NEW endpoints:
  1. /compare          – multi-stock rebased performance + correlation matrix
  2. /risk/{symbol}    – volatility, Sharpe, Sortino, beta, max drawdown, VaR
  3. /signal/{symbol}  – rule-based composite Buy/Hold/Sell score
  4. /backtest/{symbol}– MA-cross / RSI / MACD strategy vs buy & hold
  5. /montecarlo/{symbol} – 1-year price-range simulation (GBM)
  6. /market           – NIFTY, SENSEX, VIX, USD/INR + sector performance
  7. /news/{symbol}    – latest headlines with keyword sentiment

REQUIREMENTS (same as before):  pip install fastapi uvicorn yfinance pandas numpy
RUN LOCALLY:  uvicorn nse_analysis_api:app --reload --port 8000
Interactive docs (free, auto-generated):  http://127.0.0.1:8000/docs

For educational use only — not investment advice.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime

app = FastAPI(title="NSE Stock Analysis API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RISK_FREE_RATE = 0.065      # approx. Indian 10-yr G-Sec yield; change if you like
BENCHMARK = "^NSEI"         # NIFTY 50
DISCLAIMER = "Educational use only. Not investment advice."

# ── SMALL IN-MEMORY CACHE (10 min) — keeps Yahoo from rate-limiting the free tier ──
_CACHE = {}
CACHE_TTL = 600


def get_history(ticker, period="2y"):
    key = (ticker, period)
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < CACHE_TTL:
        return _CACHE[key][1].copy()
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if not hist.empty:
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        hist.index = hist.index.normalize()
        _CACHE[key] = (now, hist)
    return hist.copy()


def norm_ticker(symbol):
    s = symbol.upper().strip()
    if s.startswith("^") or s.endswith((".NS", ".BO", "=X")):
        return s
    return f"{s}.NS"


# ── ORIGINAL INDICATOR FUNCTIONS (unchanged) ─────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth (fixes the fixed-position matching bug)."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def safe_round(val, decimals=2):
    if val is None:
        return None
    try:
        if pd.isna(val) or np.isinf(val):
            return None
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Fundamentals from income statement + balance sheet (avoids blocked stock.info)."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass

    return fundamentals


def get_technical_snapshot(hist):
    """Core technical calculation shared by /analyze, /watchlist and /signal."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


# ── ORIGINAL ENDPOINTS (unchanged) ───────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API v2 is running"}


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "fundamentals": fundamentals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """OHLC + RSI, MACD, MA50, MA200 per trading day."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    months = min(max(months, 1), 24)

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        trading_days = months * 21
        recent = hist.tail(trading_days)

        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]) if date in rsi.index else None,
                "macd_line": safe_round(macd_line.loc[date]) if date in macd_line.index else None,
                "macd_signal": safe_round(signal_line.loc[date]) if date in signal_line.index else None,
                "ma_50": safe_round(ma50.loc[date]) if date in ma50.index else None,
                "ma_200": safe_round(ma200.loc[date]) if date in ma200.index else None,
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """Multi-stock snapshot: /watchlist?symbols=RELIANCE,TCS,INFY (max 10)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", auto_adjust=True)
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue

            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


# ═════════════════════════════════════════════════════════════════════════════
#                              NEW FEATURES (v2)
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. COMPARE ───────────────────────────────────────────────────────────────

@app.get("/compare")
def compare(symbols: str, months: int = 12):
    """Rebased (=100) performance + correlation matrix. /compare?symbols=TCS,INFY,WIPRO&months=12"""
    sym_list = [norm_ticker(s) for s in symbols.split(",") if s.strip()][:5]
    if len(sym_list) < 2:
        raise HTTPException(status_code=400, detail="Provide 2 to 5 symbols, e.g. ?symbols=TCS,INFY")
    months = min(max(months, 1), 24)

    closes, errors = {}, []
    for t in sym_list:
        try:
            h = get_history(t, "2y")
            if h.empty:
                errors.append(f"{t}: no data")
            else:
                closes[t] = h["Close"]
        except Exception as e:
            errors.append(f"{t}: {e}")

    if len(closes) < 2:
        raise HTTPException(status_code=404, detail=f"Need at least 2 valid symbols. Issues: {errors}")

    df = pd.DataFrame(closes).ffill().dropna().tail(months * 21)
    if len(df) < 5:
        raise HTTPException(status_code=404, detail="Not enough overlapping data.")

    rebased = df / df.iloc[0] * 100
    rets = df.pct_change().dropna()
    corr = rets.corr()

    summary = []
    for t in df.columns:
        summary.append({
            "symbol": t,
            "total_return_pct": safe_round((df[t].iloc[-1] / df[t].iloc[0] - 1) * 100),
            "annualized_volatility_pct": safe_round(rets[t].std() * np.sqrt(252) * 100),
            "best_day_pct": safe_round(rets[t].max() * 100),
            "worst_day_pct": safe_round(rets[t].min() * 100),
        })
    summary.sort(key=lambda x: (x["total_return_pct"] is None, -(x["total_return_pct"] or 0)))

    series = []
    for date, row in rebased.iterrows():
        d = {"date": date.strftime("%Y-%m-%d")}
        for t in rebased.columns:
            d[t] = safe_round(row[t])
        series.append(d)

    return {
        "symbols": list(df.columns),
        "months": months,
        "summary": summary,
        "correlation": {a: {b: safe_round(corr.loc[a, b], 3) for b in corr.columns} for a in corr.index},
        "series": series,
        "errors": errors,
    }


# ── 2. RISK METRICS ──────────────────────────────────────────────────────────

@app.get("/risk/{symbol}")
def risk_metrics(symbol: str):
    """Volatility, Sharpe, Sortino, beta vs NIFTY, max drawdown, VaR/CVaR."""
    ticker = norm_ticker(symbol)
    try:
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"].dropna()
        ret = close.pct_change().dropna()
        if len(ret) < 60:
            raise HTTPException(status_code=400, detail="Not enough history for risk metrics.")

        years = len(ret) / 252
        ann_ret = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else None

        downside = ret[ret < 0]
        downside_dev = downside.std() * np.sqrt(252) if len(downside) > 1 else None
        sortino = (ann_ret - RISK_FREE_RATE) / downside_dev if downside_dev else None

        dd = close / close.cummax() - 1
        var95 = np.percentile(ret, 5)
        cvar95 = ret[ret <= var95].mean()

        beta = corr_bench = None
        try:
            bench = get_history(BENCHMARK, "2y")["Close"].pct_change().dropna()
            al = pd.concat([ret, bench], axis=1, join="inner").dropna()
            al.columns = ["s", "b"]
            if len(al) > 30 and al["b"].var() > 0:
                beta = al["s"].cov(al["b"]) / al["b"].var()
                corr_bench = al["s"].corr(al["b"])
        except Exception:
            pass

        vol_pct = ann_vol * 100
        level = "Low" if vol_pct < 20 else "Moderate" if vol_pct < 35 else "High"

        return {
            "symbol": ticker,
            "risk_level": level,
            "annualized_return_pct": safe_round(ann_ret * 100),
            "annualized_volatility_pct": safe_round(vol_pct),
            "sharpe_ratio": safe_round(sharpe),
            "sortino_ratio": safe_round(sortino),
            "beta_vs_nifty": safe_round(beta),
            "correlation_vs_nifty": safe_round(corr_bench),
            "max_drawdown_pct": safe_round(dd.min() * 100),
            "max_drawdown_date": dd.idxmin().strftime("%Y-%m-%d"),
            "var_95_daily_pct": safe_round(var95 * 100),
            "cvar_95_daily_pct": safe_round(cvar95 * 100),
            "risk_free_rate_pct": RISK_FREE_RATE * 100,
            "drawdown_series": [
                {"date": d.strftime("%Y-%m-%d"), "drawdown_pct": safe_round(v * 100)}
                for d, v in dd.tail(252).items()
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing risk for {ticker}: {str(e)}")


# ── 3. COMPOSITE SIGNAL ──────────────────────────────────────────────────────

@app.get("/signal/{symbol}")
def composite_signal(symbol: str):
    """Rule-based score from 6 indicators: each votes +1 (bullish), 0, or -1 (bearish)."""
    ticker = norm_ticker(symbol)
    try:
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        t, _ = get_technical_snapshot(hist)
        components = []

        def add(name, value, vote, reason):
            components.append({"indicator": name, "value": value, "vote": vote, "reason": reason})

        rsi = t["rsi_14"]
        if rsi is not None:
            if rsi < 30: add("RSI (14)", rsi, 1, "Oversold (<30): potential bounce")
            elif rsi > 70: add("RSI (14)", rsi, -1, "Overbought (>70): potential pullback")
            else: add("RSI (14)", rsi, 0, "Neutral zone (30-70)")

        mh = t["macd_histogram"]
        if mh is not None:
            add("MACD", mh, 1 if mh > 0 else -1,
                "MACD above signal line: bullish momentum" if mh > 0 else "MACD below signal line: bearish momentum")

        if t["ma_trend"]:
            bull = t["ma_trend"] == "bullish"
            add("MA 50/200", t["ma_trend"], 1 if bull else -1,
                "Golden-cross regime (50 > 200)" if bull else "Death-cross regime (50 < 200)")

        pb = t["bollinger_percent_b"]
        if pb is not None:
            if pb < 0.2: add("Bollinger %B", pb, 1, "Near lower band: stretched down")
            elif pb > 0.8: add("Bollinger %B", pb, -1, "Near upper band: stretched up")
            else: add("Bollinger %B", pb, 0, "Inside the bands")

        roc = t["roc_12"]
        if roc is not None:
            add("ROC (12)", roc, 1 if roc > 0 else -1,
                "Positive 12-day momentum" if roc > 0 else "Negative 12-day momentum")

        p52 = t["pct_of_52w_high"]
        if p52 is not None:
            if p52 >= 90: add("52-week high", p52, 1, "Within 10% of 52-week high: strength")
            elif p52 < 70: add("52-week high", p52, -1, "More than 30% below 52-week high: weakness")
            else: add("52-week high", p52, 0, "Mid-range vs 52-week high")

        score = sum(c["vote"] for c in components)
        n = len(components)
        if score >= 4: label = "Strong Bullish"
        elif score >= 2: label = "Bullish"
        elif score >= -1: label = "Neutral"
        elif score >= -3: label = "Bearish"
        else: label = "Strong Bearish"

        adx = t["adx_14"]
        trend_strength = None if adx is None else ("Strong trend" if adx > 25 else "Weak / no trend")

        return {
            "symbol": ticker,
            "signal": label,
            "score": score,
            "max_score": n,
            "trend_strength": trend_strength,
            "adx_14": adx,
            "components": components,
            "disclaimer": DISCLAIMER,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building signal for {ticker}: {str(e)}")


# ── 4. BACKTEST ──────────────────────────────────────────────────────────────

def _perf_stats(r):
    r = r.dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    return {
        "total_return_pct": safe_round((eq.iloc[-1] - 1) * 100),
        "cagr_pct": safe_round(cagr * 100),
        "volatility_pct": safe_round(vol * 100),
        "sharpe": safe_round((cagr - RISK_FREE_RATE) / vol) if vol > 0 else None,
        "max_drawdown_pct": safe_round((eq / eq.cummax() - 1).min() * 100),
    }


@app.get("/backtest/{symbol}")
def backtest(symbol: str, strategy: str = "ma_cross", fast: int = 50, slow: int = 200, fee_pct: float = 0.1):
    """
    Backtest on 5 years of data vs buy & hold (signals act on the NEXT day — no look-ahead).
    strategy = ma_cross (buy when fast MA > slow MA) | rsi (buy <30, sell >55) | macd (buy when MACD > signal)
    """
    ticker = norm_ticker(symbol)
    if strategy not in ("ma_cross", "rsi", "macd"):
        raise HTTPException(status_code=400, detail="strategy must be ma_cross, rsi or macd")
    if strategy == "ma_cross" and not (2 <= fast < slow <= 250):
        raise HTTPException(status_code=400, detail="Need 2 <= fast < slow <= 250")
    fee = min(max(fee_pct, 0), 2) / 100

    try:
        hist = get_history(ticker, "5y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")
        close = hist["Close"].dropna()
        ret = close.pct_change().fillna(0)

        if strategy == "ma_cross":
            pos = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int)
            warmup = slow
            desc = f"Long when {fast}-day MA is above {slow}-day MA, else cash"
        elif strategy == "macd":
            m, s, _ = calculate_macd(close)
            pos = (m > s).astype(int)
            warmup = 35
            desc = "Long when MACD line is above signal line, else cash"
        else:
            rsi = calculate_rsi(close)
            holding, vals = 0, []
            for r in rsi:
                if not pd.isna(r):
                    if holding == 0 and r < 30: holding = 1
                    elif holding == 1 and r > 55: holding = 0
                vals.append(holding)
            pos = pd.Series(vals, index=close.index)
            warmup = 20
            desc = "Buy when RSI < 30, sell when RSI > 55, else hold position"

        if len(close) <= warmup + 30:
            raise HTTPException(status_code=400, detail="Not enough history for this strategy.")

        held = pos.shift(1).fillna(0)
        turnover = held.diff().abs().fillna(0)
        strat_ret = (held * ret - turnover * fee).iloc[warmup:]
        bh_ret = ret.iloc[warmup:]

        strat_eq = (1 + strat_ret).cumprod() * 100
        bh_eq = (1 + bh_ret).cumprod() * 100

        step = 5
        idx = list(range(0, len(strat_eq), step))
        if idx[-1] != len(strat_eq) - 1:
            idx.append(len(strat_eq) - 1)
        curve = [{
            "date": strat_eq.index[i].strftime("%Y-%m-%d"),
            "strategy": safe_round(strat_eq.iloc[i]),
            "buy_and_hold": safe_round(bh_eq.iloc[i]),
        } for i in idx]

        return {
            "symbol": ticker,
            "strategy": strategy,
            "description": desc,
            "fee_per_trade_pct": fee * 100,
            "period": {"start": strat_eq.index[0].strftime("%Y-%m-%d"), "end": strat_eq.index[-1].strftime("%Y-%m-%d")},
            "strategy_stats": _perf_stats(strat_ret),
            "buy_and_hold_stats": _perf_stats(bh_ret),
            "number_of_trades": int((pos.diff().iloc[warmup:] == 1).sum()),
            "time_in_market_pct": safe_round(held.iloc[warmup:].mean() * 100),
            "equity_curve": curve,
            "disclaimer": "Past performance does not predict future results. " + DISCLAIMER,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting {ticker}: {str(e)}")


# ── 5. MONTE CARLO ───────────────────────────────────────────────────────────

@app.get("/montecarlo/{symbol}")
def monte_carlo(symbol: str, days: int = 252, sims: int = 1000):
    """Geometric Brownian Motion simulation using the stock's own 2-year drift & volatility."""
    ticker = norm_ticker(symbol)
    days = min(max(days, 20), 504)
    sims = min(max(sims, 100), 2000)
    try:
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")
        close = hist["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) < 60:
            raise HTTPException(status_code=400, detail="Not enough history.")

        mu, sigma = log_ret.mean(), log_ret.std()
        last = float(close.iloc[-1])

        rng = np.random.default_rng()
        shocks = rng.normal(mu - 0.5 * sigma ** 2, sigma, size=(sims, days))
        paths = last * np.exp(np.cumsum(shocks, axis=1))
        final = paths[:, -1]

        qs = [5, 25, 50, 75, 95]
        pct = np.percentile(paths, qs, axis=0)

        bands = [{"day": 0, "p5": last, "p25": last, "p50": last, "p75": last, "p95": last}]
        for d in list(range(4, days, 5)) + [days - 1]:
            bands.append({"day": d + 1, **{f"p{q}": safe_round(pct[i, d]) for i, q in enumerate(qs)}})

        return {
            "symbol": ticker,
            "current_price": safe_round(last),
            "days": days,
            "simulations": sims,
            "annualized_drift_pct": safe_round(mu * 252 * 100),
            "annualized_volatility_pct": safe_round(sigma * np.sqrt(252) * 100),
            "final_price": {f"p{q}": safe_round(np.percentile(final, q)) for q in qs},
            "expected_final_price": safe_round(final.mean()),
            "prob_above_current_pct": safe_round((final > last).mean() * 100),
            "prob_loss_over_20pct": safe_round((final < last * 0.8).mean() * 100),
            "prob_gain_over_20pct": safe_round((final > last * 1.2).mean() * 100),
            "bands": bands,
            "disclaimer": "Simulation assumes past drift and volatility persist — a learning tool, not a forecast. " + DISCLAIMER,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating {ticker}: {str(e)}")


# ── 6. MARKET OVERVIEW + SECTORS ─────────────────────────────────────────────

INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX",
    "USD/INR": "INR=X",
}
SECTORS = {
    "IT": "^CNXIT",
    "Bank": "^NSEBANK",
    "Financial Services": "^CNXFIN",
    "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "Metal": "^CNXMETAL",
    "Energy": "^CNXENERGY",
    "Realty": "^CNXREALTY",
}


def _period_returns(ticker):
    h = get_history(ticker, "1y")
    if h.empty or len(h) < 2:
        return None
    c = h["Close"].dropna()

    def chg(n):
        return safe_round((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100) if len(c) > n else None

    return {
        "last": safe_round(c.iloc[-1]),
        "change_1d_pct": chg(1),
        "change_1w_pct": chg(5),
        "change_1m_pct": chg(21),
        "change_3m_pct": chg(63),
        "as_of": c.index[-1].strftime("%Y-%m-%d"),
    }


@app.get("/market")
def market_overview():
    """Headline indices + sector performance (for a heatmap). Missing tickers are skipped."""
    indices, sectors = [], []
    for name, t in INDICES.items():
        try:
            r = _period_returns(t)
            if r:
                indices.append({"name": name, "symbol": t, **r})
        except Exception:
            pass
    for name, t in SECTORS.items():
        try:
            r = _period_returns(t)
            if r:
                sectors.append({"sector": name, "symbol": t, **r})
        except Exception:
            pass
    sectors.sort(key=lambda x: (x["change_1m_pct"] is None, -(x["change_1m_pct"] or 0)))
    return {"timestamp": datetime.now().isoformat(), "indices": indices, "sectors": sectors}


# ── 7. NEWS + SENTIMENT ──────────────────────────────────────────────────────

POSITIVE = {"surge", "surges", "jump", "jumps", "rally", "rallies", "gain", "gains", "rise", "rises", "record",
            "profit", "beat", "beats", "growth", "upgrade", "upgrades", "buy", "bullish", "strong", "win", "wins",
            "order", "expansion", "dividend", "outperform", "boost", "soar", "soars", "high", "approval", "robust"}
NEGATIVE = {"fall", "falls", "drop", "drops", "plunge", "plunges", "slump", "loss", "losses", "miss", "misses",
            "downgrade", "downgrades", "sell", "bearish", "weak", "probe", "fraud", "penalty", "fine", "decline",
            "declines", "cut", "cuts", "concern", "concerns", "risk", "lawsuit", "layoffs", "default", "crash", "low"}


def _sentiment(text):
    words = [w.strip(".,:;!?()'\"").lower() for w in text.split()]
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    total = pos + neg
    score = 0.0 if total == 0 else (pos - neg) / total
    label = "Positive" if score > 0.2 else "Negative" if score < -0.2 else "Neutral"
    return round(score, 2), label


@app.get("/news/{symbol}")
def news(symbol: str, limit: int = 10):
    """Latest headlines (from Yahoo Finance) with simple keyword-based sentiment."""
    ticker = norm_ticker(symbol)
    limit = min(max(limit, 1), 20)
    try:
        raw = yf.Ticker(ticker).news or []
        articles = []
        for item in raw[:limit]:
            c = item.get("content", item)  # new yfinance nests under "content"
            title = c.get("title") or ""
            summary = c.get("summary") or ""
            if not title:
                continue
            url = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
            provider = (c.get("provider") or {}).get("displayName") or c.get("publisher")
            published = c.get("pubDate")
            if not published and c.get("providerPublishTime"):
                published = datetime.fromtimestamp(c["providerPublishTime"]).isoformat()
            score, label = _sentiment(f"{title} {summary}")
            articles.append({
                "title": title, "summary": summary or None, "publisher": provider,
                "published": published, "url": url, "sentiment_score": score, "sentiment": label,
            })

        if not articles:
            return {"symbol": ticker, "count": 0, "overall_sentiment": None, "articles": [],
                    "note": "No news returned by Yahoo Finance for this symbol right now."}

        avg = sum(a["sentiment_score"] for a in articles) / len(articles)
        overall = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
        return {
            "symbol": ticker, "count": len(articles),
            "overall_sentiment": overall, "average_score": round(avg, 2),
            "articles": articles,
            "note": "Keyword-based sentiment — a simple heuristic, not NLP-grade.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news for {ticker}: {str(e)}")
