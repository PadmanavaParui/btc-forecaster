import os
from datetime import timezone
import streamlit as st
import numpy as np
import pandas as pd
import requests, json
import scipy.stats as stats
from arch import arch_model
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Forecaster", layout="wide")
st.title("BTC/USDT — Live 1-Hour Forecast")

# ── Persistence helpers ────────────────────────────────────────────────────
def get_supabase_headers():
    return {
        "apikey"       : st.secrets["SUPABASE_KEY"],
        "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}",
        "Content-Type" : "application/json"
    }

def save_prediction(timestamp, current_price, low, high):
    """Save today's prediction. Skip if this timestamp already saved."""
    url     = st.secrets["SUPABASE_URL"] + "/rest/v1/predictions"
    headers = get_supabase_headers()

    # check if already saved
    check = requests.get(
        url, headers=headers,
        params={"timestamp": f"eq.{timestamp}", "select": "id"}
    )
    if check.json():
        return  # already saved this hour

    requests.post(url, headers=headers, json={
        "timestamp"    : timestamp,
        "current_price": current_price,
        "lower_95"     : low,
        "upper_95"     : high,
    })

def fill_actuals():
    """For past predictions with no actual, fetch real price and update."""
    url     = st.secrets["SUPABASE_URL"] + "/rest/v1/predictions"
    headers = get_supabase_headers()

    # get rows missing actuals
    rows = requests.get(
        url, headers=headers,
        params={"actual": "is.null", "select": "*"}
    ).json()

    for row in rows:
        pred_time = pd.Timestamp(row["timestamp"])
        now       = pd.Timestamp.now(tz=timezone.utc)
        if pred_time > now:
            continue  # bar hasn't closed yet

        # fetch that specific bar from Binance
        r = requests.get(
            "https://data-api.binance.vision/api/v3/klines",
            params={
                "symbol"   : "BTCUSDT",
                "interval" : "1h",
                "startTime": int(pred_time.timestamp() * 1000),
                "limit"    : 1
            }
        )
        if not r.json():
            continue

        actual_price = float(r.json()[0][4])  # close price
        width        = row["upper_95"] - row["lower_95"]
        hit          = row["lower_95"] <= actual_price <= row["upper_95"]
        alpha        = 0.05
        if actual_price < row["lower_95"]:
            winkler = width + (2/alpha) * (row["lower_95"] - actual_price)
        elif actual_price > row["upper_95"]:
            winkler = width + (2/alpha) * (actual_price - row["upper_95"])
        else:
            winkler = width

        requests.patch(
            url, headers=headers,
            params={"id": f"eq.{row['id']}"},
            json={"actual": actual_price, "hit": hit,
                  "width": width, "winkler": winkler}
        )

def load_history():
    url     = st.secrets["SUPABASE_URL"] + "/rest/v1/predictions"
    headers = get_supabase_headers()
    rows    = requests.get(
        url, headers=headers,
        params={"select": "*", "order": "created_at.desc", "limit": "200"}
    ).json()
    return pd.DataFrame(rows) if rows else pd.DataFrame()



# ── Fetch live data ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # refresh every 5 minutes
def fetch_btc(n_bars=600):
    r = requests.get(
        "https://data-api.binance.vision/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1h", "limit": n_bars},
        timeout=10
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["close"]     = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df["close"].sort_index()

# ── Model (same as backtest) ───────────────────────────────────────────────
def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

def update_params(p, sigma2, bar_sigma2, t):
    err = sigma2 - bar_sigma2
    lr  = p['eta'] / (1 + t**0.55)
    p['gamma'] = np.clip(p['gamma'] + lr * err, 0.01, 0.5)
    return p

def simulate_cyber_gbm(S0, mu, sigma_fig, H, M, params,
                       bar_sigma2, redundancy, info_filter, nu,
                       n_steps=1, dt=1, eps=1e-6):
    S = np.zeros(n_steps + 1)
    S[0] = S0
    sigma2 = sigma_fig.iloc[-1] ** 2
    H_max = max(H.max(), 1e-9)
    M_max = max(M.max(), 1e-9)
    for t in range(1, n_steps + 1):
        H_val   = min(H.iloc[-1] / H_max, 1.0)
        M_val   = min(M.iloc[-1] / M_max, 1.0)
        crisis  = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params['delta'] if crisis else 0.0
        sigma2  = (
            sigma_fig.iloc[-1]**2 * (1 + params['alpha'] * H_val + delta_t * M_val)
            + params['gamma'] * (bar_sigma2 - sigma2)
        )
        sigma2 *= max(1e-12, redundancy.iloc[-1])
        sigma2 *= 1 + 0.5 * info_filter.iloc[-1]
        sigma2  = max(eps, min(sigma2, 0.5))
        Z       = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        S[t]    = S[t-1] * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)
        params  = update_params(params, sigma2, bar_sigma2, t)
    return S

@st.cache_data(ttl=300)
def run_model(prices_values, prices_index):
    prices = pd.Series(prices_values, index=prices_index)
    log_ret = np.log(prices / prices.shift(1)).dropna()

    am  = arch_model(log_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
    res = am.fit(disp='off')
    sigma_fig = res.conditional_volatility / 100
    resid     = (log_ret * 100 - res.params['mu']) / res.conditional_volatility
    nu        = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])

    H_series    = rolling_entropy(resid)
    M_series    = log_ret.abs().rolling(60).mean()
    bar_sigma2  = (sigma_fig**2).mean()
    redundancy  = 1 + 0.1 * np.log1p(prices.rolling(5).var() / prices.rolling(20).var())
    info_filter = (H_series > H_series.mean()).astype(float)

    H_max, M_max = H_series.max(), M_series.max()
    α0, δ0 = 0.5, 0.3
    if α0 * H_max + δ0 * M_max >= 1:
        fac = 0.95 / (α0 * H_max + δ0 * M_max)
        α0 *= fac; δ0 *= fac
    base_params = {'alpha': α0, 'delta': δ0, 'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3}

    S0  = float(prices.iloc[-1])
    mu  = float(log_ret.mean())
    n_sims = 10_000
    sims = np.zeros(n_sims)
    for i in range(n_sims):
        path = simulate_cyber_gbm(
            S0, mu, sigma_fig, H_series, M_series,
            base_params.copy(), bar_sigma2, redundancy, info_filter, nu
        )
        sims[i] = path[1]

    low95, high95 = np.percentile(sims, [2.5, 97.5])
    return S0, low95, high95, float(sigma_fig.iloc[-1]) * 100

# ── Load data & run model ──────────────────────────────────────────────────
with st.spinner("Fetching live BTC data and running model..."):
    prices = fetch_btc(600)
    S0, low95, high95, cur_vol = run_model(
        prices.values.tolist(), prices.index.tolist()
    )

# ── Save this prediction + fill past actuals ───────────────────────────────
next_bar_ts = prices.index[-1].isoformat()
save_prediction(next_bar_ts, S0, low95, high95)
fill_actuals()

last_bar_time = prices.index[-1].strftime("%Y-%m-%d %H:%M UTC")

# ── Backtest metrics (hardcode from your Part A run) ──────────────────────
COVERAGE   = 0.9750
AVG_WIDTH  = 1552.62
WINKLER    = 1858.42

# ── Header metrics ─────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current BTC Price",  f"${S0:,.2f}")
col2.metric("Predicted Low",      f"${low95:,.2f}")
col3.metric("Predicted High",     f"${high95:,.2f}")
col4.metric("Range Width",        f"${high95-low95:,.2f}")
col5.metric("Current Vol/hr",     f"{cur_vol:.3f}%")

st.divider()

# col6, col7, col8 = st.columns(3)
# col6.metric("Backtest Coverage",  f"{COVERAGE:.2%}", "target: 95%")
# col7.metric("Avg Range Width",    f"${AVG_WIDTH:,.0f}")
# col8.metric("Mean Winkler Score", f"${WINKLER:,.0f}", "lower is better")

col6, col7, col8 = st.columns(3)
# col6.metric("Backtest Coverage",  f"{COVERAGE:.2%}", "target: 95%")
col6.metric("Backtest Coverage", f"{COVERAGE:.2%}", "target: 95%", delta_color="off")
col7.metric("Avg Range Width",    f"${AVG_WIDTH:,.0f}")
# col8.metric("Mean Winkler Score", f"${WINKLER:,.0f}", "lower is better")
col8.metric("Mean Winkler Score", f"${WINKLER:,.0f}", "lower is better", delta_color="off")


st.divider()

# ── Chart: last 50 bars + predicted range ribbon ───────────────────────────
recent   = prices.iloc[-50:]
next_time = prices.index[-1] + pd.Timedelta(hours=1)

fig = go.Figure()

# price line
fig.add_trace(go.Scatter(
    x=recent.index, y=recent.values,
    mode="lines", name="BTC Close",
    line=dict(color="#3266ad", width=2)
))

# shaded ribbon (current price → predicted range)
fig.add_trace(go.Scatter(
    x=[prices.index[-1], next_time, next_time, prices.index[-1]],
    y=[S0, high95, low95, S0],
    fill="toself",
    fillcolor="rgba(239,159,39,0.2)",
    line=dict(color="rgba(0,0,0,0)"),
    name="95% predicted range"
))

# bounds
fig.add_trace(go.Scatter(
    x=[prices.index[-1], next_time], y=[S0, high95],
    mode="lines", line=dict(color="#EF9F27", dash="dash", width=1.5),
    name=f"Upper: ${high95:,.0f}"
))
fig.add_trace(go.Scatter(
    x=[prices.index[-1], next_time], y=[S0, low95],
    mode="lines", line=dict(color="#EF9F27", dash="dash", width=1.5),
    name=f"Lower: ${low95:,.0f}"
))

fig.update_layout(
    title=f"Last 50 bars + next-hour forecast  |  Last bar: {last_bar_time}",
    xaxis_title="Time (UTC)",
    yaxis_title="Price (USDT)",
    hovermode="x unified",
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Model refreshes every 5 minutes. Forecast is for the hour closing at {next_time.strftime('%H:%M UTC')}.")



# ── Prediction history ─────────────────────────────────────────────────────
st.subheader("Prediction history")
hist = load_history()
if hist.empty:
    st.info("No history yet — check back after a few visits.")
else:
    def colour_hit(val):
        if val is True:  return "background-color: #d4edda"
        if val is False: return "background-color: #f8d7da"
        return ""

    display_cols = ["timestamp","current_price","lower_95",
                    "upper_95","actual","hit","winkler"]
    existing = [c for c in display_cols if c in hist.columns]
    st.dataframe(
        hist[existing].style.applymap(colour_hit, subset=["hit"]
            if "hit" in existing else []),
        use_container_width=True
    )