import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from protect import protect_app, logout

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# =====================================================
# FINANCIFY SAAS CONFIG
# =====================================================
APP_NAME = os.getenv("APP_NAME", "Hive Forecast Engine")
BACKEND_URL = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "")).rstrip("/")
SURECART_CHECKOUT_URL = st.secrets.get(
    "SURECART_CHECKOUT_URL",
    os.getenv("SURECART_CHECKOUT_URL", "https://financify.blog/buy/financify-tools"),
)
LOGO_URL = "https://financify.blog/wp-content/uploads/2026/05/445e63b6-b77c-4834-98eb-bf8e77b40f44.png"

st.set_page_config(
    page_title=f"{APP_NAME} | Financify",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# PREMIUM CSS — MOBILE + DESKTOP
# =====================================================
st.markdown(
    """
<style>
:root{
  --bg:#090909; --card:#111111; --card2:#171717; --honey:#f7c948; --honey2:#ffd86b;
  --text:#f7f2df; --muted:#c8bd93; --line:rgba(247,201,72,.22); --danger:#ff6b6b; --ok:#80ed99;
}
html, body, [data-testid="stAppViewContainer"]{
  background-color: var(--bg);
  color: var(--text);
  background-image:
    radial-gradient(circle at 20% 10%, rgba(247,201,72,.13), transparent 24%),
    radial-gradient(circle at 80% 0%, rgba(247,201,72,.08), transparent 26%),
    linear-gradient(30deg, rgba(247,201,72,.045) 12%, transparent 12.5%, transparent 87%, rgba(247,201,72,.045) 87.5%, rgba(247,201,72,.045)),
    linear-gradient(150deg, rgba(247,201,72,.045) 12%, transparent 12.5%, transparent 87%, rgba(247,201,72,.045) 87.5%, rgba(247,201,72,.045)),
    linear-gradient(30deg, rgba(247,201,72,.045) 12%, transparent 12.5%, transparent 87%, rgba(247,201,72,.045) 87.5%, rgba(247,201,72,.045)),
    linear-gradient(150deg, rgba(247,201,72,.045) 12%, transparent 12.5%, transparent 87%, rgba(247,201,72,.045) 87.5%, rgba(247,201,72,.045));
  background-size: 900px 900px, 900px 900px, 80px 140px, 80px 140px, 80px 140px, 80px 140px;
  background-position: 0 0, 0 0, 0 0, 0 0, 40px 70px, 40px 70px;
}
[data-testid="stHeader"]{background:rgba(9,9,9,.72); backdrop-filter:blur(10px)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b0b0b,#151103)!important; border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{color:var(--text)!important}
.block-container{padding-top:1.3rem; max-width:1280px}
.hero{
  padding:28px; border:1px solid var(--line); border-radius:28px;
  background:linear-gradient(135deg, rgba(247,201,72,.18), rgba(12,12,12,.94) 42%, rgba(247,201,72,.06));
  box-shadow:0 18px 50px rgba(0,0,0,.35); margin-bottom:20px;
}
.hero h1{font-size:clamp(2rem,4vw,4.5rem); line-height:1.02; margin:0; color:var(--text)}
.hero p{font-size:1.05rem; color:var(--muted); max-width:900px}
.badge{display:inline-block; padding:8px 13px; border-radius:999px; border:1px solid var(--line); background:rgba(247,201,72,.12); color:var(--honey); font-weight:700; margin-bottom:12px}
.card, .metric-card, .note-box{
  border:1px solid var(--line); border-radius:22px; background:linear-gradient(180deg,rgba(23,23,23,.98),rgba(13,13,13,.96));
  padding:20px; box-shadow:0 14px 35px rgba(0,0,0,.25); color:var(--text); margin-bottom:16px;
}
.metric-card h3{font-size:.9rem; color:var(--muted); margin:0 0 8px}.metric-card h2{margin:0; color:var(--honey2); font-size:1.55rem}.metric-card p{color:var(--muted); margin:8px 0 0; font-size:.88rem}
.note-box{background:linear-gradient(135deg,rgba(247,201,72,.12),rgba(20,20,20,.96)); color:var(--text)}
.note-box b{color:var(--honey2)}
.stButton>button, .stDownloadButton>button{
  border-radius:999px!important; border:1px solid rgba(247,201,72,.65)!important; background:linear-gradient(135deg,#f7c948,#9d6b00)!important;
  color:#080808!important; font-weight:900!important; padding:.75rem 1.1rem!important; box-shadow:0 10px 24px rgba(247,201,72,.16)
}
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], textarea{
  background:#101010!important; color:var(--text)!important; border:1px solid var(--line)!important; border-radius:14px!important;
}
label, .stMarkdown, .stDataFrame, p, span, div{color:var(--text)}
[data-testid="stMetricValue"]{color:var(--honey2)} [data-testid="stMetricLabel"]{color:var(--muted)}
.disclaimer{font-size:.88rem; color:var(--muted); border-left:3px solid var(--honey); padding-left:12px}
@media(max-width:768px){.block-container{padding-left:.85rem;padding-right:.85rem}.hero{padding:20px;border-radius:22px}.card,.metric-card,.note-box{padding:16px;border-radius:18px}.hero p{font-size:.96rem}}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# API HELPERS / AUTH
# =====================================================
def _post(path: str, payload: dict, timeout: int = 60):
    if not BACKEND_URL:
        raise RuntimeError("Backend URL is not configured.")
    r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
    if r.status_code >= 400:
        try:
            raise RuntimeError(r.json().get("detail", r.text))
        except Exception:
            raise RuntimeError(r.text)
    try:
        return r.json()
    except Exception:
        return {"ok": True}


def send_login_code(email: str):
    return _post("/auth/send-code", {"email": email.strip().lower()})


def verify_login_code(email: str, code: str):
    return _post("/auth/verify-code", {"email": email.strip().lower(), "code": code.strip()})


def check_subscription(email: str, token: str = ""):
    try:
        return _post("/subscription/status", {"email": email.strip().lower(), "token": token}, timeout=30)
    except Exception:
        # fail closed for production; local dev can use DEV_MODE
        return {"active": False, "plan": "free", "error": "Subscription check failed"}


def init_state():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("email", "")
    st.session_state.setdefault("token", "")
    st.session_state.setdefault("subscription_active", False)
    st.session_state.setdefault("plan", "free")


def auth_gate():
    init_state()
    dev_mode = st.secrets.get("DEV_MODE", os.getenv("DEV_MODE", "false")).lower() == "true"
    if dev_mode:
        st.session_state.logged_in = True
        st.session_state.email = "dev@financify.blog"
        st.session_state.subscription_active = True
        st.session_state.plan = "dev"
        return True

    with st.sidebar:
        st.image(LOGO_URL, width=140)
        st.markdown("### 🐝 Financify Access")
        if st.session_state.logged_in:
            st.success(f"Logged in: {st.session_state.email}")
            st.caption(f"Plan: {st.session_state.plan}")
            if st.button("Logout"):
                for k in ["logged_in", "email", "token", "subscription_active", "plan"]:
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            email = st.text_input("Email", placeholder="you@example.com")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Send Code"):
                    try:
                        send_login_code(email)
                        st.success("Login code sent.")
                    except Exception as e:
                        st.error(str(e))
            code = st.text_input("Login Code", placeholder="6-digit code")
            with c2:
                if st.button("Verify"):
                    try:
                        res = verify_login_code(email, code)
                        st.session_state.logged_in = True
                        st.session_state.email = email.strip().lower()
                        st.session_state.token = res.get("token", "")
                        sub = check_subscription(st.session_state.email, st.session_state.token)
                        st.session_state.subscription_active = bool(sub.get("active"))
                        st.session_state.plan = sub.get("plan", "free")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.markdown("---")
        st.markdown("**Need access?**")
        st.link_button("Upgrade to Financify Tools", SURECART_CHECKOUT_URL)

    if not st.session_state.logged_in:
        st.markdown("""
        <div class='hero'><span class='badge'>Premium Tool</span><h1>Hive Forecast Engine</h1>
        <p>Login to forecast revenue, earnings, EPS and valuation range using a bear/base/bull scenario engine.</p></div>
        """, unsafe_allow_html=True)
        return False

    if not st.session_state.subscription_active:
        st.markdown("""
        <div class='hero'><span class='badge'>Subscription Required</span><h1>Unlock Hive Forecast Engine</h1>
        <p>This tool is available for Financify Tools subscribers.</p></div>
        """, unsafe_allow_html=True)
        st.link_button("Upgrade Now", SURECART_CHECKOUT_URL)
        return False
    return True

# =====================================================
# DATA + MODEL ENGINE
# =====================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="max", auto_adjust=False)
    financials = tk.financials.T
    income = tk.income_stmt.T
    balance = tk.balance_sheet.T
    info = tk.info or {}
    return hist, financials, income, balance, info


def get_first(row, possible):
    for p in possible:
        if p in row and pd.notna(row[p]):
            return row[p]
    return np.nan


def build_annual_dataset(hist, financials, income, info):
    fin = financials.copy() if financials is not None and not financials.empty else income.copy()
    if fin is None or fin.empty:
        return pd.DataFrame()
    fin.index = pd.to_datetime(fin.index).year
    fin = fin.sort_index()
    rows = []
    for year, row in fin.iterrows():
        revenue = get_first(row, ["Total Revenue", "Operating Revenue"])
        net_income = get_first(row, ["Net Income", "Net Income Common Stockholders"])
        ebitda = get_first(row, ["EBITDA", "Normalized EBITDA"])
        diluted_eps = get_first(row, ["Diluted EPS", "Basic EPS"])
        year_prices = hist[hist.index.year == year] if hist is not None and not hist.empty else pd.DataFrame()
        price = float(year_prices["Close"].iloc[-1]) if not year_prices.empty else np.nan
        avg_price = float(year_prices["Close"].mean()) if not year_prices.empty else np.nan
        if pd.isna(diluted_eps) or diluted_eps == 0:
            shares = get_first(row, ["Diluted Average Shares", "Basic Average Shares"])
            diluted_eps = net_income / shares if pd.notna(net_income) and pd.notna(shares) and shares else np.nan
        pe = price / diluted_eps if pd.notna(price) and pd.notna(diluted_eps) and diluted_eps > 0 else np.nan
        earnings_yield = diluted_eps / price if pd.notna(price) and price > 0 and pd.notna(diluted_eps) else np.nan
        net_margin = net_income / revenue if pd.notna(net_income) and pd.notna(revenue) and revenue else np.nan
        ev_ebitda = np.nan
        rows.append({
            "year": int(year), "revenue": revenue, "net_income": net_income, "eps": diluted_eps,
            "price": price, "avg_price": avg_price, "pe": pe, "earnings_yield": earnings_yield,
            "net_margin": net_margin, "ebitda": ebitda, "ev_ebitda": ev_ebitda,
        })
    df = pd.DataFrame(rows).dropna(subset=["revenue", "net_income", "eps"], how="all")
    if df.empty:
        return df
    df = df.sort_values("year").tail(20).reset_index(drop=True)
    for col in ["revenue", "net_income", "eps", "price"]:
        df[f"{col}_growth"] = df[col].pct_change().replace([np.inf, -np.inf], np.nan)
    df["peg"] = df["pe"] / (df["eps_growth"] * 100)
    df["revenue_3y_cagr"] = (df["revenue"] / df["revenue"].shift(3)) ** (1/3) - 1
    df["eps_3y_cagr"] = (df["eps"] / df["eps"].shift(3)) ** (1/3) - 1
    return df.replace([np.inf, -np.inf], np.nan)


def safe_median(s, default=np.nan):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.median()) if len(s) else default


def winsor(x, low=-0.5, high=0.7):
    if pd.isna(x): return x
    return float(np.clip(x, low, high))


def train_predict_next(df, target):
    model_df = df.copy()
    features = ["revenue_growth", "net_margin", "eps_growth", "pe", "earnings_yield", "peg", "revenue_3y_cagr", "eps_3y_cagr"]
    for f in features:
        if f not in model_df:
            model_df[f] = np.nan
    model_df[features] = model_df[features].replace([np.inf, -np.inf], np.nan).fillna(model_df[features].median(numeric_only=True)).fillna(0)
    model_df[f"target_{target}"] = model_df[target].shift(-1)
    train = model_df.dropna(subset=[f"target_{target}"])
    if len(train) < 8:
        return np.nan, "Historical median fallback", np.nan
    X, y = train[features], train[f"target_{target}"]
    candidates = []
    if XGBOOST_AVAILABLE and len(train) >= 10:
        candidates.append(("XGBoost", XGBRegressor(n_estimators=160, max_depth=2, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=42)))
    candidates.extend([
        ("Random Forest", RandomForestRegressor(n_estimators=250, max_depth=4, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
        ("Ridge", Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])),
    ])
    best_name, best_model, best_mae = None, None, np.inf
    splits = min(3, max(2, len(train)//4))
    tscv = TimeSeriesSplit(n_splits=splits)
    for name, model in candidates:
        maes = []
        try:
            for tr, te in tscv.split(X):
                model.fit(X.iloc[tr], y.iloc[tr])
                pred = model.predict(X.iloc[te])
                maes.append(mean_absolute_error(y.iloc[te], pred))
            score = float(np.mean(maes))
            if score < best_mae:
                best_name, best_model, best_mae = name, model, score
        except Exception:
            continue
    if best_model is None:
        return np.nan, "Historical median fallback", np.nan
    best_model.fit(X, y)
    next_x = model_df[features].iloc[[-1]]
    pred = float(best_model.predict(next_x)[0])
    return pred, best_name, best_mae


def create_scenarios(df):
    latest = df.dropna(subset=["revenue", "net_income", "eps"], how="any").iloc[-1]
    next_year = int(latest["year"] + 1)

    rev_growth_pred, rev_model, rev_mae = train_predict_next(df, "revenue_growth")
    margin_pred, margin_model, margin_mae = train_predict_next(df, "net_margin")
    eps_growth_pred, eps_model, eps_mae = train_predict_next(df, "eps_growth")

    hist_rev_growth = safe_median(df["revenue_growth"], 0.05)
    hist_margin = safe_median(df["net_margin"], 0.08)
    hist_eps_growth = safe_median(df["eps_growth"], 0.06)

    base_rev_growth = winsor(rev_growth_pred if pd.notna(rev_growth_pred) else hist_rev_growth, -0.35, 0.45)
    base_margin = float(np.clip(margin_pred if pd.notna(margin_pred) else hist_margin, -0.35, 0.45))
    base_eps_growth = winsor(eps_growth_pred if pd.notna(eps_growth_pred) else hist_eps_growth, -0.5, 0.6)

    pe_20 = safe_median(df["pe"], np.nan)
    pe_10 = safe_median(df.tail(10)["pe"], pe_20)
    pe_5 = safe_median(df.tail(5)["pe"], pe_10)
    pe_clean = df["pe"].replace([np.inf, -np.inf], np.nan).dropna()
    pe_low = float(pe_clean.quantile(.25)) if len(pe_clean) else pe_10
    pe_high = float(pe_clean.quantile(.75)) if len(pe_clean) else pe_10
    base_pe = pe_5 if pd.notna(pe_5) else pe_10

    cases = {
        "Bear Case": {"growth_adj": -0.35, "margin_adj": -0.15, "pe": pe_low, "tone": "Pressure test: slower growth, weaker margins, valuation compression."},
        "Base Case": {"growth_adj": 0.0, "margin_adj": 0.0, "pe": base_pe, "tone": "Normal case: model output blended with recent historical valuation."},
        "Bull Case": {"growth_adj": 0.35, "margin_adj": 0.15, "pe": pe_high, "tone": "Optimistic case: stronger growth, better margins, valuation expansion."},
    }

    results = []
    for name, c in cases.items():
        rev_growth = winsor(base_rev_growth * (1 + c["growth_adj"]), -0.45, 0.65)
        margin = float(np.clip(base_margin * (1 + c["margin_adj"]), -0.35, 0.5))
        eps_growth = winsor(base_eps_growth * (1 + c["growth_adj"]), -0.65, 0.8)
        revenue = latest["revenue"] * (1 + rev_growth)
        net_income = revenue * margin
        eps = latest["eps"] * (1 + eps_growth)
        pe = c["pe"] if pd.notna(c["pe"]) and c["pe"] > 0 else base_pe
        price = eps * pe if pd.notna(eps) and pd.notna(pe) else np.nan
        results.append({
            "Scenario": name, "Forecast Year": next_year, "Revenue Growth": rev_growth, "Forecast Revenue": revenue,
            "Net Margin": margin, "Forecast Net Income": net_income, "EPS Growth": eps_growth,
            "Forecast EPS": eps, "Applied PE": pe, "Implied Stock Price": price, "Explanation": c["tone"]
        })
    meta = {
        "revenue_model": rev_model, "margin_model": margin_model, "eps_model": eps_model,
        "pe_20y_median": pe_20, "pe_10y_median": pe_10, "pe_5y_median": pe_5,
        "rev_mae": rev_mae, "margin_mae": margin_mae, "eps_mae": eps_mae,
    }
    return pd.DataFrame(results), meta


def fmt_big(x):
    if pd.isna(x): return "—"
    sign = "-" if x < 0 else ""
    x = abs(float(x))
    if x >= 1e12: return f"{sign}{x/1e12:.2f}T"
    if x >= 1e9: return f"{sign}{x/1e9:.2f}B"
    if x >= 1e6: return f"{sign}{x/1e6:.2f}M"
    return f"{sign}{x:,.0f}"

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x*100:.1f}%"

def fmt_num(x):
    return "—" if pd.isna(x) else f"{x:,.2f}"

# =====================================================
# UI
# =====================================================
if auth_gate():
    with st.sidebar:
        st.markdown("### Forecast Controls")
        ticker = st.text_input("Stock ticker", value="RELIANCE.NS", help="Use Yahoo Finance ticker. Examples: RELIANCE.NS, TCS.NS, AAPL, MSFT")
        st.caption("Tip: For Indian stocks, usually add `.NS` after NSE ticker.")
        run = st.button("🐝 Run Forecast", use_container_width=True)
        st.markdown("---")
        st.markdown("### What this tool does")
        st.caption("It forecasts fundamentals first, then converts EPS into valuation ranges using historical PE bands.")

    st.markdown(f"""
    <div class='hero'>
      <span class='badge'>🐝 Financify Premium</span>
      <h1>Hive Forecast Engine</h1>
      <p>Forecast next year revenue, net income, EPS and implied stock price range with bear, base and bull cases. Built for learning, not blind prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='note-box'><b>How to read this:</b> The model does not claim to know the future. It studies past fundamentals, estimates next-year revenue growth, margin and EPS growth, then applies historical valuation bands to create a sensible range.</div>
    """, unsafe_allow_html=True)

    if run or ticker:
        try:
            with st.spinner("Collecting fundamentals and training the hive..."):
                hist, financials, income, balance, info = fetch_stock_data(ticker.strip().upper())
                df = build_annual_dataset(hist, financials, income, info)
                if df.empty or len(df) < 5:
                    st.error("Not enough financial history available for this ticker. Try another stock or a ticker with longer reporting history.")
                    st.stop()
                scenarios, meta = create_scenarios(df)

            name = info.get("longName") or info.get("shortName") or ticker.upper()
            current_price = info.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else np.nan)
            currency = info.get("currency", "")

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f"<div class='metric-card'><h3>Company</h3><h2>{name[:22]}</h2><p>{ticker.upper()}</p></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-card'><h3>Current Price</h3><h2>{currency} {fmt_num(current_price)}</h2><p>Latest Yahoo Finance price</p></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='metric-card'><h3>History Used</h3><h2>{len(df)} Years</h2><p>Max 20 annual periods</p></div>", unsafe_allow_html=True)
            with c4: st.markdown(f"<div class='metric-card'><h3>Base Model</h3><h2>{meta['eps_model']}</h2><p>Selected via time split test</p></div>", unsafe_allow_html=True)

            st.subheader("Scenario Forecast")
            show = scenarios.copy()
            for col in ["Revenue Growth", "Net Margin", "EPS Growth"]:
                show[col] = show[col].map(fmt_pct)
            for col in ["Forecast Revenue", "Forecast Net Income"]:
                show[col] = show[col].map(fmt_big)
            for col in ["Forecast EPS", "Applied PE", "Implied Stock Price"]:
                show[col] = show[col].map(fmt_num)
            st.dataframe(show, use_container_width=True, hide_index=True)

            st.markdown("### Case Explanation")
            cols = st.columns(3)
            for i, row in scenarios.iterrows():
                with cols[i]:
                    st.markdown(f"""
                    <div class='card'>
                      <h3 style='color:#ffd86b;margin-top:0'>{row['Scenario']}</h3>
                      <p>{row['Explanation']}</p>
                      <p><b>EPS:</b> {fmt_num(row['Forecast EPS'])}<br><b>PE:</b> {fmt_num(row['Applied PE'])}<br><b>Price:</b> {currency} {fmt_num(row['Implied Stock Price'])}</p>
                    </div>
                    """, unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=scenarios["Scenario"], y=scenarios["Implied Stock Price"], text=[fmt_num(x) for x in scenarios["Implied Stock Price"]], textposition="outside", name="Implied Price"))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title="Bear / Base / Bull Implied Stock Price", yaxis_title=f"Price {currency}", height=430, margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Historical Fundamentals")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["year"], y=df["revenue"], mode="lines+markers", name="Revenue"))
            fig2.add_trace(go.Scatter(x=df["year"], y=df["net_income"], mode="lines+markers", name="Net Income"))
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=430, margin=dict(l=20,r=20,t=35,b=20))
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Valuation Benchmarks")
            b1, b2, b3 = st.columns(3)
            b1.metric("20Y Median PE", fmt_num(meta["pe_20y_median"]))
            b2.metric("10Y Median PE", fmt_num(meta["pe_10y_median"]))
            b3.metric("5Y Median PE", fmt_num(meta["pe_5y_median"]))

            with st.expander("See raw annual dataset"):
                st.dataframe(df, use_container_width=True, hide_index=True)

            csv = scenarios.to_csv(index=False).encode("utf-8")
            st.download_button("Download Forecast CSV", csv, file_name=f"{ticker.upper()}_hive_forecast.csv", mime="text/csv")

            st.markdown("""
            <p class='disclaimer'><b>Disclaimer:</b> This is an educational forecasting tool. It is not investment advice, not a buy/sell recommendation, and not a guarantee of future price. Always verify financial statements and valuation assumptions manually.</p>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Check the ticker format, backend subscription settings, or whether Yahoo Finance has enough data for this stock.")
