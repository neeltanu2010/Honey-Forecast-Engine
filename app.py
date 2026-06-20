# ============================================================
# Financify - Hive Forecast Engine
# app.py
# Option A + XGBoost architecture
# ============================================================

import math
import warnings
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

try:
    from protect import protect_app, logout
except Exception:
    protect_app = None
    logout = None

st.set_page_config(
    page_title="Hive Forecast Engine | Financify",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if protect_app:
    protect_app()

st.markdown(
    """
<style>
:root{--y:#f5b400;--g:#c48900;--w:#fffbea;--m:#d6c88f;--b:#050505;--card:rgba(15,15,15,.9);--border:rgba(245,180,0,.30)}
html,body,[data-testid="stAppViewContainer"]{
background:linear-gradient(rgba(0,0,0,.84),rgba(0,0,0,.94)),
radial-gradient(circle at 25% 10%,rgba(245,180,0,.16),transparent 25%),
repeating-linear-gradient(30deg,rgba(245,180,0,.10) 0px,rgba(245,180,0,.10) 2px,transparent 2px,transparent 72px),
repeating-linear-gradient(150deg,rgba(245,180,0,.08) 0px,rgba(245,180,0,.08) 2px,transparent 2px,transparent 72px),#050505!important;color:var(--w)}
.block-container{padding-top:2rem;padding-bottom:4rem;max-width:1220px} h1,h2,h3,p,label,span,div{color:var(--w)}
[data-testid="stSidebar"]{background:rgba(5,5,5,.97);border-right:1px solid var(--border)}
.hero{padding:2rem;border:1px solid var(--border);border-radius:30px;background:linear-gradient(135deg,rgba(245,180,0,.18),rgba(0,0,0,.78));box-shadow:0 20px 70px rgba(0,0,0,.45);margin-bottom:1.4rem}
.kicker{color:var(--y);font-weight:900;letter-spacing:.08em;text-transform:uppercase;font-size:.82rem}.title{font-size:clamp(2.2rem,5vw,4.4rem);font-weight:950;line-height:1.04;margin:.4rem 0}.sub{max-width:850px;color:#eadfb6;font-size:1.05rem;line-height:1.7}
.card{background:var(--card);border:1px solid var(--border);border-radius:26px;padding:1.25rem;box-shadow:0 15px 45px rgba(0,0,0,.35);min-height:138px;margin-bottom:1rem}.label{color:var(--m);font-weight:800;font-size:.9rem;margin-bottom:.55rem}.value{font-size:clamp(1.6rem,4vw,2.65rem);font-weight:950;line-height:1.1}.note{color:#d6c88f;font-size:.9rem;margin-top:.6rem}
.box{background:rgba(12,12,12,.92);border-left:5px solid var(--y);border-radius:18px;padding:1rem 1.15rem;margin:1rem 0;color:#f0e7c5;line-height:1.65}.warn{background:rgba(245,180,0,.10);border:1px solid rgba(245,180,0,.35);border-radius:20px;padding:1rem 1.2rem;color:#f8e6aa;line-height:1.6}
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,#ffd84d,#c48900)!important;color:#111!important;border:0!important;border-radius:18px!important;font-weight:900!important;padding:.85rem 1.1rem!important}.stTextInput input,.stNumberInput input{background:rgba(255,255,255,.94)!important;color:#111!important;border-radius:14px!important}
.pill{display:inline-block;padding:.42rem .75rem;border-radius:999px;background:rgba(245,180,0,.13);border:1px solid rgba(245,180,0,.32);color:#ffe79a;font-weight:800;margin:.16rem;font-size:.84rem}
@media(max-width:768px){.block-container{padding-left:1rem;padding-right:1rem}.hero{padding:1.2rem;border-radius:22px}.card{padding:1rem;border-radius:20px;min-height:118px}}
</style>
""",
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = ["Year", "Revenue", "Net Profit", "EPS", "PE"]
OPTIONAL_COLUMNS = ["EBITDA", "Operating Margin", "Net Margin", "ROE", "ROCE", "Debt/Equity", "PEG", "EV/EBITDA", "Earnings Yield", "Stock Price"]

def money(x, prefix="₹"):
    if x is None or pd.isna(x): return "—"
    x=float(x)
    if abs(x)>=1e7: return f"{prefix}{x/1e7:,.2f} Cr"
    if abs(x)>=1e5: return f"{prefix}{x/1e5:,.2f} L"
    return f"{prefix}{x:,.2f}"

def num(x, suffix=""):
    if x is None or pd.isna(x): return "—"
    return f"{float(x):,.2f}{suffix}"

def pct(x):
    if x is None or pd.isna(x): return "—"
    return f"{float(x):,.2f}%"

def clean_columns(df):
    mapping={"year":"Year","revenue":"Revenue","sales":"Revenue","net sales":"Revenue","net profit":"Net Profit","profit after tax":"Net Profit","pat":"Net Profit","eps":"EPS","pe":"PE","p e":"PE","price earnings":"PE","peg":"PEG","ev ebitda":"EV/EBITDA","ev/ebitda":"EV/EBITDA","earnings yield":"Earnings Yield","ebitda":"EBITDA","operating margin":"Operating Margin","opm":"Operating Margin","net margin":"Net Margin","npm":"Net Margin","roe":"ROE","roce":"ROCE","debt equity":"Debt/Equity","debt/equity":"Debt/Equity","stock price":"Stock Price","price":"Stock Price","close":"Stock Price"}
    rename={}
    for c in df.columns:
        k=str(c).strip().lower().replace("_"," ").replace("-"," ")
        rename[c]=mapping.get(k,str(c).strip())
    return df.rename(columns=rename)

def sample_dataset():
    years=list(range(2005,2025)); rows=[]; rev=10000; margin=9.5; shares=100
    for i,y in enumerate(years):
        growth=.10+.03*math.sin(i/2.5); rev*=(1+growth); margin+=.12*math.sin(i/3)
        profit=rev*margin/100; eps=profit/shares; pe=max(8,26+6*math.sin(i/2)+i*.15)
        rows.append({"Year":y,"Revenue":rev,"Net Profit":profit,"EPS":eps,"PE":pe,"PEG":1.2+i*.035,"EV/EBITDA":14+i*.4,"Earnings Yield":100/pe,"ROE":14+i*.4,"ROCE":16+i*.45,"Debt/Equity":.45-i*.01})
    d=pd.DataFrame(rows); d["Stock Price"]=d["EPS"]*d["PE"]; d["Net Margin"]=d["Net Profit"]/d["Revenue"]*100; d["Operating Margin"]=d["Net Margin"]+4; d["EBITDA"]=d["Revenue"]*d["Operating Margin"]/100
    return d

def prepare_dataset(df):
    df=clean_columns(df).copy()
    for c in REQUIRED_COLUMNS+OPTIONAL_COLUMNS:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=REQUIRED_COLUMNS).sort_values("Year").drop_duplicates("Year").reset_index(drop=True)
    df["Year"]=df["Year"].astype(int)
    if "Net Margin" not in df or df["Net Margin"].isna().all(): df["Net Margin"]=df["Net Profit"]/df["Revenue"]*100
    if "Earnings Yield" not in df or df["Earnings Yield"].isna().all(): df["Earnings Yield"]=100/df["PE"].replace(0,np.nan)
    if "Stock Price" not in df or df["Stock Price"].isna().all(): df["Stock Price"]=df["EPS"]*df["PE"]
    for c in OPTIONAL_COLUMNS:
        if c not in df: df[c]=np.nan
    return df

def validate_dataset(df):
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: return False, f"Missing required columns: {', '.join(missing)}"
    if len(df)<8: return False, "Minimum 8 years of data required. For strong XGBoost results, use 12-20 years."
    return True,"OK"

def add_features(df):
    d=df.copy()
    d["Revenue Growth"]=d["Revenue"].pct_change()*100; d["Profit Growth"]=d["Net Profit"].pct_change()*100; d["EPS Growth"]=d["EPS"].pct_change()*100
    d["Revenue CAGR 3Y"]=((d["Revenue"]/d["Revenue"].shift(3))**(1/3)-1)*100; d["Revenue CAGR 5Y"]=((d["Revenue"]/d["Revenue"].shift(5))**(1/5)-1)*100
    d["EPS CAGR 3Y"]=((d["EPS"]/d["EPS"].shift(3))**(1/3)-1)*100; d["EPS CAGR 5Y"]=((d["EPS"]/d["EPS"].shift(5))**(1/5)-1)*100
    d["PE Median 5Y"]=d["PE"].rolling(5,min_periods=3).median(); d["PE Median 10Y"]=d["PE"].rolling(10,min_periods=5).median()
    d["EVEBITDA Median 5Y"]=d["EV/EBITDA"].rolling(5,min_periods=3).median(); d["Earnings Yield Median 5Y"]=d["Earnings Yield"].rolling(5,min_periods=3).median()
    d["Margin Trend"]=d["Net Margin"]-d["Net Margin"].shift(3); d["ROE Trend"]=d["ROE"]-d["ROE"].shift(3); d["ROCE Trend"]=d["ROCE"]-d["ROCE"].shift(3)
    d["Next Revenue"]=d["Revenue"].shift(-1); d["Next Net Profit"]=d["Net Profit"].shift(-1); d["Next EPS"]=d["EPS"].shift(-1)
    return d.replace([np.inf,-np.inf],np.nan)

FEATURES=["Year","Revenue","Net Profit","EPS","PE","PEG","EV/EBITDA","Earnings Yield","Net Margin","Operating Margin","ROE","ROCE","Debt/Equity","Revenue Growth","Profit Growth","EPS Growth","Revenue CAGR 3Y","Revenue CAGR 5Y","EPS CAGR 3Y","EPS CAGR 5Y","PE Median 5Y","PE Median 10Y","EVEBITDA Median 5Y","Earnings Yield Median 5Y","Margin Trend","ROE Trend","ROCE Trend"]

def fill_features(d):
    d=d.copy()
    for c in FEATURES:
        if c not in d: d[c]=np.nan
    d[FEATURES]=d[FEATURES].replace([np.inf,-np.inf],np.nan)
    d[FEATURES]=d[FEATURES].fillna(d[FEATURES].median(numeric_only=True)).fillna(0)
    return d

def get_model():
    if XGBOOST_AVAILABLE:
        return XGBRegressor(n_estimators=260,max_depth=3,learning_rate=.045,subsample=.88,colsample_bytree=.88,objective="reg:squarederror",random_state=42)
    return RandomForestRegressor(n_estimators=300,max_depth=4,random_state=42)

def train_predict(d,target):
    train=d.dropna(subset=[target]).copy(); train=fill_features(train)
    X=train[FEATURES]; y=train[target]
    if len(train)<8:
        return {"prediction":float(d[target.replace("Next ","")].iloc[-1]),"model_name":"Fallback trend","r2":np.nan,"mape":np.nan}
    test_size=max(2,len(train)//5)
    Xtr,ytr=X.iloc[:-test_size],y.iloc[:-test_size]; Xte,yte=X.iloc[-test_size:],y.iloc[-test_size:]
    model=get_model(); model.fit(Xtr,ytr); pred=model.predict(Xte)
    try: mape=mean_absolute_percentage_error(yte,pred)*100
    except Exception: mape=np.nan
    try: r2=r2_score(yte,pred)
    except Exception: r2=np.nan
    final=get_model(); final.fit(X,y)
    latest=fill_features(d.tail(1))[FEATURES]
    return {"prediction":max(0,float(final.predict(latest)[0])),"model_name":"XGBoost" if XGBOOST_AVAILABLE else "Random Forest","r2":r2,"mape":mape}

def valuation_benchmarks(df):
    pe=df["PE"].dropna(); ev=df["EV/EBITDA"].dropna(); peg=df["PEG"].dropna(); ey=df["Earnings Yield"].dropna()
    def med(s,n): return float(s.tail(min(n,len(s))).median()) if len(s) else np.nan
    return {"pe_20":med(pe,20),"pe_10":med(pe,10),"pe_5":med(pe,5),"pe_low_5":float(pe.tail(min(5,len(pe))).quantile(.25)) if len(pe) else np.nan,"pe_high_5":float(pe.tail(min(5,len(pe))).quantile(.75)) if len(pe) else np.nan,"ev_10":med(ev,10),"peg_10":med(peg,10),"ey_10":med(ey,10)}

def scenario_engine(base,bench,latest_price,current_year):
    base_rev=base["revenue"]["prediction"]; base_profit=base["profit"]["prediction"]; base_eps=base["eps"]["prediction"]
    pe_base=bench.get("pe_10") if not pd.isna(bench.get("pe_10")) else bench.get("pe_5",18)
    pe_bear=bench.get("pe_low_5") if not pd.isna(bench.get("pe_low_5")) else pe_base*.80
    pe_bull=bench.get("pe_high_5") if not pd.isna(bench.get("pe_high_5")) else pe_base*1.20
    rows=[{"Scenario":"🐻 Bear","Forecast Year":current_year+1,"Revenue":base_rev*.85,"Net Profit":base_profit*.80,"EPS":base_eps*.80,"Expected PE":pe_bear*.95},{"Scenario":"⚖️ Base","Forecast Year":current_year+1,"Revenue":base_rev,"Net Profit":base_profit,"EPS":base_eps,"Expected PE":pe_base},{"Scenario":"🚀 Bull","Forecast Year":current_year+1,"Revenue":base_rev*1.15,"Net Profit":base_profit*1.20,"EPS":base_eps*1.20,"Expected PE":pe_bull*1.05}]
    out=pd.DataFrame(rows); out["Estimated Price"]=out["EPS"]*out["Expected PE"]; out["Upside/Downside"]=((out["Estimated Price"]/latest_price)-1)*100 if latest_price else np.nan
    return out

def confidence_score(df,outs):
    data_score=min(45,len(df)/20*45); mapes=[outs[k]["mape"] for k in outs if not pd.isna(outs[k].get("mape",np.nan))]
    error_score=max(0,35-min(35,np.nanmean(mapes))) if mapes else 15
    cols=["Revenue","Net Profit","EPS","PE","ROE","ROCE","EV/EBITDA","PEG"]; available=sum(1 for c in cols if c in df and df[c].notna().sum()>=min(8,len(df)))
    score=int(round(data_score+error_score+available/len(cols)*20)); score=max(0,min(100,score))
    return score,"High" if score>=80 else "Medium" if score>=60 else "Low"

def line_chart(df,cols,title):
    fig=go.Figure()
    for c in cols:
        if c in df: fig.add_trace(go.Scatter(x=df["Year"],y=df[c],mode="lines+markers",name=c))
    fig.update_layout(title=title,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#fffbea"),margin=dict(l=10,r=10,t=55,b=30),legend=dict(orientation="h"),height=360)
    return fig

def price_chart(s):
    fig=go.Figure(); fig.add_trace(go.Bar(x=s["Scenario"],y=s["Estimated Price"],text=s["Estimated Price"].round(2),textposition="outside"))
    fig.update_layout(title="Estimated Stock Price Range",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#fffbea"),margin=dict(l=10,r=10,t=55,b=30),height=360)
    return fig

with st.sidebar:
    st.markdown("## 🐝 Financify")
    st.caption("Hive Forecast Engine")
    if st.session_state.get("user_email"): st.success(st.session_state["user_email"])
    if logout and st.button("🚪 Logout",use_container_width=True): logout()
    st.markdown("---")
    st.markdown("### Required CSV columns")
    st.caption("Year, Revenue, Net Profit, EPS, PE")
    st.markdown("### Recommended")
    st.caption("PEG, EV/EBITDA, Earnings Yield, ROE, ROCE, Debt/Equity, Stock Price")

st.markdown('<div class="hero"><div class="kicker">Financify SaaS Tool</div><div class="title">🐝 Hive Forecast Engine</div><div class="sub">Forecast next year Revenue, Net Profit, EPS and valuation-based stock price range using long-term fundamentals, XGBoost and a disciplined Bear / Base / Bull scenario engine.</div></div>',unsafe_allow_html=True)
st.markdown('<div class="box"><b>How this tool thinks:</b><br>XGBoost forecasts business fundamentals first — Revenue, Net Profit and EPS. Then the valuation engine applies historical PE benchmarks to estimate a reasonable stock price range.</div>',unsafe_allow_html=True)

left,right=st.columns([1.25,.75],gap="large")
with left:
    uploaded=st.file_uploader("Upload long-term annual financial dataset CSV",type=["csv"],help="Minimum 8 years, ideal 15-20 years.")
with right:
    use_sample=st.toggle("Use demo sample dataset",value=False)
    latest_manual_price=st.number_input("Latest Stock Price",min_value=0.0,value=0.0,step=1.0,help="Leave 0 if CSV has Stock Price.")

if uploaded: raw_df=pd.read_csv(uploaded)
elif use_sample: raw_df=sample_dataset()
else:
    st.markdown('<div class="warn">Upload a CSV dataset to start. For Indian stocks, do not depend only on Yahoo Finance because it usually gives only 4-5 years of annual fundamentals. Use Screener-style exports, your own Financify database, or a paid data API.</div>',unsafe_allow_html=True)
    st.download_button("Download Sample CSV Template",sample_dataset().to_csv(index=False).encode("utf-8"),file_name="financify_forecast_template.csv",mime="text/csv")
    st.stop()

df=prepare_dataset(raw_df); valid,msg=validate_dataset(df)
if not valid:
    st.error(msg); st.dataframe(df,use_container_width=True); st.stop()

feature_df=add_features(df); bench=valuation_benchmarks(df)
model_outputs={"revenue":train_predict(feature_df,"Next Revenue"),"profit":train_predict(feature_df,"Next Net Profit"),"eps":train_predict(feature_df,"Next EPS")}
latest_year=int(df["Year"].max()); latest_price=float(latest_manual_price) if latest_manual_price>0 else float(df["Stock Price"].dropna().iloc[-1])
scenarios=scenario_engine(model_outputs,bench,latest_price,latest_year); conf_score,conf_label=confidence_score(df,model_outputs)

st.markdown("## Snapshot")
c1,c2,c3=st.columns(3)
with c1: st.markdown(f'<div class="card"><div class="label">Financial History Used</div><div class="value">{len(df)} Years</div><div class="note">Ideal: 15-20 years</div></div>',unsafe_allow_html=True)
with c2: st.markdown(f'<div class="card"><div class="label">Forecast Model</div><div class="value">{model_outputs["revenue"]["model_name"]}</div><div class="note">Fundamentals first, valuation second</div></div>',unsafe_allow_html=True)
with c3: st.markdown(f'<div class="card"><div class="label">Forecast Confidence</div><div class="value">{conf_score}/100</div><div class="note">{conf_label} confidence</div></div>',unsafe_allow_html=True)

c4,c5,c6=st.columns(3)
with c4: st.markdown(f'<div class="card"><div class="label">Latest Price Used</div><div class="value">{money(latest_price)}</div><div class="note">Manual input or CSV</div></div>',unsafe_allow_html=True)
with c5: st.markdown(f'<div class="card"><div class="label">Base EPS Forecast</div><div class="value">{num(model_outputs["eps"]["prediction"])}</div><div class="note">Predicted next year EPS</div></div>',unsafe_allow_html=True)
with c6: st.markdown(f'<div class="card"><div class="label">Base Revenue Forecast</div><div class="value">{money(model_outputs["revenue"]["prediction"])}</div><div class="note">Predicted next year revenue</div></div>',unsafe_allow_html=True)

st.markdown("## Scenario Forecast")
display=scenarios.copy(); display["Revenue"]=display["Revenue"].apply(money); display["Net Profit"]=display["Net Profit"].apply(money); display["EPS"]=display["EPS"].apply(num); display["Expected PE"]=display["Expected PE"].apply(lambda x:num(x,"x")); display["Estimated Price"]=display["Estimated Price"].apply(money); display["Upside/Downside"]=display["Upside/Downside"].apply(pct)
st.dataframe(display,use_container_width=True,hide_index=True)
st.plotly_chart(price_chart(scenarios),use_container_width=True)

st.markdown("## Valuation Benchmarks")
b1,b2,b3=st.columns(3)
with b1: st.markdown(f'<div class="card"><div class="label">20Y Median PE</div><div class="value">{num(bench["pe_20"],"x")}</div><div class="note">Uses available years up to 20</div></div>',unsafe_allow_html=True)
with b2: st.markdown(f'<div class="card"><div class="label">10Y Median PE</div><div class="value">{num(bench["pe_10"],"x")}</div><div class="note">Base valuation anchor</div></div>',unsafe_allow_html=True)
with b3: st.markdown(f'<div class="card"><div class="label">5Y Median PE</div><div class="value">{num(bench["pe_5"],"x")}</div><div class="note">Recent valuation behaviour</div></div>',unsafe_allow_html=True)
st.markdown(f'<span class="pill">10Y EV/EBITDA: {num(bench["ev_10"],"x")}</span><span class="pill">10Y PEG: {num(bench["peg_10"],"x")}</span><span class="pill">10Y Earnings Yield: {pct(bench["ey_10"])}</span>',unsafe_allow_html=True)

st.markdown("## Historical Fundamentals")
g1,g2=st.columns(2)
with g1: st.plotly_chart(line_chart(df,["Revenue","Net Profit"],"Revenue and Net Profit"),use_container_width=True)
with g2: st.plotly_chart(line_chart(df,["EPS","PE"],"EPS and PE Trend"),use_container_width=True)

st.markdown("## Model Quality")
quality=[]
for key,label in [("revenue","Revenue Model"),("profit","Net Profit Model"),("eps","EPS Model")]:
    quality.append({"Model":label,"Algorithm":model_outputs[key]["model_name"],"Test MAPE":pct(model_outputs[key]["mape"]),"Test R²":num(model_outputs[key]["r2"])})
st.dataframe(pd.DataFrame(quality),use_container_width=True,hide_index=True)
st.markdown('<div class="box"><b>Important:</b> XGBoost works best with clean 15-20 year data. With only 8-10 rows, treat the output as a structured estimate, not a prediction.</div>',unsafe_allow_html=True)

latest=df.iloc[-1]
reasons=[]
if feature_df["Revenue CAGR 5Y"].notna().any(): reasons.append(f'Revenue 5Y CAGR is {feature_df["Revenue CAGR 5Y"].dropna().iloc[-1]:.2f}%, showing long-term sales trend.')
if feature_df["EPS CAGR 5Y"].notna().any(): reasons.append(f'EPS 5Y CAGR is {feature_df["EPS CAGR 5Y"].dropna().iloc[-1]:.2f}%, which directly affects valuation.')
if not pd.isna(latest["PE"]) and not pd.isna(bench["pe_10"]): reasons.append("Current PE is below the 10Y median PE, so valuation is not stretched versus history." if latest["PE"]<bench["pe_10"] else "Current PE is above the 10Y median PE, so valuation risk is higher.")
if df["ROCE"].notna().any(): reasons.append(f'Latest ROCE is {df["ROCE"].dropna().iloc[-1]:.2f}%, useful for judging business quality.')
why="<br>".join([f"✓ {r}" for r in reasons]) if reasons else "Add ROE, ROCE, PEG and EV/EBITDA to improve explanation quality."
st.markdown("## Why This Forecast Looks This Way")
st.markdown(f'<div class="box">{why}</div>',unsafe_allow_html=True)

with st.expander("See raw annual dataset"):
    st.dataframe(df,use_container_width=True)
st.download_button("Download Forecast CSV",scenarios.to_csv(index=False).encode("utf-8"),file_name="hive_forecast_engine_output.csv",mime="text/csv")
st.markdown('<div class="warn"><b>Disclaimer:</b> This is an educational forecasting tool. It is not investment advice, not a buy/sell recommendation, and not a guarantee of future price. Always verify financial statements and valuation assumptions manually.</div>',unsafe_allow_html=True)
