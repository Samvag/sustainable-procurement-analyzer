# app.py
# Sustainable Procurement & Vendor Cost Analyzer (Sidebar + AI Insights + Agent 1: Supplier Risk Monitor)
# -----------------------------------------------------------------------------------------------------
# Pages:
#  - Category Overview
#  - Scenarios
#  - Opportunities
#  - AI Insights
#  - Agent: Supplier Risk Monitor   <-- NEW
#  - About
#
# Agent 1 capabilities:
#  - Upload (or auto-generate) supplier Events/News CSV
#  - Keyword-based risk scoring with recency/severity weights
#  - Supplier-level alerts (High/Med/Low) + suggested alternatives
#  - CSV export of alerts
#  - AI summary (uses OpenAI if key present; otherwise offline narrative)
#
# Notes:
#  - No background scheduler here (Streamlit Cloud limitation). Run agent page on demand.
#  - Add OPENAI_API_KEY in Streamlit Secrets (optional) for live AI narratives.

import os
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------
# ---------- THEME -----------
# ----------------------------
st.set_page_config(
    page_title="Sustainable Procurement & Vendor Cost Analyzer",
    page_icon="♻️",
    layout="wide"
)

st.title("Sustainable Procurement & Vendor Cost Analyzer")
st.caption("Bridge sustainability with finance — quantify Scope 3 + freight, cost levers, and quick wins.")

# ----------------------------
# ---- CONFIG / FACTORS ------
# ----------------------------

# Freight emissions factors (kg CO2e per tonne-km), illustrative
FREIGHT_EF = {
    "Air": 0.5,
    "Road": 0.12,
    "Rail": 0.03,
    "Sea": 0.015,
}

# Freight cost per tonne-km (USD), illustrative
FREIGHT_COST = {
    "Air": 1.50,
    "Road": 0.07,
    "Rail": 0.04,
    "Sea": 0.02,
}

# Purchased-goods emission factors per kg (kg CO2e/kg), illustrative
CATEGORY_PG_EF = {
    "Packaging-Plastic": 3.0,
    "Packaging-Glass": 1.5,
    "Raw-Chemicals": 4.0,
    "Fragrance": 2.5,
    "Colorants": 5.0,
    "Paper-Board": 1.2,
}

# Recycled content relative reduction (illustrative average multipliers)
RECYCLED_REDUCTION_MULTIPLIER = {
    "Packaging-Plastic": 0.65,
    "Packaging-Glass": 0.45,
    "Raw-Chemicals": 0.30,
    "Fragrance": 0.20,
    "Colorants": 0.25,
    "Paper-Board": 0.55,
}

# ----------------------------
# ------ SAMPLE DATA ----------
# ----------------------------

def build_sample_data(n_suppliers: int = 28, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    categories = list(CATEGORY_PG_EF.keys())
    modes = list(FREIGHT_EF.keys())

    rows = []
    for i in range(n_suppliers):
        cat = np.random.choice(categories)
        mode = np.random.choice(modes, p=[0.05, 0.6, 0.2, 0.15])  # mostly Road
        supplier = f"Supplier-{i+1:02d}"
        spend = float(np.random.randint(50_000, 600_000))          # Annual spend (USD)
        weight_kg = float(np.random.randint(6_000, 75_000))        # Annual purchased mass (kg)
        dist_km = float(np.random.choice(
            [120, 250, 400, 800, 1200, 1800, 2500],
            p=[0.12, 0.2, 0.2, 0.18, 0.14, 0.1, 0.06]
        ))
        rows.append({
            "Supplier": supplier,
            "Category": cat,
            "Annual_Spend_USD": spend,
            "Weight_kg": weight_kg,
            "Distance_km": dist_km,
            "Mode": mode,
        })
    return pd.DataFrame(rows)

# ----------------------------
# ------ CORE COMPUTE --------
# ----------------------------

def compute_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Tonne_km"] = (df["Weight_kg"] / 1000.0) * df["Distance_km"]
    df["Freight_Emissions_kgCO2e"] = df.apply(
        lambda r: r["Tonne_km"] * FREIGHT_EF.get(str(r["Mode"]), 0.12), axis=1
    )
    df["Freight_Cost_USD"] = df.apply(
        lambda r: r["Tonne_km"] * FREIGHT_COST.get(str(r["Mode"]), 0.07), axis=1
    )
    # Purchased goods emissions (baseline)
    df["PurchasedGoods_Emissions_kgCO2e"] = df.apply(
        lambda r: (CATEGORY_PG_EF.get(str(r["Category"]), 2.0)) * r["Weight_kg"], axis=1
    )
    return df

def scenario_localization(df: pd.DataFrame, localize_pct: float, long_haul_km=1000, new_km=200) -> pd.DataFrame:
    df = df.copy()
    mask = df["Distance_km"] > long_haul_km
    candidates = df[mask].copy()
    if candidates.empty or localize_pct <= 0:
        df["Freight_Cost_Saved_USD"] = 0.0
        df["Freight_Emissions_Saved_kgCO2e"] = 0.0
        df["Localized"] = "No"
        df["Tonne_km_after_loc"] = df["Tonne_km"]
        df["Freight_Cost_USD_after"] = df["Freight_Cost_USD"]
        df["Freight_Emissions_kgCO2e_after"] = df["Freight_Emissions_kgCO2e"]
        return df

    candidates = candidates.sort_values("Annual_Spend_USD", ascending=False)
    n_select = max(1, int(round(len(candidates) * (localize_pct / 100.0))))
    selected_suppliers = set(candidates.head(n_select)["Supplier"])

    def _tonne_km_after(row):
        if row["Supplier"] in selected_suppliers:
            return (row["Weight_kg"] / 1000.0) * new_km
        return row["Tonne_km"]

    df["Tonne_km_after_loc"] = df.apply(_tonne_km_after, axis=1)
    df["Freight_Cost_USD_after"] = df.apply(
        lambda r: r["Tonne_km_after_loc"] * FREIGHT_COST.get(str(r["Mode"]), 0.07), axis=1
    )
    df["Freight_Emissions_kgCO2e_after"] = df.apply(
        lambda r: r["Tonne_km_after_loc"] * FREIGHT_EF.get(str(r["Mode"]), 0.12), axis=1
    )

    df["Freight_Cost_Saved_USD"] = (df["Freight_Cost_USD"] - df["Freight_Cost_USD_after"]).clip(lower=0)
    df["Freight_Emissions_Saved_kgCO2e"] = (df["Freight_Emissions_kgCO2e"] - df["Freight_Emissions_kgCO2e_after"]).clip(lower=0)
    df["Localized"] = df["Supplier"].apply(lambda s: "Yes" if s in selected_suppliers else "No")
    return df

def scenario_recycled(df: pd.DataFrame, recycled_pct: float) -> pd.DataFrame:
    df = df.copy()
    rp = max(0.0, min(100.0, recycled_pct)) / 100.0

    def _pg_after(row):
        base_ef = CATEGORY_PG_EF.get(str(row["Category"]), 2.0)
        mult = RECYCLED_REDUCTION_MULTIPLIER.get(str(row["Category"]), 0.3)
        effective_ef = base_ef * (1.0 - rp * mult)
        return effective_ef * row["Weight_kg"]

    df["PurchasedGoods_Emissions_kgCO2e_after"] = df.apply(_pg_after, axis=1)
    df["PurchasedGoods_Emissions_Saved_kgCO2e"] = (
        df["PurchasedGoods_Emissions_kgCO2e"] - df["PurchasedGoods_Emissions_kgCO2e_after"]
    ).clip(lower=0)
    return df

def tail_spend_consolidation(df: pd.DataFrame, tail_share: float = 0.2, saving_rate: float = 0.03) -> pd.DataFrame:
    df = df.copy()
    by_supplier = df.groupby("Supplier", as_index=False).agg(
        Supplier_Spend=("Annual_Spend_USD", "sum"),
        Supplier_Count=("Supplier", "count"),
        Weight_kg=("Weight_kg", "sum"),
        Category=("Category", "first")
    ).sort_values("Supplier_Spend", ascending=True)

    n_tail = max(1, int(round(len(by_supplier) * tail_share)))
    tail = by_supplier.head(n_tail).copy()
    tail["Indicative_Consolidation_Saving_USD"] = tail["Supplier_Spend"] * saving_rate
    return tail

# ----------------------------
# ---------- AI LAYER --------
# ----------------------------

CASE_LIBRARY = [
    {
        "sector": "CPG",
        "pattern": "supplier localization",
        "impact": "8–15% freight CO₂e reduction; $40k–$120k logistics savings per BU",
        "source": "Industry logistics case summaries (illustrative)"
    },
    {
        "sector": "Packaging",
        "pattern": "recycled resin (20–30%)",
        "impact": "10–25% purchased-goods CO₂e reduction; cost impact neutral to +5%",
        "source": "Circular economy meta-studies (illustrative)"
    },
    {
        "sector": "Chemicals",
        "pattern": "tail-spend consolidation",
        "impact": "~3% saving on tail spend; improved OTIF and standardization",
        "source": "Procurement benchmarks (illustrative)"
    },
]

def build_ai_context(df, df_loc=None, df_rec=None, tail_df=None):
    ctx = {}
    ctx["total_spend_usd"] = float(df["Annual_Spend_USD"].sum())
    ctx["freight_cost_usd"] = float(df["Freight_Cost_USD"].sum())
    ctx["freight_kg"] = float(df["Freight_Emissions_kgCO2e"].sum())
    ctx["purchased_kg"] = float(df["PurchasedGoods_Emissions_kgCO2e"].sum())
    ctx["top_categories"] = (
        df.groupby("Category", as_index=False)
          .agg(Spend_USD=("Annual_Spend_USD","sum"),
               Freight_kg=("Freight_Emissions_kgCO2e","sum"),
               Purchased_kg=("PurchasedGoods_Emissions_kgCO2e","sum"))
          .sort_values("Spend_USD", ascending=False)
          .head(5)
          .to_dict(orient="records")
    )
    if df_loc is not None and "Freight_Cost_Saved_USD" in df_loc.columns:
        ctx["loc_savings_usd"] = float(df_loc["Freight_Cost_Saved_USD"].sum())
        ctx["loc_avoided_kg"] = float(df_loc["Freight_Emissions_Saved_kgCO2e"].sum())
    if df_rec is not None and "PurchasedGoods_Emissions_Saved_kgCO2e" in df_rec.columns:
        ctx["recycled_avoided_kg"] = float(df_rec["PurchasedGoods_Emissions_Saved_kgCO2e"].sum())
    if tail_df is not None and not tail_df.empty:
        ctx["tail_savings_usd"] = float(tail_df["Indicative_Consolidation_Saving_USD"].sum())
    return ctx

def format_ctx_for_prompt(ctx: dict) -> str:
    lines = []
    lines.append(f"Total spend (USD): {ctx.get('total_spend_usd', 0):,.0f}")
    lines.append(f"Freight cost (USD): {ctx.get('freight_cost_usd', 0):,.0f}")
    lines.append(f"Freight emissions (kg): {ctx.get('freight_kg', 0):,.0f}")
    lines.append(f"Purchased-goods emissions (kg): {ctx.get('purchased_kg', 0):,.0f}")
    if "loc_savings_usd" in ctx or "loc_avoided_kg" in ctx:
        lines.append(f"Localization savings (USD): {ctx.get('loc_savings_usd', 0):,.0f}")
        lines.append(f"Localization avoided (kg): {ctx.get('loc_avoided_kg', 0):,.0f}")
    if "recycled_avoided_kg" in ctx:
        lines.append(f"Recycled-content avoided (kg): {ctx.get('recycled_avoided_kg', 0):,.0f}")
    if "tail_savings_usd" in ctx:
        lines.append(f"Tail-spend consolidation savings (USD): {ctx.get('tail_savings_usd', 0):,.0f}")
    cats = ctx.get("top_categories", [])
    if cats:
        lines.append("Top categories (by spend):")
        for c in cats:
            lines.append(
                f"- {c.get('Category','Unknown')}: Spend ${c.get('Spend_USD',0):,.0f}, "
                f"Freight kg {c.get('Freight_kg',0):,.0f}, Purchased kg {c.get('Purchased_kg',0):,.0f}"
            )
    return "\n".join(lines)

def offline_insights(ctx: dict) -> str:
    bullets = []
    if ctx.get("loc_savings_usd", 0) > 0 or ctx.get("loc_avoided_kg", 0) > 0:
        bullets.append(
            f"**Localize long-haul lanes**: realize ≈ ${ctx.get('loc_savings_usd',0):,.0f} freight savings and avoid ≈ {ctx.get('loc_avoided_kg',0):,.0f} kg CO₂e."
        )
    if ctx.get("recycled_avoided_kg", 0) > 0:
        bullets.append(
            f"**Adopt 20–30% recycled content**: avoid ≈ {ctx.get('recycled_avoided_kg',0):,.0f} kg CO₂e in purchased goods; pilot in top categories."
        )
    if ctx.get("tail_savings_usd", 0) > 0:
        bullets.append(
            f"**Consolidate tail suppliers**: target ≈ ${ctx.get('tail_savings_usd',0):,.0f}/yr; standardize SLAs/specs."
        )
    if ctx.get("freight_cost_usd", 0) > 0:
        bullets.append(
            f"**Lane optimization**: benchmark freight spend (${ctx.get('freight_cost_usd',0):,.0f}) vs peers; evaluate mode shifts."
        )
    bullets.append("**Governance**: publish a monthly KPI pack and track pilot ROI by BU.")
    header = ":bulb: **AI Insights (Offline Demo Mode)**"
    return header + "\n" + "\n".join([f"- {b}" for b in bullets]) if bullets else header + "\n- No scenario impacts yet; adjust sliders in *Scenarios*."

def generate_ai_insights(ctx: dict) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return offline_insights(ctx)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        case_snips = [
            f"- [{c['sector']}] {c['pattern']}: {c['impact']} (Source: {c['source']})"
            for c in CASE_LIBRARY
        ]
        system_msg = (
            "You are a senior sustainability-finance analyst. "
            "Write concise, CFO-ready insights that tie actions to cost savings and Scope 3 reductions. "
            "Use bullets. Avoid hallucinations."
        )
        user_msg = (
            "DATA SUMMARY:\n"
            f"{format_ctx_for_prompt(ctx)}\n\n"
            "REFERENCE CASE PATTERNS:\n"
            f"{'\n'.join(case_snips)}\n\n"
            "TASK: Provide 5 bullet insights with quantified impact and next actions."
        )
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_msg},
                      {"role":"user","content":user_msg}],
            temperature=0.2,
        )
        return ":robot_face: **AI Insights (Live)**\n" + completion.choices[0].message.content.strip()
    except Exception as e:
        return f":warning: AI service unavailable (offline insights shown). Error: {e}\n\n" + offline_insights(ctx)

# ----------------------------
# ----- AGENT 1 (NEW) --------
# ----------------------------

# Default keywords with weights (editable in UI)
DEFAULT_KEYWORDS = {
    "violation": 3.0,
    "non-compliance": 3.0,
    "fine": 2.5,
    "recall": 2.5,
    "strike": 2.0,
    "shutdown": 3.0,
    "accident": 2.5,
    "spill": 3.0,
    "pfas": 2.0,
    "deforestation": 2.5,
    "labor": 2.0,
    "sanction": 2.5,
    "litigation": 2.0,
    "emissions up": 2.5,
    "co2 up": 2.0,
    "audit fail": 3.0
}

def build_sample_events(suppliers: pd.Series, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    samples = []
    news_pool = [
        ("minor process deviation noted in audit", 0.5),
        ("OSHA violation reported; corrective action required", 3.0),
        ("temporary line shutdown for maintenance", 1.0),
        ("PFAS found in legacy waste stream; remediation plan filed", 2.5),
        ("labor dispute resolved; no further action", 1.0),
        ("transport spill; contained with no injuries", 2.5),
        ("emissions up 12% vs last quarter", 2.0),
        ("supplier passed EcoVadis audit with bronze rating", 0.5),
        ("regulatory fine levied for late reporting", 2.0),
        ("factory expansion; increased capacity", 0.5),
    ]
    today = datetime.utcnow().date()
    for s in suppliers.sample(min(12, len(suppliers)), replace=False):
        title, sev = news_pool[np.random.randint(0, len(news_pool))]
        days_ago = int(np.random.randint(0, 90))
        samples.append({
            "Supplier": s,
            "Date": (today - timedelta(days=days_ago)).isoformat(),
            "Title": title.capitalize(),
            "Summary": title,
            "Source": "sample-news",
            "Severity": sev
        })
    return pd.DataFrame(samples)

def score_events(df_events: pd.DataFrame, keywords: dict, recency_half_life_days: float = 45.0) -> pd.DataFrame:
    """
    Score each event by:
      - keyword hits (sum of weights found in Title+Summary)
      - severity column if present
      - recency decay (half-life)
    Returns df with Event_Score column.
    """
    df = df_events.copy()
    for col in ["Title", "Summary"]:
        if col not in df.columns:
            df[col] = ""

    def kw_score(text: str) -> float:
        t = (text or "").lower()
        score = 0.0
        for k, w in keywords.items():
            if k in t:
                score += float(w)
        return score

    today = datetime.utcnow().date()

    def recency_weight(date_str: str) -> float:
        try:
            d = datetime.fromisoformat(str(date_str)).date()
            days = (today - d).days
            # half-life decay
            return 0.5 ** (days / max(1.0, recency_half_life_days))
        except Exception:
            return 1.0

    df["Keyword_Score"] = (df["Title"].fillna("") + " " + df["Summary"].fillna("")).apply(kw_score)
    df["Severity_Score"] = df["Severity"].astype(float) if "Severity" in df.columns else 1.0
    df["Recency_W"] = df["Date"].apply(recency_weight) if "Date" in df.columns else 1.0
    df["Event_Score"] = (df["Keyword_Score"] + df["Severity_Score"]) * df["Recency_W"]
    return df

def aggregate_supplier_risk(df_events_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up by supplier. Produce Risk_Bucket by simple thresholds.
    """
    if df_events_scored.empty:
        return pd.DataFrame(columns=["Supplier","Event_Count","Risk_Score","Risk_Bucket"])
    agg = df_events_scored.groupby("Supplier", as_index=False).agg(
        Event_Count=("Event_Score","count"),
        Risk_Score=("Event_Score","sum")
    ).sort_values("Risk_Score", ascending=False)
    # Buckets
    conditions = [
        agg["Risk_Score"] >= 6.0,
        agg["Risk_Score"].between(3.0, 6.0, inclusive="left"),
        agg["Risk_Score"] < 3.0
    ]
    buckets = ["High", "Medium", "Low"]
    agg["Risk_Bucket"] = np.select(conditions, buckets, default="Low")
    return agg

def suggest_lower_emission_alternatives(df_base: pd.DataFrame, supplier: str, top_n: int = 3) -> list:
    """
    From same category, pick suppliers with lower PurchasedGoods_Emissions per kg (proxy)
    """
    if supplier not in set(df_base["Supplier"]):
        return []
    row = df_base[df_base["Supplier"] == supplier].iloc[0]
    cat = row["Category"]
    peers = df_base[df_base["Category"] == cat].copy()
    if peers.empty:
        return []
    peers["Emissions_per_kg"] = CATEGORY_PG_EF.get(cat, 2.0)  # constant per category in demo
    # Use distance as additional proxy: closer is preferable
    peers["Distance_rank"] = peers["Distance_km"].rank(method="first")
    # Exclude the current supplier
    peers = peers[peers["Supplier"] != supplier]
    # Sort by shorter Distance (proxy) then spend (capacity)
    peers = peers.sort_values(["Distance_km","Annual_Spend_USD"], ascending=[True, False]).head(top_n)
    return peers["Supplier"].tolist()

def agent_offline_summary(alerts_df: pd.DataFrame) -> str:
    if alerts_df.empty:
        return ":white_check_mark: No supplier risk alerts detected in the latest scan."
    top = alerts_df.head(5)
    lines = [":bulb: **Agent Summary (Offline)**"]
    for _, r in top.iterrows():
        lines.append(f"- **{r['Supplier']}** → *{r['Risk_Bucket']}* risk (score {r['Risk_Score']:.1f}). Recommended: {r['Recommended_Action']}")
    return "\n".join(lines)

def agent_live_summary(alerts_df: pd.DataFrame) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return agent_offline_summary(alerts_df)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        rows = []
        for _, r in alerts_df.head(10).iterrows():
            rows.append(f"- {r['Supplier']} | {r['Risk_Bucket']} | {r['Risk_Score']:.1f} | {r['Recommended_Action']}")
        user_msg = "Summarize these supplier risk alerts for a procurement + ESG audience, with next steps:\n" + "\n".join(rows)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"Be concise, action-oriented, and CFO-friendly."},
                      {"role":"user","content":user_msg}],
            temperature=0.2,
        )
        return ":robot_face: **Agent Summary (Live)**\n" + completion.choices[0].message.content.strip()
    except Exception as e:
        return f":warning: AI unavailable; offline summary shown. Error: {e}\n\n" + agent_offline_summary(alerts_df)

# ----------------------------
# ---------- SIDEBAR ---------
# ----------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Category Overview", "Scenarios", "Opportunities", "AI Insights", "Agent: Supplier Risk Monitor", "About"],
        index=0
    )
    st.markdown("---")
    st.subheader("Data & Assumptions")
    file = st.file_uploader("Upload dataset (CSV)", type=["csv"], help="Columns: Supplier, Category, Annual_Spend_USD, Weight_kg, Distance_km, Mode")
    long_haul_km = st.number_input("Long-haul threshold (km)", min_value=200, max_value=5000, value=1000, step=50)
    localized_km = st.number_input("Localized distance (km)", min_value=50, max_value=1000, value=200, step=10)
    tail_share = st.slider("Tail supplier share (by count)", 0.05, 0.5, 0.20, 0.05)

# Load data
if file is not None:
    try:
        df_raw = pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df_raw = build_sample_data()
else:
    df_raw = build_sample_data()

# Compute baseline and KPIs (shared across pages)
df = compute_core_metrics(df_raw)
total_spend = df["Annual_Spend_USD"].sum()
total_freight_cost = df["Freight_Cost_USD"].sum()
total_freight_kg = df["Freight_Emissions_kgCO2e"].sum()
total_purchased_kg = df["PurchasedGoods_Emissions_kgCO2e"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Spend (USD)", f"${total_spend:,.0f}")
k2.metric("Freight Cost (USD)", f"${total_freight_cost:,.0f}")
k3.metric("Freight Emissions (kg)", f"{total_freight_kg:,.0f}")
k4.metric("Purchased-Goods Emissions (kg)", f"{total_purchased_kg:,.0f}")
st.markdown("---")

# ----------------------------
# ---------- PAGES -----------
# ----------------------------
if page == "Category Overview":
    col1, col2 = st.columns([1.2, 1])
    by_cat = (
        df.groupby("Category", as_index=False)
          .agg(Spend_USD=("Annual_Spend_USD","sum"),
               Freight_kg=("Freight_Emissions_kgCO2e","sum"),
               Purchased_kg=("PurchasedGoods_Emissions_kgCO2e","sum"))
          .sort_values("Spend_USD", ascending=False)
    )
    fig1 = px.bar(by_cat, x="Category", y="Spend_USD", title="Spend by Category (USD)")
    fig2 = px.bar(by_cat, x="Category", y=["Freight_kg","Purchased_kg"], barmode="group", title="Emissions by Category (kg)")
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df.head(20), use_container_width=True)

elif page == "Scenarios":
    st.subheader("What-if Scenarios")
    s1, s2 = st.columns(2)
    with s1:
        loc_pct = st.slider("Localize % of long-haul suppliers (by count)", 0, 100, 25, 5)
    with s2:
        rec_pct = st.slider("Adopt recycled content across categories (%)", 0, 80, 30, 5)

    df_loc = scenario_localization(df, loc_pct, long_haul_km, localized_km)
    loc_saved_usd = df_loc["Freight_Cost_Saved_USD"].sum()
    loc_avoided_kg = df_loc["Freight_Emissions_Saved_kgCO2e"].sum()

    df_rec = scenario_recycled(df, rec_pct)
    rec_avoided_kg = df_rec["PurchasedGoods_Emissions_Saved_kgCO2e"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Localization: Freight Savings (USD)", f"${loc_saved_usd:,.0f}")
    c2.metric("Localization: Avoided Freight CO₂e (kg)", f"{loc_avoided_kg:,.0f}")
    c3.metric("Recycled Content: Avoided Purchased-Goods CO₂e (kg)", f"{rec_avoided_kg:,.0f}")

    st.markdown("**Localized Suppliers (Yes = distance reduced):**")
    st.dataframe(
        df_loc[["Supplier","Category","Annual_Spend_USD","Mode","Distance_km","Localized",
                "Freight_Cost_Saved_USD","Freight_Emissions_Saved_kgCO2e"]]
        .sort_values("Freight_Cost_Saved_USD", ascending=False),
        use_container_width=True
    )

    sc1, sc2 = st.columns(2)
    by_mode = df.groupby("Mode", as_index=False).agg(Tonne_km=("Tonne_km","sum"))
    sc1.plotly_chart(px.pie(by_mode, names="Mode", values="Tonne_km", title="Baseline Tonne-km by Mode"), use_container_width=True)

    sc2.plotly_chart(px.bar(
        pd.DataFrame({
            "Lever": ["Localization Freight $", "Localization kgCO₂e", "Recycled kgCO₂e"],
            "Value": [loc_saved_usd, loc_avoided_kg, rec_avoided_kg],
            "Units": ["USD", "kg", "kg"]
        }),
        x="Lever", y="Value", title="Scenario Impacts"
    ), use_container_width=True)

    def _to_bytes_csv():
        out = io.StringIO()
        exp = df_loc[["Supplier","Category","Annual_Spend_USD","Mode","Distance_km","Localized",
                      "Freight_Cost_Saved_USD","Freight_Emissions_Saved_kgCO2e"]].copy()
        exp2 = df_rec[["Supplier","PurchasedGoods_Emissions_Saved_kgCO2e"]].copy().rename(
            columns={"PurchasedGoods_Emissions_Saved_kgCO2e":"PG_Emissions_Saved_kgCO2e"})
        merged = exp.merge(exp2, left_index=True, right_index=True, how="left")
        merged.to_csv(out, index=False)
        return out.getvalue().encode("utf-8")

    st.download_button(
        "Download Scenario Results (CSV)",
        data=_to_bytes_csv(),
        file_name=f"scenario_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    st.session_state["ai_ctx"] = build_ai_context(df, df_loc, df_rec, tail_spend_consolidation(df, tail_share, 0.03))

elif page == "Opportunities":
    st.subheader("Tail-Spend Consolidation Opportunities")
    tail = tail_spend_consolidation(df, tail_share=tail_share, saving_rate=0.03)
    total_tail_save = tail["Indicative_Consolidation_Saving_USD"].sum() if not tail.empty else 0.0

    t1, t2 = st.columns(2)
    t1.metric("Estimated Tail-Spend Savings (USD)", f"${total_tail_save:,.0f}")
    t2.caption("Assumes ~3% savings on tail supplier spend; adjust in code for client-specific rates.")

    st.dataframe(tail, use_container_width=True)

    if "ai_ctx" not in st.session_state:
        st.session_state["ai_ctx"] = build_ai_context(df, None, None, tail)

elif page == "AI Insights":
    st.subheader("AI Insights (CFO-ready)")
    has_key = bool(st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else os.getenv("OPENAI_API_KEY"))
    st.caption("Mode: " + ("**Live (OpenAI)**" if has_key else "**Offline demo** — no API key needed"))
    ctx = st.session_state.get("ai_ctx", build_ai_context(df, None, None, None))

    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing patterns and generating insights…"):
            insights = generate_ai_insights(ctx)
        st.markdown(insights)
    else:
        st.info("Click **Generate AI Insights** to convert your KPIs and scenario outcomes into concise, actionable recommendations.")

elif page == "Agent: Supplier Risk Monitor":
    st.subheader("Agent 1 — Supplier Risk & Emission Monitor")
    st.caption("Uploads or sample events are scored by keywords, severity, and recency. Alerts suggest actions and alternatives.")

    colk1, colk2 = st.columns(2)
    with colk1:
        events_file = st.file_uploader("Upload Events/News CSV", type=["csv"],
                                       help="Columns: Supplier, Date(YYYY-MM-DD), Title, Summary, Source, Severity (0.5-3.0)")
    with colk2:
        kw_text = st.text_area("Risk keywords & weights (JSON)", value=str(DEFAULT_KEYWORDS), height=160,
                               help="Edit as JSON, e.g. {\"violation\":3.0, \"pfas\":2.0}")
    st.markdown(" ")

    # Parse keywords
    try:
        if kw_text.strip().startswith("{"):
            risk_keywords = eval(kw_text)  # simple parser for demo; replace with json.loads in hardened version
        else:
            risk_keywords = DEFAULT_KEYWORDS
    except Exception:
        risk_keywords = DEFAULT_KEYWORDS

    # Load events
    if events_file is not None:
        try:
            df_events = pd.read_csv(events_file)
        except Exception as e:
            st.error(f"Failed to read Events CSV: {e}")
            df_events = build_sample_events(df["Supplier"])
    else:
        df_events = build_sample_events(df["Supplier"])

    st.markdown("**Preview events**")
    st.dataframe(df_events.head(20), use_container_width=True)

    # Score and aggregate
    df_scored = score_events(df_events, risk_keywords)
    alerts = aggregate_supplier_risk(df_scored)

    # Join category and spend for context
    if not alerts.empty:
        base = df.groupby(["Supplier","Category"], as_index=False).agg(Annual_Spend_USD=("Annual_Spend_USD","sum"),
                                                                      Distance_km=("Distance_km","mean"))
        alerts = alerts.merge(base, on="Supplier", how="left")

        # Recommended action + alternatives
        recs = []
        alts = []
        for _, r in alerts.iterrows():
            if r["Risk_Bucket"] == "High":
                recs.append("Immediate supplier review; consider regional alternative and quality audit.")
            elif r["Risk_Bucket"] == "Medium":
                recs.append("Engage supplier for CAPA; monitor monthly.")
            else:
                recs.append("Monitor quarterly; no action now.")
            alts.append(", ".join(suggest_lower_emission_alternatives(df, r["Supplier"], top_n=3)) or "—")
        alerts["Recommended_Action"] = recs
        alerts["Alt_Suppliers_Same_Category"] = alts

    st.markdown("### Alerts")
    st.dataframe(alerts, use_container_width=True)

    # Simple charts
    ch1, ch2 = st.columns(2)
    if not alerts.empty:
        ch1.plotly_chart(px.bar(alerts.sort_values("Risk_Score", ascending=False).head(10),
                                x="Supplier", y="Risk_Score", color="Risk_Bucket",
                                title="Top Suppliers by Risk Score"),
                         use_container_width=True)
        bucket_counts = alerts["Risk_Bucket"].value_counts().reset_index()
        bucket_counts.columns = ["Risk_Bucket","Count"]
        ch2.plotly_chart(px.pie(bucket_counts, names="Risk_Bucket", values="Count",
                                title="Risk Bucket Distribution"),
                         use_container_width=True)
    else:
        ch1.info("No alerts to chart."); ch2.info("No alerts to chart.")

    # Export
    if not alerts.empty:
        out_csv = io.StringIO()
        alerts.to_csv(out_csv, index=False)
        st.download_button("Download Alerts (CSV)", data=out_csv.getvalue().encode("utf-8"),
                           file_name=f"supplier_risk_alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")

    # Agent summary (AI optional)
    st.markdown("---")
    if st.button("Generate Agent Summary"):
        with st.spinner("Summarizing alerts…"):
            summary = agent_live_summary(alerts)
        st.markdown(summary)
    else:
        st.info("Click **Generate Agent Summary** to produce a concise narrative for Procurement + ESG.")

elif page == "About":
    st.markdown("""
**About this demo**

- Quantifies **freight cost** and **Scope 3 emissions** from supplier lanes and purchased goods  
- Scenario levers: **Supplier localization** and **Recycled content adoption**  
- Opportunity finder: **Tail-spend consolidation**  
- **AI Insights**: Converts KPIs into CFO-ready actions (offline mode by default; plug in `OPENAI_API_KEY` for live AI)  
- **Agent: Supplier Risk Monitor**: Scores supplier news/events and generates alerts with suggested alternatives.  
- Enterprise-ready: Can point to **Azure OpenAI** and client data lakes; no data persisted by this demo.
""")

st.markdown("---")
st.caption("Note: Emission factors and costs are illustrative. Replace with client data during onboarding.")
