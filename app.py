# app.py
# Sustainable Procurement & Vendor Cost Analyzer (with AI Insights)
# -----------------------------------------------------------------
# - One-file Streamlit app
# - Works out-of-the-box with a built-in sample dataset
# - Scenarios: Supplier localization + Recycled content adoption
# - Tail-spend consolidation estimator
# - CSV export of results
# - AI Insights tab:
#     * Uses OPENAI_API_KEY if provided (Streamlit Secrets or env var)
#     * Otherwise, returns realistic offline insights (no cost, no key)
#
# Notes:
# - All emission factors and costs are illustrative for demo purposes.
# - Replace factors with client-specific values for production.

import os
import io
import math
import json
from datetime import datetime

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

# Optional light branding (remove/replace logo as needed)
st.markdown(
    "<h2 style='margin-bottom:0'>Sustainable Procurement & Vendor Cost Analyzer</h2>"
    "<div style='color:#475569;margin-bottom:1rem'>Bridge sustainability with finance — quantify Scope 3 + freight, cost levers, and quick wins.</div>",
    unsafe_allow_html=True
)

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
# e.g., if recycled% = 30, effective EF = (1 - 0.30 * 0.6) * baseline
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
        cat = np.random.choice(categories, p=np.ones(len(categories))/len(categories))
        mode = np.random.choice(modes, p=[0.05, 0.6, 0.2, 0.15])  # mostly Road
        supplier = f"Supplier-{i+1:02d}"
        # Annual spend (USD)
        spend = float(np.random.randint(50_000, 600_000))
        # Annual purchased mass (kg)
        weight_kg = float(np.random.randint(6_000, 75_000))
        # Avg lane distance (km)
        dist_km = float(np.random.choice(
            [120, 250, 400, 800, 1200, 1800, 2500], p=[0.12, 0.2, 0.2, 0.18, 0.14, 0.1, 0.06]
        ))
        rows.append({
            "Supplier": supplier,
            "Category": cat,
            "Annual_Spend_USD": spend,
            "Weight_kg": weight_kg,
            "Distance_km": dist_km,
            "Mode": mode,
        })
    df = pd.DataFrame(rows)
    return df


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
    # Purchased goods emissions (baseline, no recycled content applied)
    df["PurchasedGoods_Emissions_kgCO2e"] = df.apply(
        lambda r: (CATEGORY_PG_EF.get(str(r["Category"]), 2.0)) * r["Weight_kg"], axis=1
    )
    return df


def scenario_localization(df: pd.DataFrame, localize_pct: float, long_haul_km=1000, new_km=200) -> pd.DataFrame:
    """
    Reduce distance for a % of suppliers with lanes > long_haul_km down to new_km.
    Returns detailed per-supplier deltas and savings.
    """
    df = df.copy()
    mask = df["Distance_km"] > long_haul_km
    candidates = df[mask].copy()
    if candidates.empty or localize_pct <= 0:
        # no change
        df["Freight_Cost_Saved_USD"] = 0.0
        df["Freight_Emissions_Saved_kgCO2e"] = 0.0
        return df

    # Select top-N fraction of long-haul by spend
    candidates = candidates.sort_values("Annual_Spend_USD", ascending=False)
    n_select = max(1, int(round(len(candidates) * (localize_pct / 100.0))))
    selected_suppliers = set(candidates.head(n_select)["Supplier"])

    # Compute "after" tonne-km for selected
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
    """
    Apply recycled content adoption across categories.
    Effective EF = baseline * (1 - recycled% * multiplier)
    """
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
    """
    Identify tail suppliers (bottom X% of total spend by supplier count) and estimate savings.
    """
    df = df.copy()
    by_supplier = df.groupby("Supplier", as_index=False).agg(
        Supplier_Spend=("Annual_Spend_USD", "sum"),
        Supplier_Count=("Supplier", "count"),
        Weight_kg=("Weight_kg", "sum"),
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
        "source": "Circularity meta-studies (illustrative)"
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
            cat_name = c.get("Category", "Unknown")
            lines.append(
                f"- {cat_name}: Spend ${c.get('Spend_USD',0):,.0f}, "
                f"Freight kg {c.get('Freight_kg',0):,.0f}, Purchased kg {c.get('Purchased_kg',0):,.0f}"
            )
    return "\n".join(lines)

def retrieve_case_snippets(keywords=("localization","recycled","tail")):
    results = []
    for case in CASE_LIBRARY:
        for k in keywords:
            if k.lower() in case["pattern"].lower():
                results.append(case)
                break
    return results[:3]

def generate_ai_insights(ctx: dict):
    """
    If OPENAI_API_KEY exists, call OpenAI (gpt-4o-mini) for live insights.
    Otherwise, return a robust offline mock summary.
    """
    api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    context_text = format_ctx_for_prompt(ctx)
    case_snips = retrieve_case_snippets()
    cases_text = "\n".join(
        [f"- [{c['sector']}] {c['pattern']}: {c['impact']} (Source: {c['source']})" for c in case_snips]
    )

    # Offline mock (no key needed)
    if not api_key:
        bullets = []
        if ctx.get("loc_savings_usd", 0) > 0 or ctx.get("loc_avoided_kg", 0) > 0:
            bullets.append(
                f"**Localize long-haul lanes**: realize ≈ ${ctx.get('loc_savings_usd',0):,.0f} freight savings and avoid ≈ {ctx.get('loc_avoided_kg',0):,.0f} kg CO₂e; align procurement + logistics to shift selected suppliers to regional hubs."
            )
        if ctx.get("recycled_avoided_kg", 0) > 0:
            bullets.append(
                f"**Adopt 20–30% recycled content**: avoid ≈ {ctx.get('recycled_avoided_kg',0):,.0f} kg CO₂e in purchased goods; pilot in top categories and validate cost neutrality with vendor quotes."
            )
        if ctx.get("tail_savings_usd", 0) > 0:
            bullets.append(
                f"**Consolidate tail suppliers**: target ≈ ${ctx.get('tail_savings_usd',0):,.0f}/yr via ~3% tail-spend saving; standardize SLAs/specs to maintain quality."
            )
        if ctx.get("freight_cost_usd", 0) > 0:
            bullets.append(
                f"**Lane optimization**: benchmark freight spend (${ctx.get('freight_cost_usd',0):,.0f}) vs peers; evaluate mode shifts (air→road/rail) to cut cost and CO₂e."
            )
        bullets.append("**Governance**: publish a monthly KPI pack (spend, freight cost, Scope 3 split) and track pilot ROI by BU.")
        header = ":bulb: **AI Insights (Offline Demo Mode)**"
        return header + "\n" + "\n".join([f"- {b}" for b in bullets]) if bullets else header + "\n- No scenario impacts yet; adjust sliders to see AI suggestions."

    # Live OpenAI call
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        system_msg = (
            "You are a senior sustainability-finance analyst. "
            "Write concise, CFO-ready insights that tie actions to cost savings and Scope 3 reductions. "
            "Reference case patterns only if provided. Use bullets. Avoid hallucinations."
        )
        user_msg = (
            "DATA SUMMARY:\n"
            f"{context_text}\n\n"
            "REFERENCE CASE PATTERNS:\n"
            f"{cases_text or 'None'}\n\n"
            "TASK: Provide 5 bullet insights. Each bullet should:\n"
            "- Identify the lever (localization, recycled content, consolidation, etc.)\n"
            "- Quantify impact using the provided numbers\n"
            "- Note trade-offs/risks briefly\n"
            "- Be actionable (who should do what next)\n"
        )
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content.strip()
        return ":robot_face: **AI Insights (Live)**\n" + content
    except Exception as e:
        return f":warning: AI service unavailable (showing offline insights). Error: {e}\n" + generate_ai_insights.__wrapped__(ctx)  # type: ignore


# ----------------------------
# ---------- UI --------------
# ----------------------------
with st.expander("Upload data (optional) & assumptions", expanded=False):
    st.write("Upload a CSV with columns: Supplier, Category, Annual_Spend_USD, Weight_kg, Distance_km, Mode")
    file = st.file_uploader("Upload supplier dataset (CSV)", type=["csv"])
    colA, colB, colC = st.columns(3)
    with colA:
        long_haul_km = st.number_input("Long-haul threshold (km)", min_value=200, max_value=5000, value=1000, step=50)
    with colB:
        localized_km = st.number_input("Localized distance (km)", min_value=50, max_value=1000, value=200, step=10)
    with colC:
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

# Compute baseline
df = compute_core_metrics(df_raw)

# KPIs
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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Category Overview", "Scenarios", "Opportunities", "AI Insights"])

# ---- Tab 1: Category Overview ----
with tab1:
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

# ---- Tab 2: Scenarios ----
with tab2:
    st.subheader("What-if Scenarios")
    s1, s2 = st.columns(2)
    with s1:
        loc_pct = st.slider("Localize % of long-haul suppliers (by count)", 0, 100, 25, 5)
    with s2:
        rec_pct = st.slider("Adopt recycled content across categories (%)", 0, 80, 30, 5)

    # Localization scenario
    df_loc = scenario_localization(df, loc_pct, long_haul_km, localized_km)
    loc_saved_usd = df_loc["Freight_Cost_Saved_USD"].sum()
    loc_avoided_kg = df_loc["Freight_Emissions_Saved_kgCO2e"].sum()

    # Recycled scenario
    df_rec = scenario_recycled(df, rec_pct)
    rec_avoided_kg = df_rec["PurchasedGoods_Emissions_Saved_kgCO2e"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Localization: Freight Savings (USD)", f"${loc_saved_usd:,.0f}")
    c2.metric("Localization: Avoided Freight CO₂e (kg)", f"{loc_avoided_kg:,.0f}")
    c3.metric("Recycled Content: Avoided Purchased-Goods CO₂e (kg)", f"{rec_avoided_kg:,.0f}")

    st.markdown("**Localized Suppliers (Yes = distance reduced):**")
    st.dataframe(
        df_loc[["Supplier","Category","Annual_Spend_USD","Mode","Distance_km","Localized","Freight_Cost_Saved_USD","Freight_Emissions_Saved_kgCO2e"]]
        .sort_values("Freight_Cost_Saved_USD", ascending=False),
        use_container_width=True
    )

    # Charts for scenario impacts
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

    # Export results
    def _to_bytes_csv(df_dict):
        # Create a multi-sheet-like zip in memory? Keep simple: single CSV combining key outputs.
        out = io.StringIO()
        # Merge key cols from loc + rec for compact export
        exp = df_loc[["Supplier","Category","Annual_Spend_USD","Mode","Distance_km","Localized",
                      "Freight_Cost_Saved_USD","Freight_Emissions_Saved_kgCO2e"]].copy()
        exp2 = df_rec[["Supplier","PurchasedGoods_Emissions_Saved_kgCO2e"]].copy().rename(
            columns={"PurchasedGoods_Emissions_Saved_kgCO2e":"PG_Emissions_Saved_kgCO2e"})
        merged = exp.merge(exp2, left_index=True, right_index=True, how="left")
        merged.to_csv(out, index=False)
        return out.getvalue().encode("utf-8")

    st.download_button(
        "Download Scenario Results (CSV)",
        data=_to_bytes_csv({"loc": df_loc, "rec": df_rec}),
        file_name=f"scenario_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# ---- Tab 3: Opportunities (Tail Spend etc.) ----
with tab3:
    st.subheader("Tail-Spend Consolidation Opportunities")
    tail = tail_spend_consolidation(df, tail_share=tail_share, saving_rate=0.03)
    total_tail_save = tail["Indicative_Consolidation_Saving_USD"].sum() if not tail.empty else 0.0

    t1, t2 = st.columns(2)
    t1.metric("Estimated Tail-Spend Savings (USD)", f"${total_tail_save:,.0f}")
    t2.caption("Assumes ~3% savings on tail supplier spend; adjust in code for client-specific rates.")

    st.dataframe(tail, use_container_width=True)

# ---- Tab 4: AI Insights ----
with tab4:
    st.subheader("AI Insights (CFO-ready)")
    has_key = bool(st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else os.getenv("OPENAI_API_KEY"))
    st.caption("Mode: " + ("**Live (OpenAI)**" if has_key else "**Offline demo** — no API key needed"))
    try:
        ctx = build_ai_context(df, df_loc, df_rec, tail)
        if st.button("Generate AI Insights"):
            with st.spinner("Analyzing patterns and generating insights…"):
                insights = generate_ai_insights(ctx)
            st.markdown(insights)
        else:
            st.info("Click **Generate AI Insights** to convert your KPIs into concise, actionable recommendations.")
    except Exception as e:
        st.error(f"AI insights unavailable: {e}")

st.markdown("---")
st.caption("Note: Factors and costs are illustrative. Replace with client-specific data during onboarding. "
           "AI can be routed to Azure OpenAI in enterprise environments; no data is persisted by this demo.")
