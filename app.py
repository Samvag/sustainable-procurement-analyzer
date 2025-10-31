# app.py
# Sustainable Procurement & Vendor Cost Analyzer (Prototype)
# ---------------------------------------------------------
# What this does:
# 1) Ingests a supplier master CSV (or uses built-in sample data).
# 2) Estimates Scope 3 purchased goods (category EF) + freight emissions (tonne-km * mode EF).
# 3) Estimates freight cost (per tkm) and flags consolidation/localization opportunities.
# 4) Runs simple scenarios: (a) Localize distant suppliers, (b) Recycled-content adoption.
# 5) Produces CFO-friendly KPIs, charts, and a downloadable report (CSV).

import io
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Sustainable Procurement & Vendor Cost Analyzer",
                   layout="wide", page_icon="♻️")

# ----------------------------
# Defaults & Example Factors
# ----------------------------
# NOTE: These are illustrative example factors and cost assumptions for demo purposes.
# Replace with client-specific/region-specific factors during onboarding.
CATEGORY_PURCHASED_GOODS_EF_KGCO2_PER_TONNE = {
    # Example cradle-to-gate factors (illustrative ranges)
    "Plastics (PE/PP/PET)": 2500,       # kg CO2e / t
    "Paper & Board": 1200,
    "Metals (Aluminum)": 10000,
    "Metals (Steel)": 1900,
    "Chemicals (General)": 3000,
    "Packaging (Mixed)": 1800,
    "Ingredients (Food/Cosmetic)": 2500,
    "Other/Unknown": 2500,
}

# Emission factors for transport (kg CO2e per tonne-km), illustrative
TRANSPORT_EF_KGCO2_PER_TKM = {
    "Road": 0.12,
    "Rail": 0.03,
    "Sea": 0.01,
    "Air": 0.60,
}

# Freight cost assumption per tonne-km (USD/tkm), illustrative
FREIGHT_COST_USD_PER_TKM = {
    "Road": 0.10,
    "Rail": 0.05,
    "Sea": 0.02,
    "Air": 0.80,
}

# Recycled-content emissions multipliers (illustrative midpoints of literature ranges)
# (applied to category purchased-goods EF)
RECYCLED_EMISSIONS_MULTIPLIER = {
    "Plastics (PE/PP/PET)": 0.55,   # 45% reduction at 100% recycled
    "Paper & Board": 0.70,
    "Metals (Aluminum)": 0.30,      # recycled Al much lower than virgin
    "Metals (Steel)": 0.65,
    "Chemicals (General)": 0.80,
    "Packaging (Mixed)": 0.60,
    "Ingredients (Food/Cosmetic)": 0.85,
    "Other/Unknown": 0.80,
}

# ----------------------------
# Sample Supplier Dataset
# ----------------------------
def sample_suppliers():
    np.random.seed(7)
    suppliers = [
        ("PolyPack Midwest", "Plastics (PE/PP/PET)", "Road", 1200, 750, 820),
        ("EcoBoard LLC", "Paper & Board", "Road", 900, 1200, 400),
        ("OceanCan Metals", "Metals (Aluminum)", "Sea", 1500, 2000, 6000),
        ("SteelWorks Ohio", "Metals (Steel)", "Rail", 2200, 1800, 900),
        ("ChemX Solutions", "Chemicals (General)", "Road", 1800, 1600, 1200),
        ("PackRight", "Packaging (Mixed)", "Road", 700, 500, 350),
        ("GreenIngredients Co", "Ingredients (Food/Cosmetic)", "Road", 1400, 900, 700),
        ("FlexiPlast East", "Plastics (PE/PP/PET)", "Road", 800, 550, 1600),
        ("BoardPro Central", "Paper & Board", "Rail", 600, 1000, 1500),
        ("AlliedChem Intl", "Chemicals (General)", "Sea", 2100, 2400, 5000),
        ("MetalOne Imports", "Metals (Steel)", "Sea", 1300, 1500, 7000),
        ("PackWest", "Packaging (Mixed)", "Road", 500, 420, 1400),
    ]
    rows = []
    for name, cat, mode, spend, weight_t, distance_km in suppliers:
        # Spread annual spend/weight with +/- random
        spend_usd = int(spend * (0.85 + 0.30 * np.random.rand()))
        weight_t = float(weight_t * (0.85 + 0.30 * np.random.rand()))
        distance = int(distance_km * (0.85 + 0.30 * np.random.rand()))
        rows.append({
            "Supplier": name,
            "Category": cat,
            "Transport_Mode": mode,
            "Annual_Spend_USD": spend_usd,
            "Estimated_Weight_tonnes": round(weight_t, 0),
            "Avg_Distance_km": distance,
            "Sustainability_Rating": np.random.choice([None, "Bronze", "Silver", "Gold", "NR"], p=[0.15,0.25,0.30,0.20,0.10]),
            "Region": np.random.choice(["Local/Regional","Domestic","Near-shore","Offshore"], p=[0.25,0.35,0.20,0.20]),
        })
    return pd.DataFrame(rows)

# ----------------------------
# Core Calculations
# ----------------------------
def calc_emissions_and_cost(df: pd.DataFrame):
    df = df.copy()
    # Validate required columns
    required_cols = [
        "Supplier","Category","Transport_Mode","Annual_Spend_USD",
        "Estimated_Weight_tonnes","Avg_Distance_km"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Map Purchased Goods EF
    df["PurchasedGoods_EF_kgCO2_t"] = df["Category"].map(CATEGORY_PURCHASED_GOODS_EF_KGCO2_PER_TONNE).fillna(
        CATEGORY_PURCHASED_GOODS_EF_KGCO2_PER_TONNE["Other/Unknown"]
    )
    # Transport EF & Cost
    df["Transport_EF_kgCO2_tkm"] = df["Transport_Mode"].map(TRANSPORT_EF_KGCO2_PER_TKM).fillna(0.12)
    df["Freight_Cost_USD_per_tkm"] = df["Transport_Mode"].map(FREIGHT_COST_USD_PER_TKM).fillna(0.10)

    # Tonne-km
    df["Tonne_km"] = df["Estimated_Weight_tonnes"] * df["Avg_Distance_km"]
    df["Freight_Emissions_kgCO2e"] = df["Tonne_km"] * df["Transport_EF_kgCO2_tkm"]
    df["Freight_Cost_USD"] = df["Tonne_km"] * df["Freight_Cost_USD_per_tkm"]

    # Purchased goods emissions
    df["PurchasedGoods_Emissions_kgCO2e"] = df["Estimated_Weight_tonnes"] * df["PurchasedGoods_EF_kgCO2_t"]

    # Totals
    df["Total_Emissions_kgCO2e"] = df["PurchasedGoods_Emissions_kgCO2e"] + df["Freight_Emissions_kgCO2e"]

    return df

def scenario_localize(df: pd.DataFrame, distance_threshold_km: int, reduce_distance_pct: float):
    df = df.copy()
    # Apply to suppliers beyond threshold
    mask = df["Avg_Distance_km"] > distance_threshold_km
    df["Avg_Distance_km_scn"] = df["Avg_Distance_km"]
    df.loc[mask, "Avg_Distance_km_scn"] = (df.loc[mask, "Avg_Distance_km"] * (1 - reduce_distance_pct/100)).round(0)

    # recompute tonne-km, freight emissions & cost for scenario
    df["Tonne_km_scn"] = df["Estimated_Weight_tonnes"] * df["Avg_Distance_km_scn"]
    df["Freight_Emissions_kgCO2e_scn"] = df["Tonne_km_scn"] * df["Transport_EF_kgCO2_tkm"]
    df["Freight_Cost_USD_scn"] = df["Tonne_km_scn"] * df["Freight_Cost_USD_per_tkm"]

    # deltas
    df["Freight_Emissions_Saved_kgCO2e"] = df["Freight_Emissions_kgCO2e"] - df["Freight_Emissions_kgCO2e_scn"]
    df["Freight_Cost_Saved_USD"] = df["Freight_Cost_USD"] - df["Freight_Cost_USD_scn"]

    return df

def scenario_recycled(df: pd.DataFrame, adoption_pct_by_category: dict):
    df = df.copy()
    # For each supplier row, adjust purchased goods EF by recycled mix
    def adjusted_ef(row):
        cat = row["Category"]
        base = row["PurchasedGoods_EF_kgCO2_t"]
        adopt_pct = adoption_pct_by_category.get(cat, 0) / 100.0
        recycled_mult = RECYCLED_EMISSIONS_MULTIPLIER.get(cat, 0.8)
        # Weighted EF = adopt% * (base*mult) + (1-adopt%) * base
        return adopt_pct * (base * recycled_mult) + (1 - adopt_pct) * base

    df["PurchasedGoods_EF_kgCO2_t_scn"] = df.apply(adjusted_ef, axis=1)
    df["PurchasedGoods_Emissions_kgCO2e_scn"] = df["Estimated_Weight_tonnes"] * df["PurchasedGoods_EF_kgCO2_t_scn"]
    df["PurchasedGoods_Emissions_Saved_kgCO2e"] = df["PurchasedGoods_Emissions_kgCO2e"] - df["PurchasedGoods_Emissions_kgCO2e_scn"]
    return df

def consolidation_opportunities(df: pd.DataFrame, min_vendor_count=2, max_tail_spend_pct=0.05):
    """
    Flags categories where there are many small suppliers (tail spend) that could be consolidated.
    - min_vendor_count: flag categories with at least this many suppliers
    - max_tail_spend_pct: suppliers below this spend share (e.g., 5%) considered tail
    """
    df = df.copy()
    total_spend = df["Annual_Spend_USD"].sum()
    df["Spend_Share"] = df["Annual_Spend_USD"] / total_spend

    by_cat = df.groupby("Category").agg(
        Suppliers=("Supplier","nunique"),
        Cat_Spend_USD=("Annual_Spend_USD","sum")
    ).reset_index()
    by_cat["Cat_Spend_Share"] = by_cat["Cat_Spend_USD"] / total_spend

    # Tail per category
    tail = []
    for cat, g in df.groupby("Category"):
        g_sorted = g.sort_values("Annual_Spend_USD", ascending=False).reset_index(drop=True)
        tail_rows = g_sorted[g_sorted["Spend_Share"] < max_tail_spend_pct].copy()
        if len(g_sorted["Supplier"].unique()) >= min_vendor_count and not tail_rows.empty:
            tail_spend = tail_rows["Annual_Spend_USD"].sum()
            tail_suppliers = list(tail_rows["Supplier"].unique())
            tail.append({
                "Category": cat,
                "Suppliers_in_Category": len(g_sorted["Supplier"].unique()),
                "Tail_Suppliers_Count": len(tail_suppliers),
                "Tail_Spend_USD": int(tail_spend),
                "Indicative_Consolidation_Saving_USD": int(0.03 * tail_spend)  # assume 3% saving potential
            })
    return pd.DataFrame(tail)

def kpi_card(label, value, helptext=None):
    st.metric(label=label, value=value, help=helptext)

def format_usd(x): return f"${x:,.0f}"
def format_tonnes(x): return f"{x:,.0f} t"
def format_kg(x): return f"{x:,.0f} kg"

# ----------------------------
# UI
# ----------------------------
st.title("Sustainable Procurement & Vendor Cost Analyzer")
st.caption("Prototype – quantifies **Scope 3** + freight impact, and identifies **cost-saving sustainable sourcing** opportunities.")

with st.sidebar:
    st.header("1) Data")
    uploaded = st.file_uploader("Upload Supplier Master CSV", type=["csv"])
    st.markdown("**Required columns:**  \n"
                "`Supplier, Category, Transport_Mode, Annual_Spend_USD, Estimated_Weight_tonnes, Avg_Distance_km`  \n"
                "_Optional:_ `Sustainability_Rating, Region`")
    if st.button("Download CSV Template"):
        template = pd.DataFrame({
            "Supplier": ["Supplier A","Supplier B"],
            "Category": ["Plastics (PE/PP/PET)","Paper & Board"],
            "Transport_Mode": ["Road","Rail"],
            "Annual_Spend_USD": [1000, 800],
            "Estimated_Weight_tonnes": [700, 500],
            "Avg_Distance_km": [1200, 900],
            "Sustainability_Rating": ["Silver","NR"],
            "Region": ["Domestic","Offshore"]
        })
        st.download_button("Save template.csv", template.to_csv(index=False).encode("utf-8"), file_name="supplier_template.csv", mime="text/csv")

    st.header("2) Scenarios")
    st.subheader("A) Localize Distant Suppliers")
    distance_threshold_km = st.slider("Distance threshold (km) for localization", 200, 8000, 1500, 100)
    reduce_distance_pct = st.slider("Reduce distance by (%) for those above threshold", 5, 80, 30, 5)

    st.subheader("B) Recycled-Content Adoption")
    default_adoption = 20
    adoption_inputs = {}
    for cat in CATEGORY_PURCHASED_GOODS_EF_KGCO2_PER_TONNE.keys():
        adoption_inputs[cat] = st.number_input(f"{cat}: recycled content adoption (%)", min_value=0, max_value=100, value=default_adoption, step=5)

    st.header("3) Display Settings")
    show_details = st.checkbox("Show supplier-level details", value=False)

# Load data
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    st.info("No file uploaded. Using built-in sample dataset for demo.")
    df_raw = sample_suppliers()

# Compute base
try:
    df = calc_emissions_and_cost(df_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

# KPIs – Base
total_spend = df["Annual_Spend_USD"].sum()
total_freight_cost = df["Freight_Cost_USD"].sum()
total_freight_emis = df["Freight_Emissions_kgCO2e"].sum()
total_purch_emis = df["PurchasedGoods_Emissions_kgCO2e"].sum()
total_emis = df["Total_Emissions_kgCO2e"].sum()

col1, col2, col3, col4 = st.columns(4)
with col1: kpi_card("Total Annual Spend", format_usd(total_spend))
with col2: kpi_card("Freight Cost (est.)", format_usd(total_freight_cost))
with col3: kpi_card("Freight Emissions", format_kg(total_freight_emis))
with col4: kpi_card("Purchased-Goods Emissions", format_kg(total_purch_emis))

st.divider()

# Charts – Spend by Category / Emissions by Category
tab1, tab2, tab3 = st.tabs(["Category Overview", "Scenarios", "Opportunities"])

with tab1:
    left, right = st.columns(2)
    by_cat = df.groupby("Category", as_index=False).agg(
        Spend_USD=("Annual_Spend_USD","sum"),
        Freight_kgCO2e=("Freight_Emissions_kgCO2e","sum"),
        Purchased_kgCO2e=("PurchasedGoods_Emissions_kgCO2e","sum")
    )
    by_cat["Total_kgCO2e"] = by_cat["Freight_kgCO2e"] + by_cat["Purchased_kgCO2e"]

    with left:
        st.subheader("Spend by Category")
        fig1 = px.bar(by_cat, x="Category", y="Spend_USD", text_auto=True)
        fig1.update_layout(yaxis_title="USD", xaxis_title="")
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        st.subheader("Emissions by Category (kg CO₂e)")
        melted = by_cat.melt(id_vars="Category", value_vars=["Freight_kgCO2e","Purchased_kgCO2e"],
                             var_name="Type", value_name="kgCO2e")
        fig2 = px.bar(melted, x="Category", y="kgCO2e", color="Type", barmode="stack", text_auto=True)
        fig2.update_layout(yaxis_title="kg CO₂e", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    if show_details:
        st.subheader("Supplier Detail (Base)")
        st.dataframe(df.sort_values("Annual_Spend_USD", ascending=False), use_container_width=True)

with tab2:
    st.subheader("Scenario A: Localize Distant Suppliers")
    df_loc = scenario_localize(df, distance_threshold_km, reduce_distance_pct)
    total_freight_saved = df_loc["Freight_Cost_Saved_USD"].sum()
    total_emis_saved = df_loc["Freight_Emissions_Saved_kgCO2e"].sum()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Freight Cost Saved (est.)", format_usd(total_freight_saved),
                  help=f"Threshold: {distance_threshold_km} km | Reduce distance: {reduce_distance_pct}%")
    with c2:
        st.metric("Freight Emissions Avoided", format_kg(total_emis_saved))

    # Chart: before vs after distance for impacted suppliers
    impacted = df_loc[df_loc["Freight_Cost_Saved_USD"] > 0].copy()
    if not impacted.empty:
        impacted["Distance_Before"] = impacted["Avg_Distance_km"]
        impacted["Distance_After"] = impacted["Avg_Distance_km_scn"]
        plot_df = impacted[["Supplier","Distance_Before","Distance_After"]].melt(
            id_vars="Supplier", var_name="Scenario", value_name="Distance_km"
        )
        fig3 = px.bar(plot_df, x="Supplier", y="Distance_km", color="Scenario", barmode="group", text_auto=True)
        fig3.update_layout(xaxis_title="", yaxis_title="km")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No suppliers exceed the distance threshold — adjust the slider to simulate a relocation benefit.")

    st.divider()
    st.subheader("Scenario B: Recycled-Content Adoption")
    df_rec = scenario_recycled(df, adoption_inputs)
    purch_saved = df_rec["PurchasedGoods_Emissions_Saved_kgCO2e"].sum()
    st.metric("Purchased-Goods Emissions Avoided", format_kg(purch_saved),
              help="Applies category-specific recycled multipliers to the adopted % of purchased goods.")

    # Category view of recycled savings
    rec_cat = df_rec.groupby("Category", as_index=False)["PurchasedGoods_Emissions_Saved_kgCO2e"].sum()
    fig4 = px.bar(rec_cat, x="Category", y="PurchasedGoods_Emissions_Saved_kgCO2e", text_auto=True,
                  title="Emissions Savings by Category (Recycled Adoption)")
    fig4.update_layout(yaxis_title="kg CO₂e", xaxis_title="")
    st.plotly_chart(fig4, use_container_width=True)

    if show_details:
        st.subheader("Supplier Detail (Scenarios)")
        show_cols = [
            "Supplier","Category","Transport_Mode","Annual_Spend_USD",
            "Estimated_Weight_tonnes","Avg_Distance_km","Avg_Distance_km_scn",
            "Freight_Cost_USD","Freight_Cost_USD_scn","Freight_Cost_Saved_USD",
            "Freight_Emissions_kgCO2e","Freight_Emissions_kgCO2e_scn","Freight_Emissions_Saved_kgCO2e",
            "PurchasedGoods_EF_kgCO2_t","PurchasedGoods_EF_kgCO2_t_scn",
            "PurchasedGoods_Emissions_kgCO2e","PurchasedGoods_Emissions_kgCO2e_scn","PurchasedGoods_Emissions_Saved_kgCO2e"
        ]
        merged = df_loc.merge(
            df_rec[["Supplier","Category","PurchasedGoods_EF_kgCO2_t_scn",
                    "PurchasedGoods_Emissions_kgCO2e_scn","PurchasedGoods_Emissions_Saved_kgCO2e"]],
            on=["Supplier","Category"], how="left"
        )
        st.dataframe(merged[show_cols].sort_values("Annual_Spend_USD", ascending=False), use_container_width=True)

with tab3:
    st.subheader("Consolidation Opportunities (Tail Spend)")
    tail = consolidation_opportunities(df, min_vendor_count=2, max_tail_spend_pct=0.05)
    if tail.empty:
        st.info("No obvious tail-spend consolidation opportunities under current thresholds. Try uploading real data.")
    else:
        tail["Tail_Spend_USD_fmt"] = tail["Tail_Spend_USD"].apply(format_usd)
        tail["Indicative_Consolidation_Saving_USD_fmt"] = tail["Indicative_Consolidation_Saving_USD"].apply(format_usd)
        st.dataframe(tail[[
            "Category","Suppliers_in_Category","Tail_Suppliers_Count",
            "Tail_Spend_USD_fmt","Indicative_Consolidation_Saving_USD_fmt"
        ]], use_container_width=True)

    st.divider()
    st.subheader("Auto-Generated Talking Points (for CFO/Procurement)")
    bullets = []
    if total_freight_cost > 0:
        bullets.append(f"Estimated annual freight cost: **{format_usd(total_freight_cost)}**.")
    if total_freight_emis > 0:
        bullets.append(f"Freight emissions contribute **{format_kg(total_freight_emis)}** to Scope 3 (Category 4/9).")
    if total_purch_emis > 0:
        bullets.append(f"Purchased goods/process emissions estimated at **{format_kg(total_purch_emis)}**.")
    if not tail.empty:
        pot = tail["Indicative_Consolidation_Saving_USD"].sum()
        bullets.append(f"Vendor consolidation could save ~**{format_usd(pot)}** (assumed 3% on tail spend).")
    if 'total_emis_saved' in locals() and total_emis_saved > 0:
        bullets.append(f"Localization scenario avoids ~**{format_kg(total_emis_saved)}** freight CO₂e "
                       f"and saves **{format_usd(total_freight_saved)}**.")
    if 'purch_saved' in locals() and purch_saved > 0:
        bullets.append(f"Recycled-content scenario avoids ~**{format_kg(purch_saved)}** purchased-goods CO₂e.")

    if bullets:
        st.markdown("\n".join([f"- {b}" for b in bullets]))
    else:
        st.write("Upload data or adjust scenarios to generate CFO-ready talking points.")

# ----------------------------
# Export Report
# ----------------------------
st.divider()
st.subheader("Export")
export_cols = [
    "Supplier","Category","Transport_Mode","Region","Sustainability_Rating",
    "Annual_Spend_USD","Estimated_Weight_tonnes","Avg_Distance_km",
    "Freight_Cost_USD","Freight_Emissions_kgCO2e","PurchasedGoods_EF_kgCO2_t","PurchasedGoods_Emissions_kgCO2e",
]
out = df.copy()
if 'df_loc' in locals():
    out["Avg_Distance_km_scn"] = df_loc["Avg_Distance_km_scn"]
    out["Freight_Cost_USD_scn"] = df_loc["Freight_Cost_USD_scn"]
    out["Freight_Emissions_kgCO2e_scn"] = df_loc["Freight_Emissions_kgCO2e_scn"]
    out["Freight_Cost_Saved_USD"] = df_loc["Freight_Cost_Saved_USD"]
    out["Freight_Emissions_Saved_kgCO2e"] = df_loc["Freight_Emissions_Saved_kgCO2e"]
    export_cols += ["Avg_Distance_km_scn","Freight_Cost_USD_scn","Freight_Emissions_kgCO2e_scn",
                    "Freight_Cost_Saved_USD","Freight_Emissions_Saved_kgCO2e"]

if 'df_rec' in locals():
    out["PurchasedGoods_EF_kgCO2_t_scn"] = df_rec["PurchasedGoods_EF_kgCO2_t_scn"]
    out["PurchasedGoods_Emissions_kgCO2e_scn"] = df_rec["PurchasedGoods_Emissions_kgCO2e_scn"]
    out["PurchasedGoods_Emissions_Saved_kgCO2e"] = df_rec["PurchasedGoods_Emissions_Saved_kgCO2e"]
    export_cols += ["PurchasedGoods_EF_kgCO2_t_scn","PurchasedGoods_Emissions_kgCO2e_scn",
                    "PurchasedGoods_Emissions_Saved_kgCO2e"]

csv_bytes = out[export_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download results (CSV)", csv_bytes, file_name=f"procurement_analyzer_{dt.date.today().isoformat()}.csv", mime="text/csv")

st.caption("**Note:** Factors and costs are illustrative for demo. Replace with client-specific data during onboarding.")
