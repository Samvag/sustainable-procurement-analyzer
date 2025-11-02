# app.py
# Sustainable Procurement & Vendor Cost Analyzer — AI-ready demo
# Runs offline with simulated data; optional live LLM via OPENAI/ANTHROPIC keys.

import os
import time
import json
import base64
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────────────────
# Page config & styles
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sustainable Procurement & Vendor Cost Analyzer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stButton>button {
        background:#003DA5; color:#fff; font-weight:600; border:none; border-radius:8px; padding:.5rem 1rem;
    }
    .stButton>button:hover { background:#002D7A; }
    .metric-card { background:linear-gradient(135deg,#667EEA 0%,#764BA2 100%); color:#fff;
        padding:1rem; border-radius:14px; box-shadow:0 2px 10px rgba(0,0,0,.06); }
    .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; color:#fff; }
    .pill.ok{background:#16a34a;} .pill.warn{background:#eab308;} .pill.err{background:#dc2626;}
    .box { border:1px solid #eee; border-radius:10px; padding:1rem; background:#fafafa; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────────
if "vendor_df" not in st.session_state: st.session_state.vendor_df = None
if "supplier_df" not in st.session_state: st.session_state.supplier_df = None
if "evidence_rows" not in st.session_state: st.session_state.evidence_rows = None
if "nav" not in st.session_state: st.session_state.nav = "Overview"

# ────────────────────────────────────────────────────────────────────────────────
# Demo data generators
# ────────────────────────────────────────────────────────────────────────────────
def gen_vendor_demo(n=50) -> pd.DataFrame:
    np.random.seed(42)
    suppliers = [f"Supplier {i:02d}" for i in range(1, n+1)]
    categories = ["Raw Materials", "Packaging", "Logistics", "Chemicals", "Tolling"]
    countries = ["US", "DE", "FR", "BE", "PL", "CN", "IN", "BR"]
    df = pd.DataFrame({
        "Supplier_Name": np.random.choice(suppliers, n, replace=False),
        "Category": np.random.choice(categories, n),
        "Country": np.random.choice(countries, n),
        "Spend_USD": np.random.lognormal(mean=14.2, sigma=0.6, size=n).round(0),  # ~ $1.4M median
        "Sustainability_Score": np.clip(np.random.normal(65, 12, n), 20, 95).round(0),
        "Carbon_Score": np.clip(np.random.normal(60, 15, n), 10, 95).round(0),
        "Waste_Score": np.clip(np.random.normal(62, 14, n), 10, 95).round(0),
        "Circularity_Score": np.clip(np.random.normal(58, 16, n), 10, 95).round(0),
        "Risk_Level": np.clip(np.random.normal(50, 20, n), 5, 100).round(0),     # used for bubble size
        "Emissions_tCO2e": np.clip(np.random.normal(1200, 400, n), 100, 4000).round(0),
        "Recycled_Content_pct": np.clip(np.random.normal(25, 12, n), 0, 80).round(1),
        "OnTime_Delivery_pct": np.clip(np.random.normal(92, 5, n), 60, 100).round(1),
        "ESG_Data_Complete": np.random.choice(["Yes", "Partial", "No"], n, p=[0.55, 0.3, 0.15]),
    })
    return df

def gen_evidence_manifest() -> pd.DataFrame:
    np.random.seed(9)
    return pd.DataFrame({
        "id": [f"EVD-{i:03d}" for i in range(1, 10)],
        "supplier_name": np.random.choice([f"Supplier {i:02d}" for i in range(1, 11)], 9),
        "evidence_type": np.random.choice(["SVHC Declaration", "Recycled Content Proof", "ISO14001"], 9),
        "submission_date": pd.date_range("2024-07-05", periods=9, freq="21D"),
        "file_name": [f"evidence_{i}.pdf" for i in range(1, 10)],
        "status": np.random.choice(["Submitted", "Pending Review", "Approved"], 9),
    })

def gen_supplier_readiness() -> pd.DataFrame:
    np.random.seed(11)
    suppliers = ["EcoMat Ltd", "PlastoChem GmbH", "ReGenPolymers NV"]
    regions = ["EU", "NA", "APAC"]
    materials = ["PCR-PE", "PCR-PET", "Bio-PP"]
    rows = []
    for s in suppliers:
        rows.append({
            "supplier_name": s,
            "region": np.random.choice(regions),
            "primary_material": np.random.choice(materials),
            "recycled_content_cert": np.random.choice(["Yes", "No", "Expired"], p=[0.6, 0.25, 0.15]),
            "svhc_declaration": np.random.choice(["Yes", " "], p=[0.8, 0.2]).strip() or "No",
            "iso14001": np.random.choice(["Yes", "No"], p=[0.7, 0.3]),
            "evidence_files": np.random.randint(1, 6),
            "last_update": pd.to_datetime("2024-09-01") + pd.to_timedelta(np.random.randint(0, 120), unit="D"),
        })
    df = pd.DataFrame(rows)
    score = 0
    score += (df["recycled_content_cert"] == "Yes") * 40
    score += (df["svhc_declaration"] == "Yes") * 30
    score += (df["iso14001"] == "Yes") * 20
    score += np.minimum(df["evidence_files"] * 2, 10)
    df["readiness_score"] = score

    def flag_risk(r):
        risks = []
        if r["recycled_content_cert"] in ("No", "Expired"):
            risks.append("Recycled-content proof missing/expired")
        if r["svhc_declaration"] == "No":
            risks.append("SVHC declaration missing")
        if r["readiness_score"] < 60:
            risks.append("Low overall readiness")
        return "; ".join(risks) if risks else "OK"
    df["risk_note"] = df.apply(flag_risk, axis=1)
    return df

# ────────────────────────────────────────────────────────────────────────────────
# API-ready stubs (LLM + Supplier API)
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class APIConfig:
    base_url: str
    api_key: Optional[str] = None

class SupplierAPIClient:
    def __init__(self, cfg: APIConfig): self.cfg = cfg
    def health(self) -> Dict[str, Any]:
        time.sleep(0.2)
        ok = self.cfg.base_url.startswith(("http://","https://"))
        return {"service":"supplier-api","ok":ok,"base_url":self.cfg.base_url}
    def fetch_suppliers(self) -> List[str]:
        time.sleep(0.2)
        return [f"Supplier {i:02d}" for i in range(1, 11)]
    def submit_magic_link(self, supplier_name: str) -> Dict[str, Any]:
        time.sleep(0.3)
        return {"supplier":supplier_name,
                "link":f"{self.cfg.base_url.rstrip('/')}/magic/{supplier_name.replace(' ','_').lower()}",
                "expires_in_hours":72, "status":"sent"}

class LLMClient:
    def __init__(self, provider: str, api_key_present: bool):
        self.provider = provider; self.api_key_present = api_key_present
    def health(self) -> Dict[str, Any]:
        time.sleep(0.2); return {"service":f"llm-{self.provider}", "ok": self.api_key_present}
    def generate(self, prompt: str) -> str:
        time.sleep(0.5)
        return f"[{self.provider.upper()} DEMO OUTPUT]\n" + prompt[:500] + "\n... (truncated)"

SUPPLIER_API = SupplierAPIClient(APIConfig(
    base_url=os.getenv("SUPPLIER_API_BASE","https://api.example.com/suppliers"),
    api_key=os.getenv("SUPPLIER_API_KEY"))
)
LLM = LLMClient(
    provider=os.getenv("LLM_PROVIDER","openai"),
    api_key_present=bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
)

# ────────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────────
def show_header():
    st.markdown("""
    <div style="background:linear-gradient(90deg,#003DA5,#0052CC);color:#fff;padding:16px 18px;border-radius:12px;margin-bottom:10px;">
      <h2 style="margin:0;">📦 Sustainable Procurement & Vendor Cost Analyzer</h2>
      <div style="opacity:.9;margin-top:4px;">Unifies supplier cost analytics, ESG performance, and AI-driven recommendations</div>
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Pages
# ────────────────────────────────────────────────────────────────────────────────
def page_overview():
    st.subheader("Executive Overview")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown("<div class='metric-card'><div>Total Vendors</div><h2>50</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric-card'><div>Sustainable Spend %</div><h2>38%</h2></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric-card'><div>Potential Savings</div><h2>€2.1M</h2></div>", unsafe_allow_html=True)
    with c4: st.markdown("<div class='metric-card'><div>Fine Risk (CSRD)</div><h2>€0.6M</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Spend by Category**")
        if st.session_state.vendor_df is None:
            demo = gen_vendor_demo(50)
            spend_by_cat = demo.groupby("Category")["Spend_USD"].sum().reset_index()
        else:
            spend_by_cat = st.session_state.vendor_df.groupby("Category")["Spend_USD"].sum().reset_index()
        fig = px.bar(spend_by_cat, x="Category", y="Spend_USD", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Average ESG Performance**")
        if st.session_state.vendor_df is None:
            demo = gen_vendor_demo(50)
        else:
            demo = st.session_state.vendor_df
        esg_avg = demo[["Carbon_Score","Waste_Score","Circularity_Score"]].mean().reset_index()
        esg_avg.columns = ["ESG_Dimension","Score"]
        fig2 = px.bar(esg_avg, x="ESG_Dimension", y="Score", range_y=[0,100])
        st.plotly_chart(fig2, use_container_width=True)

def page_supplier_analyzer():
    st.subheader("Supplier Data Upload & Analyzer")
    c1, c2 = st.columns([2,1])
    with c1:
        f = st.file_uploader("Upload supplier dataset (CSV/Excel)", type=["csv","xlsx"], key="vendor_upload")
        if st.button("Load Demo Vendors", key="load_demo"):
            st.session_state.vendor_df = gen_vendor_demo(50)
            st.success("Demo vendor dataset loaded.")
        if f is not None:
            try:
                st.session_state.vendor_df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                st.success(f"Loaded {f.name} ({len(st.session_state.vendor_df)} rows).")
            except Exception as e:
                st.error(f"Could not read file: {e}")
    with c2:
        st.markdown("**Expected Columns**")
        st.caption("Supplier_Name, Category, Country, Spend_USD, Sustainability_Score, "
                   "Carbon_Score, Waste_Score, Circularity_Score, Risk_Level, Emissions_tCO2e, "
                   "Recycled_Content_pct, OnTime_Delivery_pct, ESG_Data_Complete")
    if st.session_state.vendor_df is not None:
        df = st.session_state.vendor_df.copy()
        st.dataframe(df.head(20), use_container_width=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Vendors", len(df))
        with c2: st.metric("Total Spend", f"${df['Spend_USD'].sum():,.0f}")
        with c3: st.metric("Avg Sustain. Score", f"{df['Sustainability_Score'].mean():.0f}")
        with c4: st.metric("Avg Recycled %", f"{df['Recycled_Content_pct'].mean():.1f}%")
    else:
        st.info("Load demo data or upload your supplier file to continue.")

def page_performance_dashboard():
    st.subheader("📊 Procurement & ESG Performance Dashboard")
    if st.session_state.vendor_df is None:
        st.info("Load vendor data first in 'Supplier Analyzer'.")
        return
    df = st.session_state.vendor_df.copy()

    # Filters
    cc1, cc2 = st.columns(2)
    with cc1:
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
        country = st.selectbox("Filter by Country", countries, index=0)
    with cc2:
        cats = ["All"] + sorted(df["Category"].dropna().unique().tolist())
        cat = st.selectbox("Filter by Category", cats, index=0)
    if country != "All": df = df[df["Country"] == country]
    if cat != "All": df = df[df["Category"] == cat]

    # Scatter
    st.markdown("#### Spend vs. Sustainability Score")
    fig1 = px.scatter(
        df, x="Spend_USD", y="Sustainability_Score",
        size="Risk_Level", color="Category",
        hover_name="Supplier_Name", size_max=40,
        labels={"Spend_USD":"Spend (USD)", "Sustainability_Score":"Sustainability Score"}
    )
    fig1.update_layout(height=420)
    st.plotly_chart(fig1, use_container_width=True)

    # Bar
    st.markdown("#### Spend by Category")
    spend_by_cat = df.groupby("Category")["Spend_USD"].sum().reset_index()
    fig2 = px.bar(spend_by_cat, x="Category", y="Spend_USD", text_auto=".2s")
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

    # Heatmap
    st.markdown("#### Supplier vs ESG Dimensions (Heatmap)")
    esg_cols = ["Carbon_Score","Waste_Score","Circularity_Score"]
    if all(c in df.columns for c in esg_cols):
        pivot = df.melt(id_vars=["Supplier_Name"], value_vars=esg_cols)
        fig3 = px.density_heatmap(
            pivot, x="variable", y="Supplier_Name", z="value",
            color_continuous_scale="Blues", height=500
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Upload data including Carbon_Score, Waste_Score, and Circularity_Score for the heatmap.")

def page_savings_simulator():
    st.subheader("💰 Cost & Savings Simulator")
    if st.session_state.vendor_df is None:
        st.info("Load vendor data first in 'Supplier Analyzer'.")
        return
    df = st.session_state.vendor_df
    base_spend = df["Spend_USD"].sum()

    c1,c2,c3 = st.columns(3)
    with c1:
        consolidate = st.slider("Supplier consolidation (%)", 0, 30, 10)
        recycled_boost = st.slider("Increase recycled content (%)", 0, 40, 15)
    with c2:
        logistics_local = st.slider("Shift to regional logistics (%)", 0, 50, 20)
        ontime_improve = st.slider("Improve on-time delivery (pp)", 0, 20, 5)
    with c3:
        contract_reneg = st.slider("Contract renegotiation savings (%)", 0, 15, 6)
        spec_change = st.slider("Spec optimization savings (%)", 0, 10, 4)

    if st.button("Calculate Savings", key="calc_savings"):
        savings = 0
        savings += base_spend * (consolidate/100) * 0.6
        savings += base_spend * (contract_reneg/100)
        savings += base_spend * (spec_change/100) * 0.8
        savings += base_spend * (logistics_local/100) * 0.2
        # CO2 reduction (illustrative)
        co2_reduction = df["Emissions_tCO2e"].sum() * (recycled_boost/100) * 0.15 + (logistics_local/100) * 0.25
        st.markdown("---")
        a,b,c = st.columns(3)
        with a: st.markdown(f"<div class='box'><b>Annual Savings</b><h3>€{savings/1e6:.2f}M</h3></div>", unsafe_allow_html=True)
        with b: st.markdown(f"<div class='box'><b>Spend Reduction</b><h3>{(savings/base_spend)*100:.1f}%</h3></div>", unsafe_allow_html=True)
        with c: st.markdown(f"<div class='box'><b>CO₂ Reduction</b><h3>{co2_reduction:,.0f} tCO₂e</h3></div>", unsafe_allow_html=True)

def page_ai_insights():
    st.subheader("🤖 AI Insights & Recommendations")
    if st.session_state.vendor_df is None:
        st.info("Load vendor data first in 'Supplier Analyzer'.")
        return
    df = st.session_state.vendor_df
    prompt = f"""
You are a procurement & sustainability analyst. Based on supplier data columns
(Spend_USD, Sustainability_Score, Emissions_tCO2e, Recycled_Content_pct, Category, Country),
produce: 1) Top 3 risk findings, 2) Top 3 savings levers, 3) 2 supplier-substitution strategies (Tier-2/near-shore).
Be concise and numeric where possible.
Sample stats: total spend=${df['Spend_USD'].sum():,.0f}, avg sustain score={df['Sustainability_Score'].mean():.0f}, avg recycled%={df['Recycled_Content_pct'].mean():.1f}.
"""
    if st.button("Generate AI Insights", key="ai_btn"):
        st.info(f"LLM provider: {LLM.provider} • API key present: {LLM.api_key_present}")
        out = LLM.generate(prompt)
        st.code(out)

def _make_tiny_pdf_bytes(title: str) -> bytes:
    content = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 88 >> stream
BT /F1 24 Tf 72 700 Td (Evidence Preview:) Tj T* (""" + title.replace("(","[").replace(")","]") + """) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref 0 6
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000110 00000 n 
0000000344 00000 n 
0000000495 00000 n 
trailer << /Root 1 0 R /Size 6 >>
startxref
594
%%EOF"""
    return content.encode("latin-1")

def page_supplier_portal():
    st.subheader("🤝 Supplier Portal & Evidence")
    c1, c2 = st.columns([2,1])
    with c1:
        f = st.file_uploader("Upload supplier evidence manifest (CSV/Excel)", type=["csv","xlsx"], key="sup_upload")
        if st.button("Load Demo Evidence", key="sup_demo_btn"):
            st.session_state.evidence_rows = gen_evidence_manifest()
            st.success("Demo evidence manifest loaded.")
        if f is not None:
            st.session_state.evidence_rows = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            st.success(f"Loaded {f.name} ({len(st.session_state.evidence_rows)} rows).")
    with c2:
        st.markdown("**API-Ready Hooks**")
        if st.button("Test Supplier API Health", key="sup_health_btn"):
            res = SUPPLIER_API.health()
            ok = res.get("ok")
            st.markdown(f"Service: supplier-api • Base: {res['base_url']} • Status: " +
                        (f"<span class='pill ok'>OK</span>" if ok else f"<span class='pill err'>ERROR</span>"),
                        unsafe_allow_html=True)
        suppliers = SUPPLIER_API.fetch_suppliers()
        sel_sup = st.selectbox("Supplier (send magic link)", suppliers, key="sup_ml_sel")
        if st.button("Send Magic Link", key="sup_ml_btn"):
            resp = SUPPLIER_API.submit_magic_link(sel_sup)
            st.success(f"Sent to {resp['supplier']}. Expires in {resp['expires_in_hours']}h.")
            st.code(resp["link"])

    # Supplier Readiness Dashboard
    st.markdown("---")
    st.subheader("📊 Supplier Readiness Dashboard")
    col_demo_a, col_demo_b = st.columns([2,1])
    with col_demo_a:
        if st.button("Load Demo Supplier Readiness", key="sup_readiness_demo_btn"):
            st.session_state.supplier_df = gen_supplier_readiness()
            st.success("Demo supplier readiness loaded.")
    with col_demo_b:
        region_filter = st.selectbox("Filter by region", ["All","EU","NA","APAC"], key="sup_region_filter")

    if st.session_state.supplier_df is not None:
        df_sup = st.session_state.supplier_df.copy()
        if region_filter != "All":
            df_sup = df_sup[df_sup["region"] == region_filter]

        d1,d2,d3,d4 = st.columns(4)
        with d1: st.metric("Suppliers", len(df_sup))
        with d2: st.metric("Avg Readiness", f"{df_sup['readiness_score'].mean():.0f}")
        with d3: st.metric("Evidence Files", int(df_sup['evidence_files'].sum()))
        with d4: st.metric("High-Risk", int((df_sup['readiness_score']<60).sum()))
        st.dataframe(df_sup.sort_values("readiness_score", ascending=False), use_container_width=True)

        st.markdown("#### ✨ Opportunities & Actions (Auto)")
        for _, row in df_sup.sort_values("readiness_score").head(3).iterrows():
            st.write(f"- **{row['supplier_name']}** → {row['risk_note']}. Action: request updated recycled-content proof and SVHC declaration; consider ISO14001 audit.")

    # Evidence Inbox (inline preview)
    st.markdown("---")
    st.subheader("📥 Evidence Inbox")
    if st.session_state.evidence_rows is not None and len(st.session_state.evidence_rows):
        st.dataframe(st.session_state.evidence_rows, use_container_width=True)
        sel_eid = st.selectbox("Preview evidence id", st.session_state.evidence_rows["id"].tolist(), key="evidence_preview_sel")
        if st.button("Preview Selected", key="evidence_preview_btn"):
            pdf_bytes = _make_tiny_pdf_bytes(title=f"Evidence {sel_eid}")
            b64 = base64.b64encode(pdf_bytes).decode()
            st.download_button("⬇️ Download selected PDF", data=pdf_bytes, file_name=f"{sel_eid}.pdf", mime="application/pdf")
            st.markdown(f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='480' style='border:1px solid #eee;border-radius:8px'></iframe>", unsafe_allow_html=True)
    else:
        st.caption("Upload or load the evidence manifest to review submissions.")

    # Agent 1 — Supplier Risk Monitor (Preview)
    st.markdown("---")
    st.subheader("🤖 Agent 1 — Supplier Risk Monitor (Preview)")
    st.caption("Shows how an agent could watch news/regulatory lists and flag supplier risks or greener options.")
    col_a, col_b = st.columns([2,1])
    with col_a:
        if st.button("Run Risk Scan (Demo)", key="agent1_scan_btn"):
            if st.session_state.supplier_df is None or len(st.session_state.supplier_df) == 0:
                st.info("Load the Supplier Readiness demo above first.")
            else:
                df = st.session_state.supplier_df.sort_values("readiness_score")
                flags = df.head(2).copy()
                flags["event"] = ["Media mention: packaging complaint (APAC)", "Expired certificate in registry"]
                flags["agent_action"] = [
                    "Request updated PCR certificate; propose alternative vendor with higher recycled content.",
                    "Send magic link for re-attestation; escalate to procurement if no response in 5 days."
                ]
                st.success("Risk scan completed (demo).")
                st.dataframe(flags[[
                    "supplier_name","region","primary_material","readiness_score","risk_note","event","agent_action"
                ]], use_container_width=True)
    with col_b:
        st.markdown("**Agent Configuration (Demo)**")
        st.write("- Monitors: news RSS, SVHC updates, supplier portals\n- Triggers: certificate expiry, negative media, new SVHC list\n- Outputs: alerts, Planner/Jira tasks, supplier ‘magic links’")
        if st.button("Generate Agent Summary (LLM Demo)", key="agent1_llm_btn"):
            prompt = "Summarize 3 actions to improve supplier readiness, prioritizing recycled-content proof and SVHC declarations."
            out = LLM.generate(prompt)
            st.code(out)

def page_reports():
    st.subheader("📄 Reports & Summaries")
    if st.button("Generate Executive Summary", key="rep_exec_btn"):
        time.sleep(0.5)
        report = """# Sustainable Procurement — Executive Summary (Demo)

- Consolidation, spec optimization, and renegotiation → €2.1M modeled savings.
- High-spend vs. low-sustainability suppliers identified on dashboard.
- Supplier readiness dashboard highlights missing recycled-content proofs & SVHC declarations.
- Platform is API-ready to connect to Ariba/Coupa; supports AI narratives & insights.

*Generated by demo app; replace with automated pipeline in production.*
"""
        st.code(report, language="markdown")
        st.download_button("📥 Download (Markdown)", data=report, file_name="Sustainable_Procurement_Summary.md")

def page_api():
    st.subheader("🔗 API & Integrations (Demo)")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Check LLM & Supplier API", key="api_health_btn"):
            l = LLM.health(); s = SUPPLIER_API.health()
            st.markdown(f"LLM: {(('<span class=\"pill ok\">OK</span>') if l['ok'] else '<span class=\"pill err\">ERROR</span>')} — provider={LLM.provider}", unsafe_allow_html=True)
            st.markdown(f"Supplier API: {(('<span class=\"pill ok\">OK</span>') if s['ok'] else '<span class=\"pill err\">ERROR</span>')} — base={s['base_url']}", unsafe_allow_html=True)
        st.markdown("**Environment**")
        st.code(json.dumps({
            "SUPPLIER_API_BASE": os.getenv("SUPPLIER_API_BASE","https://api.example.com/suppliers"),
            "SUPPLIER_API_KEY": "set" if os.getenv("SUPPLIER_API_KEY") else "not set",
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER","openai"),
            "OPENAI_API_KEY": "set" if os.getenv("OPENAI_API_KEY") else "not set",
            "ANTHROPIC_API_KEY": "set" if os.getenv("ANTHROPIC_API_KEY") else "not set",
        }, indent=2))
    with c2:
        st.markdown("**OpenAPI (sample)**")
        openapi = {
            "openapi":"3.0.0",
            "info":{"title":"Supplier Evidence API","version":"0.1.0"},
            "paths":{
                "/suppliers":{"get":{"summary":"List suppliers","responses":{"200":{"description":"OK"}}}},
                "/suppliers/{name}/magic-link":{"post":{"summary":"Create evidence portal link","responses":{"200":{"description":"OK"}}}},
                "/evidence":{"post":{"summary":"Upload evidence metadata","responses":{"201":{"description":"Created"}}}},
            }
        }
        st.code(json.dumps(openapi, indent=2), language="json")
        st.markdown("**Supplier Readiness Schema (sample)**")
        readiness_schema = {
            "supplier_name":"string","region":"EU|NA|APAC","primary_material":"PCR-PE|PCR-PET|Bio-PP",
            "recycled_content_cert":"Yes|No|Expired","svhc_declaration":"Yes|No","iso14001":"Yes|No",
            "evidence_files":"int","last_update":"YYYY-MM-DD","readiness_score":"0-100","risk_note":"string"
        }
        st.code(json.dumps(readiness_schema, indent=2), language="json")

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar & Router
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Navigation")
    nav = st.radio("", [
        "Overview",
        "Supplier Analyzer",
        "Performance Dashboard",
        "Savings Simulator",
        "AI Insights",
        "Supplier Portal",
        "Reports",
        "API & Integrations",
    ], index=["Overview","Supplier Analyzer","Performance Dashboard","Savings Simulator","AI Insights","Supplier Portal","Reports","API & Integrations"].index(st.session_state.nav))
    st.session_state.nav = nav
    st.markdown("---")
    st.caption(f"Build: {datetime.now().strftime('%b %d, %Y')} • v2.3")

show_header()
if st.session_state.nav == "Overview":
    page_overview()
elif st.session_state.nav == "Supplier Analyzer":
    page_supplier_analyzer()
elif st.session_state.nav == "Performance Dashboard":
    page_performance_dashboard()
elif st.session_state.nav == "Savings Simulator":
    page_savings_simulator()
elif st.session_state.nav == "AI Insights":
    page_ai_insights()
elif st.session_state.nav == "Supplier Portal":
    page_supplier_portal()
elif st.session_state.nav == "Reports":
    page_reports()
elif st.session_state.nav == "API & Integrations":
    page_api()

st.markdown("---")
st.caption("Sustainable Procurement & Vendor Cost Analyzer — All data simulated for demonstration • © 2025")
