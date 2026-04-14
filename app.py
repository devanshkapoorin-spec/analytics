import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from utils import apply_styles, section_header, metric_card, footer

load_dotenv()

st.set_page_config(
    page_title="EXL AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_styles()

st.markdown("""
<div class="top-header">
    <div>
        <h1>📊 EXL AI Business Analytics Platform</h1>
        <p>Predictive Customer Analytics &nbsp;·&nbsp; AI-Powered Insights &nbsp;·&nbsp; Built for EXL Service</p>
    </div>
    <div class="header-badge">🤖 Powered by Groq AI</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### EXL ANALYTICS")
    st.markdown("---")
    st.markdown("### DATA INPUT")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    use_sample = st.checkbox("Use sample dataset", value=True)
    st.markdown("---")
    st.markdown("### NAVIGATION")
    st.markdown("""
- 🏠 **Home** — KPI Overview
- 📊 **1 Analytics** — Deep analysis
- 🤖 **2 ML Predictions** — Churn & CLV
- 🎯 **3 What If Simulator** — Scenarios
- 💰 **4 ROI Calculator** — Business value
- 📄 **5 AI Report** — Executive summary
    """)
    st.markdown("---")
    st.markdown("### HOW TO USE")
    st.markdown("""
1. Upload a CSV or use sample data
2. Explore the Analytics dashboard
3. Review ML churn predictions
4. Test scenarios in the simulator
5. Calculate ROI of retention plans
6. Generate & download the AI report
    """)

# ── Load data ──────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    st.success(f"Uploaded: {uploaded_file.name} — {len(df)} records")
elif use_sample:
    try:
        df = pd.read_csv("sample_data.csv")
        st.session_state['df'] = df
    except FileNotFoundError:
        st.error("sample_data.csv not found.")
        st.stop()

if 'df' not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;background:white;border-radius:14px;margin-top:40px;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
        <div style="font-size:3rem;margin-bottom:16px">📊</div>
        <h2 style="color:#0F172A;font-size:1.4rem;font-weight:700">Upload your data to get started</h2>
        <p style="color:#64748B;font-size:1rem;margin-top:8px">Check "Use sample dataset" in the sidebar, or upload your own CSV file.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state['df']

# ── KPI Metrics ────────────────────────────────────────────────────────────────
section_header("Business Overview", "Key performance indicators across your customer base")

total = len(df)
churned = int((df["Churned"] == "Yes").sum()) if "Churned" in df.columns else 0
churn_rate = round(churned / total * 100, 1)
avg_value = round(df["Monthly_Value"].mean(), 0) if "Monthly_Value" in df.columns else 0
high_value = int((df["Customer_Segment"] == "High Value").sum()) if "Customer_Segment" in df.columns else 0
revenue_at_risk = round(df[df["Churned"] == "Yes"]["Monthly_Value"].sum(), 0) if "Churned" in df.columns and "Monthly_Value" in df.columns else 0
avg_risk = round(df["Risk_Score"].mean(), 1) if "Risk_Score" in df.columns else 0

cols = st.columns(6)
cards = [
    ("👥", str(total), "Total Customers", "#2563EB"),
    ("⚠️", f"{churn_rate}%", "Churn Rate", "#DC2626"),
    ("💰", f"₹{int(avg_value):,}", "Avg Monthly Value", "#2563EB"),
    ("⭐", str(high_value), "High Value Customers", "#059669"),
    ("📉", f"₹{int(revenue_at_risk):,}", "Revenue at Risk", "#DC2626"),
    ("🎯", str(avg_risk), "Avg Risk Score", "#D97706"),
]
for col, (icon, val, label, color) in zip(cols, cards):
    with col:
        st.markdown(metric_card(icon, val, label, color), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
section_header("Analytics Dashboard", "Visual breakdown of churn, revenue, and customer behaviour")

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white", font_family="Inter",
    title_font_size=14, title_font_color="#0F172A",
    margin=dict(l=10, r=10, t=50, b=10)
)

col1, col2 = st.columns(2)

with col1:
    if "Customer_Segment" in df.columns and "Churned" in df.columns:
        churn_seg = df.groupby("Customer_Segment")["Churned"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100).reset_index()
        churn_seg.columns = ["Segment", "Churn Rate"]
        churn_seg = churn_seg.sort_values("Churn Rate")
        churn_seg["Color"] = churn_seg["Churn Rate"].apply(
            lambda v: "#059669" if v < 20 else ("#D97706" if v < 50 else "#DC2626"))
        fig = px.bar(churn_seg, x="Churn Rate", y="Segment", orientation="h",
                     color="Color", color_discrete_map="identity",
                     title="Churn Rate by Customer Segment",
                     text=churn_seg["Churn Rate"].apply(lambda v: f"{v:.1f}%"))
        fig.update_layout(showlegend=False, xaxis_title="Churn Rate (%)", yaxis_title="", **PLOTLY_LAYOUT)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if "Product" in df.columns and "Churned" in df.columns:
        churn_prod = df.groupby("Product")["Churned"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100).reset_index()
        churn_prod.columns = ["Product", "Churn Rate"]
        churn_prod = churn_prod.sort_values("Churn Rate")
        churn_prod["Color"] = churn_prod["Churn Rate"].apply(
            lambda v: "#059669" if v < 20 else ("#D97706" if v < 50 else "#DC2626"))
        fig2 = px.bar(churn_prod, x="Churn Rate", y="Product", orientation="h",
                      color="Color", color_discrete_map="identity",
                      title="Churn Rate by Product",
                      text=churn_prod["Churn Rate"].apply(lambda v: f"{v:.1f}%"))
        fig2.update_layout(showlegend=False, xaxis_title="Churn Rate (%)", yaxis_title="", **PLOTLY_LAYOUT)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    if "Region" in df.columns and "Monthly_Value" in df.columns:
        region_val = df.groupby("Region")["Monthly_Value"].mean().reset_index()
        region_val.columns = ["Region", "Avg Monthly Value"]
        region_val = region_val.sort_values("Avg Monthly Value", ascending=False)
        fig3 = px.bar(region_val, x="Region", y="Avg Monthly Value",
                      title="Avg Monthly Revenue by Region (₹)",
                      text=region_val["Avg Monthly Value"].apply(lambda v: f"₹{int(v):,}"))
        fig3.update_layout(yaxis_title="Avg Monthly Value (₹)", xaxis_title="", **PLOTLY_LAYOUT)
        fig3.update_traces(textposition="outside", marker_color="#1A3A6E")
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    if "Tenure_Months" in df.columns and "Churned" in df.columns:
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=df[df["Churned"] == "No"]["Tenure_Months"],
            name="Retained", nbinsx=12, marker_color="#1A3A6E", opacity=0.75))
        fig4.add_trace(go.Histogram(
            x=df[df["Churned"] == "Yes"]["Tenure_Months"],
            name="Churned", nbinsx=12, marker_color="#DC2626", opacity=0.75))
        fig4.update_layout(
            barmode="overlay", title="Tenure: Churned vs Retained",
            xaxis_title="Tenure (Months)", yaxis_title="Customers",
            legend=dict(orientation="h", y=1.05), **PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Gender & Region breakdown ──────────────────────────────────────────────────
if "Gender" in df.columns and "Churned" in df.columns:
    section_header("Demographic Breakdown", "Churn patterns across gender and region")
    col5, col6 = st.columns(2)

    with col5:
        gender_churn = df.groupby("Gender")["Churned"].value_counts(normalize=True).mul(100).reset_index()
        gender_churn.columns = ["Gender", "Churned", "Pct"]
        fig5 = px.bar(gender_churn, x="Gender", y="Pct", color="Churned",
                      color_discrete_map={"Yes": "#DC2626", "No": "#1A3A6E"},
                      title="Churn Distribution by Gender (%)",
                      barmode="stack", text=gender_churn["Pct"].apply(lambda v: f"{v:.1f}%"))
        fig5.update_layout(yaxis_title="%", xaxis_title="", **PLOTLY_LAYOUT)
        fig5.update_traces(textposition="inside")
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        if "Region" in df.columns:
            region_churn = df.groupby("Region")["Churned"].apply(
                lambda x: (x == "Yes").sum() / len(x) * 100).reset_index()
            region_churn.columns = ["Region", "Churn Rate"]
            fig6 = px.bar(region_churn, x="Region", y="Churn Rate",
                          title="Churn Rate by Region (%)",
                          color="Churn Rate", color_continuous_scale=["#059669", "#D97706", "#DC2626"],
                          text=region_churn["Churn Rate"].apply(lambda v: f"{v:.1f}%"))
            fig6.update_layout(coloraxis_showscale=False, yaxis_title="Churn Rate (%)", xaxis_title="", **PLOTLY_LAYOUT)
            fig6.update_traces(textposition="outside")
            st.plotly_chart(fig6, use_container_width=True)

footer()
