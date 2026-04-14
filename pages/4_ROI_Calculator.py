import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import apply_styles, check_data, section_header, footer

st.set_page_config(page_title="ROI Calculator | EXL", page_icon="💰", layout="wide")
apply_styles()

st.markdown("""
<div class="top-header">
    <div>
        <h1>💰 ROI Calculator</h1>
        <p>Quantify the business impact of churn reduction &nbsp;·&nbsp; Model retention scenarios</p>
    </div>
    <div class="header-badge">Business Value</div>
</div>
""", unsafe_allow_html=True)

df = check_data()

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white", font_family="Inter",
    title_font_size=14, title_font_color="#0F172A",
    margin=dict(l=10, r=10, t=50, b=10)
)

if "Churned" not in df.columns or "Monthly_Value" not in df.columns:
    st.warning("Need 'Churned' and 'Monthly_Value' columns for ROI calculation.")
    st.stop()

# ── Baseline metrics ───────────────────────────────────────────────────────────
section_header("Current Churn Impact", "Baseline — what churn is costing the business right now")

total         = len(df)
churned_count = int((df["Churned"] == "Yes").sum())
retained      = total - churned_count
churn_rate    = churned_count / total * 100
avg_value     = df["Monthly_Value"].mean()
monthly_lost  = df[df["Churned"] == "Yes"]["Monthly_Value"].sum()
annual_lost   = monthly_lost * 12

c1, c2, c3, c4 = st.columns(4)
c1.metric("Churned Customers",      churned_count)
c2.metric("Churn Rate",             f"{churn_rate:.1f}%")
c3.metric("Monthly Revenue Lost",   f"₹{int(monthly_lost):,}")
c4.metric("Annual Revenue Lost",    f"₹{int(annual_lost):,}")

st.markdown("---")

# ── Scenario builder ───────────────────────────────────────────────────────────
section_header("Build Your Retention Scenario", "Adjust the inputs — results update instantly")

col_in, col_out = st.columns([1, 2])

with col_in:
    reduction_pct = st.slider("Churn Reduction Target (%)", 10, 90, 30, 5,
                               help="How much of current churn can your retention programme realistically prevent?")
    cost_per_customer = st.number_input(
        "Retention Cost per Customer (₹/month)", min_value=0, max_value=10000, value=500, step=100,
        help="Total monthly cost to retain one at-risk customer (outreach, offers, CSM time)")
    horizon = st.selectbox("Time Horizon", ["1 Year", "2 Years", "3 Years"])
    discount_rate = st.slider("Discount Rate (% per annum)", 5, 25, 10,
                               help="Used to calculate NPV — typically WACC or cost of capital")

    years = int(horizon[0])
    months = years * 12

    customers_saved   = round(churned_count * reduction_pct / 100)
    monthly_recovered = customers_saved * avg_value
    total_revenue     = monthly_recovered * months
    total_cost        = customers_saved * cost_per_customer * months
    net_roi           = total_revenue - total_cost
    roi_pct           = (net_roi / total_cost * 100) if total_cost > 0 else 0
    breakeven_months  = (total_cost / months) / monthly_recovered * months if monthly_recovered > 0 else float("inf")
    npv = sum([net_roi / years / (1 + discount_rate / 100) ** yr for yr in range(1, years + 1)])

with col_out:
    r1, r2 = st.columns(2)
    r3, r4 = st.columns(2)
    r5, r6 = st.columns(2)

    r1.metric("Customers Saved",              customers_saved)
    r2.metric("Monthly Revenue Recovered",    f"₹{int(monthly_recovered):,}")
    r3.metric(f"Total Revenue ({horizon})",   f"₹{int(total_revenue):,}")
    r4.metric("Total Retention Cost",         f"₹{int(total_cost):,}")
    r5.metric("Net ROI",                      f"₹{int(net_roi):,}",
              delta=f"{roi_pct:.0f}% return on spend")
    r6.metric("NPV of Initiative",            f"₹{int(npv):,}")

    if monthly_recovered > 0:
        be = total_cost / monthly_recovered
        color = "#059669" if be <= 12 else ("#D97706" if be <= 24 else "#DC2626")
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:16px 20px;border-left:4px solid {color};
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-top:12px">
            <p style="font-size:0.72rem;color:#64748B;text-transform:uppercase;font-weight:600;margin:0">Break-even Point</p>
            <p style="font-size:1.6rem;font-weight:800;color:{color};margin:4px 0">{be:.1f} months</p>
            <p style="font-size:0.8rem;color:#64748B;margin:0">At ₹{cost_per_customer:,}/customer/month retention spend</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Scenario comparison chart ──────────────────────────────────────────────────
section_header("Scenario Comparison", "How ROI changes across different churn reduction targets")

reductions = list(range(10, 91, 10))
rows = []
for r in reductions:
    cs  = round(churned_count * r / 100)
    rev = cs * avg_value * months
    cst = cs * cost_per_customer * months
    net = rev - cst
    rows.append({"Reduction": f"{r}%", "Revenue Saved": rev, "Retention Cost": cst, "Net ROI": net})

scen_df = pd.DataFrame(rows)

fig = go.Figure()
fig.add_trace(go.Bar(name="Revenue Saved",   x=scen_df["Reduction"], y=scen_df["Revenue Saved"],   marker_color="#059669"))
fig.add_trace(go.Bar(name="Retention Cost",  x=scen_df["Reduction"], y=scen_df["Retention Cost"],  marker_color="#DC2626"))
fig.add_trace(go.Scatter(name="Net ROI",     x=scen_df["Reduction"], y=scen_df["Net ROI"],
                          mode="lines+markers", line=dict(color="#1A3A6E", width=3),
                          marker=dict(size=7), yaxis="y2"))

fig.update_layout(
    barmode="group",
    title=f"Revenue Saved vs Retention Cost ({horizon})",
    yaxis=dict(title="Amount (₹)"),
    yaxis2=dict(title="Net ROI (₹)", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=1.08),
    **PLOTLY_LAYOUT
)
st.plotly_chart(fig, use_container_width=True)

# Waterfall chart for selected scenario
st.markdown(f"**Waterfall breakdown — {reduction_pct}% reduction scenario ({horizon})**")
fig_wf = go.Figure(go.Waterfall(
    name="", orientation="v",
    measure=["absolute", "relative", "total"],
    x=["Revenue Recovered", "Retention Cost", "Net ROI"],
    y=[total_revenue, -total_cost, 0],
    connector={"line": {"color": "#CBD5E1"}},
    decreasing={"marker": {"color": "#DC2626"}},
    increasing={"marker": {"color": "#059669"}},
    totals={"marker": {"color": "#1A3A6E"}},
    text=[f"₹{int(total_revenue):,}", f"-₹{int(total_cost):,}", f"₹{int(net_roi):,}"],
    textposition="outside"
))
fig_wf.update_layout(**PLOTLY_LAYOUT)
st.plotly_chart(fig_wf, use_container_width=True)

st.markdown("---")

# ── Segment-level opportunity ──────────────────────────────────────────────────
if "Customer_Segment" in df.columns:
    section_header("Retention Opportunity by Segment", "Where to focus retention spend for maximum impact")

    seg_opp = df[df["Churned"] == "Yes"].groupby("Customer_Segment").agg(
        Churned_Count=("Churned", "count"),
        Monthly_Revenue_Lost=("Monthly_Value", "sum")
    ).reset_index()
    seg_opp["Annual_Revenue_Lost"] = seg_opp["Monthly_Revenue_Lost"] * 12
    seg_opp = seg_opp.sort_values("Annual_Revenue_Lost", ascending=False)

    fig_seg = px.bar(
        seg_opp, x="Customer_Segment", y="Annual_Revenue_Lost",
        title="Annual Revenue at Risk by Segment (₹)",
        color="Annual_Revenue_Lost",
        color_continuous_scale=["#FEF3C7", "#DC2626"],
        text=seg_opp["Annual_Revenue_Lost"].apply(lambda v: f"₹{int(v):,}")
    )
    fig_seg.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    fig_seg.update_traces(textposition="outside")
    st.plotly_chart(fig_seg, use_container_width=True)

footer()
