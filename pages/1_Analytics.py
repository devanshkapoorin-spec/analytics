import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from utils import apply_styles, check_data, section_header, footer

st.set_page_config(page_title="Analytics | Analytics Tool", page_icon="📊", layout="wide")
apply_styles()

st.markdown("""
<div class="top-header">
    <div>
        <h1>📊 Advanced Analytics</h1>
        <p>Correlation analysis &nbsp;·&nbsp; Cohort retention &nbsp;·&nbsp; Anomaly detection</p>
    </div>
    <div class="header-badge">Deep Analytics</div>
</div>
""", unsafe_allow_html=True)

df = check_data()

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white", font_family="Inter",
    title_font_size=14, title_font_color="#0F172A",
    margin=dict(l=10, r=10, t=50, b=10)
)

# ── Correlation Heatmap ────────────────────────────────────────────────────────
section_header("Correlation Matrix", "Relationships between key numeric variables")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Heatmap", aspect="auto"
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("What does this tell us?"):
        st.markdown("""
        - Values close to **+1** → strong positive correlation (both rise together)
        - Values close to **-1** → strong negative correlation (one rises as other falls)
        - Values near **0** → little relationship between the two variables
        - High correlation between a feature and **Risk_Score** suggests it's a good churn predictor
        """)

st.markdown("---")

# ── Cohort Analysis ────────────────────────────────────────────────────────────
section_header("Cohort Analysis", "Customer retention rates and value across tenure cohorts")

if "Tenure_Months" in df.columns:
    df_c = df.copy()
    bins = [0, 12, 24, 36, 48, float("inf")]
    labels = ["0–12 mo", "13–24 mo", "25–36 mo", "37–48 mo", "49+ mo"]
    df_c["Cohort"] = pd.cut(df_c["Tenure_Months"], bins=bins, labels=labels)

    agg = {"Tenure_Months": "count"}
    if "Churned" in df.columns:
        agg["Churned"] = lambda x: (x == "Yes").sum() / len(x) * 100
    if "Monthly_Value" in df.columns:
        agg["Monthly_Value"] = "mean"

    cohort_stats = df_c.groupby("Cohort", observed=True).agg(agg).reset_index()
    cohort_stats.rename(columns={"Tenure_Months": "Count"}, inplace=True)
    if "Churned" in cohort_stats.columns:
        cohort_stats.rename(columns={"Churned": "Churn Rate (%)"}, inplace=True)
    if "Monthly_Value" in cohort_stats.columns:
        cohort_stats.rename(columns={"Monthly_Value": "Avg Value (₹)"}, inplace=True)

    col1, col2 = st.columns(2)

    with col1:
        if "Churn Rate (%)" in cohort_stats.columns:
            fig_cr = px.bar(
                cohort_stats, x="Cohort", y="Churn Rate (%)",
                title="Churn Rate by Tenure Cohort (%)",
                color="Churn Rate (%)",
                color_continuous_scale=["#059669", "#D97706", "#DC2626"],
                text=cohort_stats["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%")
            )
            fig_cr.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
            fig_cr.update_traces(textposition="outside")
            st.plotly_chart(fig_cr, use_container_width=True)

    with col2:
        if "Avg Value (₹)" in cohort_stats.columns:
            fig_av = px.bar(
                cohort_stats, x="Cohort", y="Avg Value (₹)",
                title="Avg Monthly Value by Cohort (₹)",
                text=cohort_stats["Avg Value (₹)"].apply(lambda v: f"₹{int(v):,}")
            )
            fig_av.update_layout(**PLOTLY_LAYOUT)
            fig_av.update_traces(textposition="outside", marker_color="#1A3A6E")
            st.plotly_chart(fig_av, use_container_width=True)

    st.dataframe(
        cohort_stats.style.format({
            "Churn Rate (%)": "{:.1f}",
            "Avg Value (₹)": "₹{:,.0f}"
        } if "Churn Rate (%)" in cohort_stats.columns else {}),
        use_container_width=True
    )

st.markdown("---")

# ── Anomaly Detection ──────────────────────────────────────────────────────────
section_header("Anomaly Detection", "Isolation Forest flags customers with unusual behaviour patterns",
               "linear-gradient(180deg, #DC2626, #991B1B)")

ANOM_FEATURES = [c for c in ["Age", "Tenure_Months", "Monthly_Value", "Transactions_Last6Months", "Risk_Score"] if c in df.columns]

if len(ANOM_FEATURES) >= 2:
    X_anom = df[ANOM_FEATURES].dropna()
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(X_anom)
    scores = iso.score_samples(X_anom)

    anom_df = df.loc[X_anom.index].copy()
    anom_df["Status"] = ["Anomalous" if p == -1 else "Normal" for p in preds]
    anom_df["Anomaly Score"] = scores.round(4)

    n_anom = (preds == -1).sum()
    st.info(f"Isolation Forest flagged **{n_anom} anomalous customers** ({n_anom / len(X_anom) * 100:.1f}% of dataset) with unusual behaviour patterns.")

    col1, col2 = st.columns([3, 2])

    with col1:
        if "Tenure_Months" in df.columns and "Monthly_Value" in df.columns:
            fig_anom = px.scatter(
                anom_df, x="Tenure_Months", y="Monthly_Value",
                color="Status", symbol="Status",
                color_discrete_map={"Normal": "#1A3A6E", "Anomalous": "#DC2626"},
                hover_data=["CustomerID"] if "CustomerID" in anom_df.columns else None,
                size_max=10,
                title="Anomaly Detection: Tenure vs Monthly Value"
            )
            fig_anom.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_anom, use_container_width=True)

    with col2:
        flagged = anom_df[anom_df["Status"] == "Anomalous"].sort_values("Anomaly Score")
        display = [c for c in ["CustomerID", "Tenure_Months", "Monthly_Value", "Risk_Score", "Customer_Segment", "Anomaly Score"] if c in flagged.columns]
        st.markdown("**Flagged Customers**")
        st.dataframe(flagged[display].reset_index(drop=True), use_container_width=True)

    if "Risk_Score" in df.columns:
        col3, col4 = st.columns(2)
        with col3:
            fig_rs = px.histogram(
                anom_df, x="Risk_Score", color="Status",
                color_discrete_map={"Normal": "#1A3A6E", "Anomalous": "#DC2626"},
                barmode="overlay", opacity=0.75,
                title="Risk Score Distribution: Normal vs Anomalous"
            )
            fig_rs.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_rs, use_container_width=True)

        with col4:
            if "Transactions_Last6Months" in df.columns:
                fig_tx = px.box(
                    anom_df, x="Status", y="Transactions_Last6Months",
                    color="Status",
                    color_discrete_map={"Normal": "#1A3A6E", "Anomalous": "#DC2626"},
                    title="Transaction Volume: Normal vs Anomalous"
                )
                fig_tx.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_tx, use_container_width=True)

footer()
