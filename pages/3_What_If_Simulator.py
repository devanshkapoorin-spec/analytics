import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import apply_styles, check_data, section_header, footer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="What-If Simulator | EXL", page_icon="🎯", layout="wide")
apply_styles()

st.markdown("""
<div class="top-header">
    <div>
        <h1>🎯 What-If Simulator</h1>
        <p>Adjust customer attributes in real time and watch churn probability change instantly</p>
    </div>
    <div class="header-badge">Scenario Analysis</div>
</div>
""", unsafe_allow_html=True)

df = check_data()

FEATURES = [f for f in ["Age", "Tenure_Months", "Monthly_Value", "Transactions_Last6Months", "Risk_Score"] if f in df.columns]

if "Churned" not in df.columns or len(FEATURES) < 3:
    st.warning("Need a 'Churned' column and at least 3 numeric features to run the simulator.")
    st.stop()

ml_df = df[FEATURES + ["Churned"]].dropna().copy()
ml_df["Churn_Binary"] = (ml_df["Churned"] == "Yes").astype(int)
X = ml_df[FEATURES]
y = ml_df["Churn_Binary"]
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# ── Customer selector ──────────────────────────────────────────────────────────
section_header("Select a Customer", "Choose a customer to analyse — sliders pre-fill with their current values")

col_sel, col_sim = st.columns([1, 2])

with col_sel:
    if "CustomerID" in df.columns:
        selected_id = st.selectbox("Customer ID", df["CustomerID"].tolist())
        row = df[df["CustomerID"] == selected_id].iloc[0]
    else:
        idx = st.slider("Customer Index", 0, len(df) - 1, 0)
        row = df.iloc[idx]

    st.markdown("---")
    st.markdown("**Current Profile**")
    for f in FEATURES:
        val = row[f] if not pd.isna(row[f]) else "N/A"
        unit = "₹" if f == "Monthly_Value" else ""
        st.metric(f.replace("_", " "), f"{unit}{val:,.0f}" if isinstance(val, float) else val)

    if "Customer_Segment" in df.columns:
        st.metric("Segment", row["Customer_Segment"])
    if "Churned" in df.columns:
        actual = row["Churned"]
        st.metric("Actual Churn", actual)

with col_sim:
    st.markdown("**Adjust attributes — churn probability updates in real time**")

    inputs = {}
    for f in FEATURES:
        col_min = float(df[f].min())
        col_max = float(df[f].max())
        default  = float(row[f]) if not pd.isna(row[f]) else col_min
        step = 1.0 if f in ["Age", "Tenure_Months", "Transactions_Last6Months", "Risk_Score"] else 100.0
        inputs[f] = st.slider(
            f.replace("_", " "),
            min_value=col_min, max_value=col_max,
            value=default, step=step
        )

    input_arr     = np.array([[inputs[f] for f in FEATURES]])
    original_arr  = np.array([[float(row[f]) if not pd.isna(row[f]) else float(df[f].median()) for f in FEATURES]])

    churn_prob    = model.predict_proba(input_arr)[0][1] * 100
    original_prob = model.predict_proba(original_arr)[0][1] * 100

    clv_annual = inputs.get("Monthly_Value", 0) * 12 * (1 - churn_prob / 100)

    bar_color  = "#DC2626" if churn_prob >= 60 else ("#D97706" if churn_prob >= 30 else "#059669")
    risk_label = "High Risk" if churn_prob >= 60 else ("Medium Risk" if churn_prob >= 30 else "Low Risk")

    col_g, col_m = st.columns(2)

    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob,
            number={"suffix": "%", "font": {"size": 32}},
            delta={
                "reference": original_prob,
                "increasing": {"color": "#DC2626"},
                "decreasing": {"color": "#059669"},
                "suffix": "%"
            },
            title={"text": "Churn Probability", "font": {"size": 15, "color": "#0F172A"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": bar_color, "thickness": 0.25},
                "steps": [
                    {"range": [0, 30],  "color": "#D1FAE5"},
                    {"range": [30, 60], "color": "#FEF3C7"},
                    {"range": [60, 100],"color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": "#0A1628", "width": 3}, "thickness": 0.8, "value": churn_prob}
            }
        ))
        fig_gauge.update_layout(height=280, paper_bgcolor="white", font_family="Inter",
                                margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_m:
        delta_str = f"{'▲' if churn_prob > original_prob else '▼'} {abs(churn_prob - original_prob):.1f}% vs original"
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:20px;border-left:4px solid {bar_color};box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-top:10px">
            <p style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;margin:0 0 2px 0">Risk Level</p>
            <p style="font-size:1.8rem;font-weight:800;color:{bar_color};margin:0 0 4px 0">{risk_label}</p>
            <p style="font-size:0.75rem;color:#94A3B8;margin:0">{delta_str}</p>
            <hr style="border:none;border-top:1px solid #E2E8F0;margin:14px 0">
            <p style="font-size:0.7rem;color:#64748B;text-transform:uppercase;font-weight:600;margin:0 0 2px 0">Projected Annual CLV</p>
            <p style="font-size:1.5rem;font-weight:700;color:#059669;margin:0">₹{int(clv_annual):,}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP individual explanation ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Why is this prediction made?**")

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        input_df = pd.DataFrame([inputs], columns=FEATURES)
        sv = explainer.shap_values(input_df)
        # SHAP <0.46: list -> sv[1][0] shape (n_features,)
        # SHAP >=0.46: array (1, n_features, n_classes) -> sv[0, :, 1]
        if isinstance(sv, list):
            sv_class1 = sv[1][0]
        else:
            sv_class1 = sv[0, :, 1]

        shap_exp = pd.DataFrame({"Feature": FEATURES, "SHAP Value": sv_class1})
        shap_exp["Direction"] = shap_exp["SHAP Value"].apply(
            lambda v: "Increases Churn Risk" if v > 0 else "Reduces Churn Risk")
        shap_exp = shap_exp.sort_values("SHAP Value")

        fig_sw = go.Figure(go.Bar(
            x=shap_exp["SHAP Value"],
            y=shap_exp["Feature"],
            orientation="h",
            marker_color=["#DC2626" if v > 0 else "#059669" for v in shap_exp["SHAP Value"]],
            text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in shap_exp["SHAP Value"]],
            textposition="outside"
        ))
        fig_sw.update_layout(
            title="Feature Contributions for This Customer",
            xaxis_title="SHAP Value (red = increases churn risk, green = reduces it)",
            plot_bgcolor="white", paper_bgcolor="white", font_family="Inter",
            margin=dict(l=10, r=60, t=50, b=10), height=280,
            title_font_size=13, title_font_color="#0F172A"
        )
        st.plotly_chart(fig_sw, use_container_width=True)

        top_risk = shap_exp[shap_exp["SHAP Value"] > 0].sort_values("SHAP Value", ascending=False)
        if not top_risk.empty:
            top_feature = top_risk.iloc[0]["Feature"]
            st.info(f"**Biggest churn driver for this customer:** {top_feature.replace('_', ' ')} — reducing this attribute would have the most impact on retention.")
    else:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance")
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
            marker_color="#1A3A6E",
            text=[f"{v:.3f}" for v in imp_df["Importance"]], textposition="outside"
        ))
        fig_imp.update_layout(title="Global Feature Importance (install shap for individual explanations)",
                              plot_bgcolor="white", paper_bgcolor="white", font_family="Inter",
                              margin=dict(l=10, r=60, t=50, b=10), height=280)
        st.plotly_chart(fig_imp, use_container_width=True)

footer()
