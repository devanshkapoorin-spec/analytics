import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import apply_styles, check_data, section_header, footer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="ML Predictions | Analytics Tool", page_icon="🤖", layout="wide")
apply_styles()

st.markdown("""
<div class="top-header">
    <div>
        <h1>🤖 ML Predictions</h1>
        <p>Churn prediction &nbsp;·&nbsp; SHAP explainability &nbsp;·&nbsp; CLV &nbsp;·&nbsp; K-Means segmentation</p>
    </div>
    <div class="header-badge">Machine Learning</div>
</div>
""", unsafe_allow_html=True)

df = check_data()

FEATURES = [f for f in ["Age", "Tenure_Months", "Monthly_Value", "Transactions_Last6Months", "Risk_Score"] if f in df.columns]

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white", font_family="Inter",
    title_font_size=14, title_font_color="#0F172A",
    margin=dict(l=10, r=10, t=50, b=10)
)

# ── Train model ────────────────────────────────────────────────────────────────
section_header("Churn Prediction Model", "Random Forest — identifies which customers are most likely to leave",
               "linear-gradient(180deg, #7C3AED, #4F46E5)")

if "Churned" not in df.columns or len(FEATURES) < 3:
    st.warning("Need a 'Churned' column and at least 3 numeric features to run the model.")
    st.stop()

ml_df = df[FEATURES + ["Churned"]].dropna().copy()
ml_df["Churn_Binary"] = (ml_df["Churned"] == "Yes").astype(int)
X = ml_df[FEATURES]
y = ml_df["Churn_Binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

churn_probs = model.predict_proba(X)[:, 1]
result_df = df.loc[ml_df.index].copy()
result_df["Churn_Probability"] = (churn_probs * 100).round(1)
result_df["Risk_Level"] = result_df["Churn_Probability"].apply(
    lambda v: "High" if v >= 60 else ("Medium" if v >= 30 else "Low"))
if "Monthly_Value" in result_df.columns:
    result_df["CLV_Annual"] = (result_df["Monthly_Value"] * 12 * (1 - churn_probs)).round(0).astype(int)

# Share with other pages
st.session_state["model"] = model
st.session_state["features"] = FEATURES
st.session_state["result_df"] = result_df

m1, m2, m3 = st.columns(3)
m1.metric("High Risk Customers",  int((churn_probs >= 0.6).sum()))
m2.metric("Medium Risk Customers", int(((churn_probs >= 0.3) & (churn_probs < 0.6)).sum()))
m3.metric("Low Risk Customers",   int((churn_probs < 0.3).sum()))

tab1, tab2, tab3 = st.tabs(["🚨 High Risk Customers", "📊 SHAP Feature Importance", "💎 Customer Lifetime Value"])

# ── Tab 1: High Risk ───────────────────────────────────────────────────────────
with tab1:
    high_risk = result_df[result_df["Churn_Probability"] >= 60].sort_values("Churn_Probability", ascending=False)
    display_cols = [c for c in ["CustomerID", "Customer_Segment", "Product", "Region",
                                 "Tenure_Months", "Monthly_Value", "Risk_Score",
                                 "Churn_Probability", "Risk_Level", "CLV_Annual"] if c in high_risk.columns]

    st.markdown(f"**{len(high_risk)} customers** have a churn probability ≥ 60%. These require immediate retention action.")

    def risk_color(val):
        if val == "High":   return "background-color: #FEE2E2; color: #991B1B; font-weight: 600"
        if val == "Medium": return "background-color: #FEF3C7; color: #92400E; font-weight: 600"
        return "background-color: #D1FAE5; color: #065F46; font-weight: 600"

    st.dataframe(
        high_risk[display_cols].reset_index(drop=True).style.applymap(
            risk_color, subset=["Risk_Level"] if "Risk_Level" in display_cols else []
        ),
        use_container_width=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if "Customer_Segment" in high_risk.columns:
            seg_dist = high_risk["Customer_Segment"].value_counts().reset_index()
            seg_dist.columns = ["Segment", "Count"]
            fig = px.pie(seg_dist, names="Segment", values="Count",
                         title="High Risk by Segment", hole=0.4,
                         color_discrete_sequence=["#DC2626", "#D97706", "#1A3A6E", "#059669"])
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if "Product" in high_risk.columns:
            prod_dist = high_risk["Product"].value_counts().reset_index()
            prod_dist.columns = ["Product", "Count"]
            fig2 = px.bar(prod_dist, x="Count", y="Product", orientation="h",
                          title="High Risk by Product", color_discrete_sequence=["#DC2626"])
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: SHAP ────────────────────────────────────────────────────────────────
with tab2:
    if SHAP_AVAILABLE:
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            # SHAP <0.46: list of [class0, class1] arrays each (n, f)
            # SHAP >=0.46: single array (n, f, n_classes)
            if isinstance(shap_values, list):
                sv_class1 = shap_values[1]          # (n_samples, n_features)
            else:
                sv_class1 = shap_values[:, :, 1]    # (n_samples, n_features)
            mean_shap = np.abs(sv_class1).mean(axis=0)

        shap_df = pd.DataFrame({"Feature": FEATURES, "SHAP Importance": mean_shap}).sort_values("SHAP Importance")
        fig_shap = px.bar(
            shap_df, x="SHAP Importance", y="Feature", orientation="h",
            title="SHAP Feature Importance — What Drives Churn?",
            color="SHAP Importance", color_continuous_scale=["#93C5FD", "#1A3A6E"],
            text=shap_df["SHAP Importance"].apply(lambda v: f"{v:.3f}")
        )
        fig_shap.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
        fig_shap.update_traces(textposition="outside")
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("SHAP (SHapley Additive exPlanations) measures each feature's actual contribution to churn predictions — more trustworthy than standard feature importance as it accounts for feature interactions.")
    else:
        importances = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance")
        fig_imp = px.bar(importances, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance — What Drives Churn?",
                         text=importances["Importance"].apply(lambda v: f"{v*100:.1f}%"))
        fig_imp.update_layout(**PLOTLY_LAYOUT)
        fig_imp.update_traces(textposition="outside", marker_color="#1A3A6E")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Install `shap` for more accurate SHAP-based importance: `pip install shap`")

# ── Tab 3: CLV ─────────────────────────────────────────────────────────────────
with tab3:
    section_header("Customer Lifetime Value", "Predicted annual CLV adjusted for individual churn risk",
                   "linear-gradient(180deg, #059669, #047857)")

    if "CLV_Annual" not in result_df.columns:
        st.warning("Need 'Monthly_Value' column for CLV calculation.")
    else:
        total_clv = int(result_df["CLV_Annual"].sum())
        avg_clv   = int(result_df["CLV_Annual"].mean())
        at_risk_clv = int(result_df[result_df["Risk_Level"] == "High"]["CLV_Annual"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Portfolio CLV (Annual)", f"₹{total_clv:,}")
        c2.metric("Average Customer CLV", f"₹{avg_clv:,}")
        c3.metric("CLV at Risk (High Risk)", f"₹{at_risk_clv:,}")

        col1, col2 = st.columns(2)

        with col1:
            if "Customer_Segment" in result_df.columns:
                clv_seg = result_df.groupby("Customer_Segment")["CLV_Annual"].mean().reset_index()
                clv_seg.columns = ["Segment", "Avg CLV"]
                fig_clv = px.bar(clv_seg, x="Segment", y="Avg CLV",
                                 title="Average Annual CLV by Segment (₹)",
                                 text=clv_seg["Avg CLV"].apply(lambda v: f"₹{int(v):,}"))
                fig_clv.update_layout(**PLOTLY_LAYOUT)
                fig_clv.update_traces(textposition="outside", marker_color="#059669")
                st.plotly_chart(fig_clv, use_container_width=True)

        with col2:
            fig_scatter = px.scatter(
                result_df, x="Churn_Probability", y="CLV_Annual",
                color="Customer_Segment" if "Customer_Segment" in result_df.columns else None,
                title="CLV vs Churn Probability",
                hover_data=["CustomerID"] if "CustomerID" in result_df.columns else None,
                labels={"Churn_Probability": "Churn Probability (%)", "CLV_Annual": "Annual CLV (₹)"}
            )
            fig_scatter.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_scatter, use_container_width=True)

        disp = [c for c in ["CustomerID", "Customer_Segment", "Monthly_Value", "Churn_Probability", "CLV_Annual"] if c in result_df.columns]
        st.dataframe(result_df[disp].sort_values("CLV_Annual", ascending=False).reset_index(drop=True), use_container_width=True)

st.markdown("---")

# ── K-Means Clustering ─────────────────────────────────────────────────────────
section_header("K-Means Customer Segmentation", "Data-driven clustering — discover natural customer groups",
               "linear-gradient(180deg, #D97706, #B45309)")

if len(FEATURES) >= 2:
    cluster_df = df[FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    k = st.slider("Number of Clusters (k)", 2, 6, 4)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    viz_df = df.loc[cluster_df.index].copy()
    viz_df["Cluster"] = [f"Cluster {i+1}" for i in cluster_labels]
    viz_df["PC1"] = X_pca[:, 0]
    viz_df["PC2"] = X_pca[:, 1]

    col1, col2 = st.columns([3, 2])

    with col1:
        fig_km = px.scatter(
            viz_df, x="PC1", y="PC2", color="Cluster",
            title=f"K-Means (k={k}) — PCA Projection ({var_explained:.1f}% variance explained)",
            hover_data=["CustomerID"] if "CustomerID" in viz_df.columns else None,
            color_discrete_sequence=["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0891B2"]
        )
        fig_km.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_km, use_container_width=True)

    with col2:
        profiles = viz_df.groupby("Cluster")[FEATURES].mean().round(1)
        if "Churned" in viz_df.columns:
            profiles["Churn Rate (%)"] = viz_df.groupby("Cluster")["Churned"].apply(
                lambda x: round((x == "Yes").sum() / len(x) * 100, 1))
        profiles["Count"] = viz_df.groupby("Cluster").size()
        st.markdown("**Cluster Profiles**")
        st.dataframe(profiles, use_container_width=True)

footer()
