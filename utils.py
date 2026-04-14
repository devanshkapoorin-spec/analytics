import streamlit as st


def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #F0F4F8; }
    section[data-testid="stSidebar"] { background-color: #0A1628; }
    section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1.5px; }

    .top-header { background: linear-gradient(135deg, #0A1628 0%, #1A3A6E 100%); padding: 28px 36px; border-radius: 14px; margin-bottom: 28px; display: flex; align-items: center; justify-content: space-between; }
    .top-header h1 { color: #FFFFFF; font-size: 1.75rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .top-header p { color: #93C5FD; font-size: 0.9rem; margin: 6px 0 0 0; }
    .header-badge { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2); color: #FFFFFF; padding: 8px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; white-space: nowrap; }

    .section-header { display: flex; align-items: center; gap: 12px; margin: 32px 0 20px 0; }
    .section-bar { width: 4px; height: 28px; border-radius: 2px; flex-shrink: 0; }
    .section-title { font-size: 1.2rem; font-weight: 700; color: #0F172A; margin: 0; }
    .section-subtitle { font-size: 0.82rem; color: #64748B; margin: 2px 0 0 0; }

    .metric-card { background: #FFFFFF; border-radius: 12px; padding: 22px 18px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06); border-top: 3px solid #2563EB; transition: transform 0.15s; }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-icon { font-size: 1.4rem; margin-bottom: 8px; }
    .metric-value { font-size: 1.9rem; font-weight: 800; color: #0F172A; line-height: 1.1; }
    .metric-label { font-size: 0.78rem; color: #64748B; margin-top: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

    .report-card { background: #FFFFFF; border-radius: 14px; padding: 36px 40px; box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 8px 32px rgba(0,0,0,0.08); border-top: 4px solid #0A1628; line-height: 1.75; color: #1E293B; }
    .report-card h2 { font-size: 1.1rem; font-weight: 700; color: #0A1628; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; margin-top: 28px; }
    .report-card h2:first-child { margin-top: 0; }
    .report-card strong { color: #0A1628; }
    .report-card table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .report-card th { background: #F1F5F9; padding: 10px 14px; text-align: left; font-weight: 600; color: #475569; font-size: 0.8rem; text-transform: uppercase; }
    .report-card td { padding: 10px 14px; border-bottom: 1px solid #F1F5F9; }

    .stButton > button { background: linear-gradient(135deg, #0A1628, #1A3A6E) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 14px 28px !important; font-size: 1rem !important; font-weight: 700 !important; width: 100% !important; transition: opacity 0.2s !important; }
    .stButton > button:hover { opacity: 0.9 !important; }
    hr { border: none; border-top: 1px solid #E2E8F0; margin: 28px 0; }
    .footer { text-align: center; padding: 24px; color: #94A3B8; font-size: 0.78rem; margin-top: 40px; border-top: 1px solid #E2E8F0; }
    </style>
    """, unsafe_allow_html=True)


def check_data():
    if 'df' not in st.session_state:
        st.warning("Please go to the **Home** page and upload or select a dataset first.")
        st.stop()
    return st.session_state['df']


def section_header(title, subtitle="", color="linear-gradient(180deg, #2563EB, #1A3A6E)"):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-bar" style="background:{color}"></div>
        <div>
            <p class="section-title">{title}</p>
            <p class="section-subtitle">{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(icon, value, label, color="#2563EB"):
    val_color = "#DC2626" if color == "#DC2626" else ("#059669" if color == "#059669" else "#0F172A")
    return f"""<div class="metric-card" style="border-top-color:{color}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color:{val_color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def footer():
    st.markdown("""
    <div class="footer">
        EXL AI Business Analytics Platform &nbsp;·&nbsp; Powered by Groq AI &nbsp;·&nbsp; Built by Devansh Kapoor
    </div>
    """, unsafe_allow_html=True)
