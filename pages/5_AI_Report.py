import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import re
from datetime import datetime
import google.generativeai as genai
import markdown as md
from dotenv import load_dotenv
from utils import apply_styles, check_data, section_header, footer

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_root, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AI Report | Analytics Tool", page_icon="📄", layout="wide")
apply_styles()

st.markdown("""
<div class="top-header">
    <div>
        <h1>📄 AI Consulting Report</h1>
        <p>Executive-level analysis written by Gemini AI — ready to share with stakeholders</p>
    </div>
    <div class="header-badge">🤖 Gemini AI</div>
</div>
""", unsafe_allow_html=True)

df = check_data()


def build_summary(df):
    total        = len(df)
    churned      = int((df["Churned"] == "Yes").sum()) if "Churned" in df.columns else 0
    churn_rate   = round(churned / total * 100, 1)
    avg_value    = round(df["Monthly_Value"].mean(), 0) if "Monthly_Value" in df.columns else 0
    high_value   = int((df["Customer_Segment"] == "High Value").sum()) if "Customer_Segment" in df.columns else 0
    rev_at_risk  = round(df[df["Churned"] == "Yes"]["Monthly_Value"].sum(), 0) if "Churned" in df.columns and "Monthly_Value" in df.columns else 0

    s = f"""DATASET: {total} financial services customers
KEY METRICS:
- Churn rate: {churn_rate}%
- Customers churned: {churned}/{total}
- Avg monthly value: ₹{int(avg_value):,}
- Monthly revenue at risk: ₹{int(rev_at_risk):,}
- High Value customers: {high_value}
CHURN BY SEGMENT:\n"""

    if "Customer_Segment" in df.columns and "Churned" in df.columns:
        for seg, grp in df.groupby("Customer_Segment"):
            c = (grp["Churned"] == "Yes").sum()
            r = round(c / len(grp) * 100, 1)
            s += f"  - {seg}: {c}/{len(grp)} churned ({r}%)\n"

    s += "CHURN BY PRODUCT:\n"
    if "Product" in df.columns and "Churned" in df.columns:
        for prod, grp in df.groupby("Product"):
            c = (grp["Churned"] == "Yes").sum()
            r = round(c / len(grp) * 100, 1)
            s += f"  - {prod}: {c}/{len(grp)} churned ({r}%)\n"

    if "Tenure_Months" in df.columns and "Churned" in df.columns:
        tc = round(df[df["Churned"] == "Yes"]["Tenure_Months"].mean(), 1)
        tr = round(df[df["Churned"] == "No"]["Tenure_Months"].mean(), 1)
        s += f"TENURE: Avg churned={tc}mo, avg retained={tr}mo\n"

    if "Risk_Score" in df.columns and "Churned" in df.columns:
        rc = round(df[df["Churned"] == "Yes"]["Risk_Score"].mean(), 1)
        rr = round(df[df["Churned"] == "No"]["Risk_Score"].mean(), 1)
        s += f"RISK SCORES: Avg churned={rc}, avg retained={rr}\n"

    return s


def generate_pdf(report_text, df):
    try:
        from fpdf import FPDF

        total       = len(df)
        churned     = int((df["Churned"] == "Yes").sum()) if "Churned" in df.columns else 0
        churn_rate  = round(churned / total * 100, 1)
        rev_at_risk = round(df[df["Churned"] == "Yes"]["Monthly_Value"].sum(), 0) if "Churned" in df.columns and "Monthly_Value" in df.columns else 0

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header band
        pdf.set_fill_color(10, 22, 40)
        pdf.rect(0, 0, 210, 48, "F")
        pdf.set_xy(15, 10)
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 12, "Analytics Tool Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(147, 197, 253)
        pdf.set_x(15)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y')}    |    {total} Customers", ln=True)

        # Key metrics
        pdf.set_xy(15, 58)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(10, 22, 40)
        pdf.cell(0, 8, "Key Metrics", ln=True)
        pdf.set_draw_color(37, 99, 235)
        pdf.set_line_width(0.4)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(3)

        for label, val in [
            ("Total Customers", str(total)),
            ("Churn Rate", f"{churn_rate}%"),
            ("Monthly Revenue at Risk", f"Rs {int(rev_at_risk):,}"),
        ]:
            pdf.set_x(15)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(70, 7, label)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(10, 22, 40)
            pdf.cell(0, 7, val, ln=True)

        pdf.ln(6)

        # Report body
        for line in report_text.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(3)
                continue

            if line.startswith("## "):
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(10, 22, 40)
                pdf.set_x(15)
                pdf.cell(0, 8, line[3:], ln=True)
                pdf.set_draw_color(37, 99, 235)
                pdf.set_line_width(0.3)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(2)
                continue

            # Skip table separator rows
            if re.match(r"^[\|\s\-:]+$", line):
                continue

            # Table data rows — render as plain text
            if line.startswith("|"):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if cells:
                    clean = "  |  ".join(re.sub(r"\*\*(.*?)\*\*", r"\1", c) for c in cells)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(30, 41, 59)
                    pdf.set_x(15)
                    try:
                        pdf.multi_cell(180, 5, clean.encode("latin-1", "replace").decode("latin-1"))
                    except Exception:
                        pdf.multi_cell(180, 5, clean.encode("ascii", "ignore").decode("ascii"))
                continue

            clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            clean = re.sub(r"\*(.*?)\*",     r"\1", clean)
            clean = re.sub(r"`(.*?)`",        r"\1", clean)

            is_bold = line.startswith("**") or (line.startswith("**") and line.endswith("**"))
            pdf.set_font("Helvetica", "B" if is_bold else "", 10)
            pdf.set_text_color(10, 22, 40 if is_bold else 30)
            pdf.set_x(15)
            try:
                pdf.multi_cell(180, 5.5, clean.encode("latin-1", "replace").decode("latin-1"))
            except Exception:
                pdf.multi_cell(180, 5.5, clean.encode("ascii", "ignore").decode("ascii"))

        # Footer
        pdf.set_y(-18)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 10, "Analytics Tool  ·  Powered by Gemini AI  ·  Built by Devansh Kapoor", align="C")

        return bytes(pdf.output())

    except ImportError:
        return None


# ── Report section ─────────────────────────────────────────────────────────────
section_header("Generate Executive Report", "AI-written consulting report — click to generate")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Add it to your .env file.")
    st.stop()

if st.button("🚀 Generate Executive Report", type="primary"):
    with st.spinner("Gemini AI is analysing your data and writing the report..."):
        summary = build_summary(df)
        prompt = f"""You are a senior analytics consultant at a leading analytics firm presenting findings to a C-suite client in financial services.
Write a sharp, professional consulting report based on this data. Be direct, specific, and opinionated.

{summary}

Write the report in this exact structure using markdown formatting:

## Executive Summary
3 sentences. State the most critical finding with the exact number. State what it costs the business. State the single most urgent action.

## Key Findings
Exactly 5 findings. Format each as:
**Finding [N]: [Bold headline claim]**
[One sentence with the specific data that proves it and what it means for the business.]

## Segment & Product Analysis
Analyse each customer segment and product. Be specific about which are profitable vs at-risk. Identify the biggest opportunity and the biggest threat.

## Root Cause Analysis
Why are customers churning? Use the tenure and risk score data to build a hypothesis. What does the data suggest about the profile of a churning customer?

## Strategic Recommendations
4 recommendations. Format each as:
**[N]. [Action verb + specific action]**
*Impact:* [What business outcome this drives]
*How:* [One specific step to execute this]

## Risk Assessment
| Risk | Likelihood | Business Impact | Urgency |
|------|-----------|-----------------|---------|
List 3 specific risks in this table format.

## Bottom Line
**[One sentence. The single most important thing this business must do in the next 30 days. No caveats. No hedging.]**

Rules:
- Every claim must have a specific number from the data.
- Active voice only. No phrases like "it could be argued" or "it may be worth considering."
- Write like you are billing $500/hour for this analysis.
- Do not repeat the same point twice.
"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini = genai.GenerativeModel("gemini-pro")
            response = gemini.generate_content(prompt)
            st.session_state["report"] = response.text
        except Exception as e:
            st.error(f"Error generating report: {e}")

if "report" in st.session_state:
    report = st.session_state["report"]
    report_html = md.markdown(report, extensions=["tables"])
    st.markdown(f'<div class="report-card">{report_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download as TXT",
            data=report,
            file_name=f"exl_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col2:
        pdf_bytes = generate_pdf(report, df)
        if pdf_bytes:
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_bytes,
                file_name=f"exl_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.caption("Install fpdf2 for PDF export: `pip3 install fpdf2`")

footer()
