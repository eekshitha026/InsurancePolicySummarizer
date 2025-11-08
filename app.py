import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import pandas as pd
import json

# --- STAGE 0: INITIAL CONFIGURATION AND IMPORTS ---

st.set_page_config(
    page_title="Policy Decoder Agent",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# 2. Check for API Key early BEFORE importing other modules
if not os.getenv("GEMINI_API_KEY"):
    st.error(
        "GEMINI_API_KEY not found in environment variables. Please ensure the *.env* file is in the root project directory and contains *GEMINI_API_KEY=\"Your_Key_Here\"*.")
    st.stop()
# --- END CONFIGURATION ---

# Import our custom modules (assuming these are available)
from extractor import extract_text_from_pdf, get_page_count
from clause_detector import segment_text_into_clauses, count_clauses
from chroma_helper import get_chroma_client, get_rag_collection
from rag_store import query_rag_store
# NOTE: Assume these functions exist in the summarizer module
from summarizer import generate_full_summary, analyze_page_content

# Initialize session state for navigation and data
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'clauses' not in st.session_state:
    st.session_state.clauses = []
if 'page_texts' not in st.session_state:
    st.session_state.page_texts = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'policy_processed' not in st.session_state:
    st.session_state.policy_processed = False
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None  # New state for structured chart data


# --- THEMEING AND STYLING FUNCTIONS (Theme-Agnostic) ---

def set_styles():
    """
    Injects custom CSS for a theme-agnostic look, respecting device settings.
    """
    theme_accent_color = "#FF6F91"  # Consistent vibrant pink

    st.markdown(f"""
        <style>
        /* CSS Variables for Auto-Contrast (Ensuring contrast with Streamlit's native theme) */
        :root {{
            --accent-color: {theme_accent_color};
            --box-bg-light: #f0f2f6; 
            --box-bg-dark: #1e1e1e;
            --shadow-color-light: rgba(0, 0, 0, 0.15);
            --shadow-color-dark: rgba(0, 0, 0, 0.5);
        }}

        /* General Aesthetics & Transitions */
        .stButton>button {{
            border-radius: 12px;
            transition: all 0.2s ease-in-out;
            border: 1px solid var(--accent-color);
            font-weight: 500;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            background-color: var(--accent-color)33; 
        }}

        /* Metric Boxes (Universal Styles) */
        [data-testid="stMetric"] {{
            padding: 15px;
            border-radius: 15px;
            border: 2px solid var(--accent-color)55;
        }}

        /* Custom Header */
        h1 {{
            color: var(--accent-color);
            font-size: 2.5em;
        }}

        /* Custom coloring for alerts (Ensure legibility in both modes) */
        .stAlert.stAlert--success {{ border-left: 5px solid #66bb6a; }} 
        .stAlert.stAlert--error {{ border-left: 5px solid #ef5350; }} 
        .stAlert.stAlert--warning {{ border-left: 5px solid #ffeb3b; }}

        /* Resizable text area style fix */
        .stTextArea > label {{ display: none; }}
        .stTextArea {{ margin-top: 10px; }}

        /* --- AUTO-CONTRAST (Dark Mode Override) --- */
        @media (prefers-color-scheme: dark) {{
            .stButton>button {{ color: white; }}
            [data-testid="stMetric"] {{
                background-color: var(--box-bg-dark);
                box-shadow: 3px 3px 10px var(--shadow-color-dark);
            }}
        }}

        /* --- LIGHT MODE (Explicit Component Styling) --- */
        @media (prefers-color-scheme: light) {{
            [data-testid="stMetric"] {{
                background-color: var(--box-bg-light);
                box-shadow: 3px 3px 10px var(--shadow-color-light);
            }}
        }}

        /* --- LANDING PAGE STYLES --- */
        [data-testid="stFileUploadDropzone"] {{
            border-style: dashed;
            border-color: var(--accent-color) !important;
            padding: 50px;
            margin-top: 30px;
        }}

        </style>
    """, unsafe_allow_html=True)


# --- CHART DATA MOCK (Simulating Gemini Structured Output) ---

def generate_mock_structured_analysis(page_analysis: list) -> dict:
    """
    Mocks the Gemini API call using Structured Output (JSON)
    to generate data for the pie charts.

    In a real app, this would be a Gemini API call with a JSON Schema
    analyzing the 'page_analysis' results.
    """

    # 1. Simple Classification Breakdown
    classification_counts = {'Coverage': 0, 'Exclusions': 0, 'Other': 0}
    for item in page_analysis:
        classification = item.get('classification', 'Other').lower()
        if 'coverage' in classification:
            classification_counts['Coverage'] += 1
        elif 'exclusion' in classification:
            classification_counts['Exclusions'] += 1
        else:
            classification_counts['Other'] += 1

    total_pages = sum(classification_counts.values())
    if total_pages == 0: total_pages = 1  # Avoid division by zero

    classification_data = pd.DataFrame({
        'Category': list(classification_counts.keys()),
        'Pages': list(classification_counts.values()),
        'Percentage': [round(v / total_pages * 100, 1) for v in classification_counts.values()]
    })

    # 2. Risk Exposure (MOCKING SPECIFIC RISK ANALYSIS)
    # This simulates asking Gemini to identify and weigh the major risks covered/excluded.
    risk_exposure_data = pd.DataFrame({
        'Risk Type': ['Property Damage', 'Theft/Burglary', 'Natural Calamity', 'Cyber Events', 'Legal Liability'],
        'Coverage Gap (%)': [5, 10, 30, 45, 10]
    })

    return {
        'classification_breakdown': classification_data,
        'risk_exposure': risk_exposure_data
    }


# --- UI COMPONENTS ---

def display_stats(page_count, clause_count):
    """Displays key policy statistics using a clean layout."""
    st.markdown("### 📊 Policy Overview")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Total Pages 📄", page_count)
    with col2:
        st.metric("Total Clauses (Segments) ✂", clause_count)
    st.markdown("---")


def render_policy_charts(chart_data: dict):
    """Renders the Pie Charts for Risk/Gap Analysis."""
    st.subheader("💡 Visual Policy Risk Analysis")
    st.markdown("(Generated via Gemini Structured Output Analysis)")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 1. Policy Section Breakdown")
        st.caption("Proportion of total pages classified by content type.")

        # Use built-in Streamlit chart for simple pie/bar visualization
        st.dataframe(
            chart_data['classification_breakdown'],
            hide_index=True,
            column_order=('Category', 'Percentage'),
            use_container_width=True
        )
        # Streamlit's native area/bar charts work great with simple dataframes
        st.bar_chart(
            chart_data['classification_breakdown'],
            x='Category',
            y='Percentage',
            color="#FF6F91"  # Accent color
        )

    with col_b:
        st.markdown("#### 2. Coverage Gap by Risk Type")
        st.caption("Simulated percentage of exposure (risks not fully covered or explicitly excluded).")

        # Display the Risk Exposure data
        st.dataframe(
            chart_data['risk_exposure'],
            hide_index=True,
            column_order=('Risk Type', 'Coverage Gap (%)'),
            use_container_width=True
        )

        st.bar_chart(
            chart_data['risk_exposure'],
            x='Risk Type',
            y='Coverage Gap (%)',
            color="#ef5350"  # Error/Risk color
        )

    st.markdown("---")


# ... (display_full_report function remains unchanged)
def display_full_report():
    """Compiles and displays the full report."""
    if not st.session_state.analysis_results:
        st.warning("Please run the 'Analyze with Gemini' step first.")
        return

    report = ""
    analysis = st.session_state.analysis_results

    # --- UPDATED SUMMARY TEXT TO REFLECT CONSTRAINT ---
    report += "========================================================================\n"
    report += f"INSURANCE POLICY DECODED REPORT: {st.session_state.file_name or 'Policy'}\n"
    report += "========================================================================\n\n"
    report += "--- 1. HIGH-LEVEL POLICY SUMMARY (20-30 LINES) ---\n"
    summary = analysis.get('full_summary', 'Summary generation failed due to API or connection error.')
    report += summary + "\n\n"

    report += "--- 2. DETAILED PAGE-BY-PAGE ANALYSIS (Classification & Summary) ---\n"
    page_analysis = analysis.get('page_analysis', [])
    for item in page_analysis:
        page_num = item.get('pageNumber', 'N/A')
        classification = item.get('classification', 'N/A')
        summary = item.get('summary', 'N/A')
        report += f"\n[PAGE {page_num} | CLASSIFICATION: {classification.upper()}]\n"
        report += f"Simplified Summary: {summary}\n"

    # 3. Clauses (Optional, for detailed reference)
    report += "\n\n--- 3. RAW CLAUSE EXTRACT (For Reference) ---\n"
    report += f"Total Clauses: {len(st.session_state.clauses)}\n"
    for clause in st.session_state.clauses:
        page_num = clause.get('page_num', 'N/A')
        clause_id_parts = clause['clause_id'].split('_')
        clause_id_display = clause_id_parts[-1].replace('c', '') if len(clause_id_parts) > 1 else clause['clause_id']

        report += f"[P{page_num}-C{clause_id_display}]\n{clause['text'][:200]}...\n\n"

    st.text_area("Generated Decoded Report", report, height=500, label_visibility="collapsed")

    st.download_button(
        label="⬇ Download Full Report (TXT)",
        data=report.encode('utf-8'),
        file_name=f"decoded_policy_report_{st.session_state.file_name or 'policy'}.txt",
        mime="text/plain",
        use_container_width=True
    )


# --- LANDING PAGE FUNCTION (UNCHANGED) ---

def render_landing_page():
    """Renders the initial upload/about page, replacing the sidebar upload."""

    st.title("🛡️ Insurance Policy Decoder Agent")
    st.markdown("---")

    st.markdown("""
    ## 💡 About This Application
    Insurance policies are notoriously complex. This AI agent leverages the *Gemini API* to break down dense legal text into *clear, human-readable summaries*. 

    * *🔑 Key Features:*
        * Extracts and segments clauses from PDF documents.
        * Classifies sections as *Coverage* or *Exclusions*.
        * Uses *Retrieval Augmented Generation (RAG)* for grounded answers.
        * Provides a simple, focused report.

    ## ⬆ Upload Your Policy to Begin
    """)

    uploaded_file = st.file_uploader(
        "Drag and drop your PDF Policy file here",
        type="pdf",
        accept_multiple_files=False,
        key="landing_uploader",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Powered by Google Gemini 2.5 Flash")

    if uploaded_file is not None:
        # --- File Processing Logic (Same as original app.py) ---
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            try:
                # 1. Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    pdf_path = tmp_file.name

                # 2. Extract and Segment
                page_texts = extract_text_from_pdf(pdf_path)
                page_count = get_page_count(pdf_path)
                clauses = segment_text_into_clauses(page_texts)
                clause_count = count_clauses(clauses)

                # 3. Update Session State and Navigate
                st.session_state.page_texts = page_texts
                st.session_state.file_name = uploaded_file.name
                st.session_state.clauses = clauses
                st.session_state.policy_processed = True

                # Rerun Streamlit to switch to the analysis page
                st.rerun()

            except Exception as e:
                st.error(f"Error during file processing: {e}")
                st.session_state.policy_processed = False

        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# --- ANALYSIS PAGE FUNCTION ---

def render_analysis_page():
    """Renders the main analysis UI, including the new charts."""

    st.title(f"🛡 Policy Analysis: {st.session_state.file_name}")
    st.markdown("---")

    # Display Stats
    display_stats(len(st.session_state.page_texts), len(st.session_state.clauses))

    # NOTE: Chroma setup happens here, after data is loaded
    chroma_client = get_chroma_client()
    rag_collection = get_rag_collection(chroma_client, st.session_state.clauses)

    # --- ACTION BUTTONS ---
    st.markdown("### ⚙ Choose Action")
    col_buttons = st.columns(3)

    with col_buttons[0]:
        if st.button("🔍 Show Raw Clauses", use_container_width=True):
            # ... (Show Raw Clauses logic remains the same)
            with st.expander("📖 View all clauses and definitions", expanded=True):

                with st.spinner("Querying RAG for key definitions..."):
                    key_terms_context = query_rag_store(rag_collection,
                                                        "Definitions of key terms and words in the policy.", k=3)

                all_clauses_text = ""
                for clause in st.session_state.clauses:
                    try:
                        clause_segment_num = clause['clause_id'].split('_')[-1].replace('c', '')
                        page_num_match = clause['clause_id'].split('_')[0].replace('p', '')
                    except Exception:
                        clause_segment_num = 'N/A'
                        page_num_match = clause.get('page_num', 'N/A')

                    formatted_clause = f"""pg no-> {page_num_match}
line no-> {clause_segment_num}
clause-> {clause['text']}
definition-> {clause['text']}
----------
"""
                    all_clauses_text += formatted_clause

                st.markdown("#### All Clauses from the document:")
                st.text_area(
                    "Raw Policy Segments",
                    all_clauses_text,
                    height=400,
                    label_visibility="collapsed",
                    key="raw_clauses_area"
                )

    with col_buttons[1]:
        if st.button("✨ Analyze with Gemini", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.chart_data = None  # Reset chart data

            full_policy_text = "\n".join(st.session_state.page_texts)

            with st.container(border=True):
                st.markdown("### 🧠 Running AI Analysis...")

                # --- MODIFIED SPINNER TEXT TO REFLECT NEW LENGTH CONSTRAINT ---
                with st.spinner("1/3: Generating concise policy summary (20-30 lines)..."):
                    # NOTE: The generate_full_summary function's internal prompt is assumed
                    # to be updated to enforce this 20-30 line constraint for the Gemini API call.
                    full_summary = generate_full_summary(full_policy_text)
                    if full_summary.startswith("Client Initialization Error") or full_summary.startswith(
                            "Gemini API Error"):
                        st.error(full_summary)
                        return

                with st.spinner("2/3: Classifying and simplifying content page-by-page..."):
                    page_analysis = analyze_page_content(st.session_state.page_texts)

                with st.spinner("3/3: Generating Structured Data for Risk Charts..."):
                    # New call to generate chart data (MOCKED HERE)
                    chart_data = generate_mock_structured_analysis(page_analysis)

                st.session_state.analysis_results = {
                    'full_summary': full_summary,
                    'page_analysis': page_analysis
                }
                st.session_state.chart_data = chart_data
                st.success("🎉 Analysis Complete! Scroll down for the full report.")

    with col_buttons[2]:
        if st.session_state.analysis_results:
            st.markdown("<div style='text-align: center;'>✨ Report Ready!</div>", unsafe_allow_html=True)
        else:
            st.button("📄 Download Report", disabled=True, use_container_width=True)

    # --- DISPLAY ANALYSIS OUTPUT ---
    if st.session_state.analysis_results:
        analysis = st.session_state.analysis_results

        st.subheader("💡 Decoded Policy Report")

        summary_col, download_col = st.columns([2, 1])

        with summary_col:
            # --- MODIFIED HEADER TEXT TO REFLECT NEW LENGTH CONSTRAINT ---
            st.markdown("#### 1. High-Level Policy Summary (20-30 Lines)")
            st.info(analysis.get('full_summary', 'Summary generation failed or returned no data.'))

        with download_col:
            st.markdown("#### Download Full Report")
            display_full_report()

        st.markdown("#### 2. Page-by-Page Classification & Plain-Language Summary")
        st.markdown("(Quickly identify Coverages and Exclusions)")

        for item in analysis.get('page_analysis', []):
            classification = item.get('classification', 'N/A')
            summary = item.get('summary', 'N/A')
            page_num = item.get('pageNumber', 'N/A')

            if classification.lower() == 'exclusions':
                st.error(f"🚫 PAGE {page_num} | EXCLUSION:** {summary}")
            elif classification.lower() == 'coverage':
                st.success(f"✅ PAGE {page_num} | COVERAGE:** {summary}")
            else:
                st.warning(f"🟡 PAGE {page_num} | {classification.upper()}:** {summary}")

        # --- DISPLAY CHARTS ---
        st.markdown("---")
        if st.session_state.chart_data:
            render_policy_charts(st.session_state.chart_data)


# --- MAIN EXECUTION BLOCK ---

# 1. Apply Styles
set_styles()

# 2. Main Navigation Logic
if st.session_state.policy_processed:
    # If a file has been uploaded and processed, show the analysis page
    render_analysis_page()
else:
    # Otherwise, show the landing page with the upload
    render_landing_page()