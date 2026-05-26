import os
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence Platform Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Design
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Gradient Header */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -30%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }
    
    .header-title {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Metrics Cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #eef2f6;
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        font-family: 'Outfit', sans-serif;
    }
    
    .band-high {
        color: #10b981;
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.02) 100%);
        border: 1px solid rgba(16,185,129,0.2);
    }
    
    .band-medium {
        color: #f59e0b;
        background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(245,158,11,0.02) 100%);
        border: 1px solid rgba(245,158,11,0.2);
    }
    
    .band-low {
        color: #ef4444;
        background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(239,68,68,0.02) 100%);
        border: 1px solid rgba(239,68,68,0.2);
    }

    /* Source citations */
    .source-container {
        padding: 1rem;
        border-left: 4px solid #3b82f6;
        background-color: #f8fafc;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    
    .source-header {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
        font-size: 0.95rem;
    }
    
    .source-snippet {
        font-size: 0.9rem;
        color: #475569;
        font-style: italic;
    }
    
    /* Preset buttons container */
    .preset-container {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Settings & Navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
st.sidebar.title("Configuration")

# Detect environment: default to Cloud (Azure) if running on Azure, otherwise Local (Development)
default_env_idx = 1
if "WEBSITE_HOSTNAME" in os.environ or os.environ.get("ENVIRONMENT") == "production":
    default_env_idx = 0

# Environment selector
env = st.sidebar.selectbox(
    "API Environment",
    ["Cloud (Azure)", "(Development)"],
    index=default_env_idx
)

# API URL setups
if env == "Cloud (Azure)":
    default_conv_url = os.getenv("CONVERSION_SERVICE_URL", "https://cip-app-13.azurewebsites.net")
    default_rag_url  = os.getenv("RAG_SERVICE_URL",        "https://cip-rag-13.azurewebsites.net")
else:
    default_conv_url = "http://localhost:8000"
    default_rag_url  = "http://localhost:8001"

conv_url = st.sidebar.text_input("Conversion Service API", value=default_conv_url)
rag_url = st.sidebar.text_input("RAG Service API", value=default_rag_url)

# Load Groq API Key from environment or .env if present
default_groq_api_key = os.getenv("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input("Groq API Key (Optional)", value=default_groq_api_key, type="password", help="Enables fast cloud LLM generation instead of local model")

# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🔮 Conversion Profiler", "💬 RAG Resolution Assistant", "📊 System Health & Telemetry"],
    index=0
)

# Customer Presets Definition
PRESETS = {
    "VIP High-Engagement": {
        "age": 45.0, "income": 95000.0, "num_campaigns": 2.0, "num_clicks": 18.0, 
        "num_opens": 28.0, "recency_days": 5.0, "tenure_days": 900.0, 
        "gender": "F", "channel": "email", "product_category": "electronics", "region": "north"
    },
    "New Cold Prospect": {
        "age": 24.0, "income": 30000.0, "num_campaigns": 5.0, "num_clicks": 1.0, 
        "num_opens": 2.0, "recency_days": 80.0, "tenure_days": 45.0, 
        "gender": "M", "channel": "sms", "product_category": "fashion", "region": "south"
    },
    "At-Risk Customer": {
        "age": 35.0, "income": 58000.0, "num_campaigns": 3.0, "num_clicks": 4.0, 
        "num_opens": 8.0, "recency_days": 45.0, "tenure_days": 350.0, 
        "gender": "M", "channel": "email", "product_category": "electronics", "region": "east"
    }
}

# Initialize Session State for Form Fields
for field_key in PRESETS["VIP High-Engagement"].keys():
    if field_key not in st.session_state:
        st.session_state[field_key] = PRESETS["VIP High-Engagement"][field_key]

def apply_preset(preset_name):
    for key, val in PRESETS[preset_name].items():
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Page 1: Conversion Profiler
# ---------------------------------------------------------------------------
if page == "🔮 Conversion Profiler":
    # Header Banner
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🔮 Customer Conversion Profiler</h1>
        <p class="header-subtitle">Analyze marketing features, score conversion probabilities, and view synthesized complaint profiles using our XGBoost Conversion service.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Presets Buttons
    st.subheader("Select a Demo Profile Preset:")
    cols_presets = st.columns(len(PRESETS))
    for i, name in enumerate(PRESETS.keys()):
        if cols_presets[i].button(name, use_container_width=True):
            apply_preset(name)
            st.rerun()

    st.markdown("---")

    # Split page: Inputs vs Diagnostics
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Customer Characteristics")
        
        # Form
        with st.form("profiler_form"):
            # Numeric inputs (grouped)
            st.markdown("##### **Quantitative Features**")
            col_age_inc = st.columns(2)
            age = col_age_inc[0].number_input("Age", min_value=18.0, max_value=100.0, key="age", step=1.0)
            income = col_age_inc[1].number_input("Annual Income ($)", min_value=0.0, max_value=1000000.0, key="income", step=1000.0)

            col_clicks_opens = st.columns(2)
            num_clicks = col_clicks_opens[0].number_input("Campaign Clicks", min_value=0.0, max_value=500.0, key="num_clicks", step=1.0)
            num_opens = col_clicks_opens[1].number_input("Campaign Opens", min_value=0.0, max_value=500.0, key="num_opens", step=1.0)

            col_days = st.columns(3)
            num_campaigns = col_days[0].number_input("Campaigns Sent", min_value=1.0, max_value=100.0, key="num_campaigns", step=1.0)
            recency_days = col_days[1].number_input("Recency (Days)", min_value=0.0, max_value=365.0, key="recency_days", step=1.0)
            tenure_days = col_days[2].number_input("Tenure (Days)", min_value=0.0, max_value=5000.0, key="tenure_days", step=1.0)

            # Categorical inputs
            st.markdown("##### **Categorical Context**")
            col_gender_chan = st.columns(2)
            gender = col_gender_chan[0].selectbox("Gender", ["M", "F"], key="gender")
            channel = col_gender_chan[1].selectbox("Primary Channel", ["email", "sms", "push", "social", "web"], key="channel")

            col_cat_reg = st.columns(2)
            product_category = col_cat_reg[0].selectbox("Product Interest", ["electronics", "fashion", "home", "grocery", "leisure"], key="product_category")
            region = col_cat_reg[1].selectbox("Region", ["north", "south", "east", "west"], key="region")
            
            submit_btn = st.form_submit_button("Analyze & Score Profile", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Predictive Diagnostics")
        
        if submit_btn:
            payload = {
                "age": age,
                "income": income,
                "num_campaigns": num_campaigns,
                "num_clicks": num_clicks,
                "num_opens": num_opens,
                "recency_days": recency_days,
                "tenure_days": tenure_days,
                "gender": gender,
                "channel": channel,
                "product_category": product_category,
                "region": region
            }
            
            # API Call
            with st.spinner("Scoring customer profile via ML endpoint..."):
                try:
                    # R11 endpoint: /customer-intel returns conversion band and themes
                    intel_res = requests.post(f"{conv_url}/customer-intel", json=payload, timeout=10)
                    predict_res = requests.post(f"{conv_url}/predict", json=payload, timeout=10)
                    
                    if intel_res.status_code == 200 and predict_res.status_code == 200:
                        intel_data = intel_res.json()
                        predict_data = predict_res.json()
                        
                        prob = predict_data["conversion_prob"]
                        band = intel_data["conversion_band"]
                        
                        # Set band styling class
                        band_class = "band-low"
                        if band == "High":
                            band_class = "band-high"
                        elif band == "Medium":
                            band_class = "band-medium"

                        # Columns for metrics
                        col_card1, col_card2 = st.columns(2)
                        with col_card1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Predicted Probability</div>
                                <div class="metric-value" style="color: #1e293b;">{prob:.2%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_card2:
                            st.markdown(f"""
                            <div class="metric-card {band_class}">
                                <div class="metric-label">Conversion Band</div>
                                <div class="metric-value">{band}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Plotly Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Conversion Propensity Indicator", 'font': {'size': 18, 'family': 'Outfit'}},
                            gauge = {
                                'axis': {'range': [0, 1], 'tickformat': ',.0%'},
                                'bar': {'color': "#1e3c72"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "#cbd5e1",
                                'steps': [
                                    {'range': [0, 0.4], 'color': 'rgba(239, 68, 68, 0.15)'},
                                    {'range': [0.4, 0.7], 'color': 'rgba(245, 158, 11, 0.15)'},
                                    {'range': [0.7, 1.0], 'color': 'rgba(16, 185, 129, 0.15)'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.5
                                }
                            }
                        ))
                        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                        # Synced Complaint Themes
                        st.markdown("##### **Top Complaint Themes (Synthesized)**")
                        themes = intel_data.get("top_complaint_themes", [])
                        if themes:
                            for theme in themes:
                                st.markdown(f"""
                                <div style="padding: 0.75rem; background-color: #f8fafc; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e2e8f0; display: flex; justify-content: space-between;">
                                    <span style="font-weight: 600; color: #334155;">⚠️ {theme['theme']}</span>
                                    <span style="font-size: 0.85rem; color: #64748b; font-family: monospace; background: #f1f5f9; padding: 2px 6px; border-radius: 4px;">{theme['cited_id']}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No active complaint logs found for similar demographics.")

                        # Raw API response expanders
                        with st.expander("Show Model Specifications"):
                            st.json({
                                "service": "conversion-service",
                                "model_version": predict_data["model_version"],
                                "features_scored": predict_data["feature_count"],
                                "imputation": "missing numeric features imputed with 0, missing categoricals mapped to unseen (-1)"
                            })
                            
                    else:
                        st.error(f"Error communicating with API. Code: {predict_res.status_code}")
                except Exception as ex:
                    st.error(f"Could not connect to Conversion API at {conv_url}. Details: {ex}")
        else:
            st.info("💡 Adjust customer characteristics in the left panel and click **Analyze & Score Profile** to run the ML inference.")

# ---------------------------------------------------------------------------
# Page 2: RAG Q&A Assistant
# ---------------------------------------------------------------------------
elif page == "💬 RAG Resolution Assistant":
    # Header Banner
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">💬 Complaint Intelligence RAG</h1>
        <p class="header-subtitle">Search historical complaints and resolve issues using a <strong>Groq-powered</strong> LLM (llama-3.1-8b-instant) with FAISS semantic retrieval and full citation transparency.</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset/Suggested queries
    st.subheader("💡 Suggested Queries:")
    cols_q = st.columns(3)
    suggested = [
        "What are the most common billing issues?",
        "What complaints are there about service outages?",
        "Why are customers demanding subscription refunds?"
    ]
    query_input = ""
    for i, sug_q in enumerate(suggested):
        if cols_q[i].button(sug_q, use_container_width=True):
            st.session_state["rag_query"] = sug_q
            
    # Input field
    st.markdown("---")
    user_query = st.text_input("Ask the RAG Assistant a question regarding complaints:", key="rag_query", placeholder="Enter your query here...")
    
    col_k, col_submit = st.columns([1, 4])
    top_k = col_k.slider("Top Chunks (k)", min_value=1, max_value=10, value=5)
    submit_query = col_submit.button("Retrieve & Generate Answer", type="primary", use_container_width=True)

    if submit_query and user_query:
        # Check query length
        if len(user_query) < 5:
            st.warning("Please enter a question with at least 5 characters.")
        else:
            with st.spinner("Retrieving relevant complaint passages and generating answer via Groq..."):
                try:
                    # Use /answer with groq_api_key passed in body — works on all RAG versions
                    payload = {
                        "query": user_query,
                        "top_k": top_k,
                        "groq_api_key": groq_api_key if groq_api_key else None
                    }
                    rag_res = requests.post(f"{rag_url}/answer", json=payload, timeout=90)

                    if rag_res.status_code == 200:
                        data = rag_res.json()
                        answer = data["answer"]
                        sources = data["sources"]

                        st.caption("⚡ Powered by **Groq** `llama-3.1-8b-instant` via FAISS retrieval")

                        # Render Answer
                        st.subheader("🤖 Generated Answer")
                        st.markdown(f"""
                        <div style="background-color: #eff6ff; padding: 1.5rem; border-radius: 12px; border: 1px solid #bfdbfe; color: #1e3a8a; margin-bottom: 2rem; line-height: 1.6;">
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)

                        # Render Citations / Sources
                        st.subheader("📄 Grounded Sources / Citations")
                        if sources:
                            for idx, src in enumerate(sources):
                                with st.expander(f"Source #{idx+1}: {src['source_file']} (ID: {src['chunk_id'].split('::')[-1]})"):
                                    st.markdown(f"""
                                    <div class="source-container">
                                        <div class="source-header">File: {src['source_file']}</div>
                                        <div class="source-snippet">"{src['snippet']}"</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("⚠️ No sources cited — the model refused due to low similarity or out-of-domain question.")

                    elif rag_res.status_code == 422:
                        st.error("Query validation failed: please enter at least 5 characters.")
                    else:
                        st.error(f"Error from RAG Service. Code: {rag_res.status_code} — {rag_res.text[:300]}")
                except Exception as ex:
                    st.error(f"Could not connect to RAG API at {rag_url}. Details: {ex}")

                    
    elif submit_query and not user_query:
        st.warning("Please enter a search query or click one of the suggested prompts above.")

# ---------------------------------------------------------------------------
# Page 3: System Status
# ---------------------------------------------------------------------------
else:
    # Header Banner
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📊 Service Health & Telemetry</h1>
        <p class="header-subtitle">Monitor microservice uptime, check model versions, and verify active configurations across the platform.</p>
    </div>
    """, unsafe_allow_html=True)

    # Perform health checks
    status_data = []

    # Check Conversion
    try:
        conv_res = requests.get(f"{conv_url}/health", timeout=3)
        if conv_res.status_code == 200:
            c_data = conv_res.json()
            status_data.append({
                "Service": "🔮 Conversion Prediction",
                "Endpoint": conv_url,
                "Status": "ONLINE",
                "Model Loaded": "✅ Yes" if c_data.get("model_loaded") else "❌ No",
                "Active Version": c_data.get("model_version", "unknown"),
                "Index Size / Vectors": "N/A"
            })
        else:
            status_data.append({
                "Service": "🔮 Conversion Prediction",
                "Endpoint": conv_url,
                "Status": f"ERROR ({conv_res.status_code})",
                "Model Loaded": "Unknown", "Active Version": "N/A", "Index Size / Vectors": "N/A"
            })
    except Exception as ex:
        status_data.append({
            "Service": "🔮 Conversion Prediction",
            "Endpoint": conv_url,
            "Status": "OFFLINE",
            "Model Loaded": "❌ No", "Active Version": "N/A", "Index Size / Vectors": "N/A"
        })

    # Check RAG
    try:
        rag_res = requests.get(f"{rag_url}/health", timeout=5)
        if rag_res.status_code == 200:
            r_data = rag_res.json()
            status_data.append({
                "Service": "💬 Complaint RAG",
                "Endpoint": rag_url,
                "Status": "ONLINE" if r_data.get("status") == "ok" else f"DEGRADED ({r_data.get('status')})",
                "Model Loaded": "✅ Yes",
                "Active Version": "TinyLlama-1.1B",
                "Index Size / Vectors": f"{r_data.get('index_vectors', 0)} vectors"
            })
        else:
            status_data.append({
                "Service": "💬 Complaint RAG",
                "Endpoint": rag_url,
                "Status": f"ERROR ({rag_res.status_code})",
                "Model Loaded": "Unknown", "Active Version": "N/A", "Index Size / Vectors": "N/A"
            })
    except Exception as ex:
        status_data.append({
            "Service": "💬 Complaint RAG",
            "Endpoint": rag_url,
            "Status": "OFFLINE",
            "Model Loaded": "❌ No", "Active Version": "N/A", "Index Size / Vectors": "N/A"
        })

    # Render Table
    df_status = pd.DataFrame(status_data)
    
    # Custom colored table via Markdown
    st.subheader("Microservice Overview")
    
    for item in status_data:
        color = "#10b981" # green
        if "OFFLINE" in item["Status"] or "ERROR" in item["Status"]:
            color = "#ef4444" # red
        elif "DEGRADED" in item["Status"]:
            color = "#f59e0b" # amber
            
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.02); display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; font-size: 1.15rem; color: #1e293b;">{item['Service']}</h4>
                <p style="margin: 4px 0 0 0; font-family: monospace; font-size: 0.85rem; color: #64748b;">{item['Endpoint']}</p>
            </div>
            <div style="display: flex; gap: 2rem; align-items: center;">
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Active Version</div>
                    <div style="font-weight: 600; color: #334155;">{item['Active Version']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Index / Store</div>
                    <div style="font-weight: 600; color: #334155;">{item['Index Size / Vectors']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Model Status</div>
                    <div style="font-weight: 600; color: #334155;">{item['Model Loaded']}</div>
                </div>
                <div style="background-color: {color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;">
                    {item['Status']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # General configuration report
    st.markdown("---")
    st.subheader("💡 Telemetry Info")
    st.info("The application endpoints publish performance histograms, active prediction counts, and inference latency statistics to Azure Application Insights via OpenTelemetry (when the connection string is defined).")
