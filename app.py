import streamlit as st
import pandas as pd
import plotly.express as px
from analyzer import detect_cherry_picking

# Page Configuration
st.set_page_config(
    page_title="Cherry Picking Detection Dashboard",
    layout="wide",
    page_icon="🍒"
)

# Custom Dark Theme
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .main-header { font-size: 2.5rem; font-weight: 700; text-align: center; color: #f8fafc; }
    .sub-header { text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 30px; }
    .card { background-color: #1e2937; padding: 20px; border-radius: 12px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍒 Cherry Picking Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect whether company growth claims are genuine or cherry-picked</p>', unsafe_allow_html=True)

# ====================== INPUT SECTION ======================
st.subheader("Data Analysis Portal")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"])

with col2:
    claimed_growth = st.number_input("Enter company's claimed growth (%)", 
                                   min_value=0.0, value=30.0, step=0.5)

analyze_button = st.button("Analyze Data and View Results", type="primary", use_container_width=True)

# ====================== ANALYSIS ======================
if uploaded_file and analyze_button:
    try:
        # Load file
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # ================== Handle Amazon Dataset Columns ==================
        if 'OrderDate' in df.columns and 'TotalAmount' in df.columns:
            df = df.rename(columns={'OrderDate': 'date', 'TotalAmount': 'sales'})
            st.info("✅ Amazon Sales Dataset detected. Using OrderDate and TotalAmount columns.")
        elif 'date' not in df.columns or 'sales' not in df.columns:
            st.error("❌ Required columns not found. Please make sure your file has 'date' and 'sales' columns, or use the Amazon dataset (OrderDate + TotalAmount).")
            st.stop()

        # Convert date and aggregate to monthly sales (very important for time-series)
        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby(df['date'].dt.to_period('M'))['sales'].sum().reset_index()
        df['date'] = df['date'].dt.to_timestamp()

        df = df.sort_values('date').reset_index(drop=True)

        st.success(f"✅ Loaded {len(df)} monthly records from {df['date'].min().date()} to {df['date'].max().date()}")

        # Run Analysis
        with st.spinner("Analyzing using Statistical + ML methods..."):
            result = detect_cherry_picking(df, claimed_growth)

        # ====================== CLAIM EXPLANATION SECTION ======================
        st.subheader("Claim Explanation Section")

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Claimed Growth %", f"{result['claimed_growth']}%")
        with colB:
            st.metric("Average Growth %", f"{result['avg_monthly_growth']}%")
        with colC:
            st.metric("Overall CAGR", f"{result['cagr']}%")

        # Verdict Card
        risk_color = "#22c55e" if result['risk_score'] <= 35 else "#eab308" if result['risk_score'] <= 70 else "#ef4444"
        st.markdown(f"""
        <div style="background-color:#1e2937; padding:25px; border-radius:12px; text-align:center; border:3px solid {risk_color}; margin:20px 0;">
            <h2 style="color:{risk_color};">{result['verdict']}</h2>
            <h3>Risk Score: <strong>{result['risk_score']}/100</strong></h3>
        </div>
        """, unsafe_allow_html=True)

        st.info(result['explanation'])

        # ====================== INTERACTIVE GRAPHS ======================
        st.subheader("Interactive Graph Section")

        df['growth_rate'] = df['sales'].pct_change() * 100

        # Growth Rate Chart
        fig = px.line(df, x='date', y='growth_rate', template="plotly_dark",
                      title="Monthly Growth Rate (%) vs Claimed Growth")
        fig.add_hline(y=claimed_growth, line_dash="dash", line_color="#f87171",
                      annotation_text=f"Claimed {claimed_growth}%")
        fig.update_layout(height=520, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Sales Trend Chart
        fig2 = px.line(df, x='date', y='sales', template="plotly_dark",
                       title="Monthly Sales Trend Over Time")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("👆 Upload the Amazon Sales Dataset (or any sales CSV) and click the Analyze button.")

st.caption("Cherry Picked Sales Trend Detection using Statistical Analysis and ML")
