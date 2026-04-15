import streamlit as st
import pandas as pd
import plotly.express as px
from analyzer import detect_cherry_picking, parse_dates

st.set_page_config(page_title="Cherry Picking Detector", layout="wide", page_icon="🍒")

st.markdown("""
<style>
    .stApp { background-color: #0a0f1c; color: #f8fafc; }
    .main-header { font-size: 2.8rem; font-weight: 700; text-align: center; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍒 Cherry Picking Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#94a3b8;">Detecting Cherry-Picked Growth Claims using Stats & ML</p>', unsafe_allow_html=True)

st.subheader("Data Analysis Portal")

col1, col2, col3 = st.columns([3, 1.4, 1.4])

with col1:
    uploaded_file = st.file_uploader("Upload Sales Dataset (CSV or XLSX)", type=["csv", "xlsx"])

with col2:
    claimed_growth = st.number_input("Claimed Growth (%)", min_value=0.0, value=20.0, step=1.0)

with col3:
    claim_period = st.number_input("Claim Period (last X months)", min_value=1, value=12, step=1)

if st.button("Analyze Data and View Results", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("Please upload your sales dataset")
    else:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                df_raw = pd.read_excel(uploaded_file)
            else:
                df_raw = pd.read_csv(uploaded_file)

            df = df_raw.copy()
            df = parse_dates(df)
            df = df.dropna(subset=['date', 'sales']).sort_values('date').reset_index(drop=True)

            # Monthly aggregation
            df['month_period'] = df['date'].dt.to_period('M')
            df_monthly = df.groupby('month_period')['sales'].sum().reset_index()
            df_monthly['date'] = df_monthly['month_period'].dt.to_timestamp()
            df_monthly = df_monthly.sort_values('date').reset_index(drop=True)

            st.success("✅ Dataset loaded successfully.")

            with st.spinner("Analyzing for cherry-picking..."):
                result = detect_cherry_picking(df_monthly, claimed_growth, claim_period)

            # Results Section
            st.subheader("Claim Explanation Section")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Claimed Growth", f"{result['claimed_growth']}% in last {result['claim_period']} months")
            with c2:
                st.metric("Actual Growth", f"{result['actual_growth_in_period']}%")
            with c3:
                st.metric("Overall CAGR", f"{result['actual_cagr']}%")

            # Verdict
            verdict = result['verdict']
            color = "#ef4444" if "High" in verdict else "#eab308" if "Medium" in verdict else "#22c55e"
            emoji = "❌" if "High" in verdict else "⚠️" if "Medium" in verdict else "✅"

            st.markdown(f"""
            <div style="background:#1e2937; padding:30px; border-radius:14px; text-align:center; border:3px solid {color};">
                <h2 style="color:{color};">{emoji} {verdict}</h2>
                <h3>Risk Score: <strong>{result['risk_score']}/100</strong></h3>
            </div>
            """, unsafe_allow_html=True)

            st.info(result.get('explanation', ''))

            # Graphs
            st.subheader("Interactive Graphs")

            df_monthly['growth_rate'] = df_monthly['sales'].pct_change() * 100
            claimed_df = df_monthly.tail(result['claim_period'])

            # Graph 1
            fig1 = px.line(df_monthly, x='date', y='sales', title="1. Monthly Sales Trend", template="plotly_dark")
            if len(claimed_df) > 0:
                fig1.add_vrect(x0=claimed_df['date'].iloc[0], x1=claimed_df['date'].iloc[-1],
                               fillcolor="rgba(248, 113, 113, 0.3)", annotation_text="Claimed Period")
            st.plotly_chart(fig1, use_container_width=True)

            # Graph 2
            fig2 = px.line(df_monthly, x='date', y='growth_rate', title="2. Monthly Growth Rate (%)", template="plotly_dark")
            if len(claimed_df) > 0:
                fig2.add_scatter(x=claimed_df['date'], y=claimed_df['growth_rate'],
                                 mode='lines', line=dict(color='red', width=4), name='Claimed Period')
            st.plotly_chart(fig2, use_container_width=True)

            # ==================== UPDATED GRAPH 3 ====================
            st.subheader("3. Growth in Claimed Period - Likely Cherry-Picked Months Highlighted")

            claimed_g = claimed_df.copy()
            claimed_g['growth_rate'] = claimed_g['sales'].pct_change() * 100

            # Define cherry-picked months: growth rate is significantly high (≥ 70% of claimed growth)
            threshold = claimed_growth * 0.7
            claimed_g['is_cherry_picked'] = claimed_g['growth_rate'] >= threshold

            # Plot with custom colors: Normal = Green, Cherry-Picked = Red
            fig3 = px.bar(claimed_g, x='date', y='growth_rate', 
                          color='is_cherry_picked',
                          color_discrete_map={True: '#ef4444', False: '#22c55e'},  # Red for cherry-picked, Green for normal
                          title="3. Growth in Claimed Period - Likely Cherry-Picked Months Highlighted",
                          template="plotly_dark")

            fig3.update_layout(showlegend=True, height=420)
            fig3.update_traces(marker_line_width=0)

            st.plotly_chart(fig3, use_container_width=True)

            # Show highlighted months
            cherry_months_list = claimed_g[claimed_g['is_cherry_picked']]['date'].dt.strftime('%b %Y').tolist()
            if cherry_months_list:
                st.caption(f"**Likely Cherry-Picked Months (growth ≥ {threshold:.1f}%):** {', '.join(cherry_months_list)}")
            else:
                st.caption("No months in this period showed unusually high growth.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.caption("Cherry Picking Detection Dashboard")
