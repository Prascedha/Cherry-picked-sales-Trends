import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzer import detect_cherry_picking

st.set_page_config(page_title="Cherry Picking Detection Dashboard", layout="wide", page_icon="🍒")

st.markdown('<h1 style="text-align:center; font-size:2.5rem;">🍒 Cherry Picking Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#94a3b8;">Detecting & Highlighting Cherry-Picked Months</p>', unsafe_allow_html=True)

st.subheader("Data Analysis Portal")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Amazon Sales Dataset (CSV)", type=["csv", "xlsx"])

with col2:
    claimed_growth = st.number_input("Claimed Growth (%)", min_value=0.0, value=50.0, step=1.0)

with col3:
    claim_period = st.number_input("Claim Period (last X months)", min_value=1, value=4, step=1)

if st.button("Analyze Data and View Results", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("Please upload the Amazon Sales Dataset")
    else:
        try:
            df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

            if 'OrderDate' in df_raw.columns and 'TotalAmount' in df_raw.columns:
                df = df_raw.rename(columns={'OrderDate': 'date', 'TotalAmount': 'sales'})
                st.success("✅ Amazon Sales Dataset detected.")
            else:
                df = df_raw

            df['date'] = pd.to_datetime(df['date'])
            df = df.groupby(df['date'].dt.to_period('M'))['sales'].sum().reset_index()
            df['date'] = df['date'].dt.to_timestamp()
            df = df.sort_values('date').reset_index(drop=True)

            with st.spinner("Analyzing and highlighting cherry-picked months..."):
                result = detect_cherry_picking(df, claimed_growth, claim_period)

            # ====================== CLAIM EXPLANATION ======================
            st.subheader("Claim Explanation Section")

            colA, colB, colC = st.columns(3)
            with colA: st.metric("Claimed Growth", f"{result['claimed_growth']}% in last {result['claim_period']} months")
            with colB: st.metric("Actual Growth in Period", f"{result['actual_growth_in_period']}%")
            with colC: st.metric("Overall CAGR", f"{result['actual_cagr']}%")

            color = "#ef4444" if "High Risk" in result['verdict'] else "#eab308" if "Medium" in result['verdict'] else "#22c55e"
            st.markdown(f"""
            <div style="background-color:#1e2937; padding:25px; border-radius:12px; text-align:center; border:3px solid {color};">
                <h2 style="color:{color};">{result['verdict']}</h2>
                <h3>Risk Score: <strong>{result['risk_score']}/100</strong></h3>
            </div>
            """, unsafe_allow_html=True)

            st.info(result['explanation'])

            # ====================== 3 GRAPHS ======================
            st.subheader("Interactive Graphs")

            df['growth_rate'] = df['sales'].pct_change() * 100
            claimed_df = df.tail(result['claim_period'])

            # Graph 1: Growth Rate with Cherry-Picked Highlight
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df['date'], y=df['growth_rate'], 
                                    mode='lines', name='All Months', line=dict(color='#60a5fa')))
            fig1.add_trace(go.Scatter(x=claimed_df['date'], y=claimed_df['growth_rate'],
                                    mode='lines', name='Cherry-Picked Months', 
                                    line=dict(color='#f87171', width=5)))
            fig1.add_hline(y=((1 + claimed_growth/100)**(1/12) - 1)*100, line_dash="dash", line_color="red")
            fig1.update_layout(title="Graph 1: Monthly Growth Rate (%) - Cherry-Picked Months Highlighted", 
                               template="plotly_dark", height=450)
            st.plotly_chart(fig1, use_container_width=True)

            # Graph 2: Sales Trend with shaded region
            fig2 = px.line(df, x='date', y='sales', template="plotly_dark", title="Graph 2: Monthly Sales Trend")
            fig2.add_vrect(x0=claimed_df['date'].iloc[0], x1=claimed_df['date'].iloc[-1],
                           fillcolor="rgba(248, 113, 113, 0.3)", opacity=0.5,
                           annotation_text="Claimed / Cherry-Picked Period")
            st.plotly_chart(fig2, use_container_width=True)

            # Graph 3: Bar chart of growth in claimed period (to see which months contributed most)
            claimed_df = claimed_df.copy()
            claimed_df['growth_rate'] = claimed_df['sales'].pct_change() * 100
            fig3 = px.bar(claimed_df, x='date', y='growth_rate', 
                          title="Graph 3: Month-by-Month Growth in Claimed Period",
                          labels={"growth_rate": "Growth Rate (%)"},
                          template="plotly_dark")
            fig3.update_traces(marker_color='#f87171')
            st.plotly_chart(fig3, use_container_width=True)

            st.caption(f"**Highlighted Cherry-Picked Months:** {', '.join(result['claimed_months'])}")

        except Exception as e:
            st.error(f"Error: {e}")

st.caption("Cherry Picked Sales Trend Detection using Statistical Analysis and Machine Learning")