import streamlit as st
import pandas as pd
from analyzer import detect_cherry_picking
st.set_page_config(page_title="Cherry Picked Sales Trend Detection", layout="wide")
st.title("🍒 Cherry Picked Sales Trend Detection")
st.subheader("Using Statistical Analysis and Machine Learning")

uploaded_file = st.file_uploader("Upload Company Sales Data (CSV with 'date' and 'sales' columns)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        st.success(f"✅ Loaded {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")
        
        claimed_growth = st.number_input("Enter Claimed Growth Percentage (%)", 
                                       min_value=0.0, value=90.0, step=0.5)
        
        if st.button("🔍 Analyze for Cherry Picking"):
            with st.spinner("Performing Statistical Analysis + ML Detection..."):
                result = detect_cherry_picking(df, claimed_growth)
            
            # Display Result
            st.subheader("Analysis Result")
            color = "green" if "Low Risk" in result["verdict"] else "orange" if "Medium" in result["verdict"] else "red"
            st.markdown(f"<h2 style='color:{color};'>{result['verdict']}</h2>", unsafe_allow_html=True)
            
            st.metric("Cherry-Picking Risk Score", f"{result['risk_score']}/100")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Claimed Growth", f"{result['claimed_growth']}%")
            with col2:
                st.metric("Overall CAGR", f"{result['cagr']}%")
            with col3:
                st.metric("Avg Monthly Growth", f"{result['avg_monthly_growth']}%")
            
            st.write("**Explanation:**")
            st.write(result['explanation'])
            
            # Charts
            st.subheader("Visualizations")
            colA, colB = st.columns(2)
            with colA:
                st.line_chart(df.set_index('date')['sales'], use_container_width=True)
                st.caption("Sales Trend Over Time")
            with colB:
                growth_rate = df['sales'].pct_change() * 100
                chart_data = pd.DataFrame({"Growth Rate (%)": growth_rate.values, 
                                         "Claimed Growth": [claimed_growth]*len(growth_rate)}, 
                                        index=df['date'])
                st.line_chart(chart_data, use_container_width=True)
                st.caption("Monthly Growth Rate vs Claimed Growth")
            
            st.info("⚠️ This analysis is based only on the uploaded data.")
            
    except Exception as e:
        st.error(f"Error: {str(e)}. Please make sure your CSV has 'date' and 'sales' columns.")
