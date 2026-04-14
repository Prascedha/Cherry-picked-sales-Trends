import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest

def calculate_cagr(df):
    if len(df) < 2:
        return 0.0
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    return ((df['sales'].iloc[-1] / df['sales'].iloc[0]) ** (1 / years) - 1) * 100

def detect_cherry_picking(df: pd.DataFrame, claimed_growth: float):
    df = df.copy()
    df['growth_rate'] = df['sales'].pct_change() * 100
    
    # Statistical Metrics
    avg_growth = df['growth_rate'].mean()
    cagr = calculate_cagr(df)
    std_growth = df['growth_rate'].std() if df['growth_rate'].std() > 0 else 1
    z_score = (claimed_growth - avg_growth) / std_growth
    
    # Temporal Coverage
    tolerance = 8.0
    coverage = (abs(df['growth_rate'] - claimed_growth) <= tolerance).mean() * 100
    
    # Rolling Window (6-month)
    rolling_6m = df['sales'].rolling(window=6, min_periods=3).apply(
        lambda x: ((x.iloc[-1] / x.iloc[0]) - 1) * 100 if len(x) > 1 else 0, raw=True)
    max_rolling_6m = rolling_6m.max()
    
    # ML Part: STL + Isolation Forest on residuals
    df_temp = df.set_index('date').asfreq('MS').interpolate()
    stl = STL(df_temp['sales'], seasonal=13, robust=True)
    res = stl.fit()
    residuals = res.resid.values.reshape(-1, 1)
    
    iso = IsolationForest(contamination=0.08, random_state=42)
    anomalies = iso.fit_predict(residuals)
    anomaly_pct = (anomalies == -1).mean() * 100
    
    # Risk Score
    risk_score = 0
    if abs(z_score) > 2.0: risk_score += 35
    if coverage < 18: risk_score += 30
    if anomaly_pct > 18: risk_score += 20
    if max_rolling_6m > claimed_growth * 1.4 and coverage < 25: risk_score += 25
    risk_score = min(100, max(0, risk_score))
    
    # Verdict
    if risk_score <= 35:
        verdict = "✅ Low Risk - Likely Representative Trend"
    elif risk_score <= 70:
        verdict = "⚠️ Medium Risk - Partially Supported (Possible Cherry Picking)"
    else:
        verdict = "❌ High Risk - Likely Cherry Picked Sales Trend"
    
    return {
        "verdict": verdict,
        "risk_score": round(risk_score, 1),
        "claimed_growth": claimed_growth,
        "cagr": round(cagr, 2),
        "avg_monthly_growth": round(avg_growth, 2),
        "temporal_coverage_pct": round(coverage, 1),
        "anomaly_percentage": round(anomaly_pct, 1),
        "max_6m_rolling": round(max_rolling_6m, 1),
        "explanation": f"The claimed {claimed_growth}% growth appears in only {coverage:.1f}% of all periods. "
                      f"Overall CAGR is {cagr:.2f}%. Highest 6-month growth reached {max_rolling_6m:.1f}%. "
                      f"Anomalies detected after removing seasonality: {anomaly_pct:.1f}%."
    }
