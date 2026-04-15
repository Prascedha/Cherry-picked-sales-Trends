import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest

def parse_dates(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    if df['date'].isna().any():
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    return df

def calculate_cagr(df):
    if len(df) < 2:
        return 0.0
    start = float(df['sales'].iloc[0])
    end = float(df['sales'].iloc[-1])
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    if years <= 0:
        return 0.0
    return ((end / start) ** (1 / years) - 1) * 100

def detect_cherry_picking(df: pd.DataFrame, claimed_growth: float, claim_period_months: int = 12):
    df = parse_dates(df)
    df = df.sort_values('date').reset_index(drop=True)
    df['growth_rate'] = df['sales'].pct_change() * 100

    actual_cagr = calculate_cagr(df)
    growth_in_period = ((df.tail(claim_period_months)['sales'].iloc[-1] / 
                        df.tail(claim_period_months)['sales'].iloc[0]) - 1) * 100 if len(df) >= claim_period_months else 0

    rolling_6m = df['sales'].rolling(window=6, min_periods=3).apply(
        lambda x: ((x[-1]/x[0])-1)*100 if len(x)>1 else 0, raw=True)
    max_6m = rolling_6m.max()

    # ==================== MACHINE LEARNING PART ====================
    anomaly_pct = 0.0
    try:
        df_temp = df.set_index('date').asfreq('MS').interpolate()
        stl = STL(df_temp['sales'], seasonal=13, robust=True)
        res = stl.fit()
        residuals = res.resid.values.reshape(-1, 1)

        iso = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso.fit_predict(residuals)
        anomaly_pct = (anomalies == -1).mean() * 100
    except:
        pass

    # Risk Score
    risk_score = 0
    if growth_in_period < 0 and claimed_growth > 20: risk_score += 40
    if growth_in_period < claimed_growth * 0.4: risk_score += 35
    if growth_in_period > claimed_growth * 1.5: risk_score += 25
    if actual_cagr < 0 and claimed_growth > 15: risk_score += 30
    if anomaly_pct > 18: risk_score += 15
    risk_score = min(100, max(0, risk_score))

    verdict = "❌ High Risk - Likely Cherry-Picked" if risk_score >= 65 else \
              "⚠️ Medium Risk - Partially Supported" if risk_score >= 45 else \
              "✅ Low Risk - Likely Genuine"

    explanation = f"Claimed {claimed_growth}% growth in last {claim_period_months} months. Actual: {growth_in_period:.1f}%. CAGR: {actual_cagr:.2f}%. Anomalies detected: {anomaly_pct:.1f}%."

    claimed_months = df.tail(claim_period_months)['date'].dt.strftime('%b %Y').tolist()

    return {
        "verdict": verdict,
        "risk_score": round(risk_score, 1),
        "claimed_growth": claimed_growth,
        "claim_period": claim_period_months,
        "actual_growth_in_period": round(growth_in_period, 1),
        "actual_cagr": round(actual_cagr, 2),
        "claimed_months": claimed_months,
        "explanation": explanation,
        "anomaly_pct": round(anomaly_pct, 1)
    }
