import pandas as pd

def calculate_cagr(df):
    if len(df) < 2:
        return 0.0
    start_sales = float(df['sales'].iloc[0])
    end_sales = float(df['sales'].iloc[-1])
    years = (pd.to_datetime(df['date'].iloc[-1]) - pd.to_datetime(df['date'].iloc[0])).days / 365.25
    if years <= 0:
        return 0.0
    return ((end_sales / start_sales) ** (1 / years) - 1) * 100

def calculate_growth_in_period(df, months):
    if len(df) < months:
        return 0.0
    recent = df.tail(months)
    return ((recent['sales'].iloc[-1] / recent['sales'].iloc[0]) - 1) * 100

def detect_cherry_picking(df: pd.DataFrame, claimed_growth: float, claim_period_months: int = 4):
    df = df.copy()
    df['growth_rate'] = df['sales'].pct_change() * 100
    
    actual_cagr = calculate_cagr(df)
    growth_in_claimed_period = calculate_growth_in_period(df, claim_period_months)
    
    # Rolling 6-month growth for comparison
    rolling_6m = df['sales'].rolling(window=6, min_periods=3).apply(
        lambda x: ((x[-1] / x[0]) - 1) * 100 if len(x) > 1 else 0, raw=True)
    max_historical_6m = rolling_6m.max()
    
    # Claimed months names for display
    claimed_months = df.tail(claim_period_months)['date'].dt.strftime('%b %Y').tolist()
    
    # Risk Score - Focused on Cherry-Picking
    risk_score = 0
    if growth_in_claimed_period > claimed_growth * 1.3:
        risk_score += 30
    if growth_in_claimed_period > max_historical_6m * 1.4:
        risk_score += 35
    if actual_cagr < claimed_growth * 0.4:
        risk_score += 25
    if actual_cagr < 0 and claimed_growth > 0:
        risk_score += 20
    risk_score = min(100, max(0, risk_score))

    # Verdict
    if risk_score >= 65:
        verdict = "❌ High Risk - Likely Cherry-Picked"
    elif risk_score >= 40:
        verdict = "⚠️ Medium Risk - Partially Supported"
    else:
        verdict = "✅ Low Risk - Likely Genuine"

    explanation = f"Company claims **{claimed_growth}% growth in last {claim_period_months} months**. "
    explanation += f"Actual growth in that period: **{growth_in_claimed_period:.1f}%**. "
    explanation += f"Overall CAGR: {actual_cagr:.2f}%. Highest historical 6-month growth: {max_historical_6m:.1f}%."

    if growth_in_claimed_period > max_historical_6m * 1.3:
        explanation += f" The last {claim_period_months} months show unusually high growth compared to history — strong sign of cherry-picking."

    return {
        "verdict": verdict,
        "risk_score": round(risk_score, 1),
        "claimed_growth": claimed_growth,
        "claim_period": claim_period_months,
        "actual_growth_in_period": round(growth_in_claimed_period, 1),
        "actual_cagr": round(actual_cagr, 2),
        "max_historical_6m": round(max_historical_6m, 1),
        "claimed_months": claimed_months,
        "explanation": explanation
    }