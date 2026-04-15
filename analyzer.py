import pandas as pd

def parse_dates(df):
    """Handles DD-MM-YYYY, MM-DD-YYYY, and YYYY-MM-DD automatically"""
    df = df.copy()
    
    # Best automatic parsing
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    
    # Fallback for DD-MM-YYYY (your current format)
    if df['date'].isna().any():
        mask = df['date'].isna()
        df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'date'], dayfirst=True, errors='coerce')
    
    # Last fallback for MM-DD-YYYY
    if df['date'].isna().any():
        mask = df['date'].isna()
        df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'date'], dayfirst=False, errors='coerce')
    
    if df['date'].isna().any():
        print("⚠️ Warning: Some dates could not be parsed.")
    
    return df


def calculate_cagr(df):
    if len(df) < 2:
        return 0.0
    start_sales = float(df['sales'].iloc[0])
    end_sales = float(df['sales'].iloc[-1])
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    if years <= 0:
        return 0.0
    return ((end_sales / start_sales) ** (1 / years) - 1) * 100


def calculate_growth_in_period(df, months=4):
    if len(df) < months:
        return 0.0
    recent = df.tail(months)
    return ((recent['sales'].iloc[-1] / recent['sales'].iloc[0]) - 1) * 100


def detect_cherry_picking(df: pd.DataFrame, claimed_growth: float, claim_period_months: int = 4):
    df = parse_dates(df)
    df = df.sort_values('date').reset_index(drop=True)
    
    df['growth_rate'] = df['sales'].pct_change() * 100

    actual_cagr = calculate_cagr(df)
    growth_in_claimed_period = calculate_growth_in_period(df, claim_period_months)

    rolling_6m = df['sales'].rolling(window=6, min_periods=3).apply(
        lambda x: ((x[-1] / x[0]) - 1) * 100 if len(x) > 1 else 0, raw=True)
    max_historical_6m = rolling_6m.max()

    claimed_months = df.tail(claim_period_months)['date'].dt.strftime('%b %Y').tolist()

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

    if risk_score >= 65:
        verdict = "❌ High Risk - Likely Cherry-Picked"
    elif risk_score >= 40:
        verdict = "⚠️ Medium Risk - Partially Supported"
    else:
        verdict = "✅ Low Risk - Likely Genuine"

    explanation = (f"Company claims **{claimed_growth}%** growth in last {claim_period_months} months. "
                   f"Actual: **{growth_in_claimed_period:.1f}%**. "
                   f"CAGR: {actual_cagr:.2f}%. Max 6M growth: {max_historical_6m:.1f}%.")

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


def analyze_cherry_picked_growth(df: pd.DataFrame, cherry_picked_months: list):
    df = parse_dates(df)
    df = df.sort_values('date').reset_index(drop=True)
    
    results = []
    
    for month_str in cherry_picked_months:
        try:
            if len(month_str.split('-')[0]) == 4:
                target_date = pd.to_datetime(month_str + '-01')
            else:
                target_date = pd.to_datetime(month_str + '-01', dayfirst=True)
        except:
            continue
        
        start_date = target_date - pd.DateOffset(months=2)
        end_date   = target_date + pd.DateOffset(months=2)
        
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        window = df[mask].copy()
        
        if len(window) < 2:
            continue
            
        window = window.sort_values('date')
        window['month_name'] = window['date'].dt.strftime('%b %Y')
        window['mom_growth_%'] = window['sales'].pct_change() * 100
        
        window_growth = ((window['sales'].iloc[-1] / window['sales'].iloc[0]) - 1) * 100
        
        cherry_row = window[window['date'].dt.to_period('M') == target_date.to_period('M')]
        cherry_sales = cherry_row['sales'].iloc[0] if not cherry_row.empty else None
        
        results.append({
            "cherry_picked_month": target_date.strftime('%b %Y'),
            "window_start": window['month_name'].iloc[0],
            "window_end": window['month_name'].iloc[-1],
            "window_growth_pct": round(window_growth, 1),
            "cherry_picked_sales": round(cherry_sales, 2) if cherry_sales is not None else None,
            "monthly_data": window[['month_name', 'sales', 'mom_growth_%']].round(2).to_dict('records')
        })
    
    return results