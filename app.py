from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

if not os.path.exists("uploads"):
    os.makedirs("uploads")


# ========================= CORE ANALYSIS FUNCTION =========================
def analyze_timeline(file_path, claimed_growth=None):

    # -------- LOAD FILE --------
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # -------- HANDLE DIFFERENT DATASETS --------
    if "Order Date" in df.columns and "Total Sales" in df.columns:
        df = df.rename(columns={"Order Date": "date", "Total Sales": "sales"})

    elif "OrderDate" in df.columns and "TotalAmount" in df.columns:
        df = df.rename(columns={"OrderDate": "date", "TotalAmount": "sales"})

    elif "date" not in df.columns or "sales" not in df.columns:
        raise Exception("Required columns not found. Use (date + sales) OR Amazon dataset.")

    # -------- CLEAN DATA --------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(subset=["date", "sales"])
    df = df.sort_values("date")

    # -------- MONTHLY AGGREGATION --------
    monthly = df.groupby(
        pd.Grouper(key="date", freq="ME")
    )["sales"].sum().reset_index()

    monthly.columns = ["date", "sales"]

    # -------- GROWTH --------
    monthly["growth"] = monthly["sales"].pct_change() * 100

    avg_growth = monthly["growth"].mean()
    std_growth = monthly["growth"].std()

    # -------- ML (FIXED NO .iloc ISSUE) --------
    model = IsolationForest(contamination=0.2, random_state=42)

    # IMPORTANT FIX: store in dataframe (NOT numpy)
    monthly["ml_flag"] = model.fit_predict(monthly[["sales"]])
    monthly["ml_flag"] = (monthly["ml_flag"] == -1).astype(int)

    anomaly_ratio = monthly["ml_flag"].mean()

    # -------- CLAIM LOGIC --------
    verdict = "No Claim Provided"
    insight = "Please enter a claimed growth percentage."
    coverage_percent = 0

    if claimed_growth is not None:

        z_claim = (claimed_growth - avg_growth) / std_growth if std_growth != 0 else 0

        tolerance = max(5, claimed_growth * 0.15)

        matches = monthly[
            (monthly["growth"] >= claimed_growth - tolerance) &
            (monthly["growth"] <= claimed_growth + tolerance)
        ]

        coverage = len(matches) / len(monthly) if len(monthly) > 0 else 0
        coverage_percent = round(coverage * 100, 2)

        if z_claim > 2 and coverage < 0.2:
            verdict = "Cherry Picked / Misleading"
            insight = "The reported growth is significantly higher than the overall trend and appears only in limited time periods."

        elif abs(z_claim) < 1 and coverage > 0.5:
            verdict = "Valid Claim"
            insight = "The reported growth is consistent across the dataset."

        else:
            verdict = "Partially Supported"
            insight = "The reported growth is not consistently supported across the dataset."

        if anomaly_ratio > 0.3:
            insight += " Data shows seasonal or irregular spikes."

    # -------- SUMMARY --------
    total_sales = monthly["sales"].sum()
    avg_sales = monthly["sales"].mean()

    # -------- GRAPH --------
    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=monthly["date"],
        y=monthly["sales"],
        mode='lines',
        line=dict(color='#38bdf8', width=4, shape='spline'),
        customdata=monthly["growth"],
        hovertemplate="Date: %{x}<br>Sales: %{y}<br>Growth: %{customdata:.2f}%"
    ))

    # Highlight anomalies (ML)
    anomalies = monthly[monthly["ml_flag"] == 1]

    fig.add_trace(go.Scatter(
        x=anomalies["date"],
        y=anomalies["sales"],
        mode='markers',
        marker=dict(color='red', size=10),
        name="Irregular / Peak Periods"
    ))

    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=6, label="6M", step="month"),
                    dict(count=1, label="1Y", step="year"),
                    dict(step="all", label="All")
                ]
            )
        ),
        hovermode="x unified"
    )

    graph_html = pio.to_html(fig, full_html=False)

    return (
        verdict,
        insight,
        graph_html,
        total_sales,
        avg_sales,
        avg_growth,
        coverage_percent
    )


# ========================= ROUTE =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files["file"]

            if file.filename == "":
                return "No file selected"

            claimed_growth = request.form.get("claimed_growth")
            claimed_growth = float(claimed_growth) if claimed_growth else None

            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            result = analyze_timeline(path, claimed_growth)

            return render_template(
                "index.html",
                verdict=result[0],
                insight=result[1],
                graph=result[2],
                total_sales=round(result[3], 2),
                avg_sales=round(result[4], 2),
                avg_growth=round(result[5], 2),
                coverage_percent=result[6],
                claimed_growth=claimed_growth,
                show_results=True
            )

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html", show_results=False)


# ========================= RUN =========================
if __name__ == "__main__":
    app.run(debug=True)
