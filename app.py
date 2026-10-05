import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# Set page layout
st.set_page_config(
    page_title="Apex Athletics — Demand Forecasting & Inventory Planner",
    page_icon="👟",
    layout="wide"
)

# Custom CSS for Midnight Command Theme & Pixel-Perfect Match with Simulation
st.markdown("""
<style>
    /* Adjust main container padding so content clears Streamlit header bar */
    .block-container {
        padding-top: 3.2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 96% !important;
    }
    
    /* Global dark backgrounds */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b !important;
    }

    /* Container cards in Midnight Command */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #162032 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }
    
    /* Direct number input styling for Midnight Command theme */
    div[data-testid="stNumberInput"] input {
        background-color: #0f172a !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-family: monospace, ui-monospace, sans-serif !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        text-align: center !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background-color: #334155 !important;
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
    }

    /* Selectbox styling */
    div[data-testid="stSelectbox"] > div {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }

    /* Preset buttons */
    div[data-testid="stButton"] > button {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        color: #e2e8f0 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #334155 !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* Responsive metric cards styling */
    [data-testid="stMetric"] {
        background-color: #162032 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        white-space: normal !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        font-family: monospace, ui-monospace, sans-serif !important;
    }

    /* Table styling */
    div[data-testid="stDataFrame"] {
        width: 100% !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Header / Brand Banner matching preview
st.markdown("""
<div style="border-bottom: 1px solid #334155; padding-bottom: 14px; margin-bottom: 20px;">
    <h1 style="font-size: 1.75rem; font-weight: 800; color: #ffffff; margin: 0; padding: 0; display: flex; align-items: center; gap: 10px;">
        <span>👟</span>
        <span>Apex Athletics — Demand Forecasting & Inventory Planner</span>
    </h1>
    <p style="font-size: 0.85rem; color: #94a3b8; margin: 6px 0 0 0;">
        Central Supply Chain & Distribution Replenishment Cockpit (<strong>Category: Athletic Footwear & Sneakers</strong>). Leverages econometric modeling and macroeconomic scenario planning to optimize sneaker inventory replenishment.
    </p>
</div>
""", unsafe_allow_html=True)

# Paths
DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    df_clean = pd.read_csv(os.path.join(DIR, "data", "data_clean.csv"))
    df_train = pd.read_csv(os.path.join(DIR, "data", "data_train.csv"))
    df_test = pd.read_csv(os.path.join(DIR, "data", "data_test.csv"))
    df_future = pd.read_csv(os.path.join(DIR, "data", "future_forecast.csv"))
    df_test_preds = pd.read_csv(os.path.join(DIR, "data", "best_model_test_preds.csv"))
    df_models = pd.read_csv(os.path.join(DIR, "data", "model_comparison.csv"))
    
    # Ensure dates are datetime
    for df in [df_clean, df_train, df_test, df_future]:
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
    if 'DATE' in df_test_preds.columns:
        df_test_preds['DATE'] = pd.to_datetime(df_test_preds['DATE'])
    elif 'date' in df_test_preds.columns:
        df_test_preds['date'] = pd.to_datetime(df_test_preds['date'])

    return df_clean, df_train, df_test, df_future, df_test_preds, df_models

@st.cache_resource
def load_model():
    model_path = os.path.join(DIR, "models", "fitted_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

try:
    df_clean, df_train, df_test, df_future_base, df_test_preds, df_models = load_data()
    model_payload = load_model()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

# Initialize session state for scenario inputs
if 'cpi_input' not in st.session_state:
    st.session_state['cpi_input'] = 0.0
if 'sent_input' not in st.session_state:
    st.session_state['sent_input'] = 0.0

# --- Sidebar: Enterprise Portal Info ---
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 12px; border-bottom: 1px solid #1e293b; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.3rem;">👟</span>
            <span style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">Apex Athletics</span>
        </div>
        <span style="font-size: 0.75rem; font-weight: 600; color: #38bdf8; display: block; margin-top: 2px;">
            Central Supply Chain Portal
        </span>
        <p style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px; line-height: 1.4;">
            Autonomous econometric forecasting & inventory replenishment engine.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Forecasting Engine Card
    st.markdown("""
    <div style="background-color: #162032; border: 1px solid #334155; border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;">
        <span style="font-size: 0.7rem; font-weight: 700; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 8px;">
            ⚡ Active Forecasting Engine
        </span>
        <div style="font-family: monospace; font-size: 0.78rem; line-height: 1.7;">
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">Model:</span><span style="color: #38bdf8; font-weight: 700;">SARIMAX + Exog</span></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">Holdout RMSE:</span><span style="color: #34d399; font-weight: 700;">$5,778M</span></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">Error Reduction:</span><span style="color: #34d399; font-weight: 700;">+80.0%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Scope Parameters
    st.markdown("""
    <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 16px;">
        <span style="font-size: 0.7rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 8px;">
            📦 Scope Parameters
        </span>
        <div style="display: flex; flex-direction: column; gap: 6px;">
            <div>• <strong>Category:</strong> Performance Sneakers</div>
            <div>• <strong>Dataset:</strong> FRED RSAFS (176 Mo)</div>
            <div>• <strong>Frequency:</strong> Monthly (MS)</div>
            <div>• <strong>Horizon:</strong> 6-Month Forward Cycle</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("OPIM-5671 Project #1 — Team 5")

# --- Section 1: Historical Data & Forecast Visualization ---
sec1_title_col, sec1_tog_col = st.columns([3, 1])
with sec1_title_col:
    st.header("📈 Historical Data & Dynamic Forecast Visualization")
    st.caption("Category Demand Index vs. Scenario-Adjusted SARIMAX Forecast")
with sec1_tog_col:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    show_ci = st.toggle("Show Confidence Interval", value=True)

# Macroeconomic Direct Inputs Panel (Directly above Chart)
with st.container(border=True):
    col_p_title, col_p_btn1, col_p_btn2, col_p_btn3 = st.columns([2.2, 1.0, 1.2, 1.2], gap="small")
    with col_p_title:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 6px; padding-top: 4px;">
            <span style="font-size: 1rem;">🌍</span>
            <span style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #38bdf8;">Macroeconomic Sensitivity Inputs</span>
            <span style="font-size: 0.72rem; color: #94a3b8;">(Directly shifts 6-mo curve)</span>
        </div>
        """, unsafe_allow_html=True)
    with col_p_btn1:
        if st.button("Base (0%)", use_container_width=True, help="Reset to baseline 0% shift"):
            st.session_state['cpi_input'] = 0.0
            st.session_state['sent_input'] = 0.0
            st.rerun()
    with col_p_btn2:
        if st.button("🔥 Inflation Surge", use_container_width=True, help="Simulate +2.5% CPI, -5.0% Sentiment"):
            st.session_state['cpi_input'] = 2.5
            st.session_state['sent_input'] = -5.0
            st.rerun()
    with col_p_btn3:
        if st.button("🚀 Consumer Boom", use_container_width=True, help="Simulate -1.0% CPI, +10.0% Sentiment"):
            st.session_state['cpi_input'] = -1.0
            st.session_state['sent_input'] = 10.0
            st.rerun()
            
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    
    macro_c1, macro_c2 = st.columns(2, gap="medium")
    with macro_c1:
        cpi_change_pct = st.number_input(
            "Expected CPI (Inflation) Shift (%)",
            min_value=-5.0,
            max_value=5.0,
            value=float(st.session_state['cpi_input']),
            step=0.5,
            format="%.1f",
            help="Simulate inflation acceleration or deceleration relative to current trend (-5.0% to +5.0%)."
        )
        st.session_state['cpi_input'] = cpi_change_pct
        
    with macro_c2:
        sentiment_change_pct = st.number_input(
            "Expected Consumer Sentiment Shift (%)",
            min_value=-20.0,
            max_value=20.0,
            value=float(st.session_state['sent_input']),
            step=1.0,
            format="%.1f",
            help="Simulate discretionary consumer optimism or hesitation (-20.0% to +20.0%)."
        )
        st.session_state['sent_input'] = sentiment_change_pct

# Compute dynamic scenario forecast using inputs
if model_payload and (model_payload.get('uses_exog', False) or bool(model_payload.get('exog_cols', []))):
    last_cpi = model_payload['last_cpi']
    last_sentiment = model_payload['last_umcsent']
    last_date = pd.to_datetime(model_payload['last_date'])
    
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
    
    # Apply shifts
    scenario_cpi = last_cpi * (1 + (cpi_change_pct / 100.0))
    scenario_sentiment = last_sentiment * (1 + (sentiment_change_pct / 100.0))
    
    future_exog_scenario = pd.DataFrame({
        'UMCSENT': [scenario_sentiment] * 6,
        'CPIAUCSL': [scenario_cpi] * 6
    }, index=future_dates)
    
    # Get dynamic forecast
    full_model = model_payload['model']
    forecast_obj = full_model.get_forecast(steps=6, exog=future_exog_scenario)
    df_future = pd.DataFrame({
        'date': future_dates,
        'forecast': forecast_obj.predicted_mean.values,
        'lower_ci': forecast_obj.conf_int().iloc[:, 0].values,
        'upper_ci': forecast_obj.conf_int().iloc[:, 1].values
    })
else:
    df_future = df_future_base.copy()

# Plotly Forecast Visualization Header Badge
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 6px;">
    <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">Interactive Multi-Series Trajectory:</span>
    <span style="background-color: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.4); color: #fda4af; padding: 2px 10px; border-radius: 9999px; font-family: monospace; font-size: 0.74rem; font-weight: 600;">
        Active Scenario: {cpi_change_pct:+.1f}% CPI, {sentiment_change_pct:+.1f}% Sentiment
    </span>
</div>
""", unsafe_allow_html=True)

# Plotly Forecast Visualization
fig = go.Figure()
# Train data
train_date = 'DATE' if 'DATE' in df_train.columns else 'date'
if train_date in df_train.columns and 'RSAFS' in df_train.columns:
    fig.add_trace(go.Scatter(x=df_train[train_date], y=df_train['RSAFS'], mode='lines', name='Historical Sales (Train)', line=dict(color='#38bdf8', width=1.5)))
# Test data
test_date = 'DATE' if 'DATE' in df_test.columns else 'date'
if test_date in df_test.columns and 'RSAFS' in df_test.columns:
    fig.add_trace(go.Scatter(x=df_test[test_date], y=df_test['RSAFS'], mode='lines', name='Actual Sales (Test)', line=dict(color='#34d399', width=2)))
    
# Test predictions
pred_col = 'Prediction' if 'Prediction' in df_test_preds.columns else ('forecast' if 'forecast' in df_test_preds.columns else df_test_preds.columns[1])
date_col = 'DATE' if 'DATE' in df_test_preds.columns else ('date' if 'date' in df_test_preds.columns else df_test_preds.columns[0])
fig.add_trace(go.Scatter(x=df_test_preds[date_col], y=df_test_preds[pred_col], mode='lines', name='Model Prediction (Test)', line=dict(color='#fbbf24', width=2, dash='dash')))

# Future Forecast
f_date = 'date'
f_pred = 'forecast'
f_lower = 'lower_ci'
f_upper = 'upper_ci'

fig.add_trace(go.Scatter(x=df_future[f_date], y=df_future[f_pred], mode='lines', name='Future Forecast (Scenario)', line=dict(color='#f43f5e', width=3)))

if show_ci and f_lower in df_future.columns and f_upper in df_future.columns:
    if not df_future[f_lower].isna().all():
        fig.add_trace(go.Scatter(
            x=pd.concat([df_future[f_date], df_future[f_date][::-1]]),
            y=pd.concat([df_future[f_upper], df_future[f_lower][::-1]]),
            fill='toself',
            fillcolor='rgba(244, 63, 94, 0.22)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

fig.update_layout(
    height=480,
    xaxis_title="Date",
    yaxis_title="Retail Sales (Millions $)",
    template="plotly_dark",
    paper_bgcolor="#1e293b",
    plot_bgcolor="#1e293b",
    font=dict(color="#f8fafc"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=45, r=20, t=35, b=40),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# --- Section 2: Footwear Inventory Reorder Advisor ---
st.header("📦 Footwear Inventory Reorder Advisor")
st.caption("Determine recommended factory purchase orders for Apex Athletics Distribution Centers based on scenario-adjusted demand forecasts.")

col_plan, col_rec = st.columns([1.1, 1.9], gap="medium")

with col_plan:
    st.subheader("Planning Inputs")
    with st.container(border=True):
        current_inv = st.number_input(
            "Current Warehouse Stock (pairs)",
            min_value=0,
            value=50000,
            step=1000,
            help="Current inventory on hand across regional distribution centers."
        )
        safety_stock_pct = st.number_input(
            "Safety Stock Buffer (%)",
            min_value=5,
            max_value=40,
            value=15,
            step=1,
            help="Buffer stock protecting against holiday sneaker demand surges (Range: 5% - 40%)."
        ) / 100.0
        lead_time_weeks = st.number_input(
            "Factory & Shipping Lead Time (weeks)",
            min_value=1,
            max_value=12,
            value=4,
            step=1,
            help="Overseas footwear manufacturing and container transit time to regional distribution centers (Range: 1 - 12 wks)."
        )
        horizon_months = st.selectbox(
            "Planning Horizon (Drop Cycle)",
            [1, 2, 3, 6],
            index=2,
            format_func=lambda x: f"{x} Months ({'Quarterly Drop' if x==3 else ('Bimonthly' if x==2 else ('Single Month' if x==1 else 'Full Season'))})"
        )

with col_rec:
    # Get forecasted demand for horizon
    forecast_horizon = df_future.head(horizon_months)
    expected_demand = forecast_horizon[f_pred].sum()
    safety_stock = expected_demand * safety_stock_pct
    planning_weeks = horizon_months * 4.333  # Approx weeks in a month
    reorder_point = (expected_demand / planning_weeks) * lead_time_weeks + safety_stock
    recommended_order = max(0, reorder_point - current_inv)
    
    if current_inv > reorder_point * 1.2:
        status_label = "🟢 Stock Adequate"
        status_color = "#34d399"
        status_bg = "rgba(16, 185, 129, 0.15)"
        status_border = "rgba(16, 185, 129, 0.4)"
    elif current_inv > reorder_point:
        status_label = "🟡 Reorder Soon"
        status_color = "#fbbf24"
        status_bg = "rgba(245, 158, 11, 0.15)"
        status_border = "rgba(245, 158, 11, 0.4)"
    else:
        status_label = "🔴 Reorder Now"
        status_color = "#f43f5e"
        status_bg = "rgba(244, 63, 94, 0.15)"
        status_border = "rgba(244, 63, 94, 0.4)"
        
    st.subheader("Recommendation Output")
    
    # 2x2 Custom Styled Metric Grid matching simulation preview
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px;">
        <div style="background-color: #162032; border: 1px solid #334155; border-radius: 10px; padding: 12px 16px;">
            <span style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">Projected Footwear Demand</span>
            <div style="font-size: 1.35rem; font-weight: 700; font-family: monospace; color: #38bdf8;">
                {expected_demand:,.0f} <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">pairs</span>
            </div>
        </div>
        <div style="background-color: #162032; border: 1px solid #334155; border-radius: 10px; padding: 12px 16px;">
            <span style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">Reorder Point (ROP)</span>
            <div style="font-size: 1.35rem; font-weight: 700; font-family: monospace; color: #38bdf8;">
                {reorder_point:,.0f} <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">pairs</span>
            </div>
        </div>
        <div style="background-color: #162032; border: 1px solid #334155; border-radius: 10px; padding: 12px 16px;">
            <span style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">Recommended Factory Order</span>
            <div style="font-size: 1.35rem; font-weight: 700; font-family: monospace; color: #34d399;">
                {recommended_order:,.0f} <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">pairs</span>
            </div>
        </div>
        <div style="background-color: #162032; border: 1px solid #334155; border-radius: 10px; padding: 12px 16px;">
            <span style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 4px;">Warehouse Stock Status</span>
            <div style="display: inline-flex; align-items: center; padding: 4px 10px; background-color: {status_bg}; border: 1px solid {status_border}; border-radius: 6px; color: {status_color}; font-size: 0.95rem; font-weight: 700; margin-top: 2px;">
                {status_label}
            </div>
        </div>
    </div>
    
    <div style="background: rgba(8, 51, 68, 0.45); border: 1px solid #0891b2; border-radius: 10px; padding: 14px 16px; color: #a5f3fc; font-size: 0.82rem; line-height: 1.6;">
        <strong>📌 Action Statement:</strong> Based on projected category demand of <strong>{expected_demand:,.0f} pairs</strong> over the next <strong>{horizon_months} months</strong> under your selected macroeconomic scenario, Apex Athletics distribution centers should place an overseas factory order of <strong>{recommended_order:,.0f} pairs</strong> today to avoid stockouts before the next product drop.
    </div>
    """, unsafe_allow_html=True)

# --- Section 3: Model Performance Summary ---
with st.expander("📊 Model Performance Summary (Benchmark vs. Candidates)", expanded=True):
    st.caption("Comparing forecasting models evaluated on the out-of-sample test set (last 12 months).")
    st.dataframe(df_models, use_container_width=True, hide_index=True)

# --- Section 4: Limitations & Notes ---
with st.expander("⚠️ Limitations & Operational Notes", expanded=False):
    st.markdown("""
    - **Prototype Status:** Developed for academic case study purposes (OPIM-5671, Team 5) modeling Apex Athletics footwear supply chain operations.
    - **Model Uncertainty:** Forecasts contain inherent uncertainty. Confidence intervals represent expected variance in consumer discretionary spending.
    - **Residual Autocorrelation:** The primary SARIMAX model captures trend and holiday seasonality, though residual variance remains from historical anomalies like the 2020 retail disruption.
    - **Scenario Assumptions:** Exogenous inputs (CPI and Sentiment shifts) are modeled as steady-state deviations across the 6-month replenishment horizon.
    - **Production Roadmap:** A full enterprise deployment would incorporate SKU-level shoe size curves (e.g., sizes 8-12), direct EDI factory integration, and regional DC routing.
    """)
