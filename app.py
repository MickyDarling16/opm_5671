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

    /* Collapsible Expander styling */
    [data-testid="stExpander"] {
        background-color: #162032 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        color: #f8fafc !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #38bdf8 !important;
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
if 'drop_shocks' not in st.session_state:
    st.session_state['drop_shocks'] = {}
if 'selected_drop_cycle' not in st.session_state:
    st.session_state['selected_drop_cycle'] = 'Drop 3: Back-to-School Surge (Jul–Sep)'

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
with st.expander("📈 Section 1: Historical Data & Dynamic Forecast Visualization", expanded=True):
    sec1_sub_col, sec1_tog_col = st.columns([3, 1])
    with sec1_sub_col:
        st.caption("Category Demand Index vs. Scenario-Adjusted SARIMAX Forecast")
    with sec1_tog_col:
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

# --- Section 2: Footwear Drop-Cycle & Replenishment Velocity Planner ---
with st.expander("⚡ Section 2: Footwear Drop-Cycle & Replenishment Velocity Planner", expanded=True):
    st.caption("Strategic quarterly merchandising cycles (Spring Launch, Summer Athletics, Back-to-School, Holiday Capsule) and supply chain shock stress-testing.")

    # Drop cycle definitions for athletic footwear merchandising
    drop_config = {
        'Drop 1: Spring Running Launch (Jan–Mar)': {
            'months': ['Jan', 'Feb', 'Mar'],
            'lead_time_wks': 4,
            'merch_focus': 'Spring outdoor running series launch & winter clearance transition',
            'dc_directive': 'Maintain lean safety buffer; stage initial running shoe catalog'
        },
        'Drop 2: Summer Athletic Series (Apr–Jun)': {
            'months': ['Apr', 'May', 'Jun'],
            'lead_time_wks': 5,
            'merch_focus': 'Marathon season, training footwear & lightweight mesh drops',
            'dc_directive': 'Build DC buffer 5 weeks prior; high weekly stock velocity'
        },
        'Drop 3: Back-to-School Surge (Jul–Sep)': {
            'months': ['Jul', 'Aug', 'Sep'],
            'lead_time_wks': 6,
            'merch_focus': 'Youth, collegiate athletic training & fall footwear catalog peak',
            'dc_directive': 'Prioritize youth/collegiate size curves (sizes 7-11); high turnover'
        },
        'Drop 4: Holiday & Black Friday Capsule (Oct–Dec)': {
            'months': ['Oct', 'Nov', 'Dec'],
            'lead_time_wks': 8,
            'merch_focus': 'Premium sneakerhead collaborations, Black Friday releases & holiday gifting',
            'dc_directive': 'Maximum safety buffer (+20%); pre-stage Black Friday inventory in October'
        }
    }

    # Baseline monthly sneaker demand (units: pairs)
    monthly_base_pairs = {
        'Jan': 295000, 'Feb': 310000, 'Mar': 385000,
        'Apr': 410000, 'May': 435000, 'Jun': 480000,
        'Jul': 520000, 'Aug': 545000, 'Sep': 425000,
        'Oct': 395000, 'Nov': 475000, 'Dec': 585000
    }
    prev_year_total_pairs = 5080000  # 5.08M pairs in 2025

    # Map months to their parent drop cycle
    month_to_drop = {}
    for drop_name, meta in drop_config.items():
        for m in meta['months']:
            month_to_drop[m] = drop_name

    # Calculate active monthly demand accounting for drop-level shocks
    active_monthly_pairs = {}
    for m, b_val in monthly_base_pairs.items():
        parent_drop = month_to_drop[m]
        shock_pct = st.session_state['drop_shocks'].get(parent_drop, 0.0)
        active_monthly_pairs[m] = b_val * (1.0 + shock_pct / 100.0)

    total_planned_pairs = sum(active_monthly_pairs.values())
    baseline_annual_pairs = sum(monthly_base_pairs.values())
    yoy_growth_pct = ((total_planned_pairs - prev_year_total_pairs) / prev_year_total_pairs) * 100.0
    avg_weekly_run_rate = total_planned_pairs / 52.0
    avg_drop_volume = total_planned_pairs / 4.0

    # Calculate volume by drop cycle
    drop_volumes = {}
    drop_base_volumes = {}
    for drop_name, meta in drop_config.items():
        drop_volumes[drop_name] = sum(active_monthly_pairs[m] for m in meta['months'])
        drop_base_volumes[drop_name] = sum(monthly_base_pairs[m] for m in meta['months'])

    peak_drop_name = max(drop_volumes, key=drop_volumes.get)
    peak_drop_share = (drop_volumes[peak_drop_name] / total_planned_pairs) * 100.0

    active_shocks_count = sum(1 for d, p in st.session_state['drop_shocks'].items() if abs(p) > 0.001)

    # Format KPI display strings
    vol_disp_str = f"{total_planned_pairs/1e6:.2f}M pairs"
    yoy_disp_str = f"{yoy_growth_pct:+.1f}% YoY vs 2025 (5.08M)"
    weekly_run_rate_str = f"{avg_weekly_run_rate/1000:,.1f}k pairs / wk"
    avg_drop_str = f"Avg Drop: {avg_drop_volume/1e6:.2f}M pairs"

    short_peak_name = peak_drop_name.split(':')[0]
    peak_vol_str = f"{drop_volumes[peak_drop_name]/1e6:.2f}M pairs"
    peak_meta_str = f"{short_peak_name} ({peak_drop_share:.1f}% of annual volume)"

    if active_shocks_count == 0:
        shock_status_title = "Steady-State"
        shock_status_sub = "Baseline plan (No shocks)"
        shock_badge_color = "#34d399"
    else:
        shock_status_title = f"{active_shocks_count} Shock(s) Active"
        shock_list = [f"{d.split(':')[0]}: {st.session_state['drop_shocks'][d]:+.0f}%" for d in drop_config if abs(st.session_state['drop_shocks'].get(d, 0.0)) > 0.001]
        shock_status_sub = ", ".join(shock_list)
        shock_badge_color = "#fbbf24"

    # 1. Executive Strategic KPI Cards (Distinct, original layout)
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 16px;">
        <div style="background-color: #162032; border: 1px solid #334155; border-radius: 12px; padding: 14px 18px;">
            <span style="font-size: 0.78rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 4px;">Planned 2026 Volume</span>
            <div style="font-size: 1.85rem; font-weight: 800; color: #38bdf8; font-family: monospace; line-height: 1.1; margin-bottom: 4px;">{vol_disp_str}</div>
            <span style="font-size: 0.78rem; color: #94a3b8;">{yoy_disp_str}</span>
        </div>
        <div style="background-color: #162032; border: 1px solid #334155; border-radius: 12px; padding: 14px 18px;">
            <span style="font-size: 0.78rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 4px;">Weekly DC Run-Rate</span>
            <div style="font-size: 1.85rem; font-weight: 800; color: #ffffff; font-family: monospace; line-height: 1.1; margin-bottom: 4px;">{weekly_run_rate_str}</div>
            <span style="font-size: 0.78rem; color: #94a3b8;">{avg_drop_str}</span>
        </div>
        <div style="background-color: #162032; border: 1px solid #334155; border-radius: 12px; padding: 14px 18px;">
            <span style="font-size: 0.78rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 4px;">Peak Drop Horizon</span>
            <div style="font-size: 1.85rem; font-weight: 800; color: #34d399; font-family: monospace; line-height: 1.1; margin-bottom: 4px;">{peak_vol_str}</div>
            <span style="font-size: 0.78rem; color: #94a3b8;">{peak_meta_str}</span>
        </div>
        <div style="background-color: #162032; border: 1px solid #334155; border-radius: 12px; padding: 14px 18px;">
            <span style="font-size: 0.78rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 4px;">Stress-Test Cadence</span>
            <div style="font-size: 1.85rem; font-weight: 800; color: {shock_badge_color}; font-family: monospace; line-height: 1.1; margin-bottom: 4px;">{shock_status_title}</div>
            <span style="font-size: 0.78rem; color: #94a3b8;">{shock_status_sub}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Multi-Perspective Planning Tabs
    tab_velocity, tab_allocation = st.tabs([
        "📈 Replenishment Velocity & Buffer Trajectory",
        "📊 Drop-Cycle Allocation & Factory Lead Times"
    ])

    all_months = list(monthly_base_pairs.keys())
    planned_curve = [monthly_base_pairs[m] for m in all_months]
    active_curve = [active_monthly_pairs[m] for m in all_months]
    safety_ceiling = [v * 1.15 for v in active_curve]

    with tab_velocity:
        fig_vel = go.Figure()

        # Baseline Plan Area
        fig_vel.add_trace(go.Scatter(
            x=all_months,
            y=planned_curve,
            mode='lines+markers',
            name='Baseline Planned Demand',
            line=dict(color='#38bdf8', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(56, 189, 248, 0.10)',
            marker=dict(size=6, color='#38bdf8'),
            hovertemplate='<b>%{x}</b><br>Baseline Demand: %{y:,.0f} pairs<extra></extra>'
        ))

        # Shock-Adjusted Trajectory (if active)
        if active_shocks_count > 0:
            fig_vel.add_trace(go.Scatter(
                x=all_months,
                y=active_curve,
                mode='lines+markers',
                name='Shock-Adjusted Demand',
                line=dict(color='#f43f5e', width=3, dash='solid'),
                marker=dict(size=8, color='#fbbf24', symbol='diamond'),
                hovertemplate='<b>%{x} (Shock-Adjusted)</b><br>Demand: %{y:,.0f} pairs<extra></extra>'
            ))

        # Safety Buffer Ceiling
        fig_vel.add_trace(go.Scatter(
            x=all_months,
            y=safety_ceiling,
            mode='lines',
            name='Target Buffer Threshold (+15%)',
            line=dict(color='#34d399', width=1.5, dash='dash'),
            hovertemplate='<b>%{x} Buffer Ceiling</b>: %{y:,.0f} pairs<extra></extra>'
        ))

        # Vertical dividing lines for Quarterly Drop Horizons
        fig_vel.add_vline(x=2.5, line_width=1, line_dash="dot", line_color="#475569")
        fig_vel.add_vline(x=5.5, line_width=1, line_dash="dot", line_color="#475569")
        fig_vel.add_vline(x=8.5, line_width=1, line_dash="dot", line_color="#475569")

        # Drop annotations across the chart
        fig_vel.add_annotation(x=1.0, y=max(safety_ceiling)*0.97, text="🌸 Drop 1 (Spring)", showarrow=False, font=dict(size=11, color="#94a3b8"))
        fig_vel.add_annotation(x=4.0, y=max(safety_ceiling)*0.97, text="☀️ Drop 2 (Summer)", showarrow=False, font=dict(size=11, color="#94a3b8"))
        fig_vel.add_annotation(x=7.0, y=max(safety_ceiling)*0.97, text="🎒 Drop 3 (BTS)", showarrow=False, font=dict(size=11, color="#94a3b8"))
        fig_vel.add_annotation(x=10.0, y=max(safety_ceiling)*0.97, text="🎁 Drop 4 (Holiday)", showarrow=False, font=dict(size=11, color="#94a3b8"))

        fig_vel.update_layout(
            height=390,
            template="plotly_dark",
            paper_bgcolor="#162032",
            plot_bgcolor="#162032",
            margin=dict(l=45, r=25, t=35, b=40),
            xaxis_title="Replenishment Horizon (2026 Monthly)",
            yaxis_title="Sneaker Volume (Pairs)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    with tab_allocation:
        col_bar_alloc, col_tbl_alloc = st.columns([1.2, 1.8], gap="medium")
        with col_bar_alloc:
            drop_labels = [d.split(':')[0] for d in drop_config]
            drop_vols_list = [drop_volumes[d] for d in drop_config]
            colors_list = ['#38bdf8', '#34d399', '#fbbf24', '#f43f5e']

            fig_drop_bar = go.Figure(go.Bar(
                x=drop_labels,
                y=drop_vols_list,
                marker=dict(color=colors_list, line=dict(color='#334155', width=1)),
                text=[f"{v/1e6:.2f}M" for v in drop_vols_list],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f} pairs<extra></extra>'
            ))
            fig_drop_bar.update_layout(
                height=320,
                template="plotly_dark",
                paper_bgcolor="#162032",
                plot_bgcolor="#162032",
                margin=dict(l=35, r=20, t=30, b=35),
                yaxis_title="Pairs per Drop",
                xaxis_title="Drop Cycle"
            )
            st.plotly_chart(fig_drop_bar, use_container_width=True)

        with col_tbl_alloc:
            table_rows = []
            for drop_name, meta in drop_config.items():
                vol = drop_volumes[drop_name]
                share = (vol / total_planned_pairs) * 100.0
                shk = st.session_state['drop_shocks'].get(drop_name, 0.0)
                status_str = f"{shk:+.0f}% Shock" if abs(shk) > 0.001 else "Baseline"
                table_rows.append({
                    "Drop Cycle": drop_name.split(':')[0],
                    "Window": drop_name.split('(')[1].replace(')', ''),
                    "Planned Volume": f"{vol:,.0f} pairs",
                    "Share": f"{share:.1f}%",
                    "Lead Time": f"{meta['lead_time_wks']} Weeks",
                    "DC Replenishment Directive": meta['dc_directive'],
                    "Status": status_str
                })
            df_drop_matrix = pd.DataFrame(table_rows)
            st.dataframe(df_drop_matrix, use_container_width=True, hide_index=True)

    # 3. Authentic Footwear Supply Chain Shock Stress-Testing Suite
    with st.container(border=True):
        col_ctrl_hd, col_ctrl_meta = st.columns([2.5, 1.5])
        with col_ctrl_hd:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.15rem;">🚢</span>
                <span style="font-weight: 700; font-size: 0.92rem; color: #ffffff;">Supply Chain Shock Stress-Testing Suite:</span>
                <span style="font-size: 0.8rem; color: #94a3b8;">Simulate factory delays, transit bottlenecks, or collab demand surges</span>
            </div>
            """, unsafe_allow_html=True)
        with col_ctrl_meta:
            drop_keys = list(drop_config.keys())
            if st.session_state.get('selected_drop_cycle') not in drop_keys:
                st.session_state['selected_drop_cycle'] = drop_keys[2]
            cur_active_drop = st.session_state['selected_drop_cycle']
            cur_drop_shock = st.session_state['drop_shocks'].get(cur_active_drop, 0.0)
            st.markdown(f"""
            <div style="text-align: right; font-size: 0.8rem;">
                <span style="color: #94a3b8;">Active Horizon: </span>
                <strong style="color: #38bdf8;">{cur_active_drop.split(':')[0]}</strong>
                <span style="color: {'#fbbf24' if abs(cur_drop_shock) > 0.001 else '#34d399'}; font-family: monospace; font-size: 0.75rem; margin-left: 6px; padding: 2px 6px; background-color: rgba(255,255,255,0.06); border-radius: 4px;">
                    {f'Shock: {cur_drop_shock:+.0f}%' if abs(cur_drop_shock) > 0.001 else 'Baseline (0%)'}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

        # Drop Selection Pills
        selected_drop_pill = st.pills(
            "Select Merchandising Drop Cycle to Stress-Test:",
            drop_keys,
            default=st.session_state['selected_drop_cycle'],
            key="drop_pills_selector"
        )
        if selected_drop_pill and selected_drop_pill != st.session_state['selected_drop_cycle']:
            st.session_state['selected_drop_cycle'] = selected_drop_pill
            st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Shock Slider & Presets
        c_shock_input, c_shock_presets = st.columns([1.2, 2.8], gap="medium")
        with c_shock_input:
            curr_shock_val = float(st.session_state['drop_shocks'].get(cur_active_drop, 0.0))
            new_shock_val = st.number_input(
                f"Demand Shock for {cur_active_drop.split(':')[0]} (%)",
                min_value=-60.0,
                max_value=60.0,
                value=curr_shock_val,
                step=5.0,
                format="%.1f",
                key=f"shock_stepper_{cur_active_drop.split(':')[0]}",
                help="Apply positive demand surge or negative supply disruption (-60% to +60%)."
            )
            if abs(new_shock_val - curr_shock_val) > 0.001:
                st.session_state['drop_shocks'][cur_active_drop] = new_shock_val
                st.rerun()

        with c_shock_presets:
            st.caption("Footwear Industry Shock Scenarios:")
            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5, gap="small")
            with btn_col1:
                if st.button("🚢 Port Delay (-20%)", use_container_width=True, help="Container shipping & ocean freight bottleneck in Q3 Back-to-School"):
                    st.session_state['drop_shocks']['Drop 3: Back-to-School Surge (Jul–Sep)'] = -20.0
                    st.session_state['selected_drop_cycle'] = 'Drop 3: Back-to-School Surge (Jul–Sep)'
                    st.rerun()
            with btn_col2:
                if st.button("🔥 Collab Craze (+25%)", use_container_width=True, help="Viral athlete collaboration surge in Q2 Summer series"):
                    st.session_state['drop_shocks']['Drop 2: Summer Athletic Series (Apr–Jun)'] = 25.0
                    st.session_state['selected_drop_cycle'] = 'Drop 2: Summer Athletic Series (Apr–Jun)'
                    st.rerun()
            with btn_col3:
                if st.button("❄️ DC Freeze (-15%)", use_container_width=True, help="Severe winter storm halting Midwestern distribution center fulfillment"):
                    st.session_state['drop_shocks']['Drop 1: Spring Running Launch (Jan–Mar)'] = -15.0
                    st.session_state['selected_drop_cycle'] = 'Drop 1: Spring Running Launch (Jan–Mar)'
                    st.rerun()
            with btn_col4:
                if st.button("🎁 Holiday Rush (+20%)", use_container_width=True, help="Unexpected Black Friday / Cyber Week retail footwear demand spike"):
                    st.session_state['drop_shocks']['Drop 4: Holiday & Black Friday Capsule (Oct–Dec)'] = 20.0
                    st.session_state['selected_drop_cycle'] = 'Drop 4: Holiday & Black Friday Capsule (Oct–Dec)'
                    st.rerun()
            with btn_col5:
                if st.button("🔄 Reset Baseline", use_container_width=True, help="Clear all applied shocks across all drop cycles"):
                    st.session_state['drop_shocks'] = {}
                    st.rerun()

        # Dynamic Operational Feedback Alert
        if active_shocks_count > 0:
            net_pair_shift = total_planned_pairs - baseline_annual_pairs
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; padding: 12px 16px; margin-top: 10px; font-size: 0.82rem; color: #fde68a; line-height: 1.5;">
                <strong>⚠️ Active Supply Chain Shock Impact:</strong> {active_shocks_count} drop cycle(s) modified ({shock_status_sub}). Full-year demand deviation: <strong>{net_pair_shift:+,.0f} pairs</strong> ({yoy_growth_pct:+.1f}% vs 2025).
                <br>
                <span style="font-size: 0.78rem; color: #cbd5e1;"><strong>Operational Directives for Section 3:</strong> {'Elevate safety stock buffers in the Reorder Advisor to absorb supplier stockout risk.' if net_pair_shift < 0 else 'Accelerate factory purchase orders and reserve additional regional warehouse pallet positions.'}</span>
            </div>
            """, unsafe_allow_html=True)


# --- Section 3: Footwear Inventory Reorder Advisor ---
with st.expander("📦 Section 3: Footwear Inventory Reorder Advisor", expanded=True):
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

# --- Section 4: Model Performance Summary ---
with st.expander("📊 Section 4: Model Performance Summary (Benchmark vs. Candidates)", expanded=True):
    st.caption("Comparing forecasting models evaluated on the out-of-sample test set (last 12 months).")
    st.dataframe(df_models, use_container_width=True, hide_index=True)

# --- Section 5: Limitations & Notes ---
with st.expander("⚠️ Section 5: Limitations & Operational Notes", expanded=False):
    st.markdown("""
    - **Prototype Status:** Developed for academic case study purposes (OPIM-5671, Team 5) modeling Apex Athletics footwear supply chain operations.
    - **Model Uncertainty:** Forecasts contain inherent uncertainty. Confidence intervals represent expected variance in consumer discretionary spending.
    - **Residual Autocorrelation:** The primary SARIMAX model captures trend and holiday seasonality, though residual variance remains from historical anomalies like the 2020 retail disruption.
    - **Scenario Assumptions:** Exogenous inputs (CPI and Sentiment shifts) are modeled as steady-state deviations across the 6-month replenishment horizon.
    - **Production Roadmap:** A full enterprise deployment would incorporate SKU-level shoe size curves (e.g., sizes 8-12), direct EDI factory integration, and regional DC routing.
    """)
