import os
import sys
import datetime

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
from FoodDeliveryTimePrediction.Utils.main_utils import load_object, TimePredictionModel

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SpeedyBites | Delivery Time Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS Styling: High Contrast Cyber Black & Neo Blue
# ---------------------------------------------------------
custom_css = """
<style>
    /* Main Theme Setup - Deep Cyber Black */
    .stApp {
        background-color: #030712 !important;
        color: #ffffff !important;
    }
    
    /* Remove/Style Streamlit top header bar */
    header[data-testid="stHeader"] {
        background-color: #030712 !important;
    }
    .stAppHeader {
        background-color: #030712 !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    div[data-testid="stToolbar"] {
        color: #ffffff !important;
    }

    /* Sidebar Background & High-Contrast Text */
    section[data-testid="stSidebar"] {
        background-color: #090e1a !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] p {
        color: #cbd5e1 !important;
    }
    
    /* Radio Button Labels & Options Visibility */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] p,
    div[data-testid="stRadio"] label p {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Input Labels Visibility */
    label[data-testid="stWidgetLabel"] p,
    .stMarkdown p,
    label p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Tab Header Visibility */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px 8px 0 0 !important;
    }
    button[data-baseweb="tab"] div p {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #00f0ff !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] div p {
        color: #00f0ff !important;
        font-size: 1.05rem !important;
    }

    /* Selectbox & Input Fields Visibility */
    div[data-baseweb="select"] > div,
    input {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-color: #334155 !important;
    }
    div[data-baseweb="select"] * {
        color: #ffffff !important;
    }

    /* Main Header Banner Styling */
    .main-header {
        background: linear-gradient(135deg, #090d16 0%, #0369a1 50%, #00f0ff 100%);
        padding: 26px 36px;
        border-radius: 16px;
        color: #ffffff;
        margin-top: 0px;
        margin-bottom: 24px;
        border: 1px solid rgba(0, 240, 255, 0.4);
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.25);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        text-shadow: 0 0 14px rgba(0, 240, 255, 0.7);
        margin-bottom: 6px !important;
    }
    .main-header p {
        color: #e0f2fe !important;
        font-size: 1.05rem !important;
        margin-bottom: 0px !important;
    }

    /* Card Containers */
    .css-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }

    /* Metric Display Box */
    .metric-box {
        background: linear-gradient(145deg, #070e1b, #0f1c35);
        border: 2px solid #00f0ff;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.3);
    }
    .metric-value {
        font-size: 3.4rem;
        font-weight: 900;
        color: #00f0ff !important;
        background: linear-gradient(90deg, #00f0ff, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
        text-shadow: 0 0 30px rgba(0, 240, 255, 0.4);
    }
    .metric-label {
        font-size: 1rem;
        color: #cbd5e1 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        background: linear-gradient(135deg, #0284c7, #00f0ff) !important;
        color: #030712 !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.7) !important;
        transform: translateY(-1px);
    }
    
    /* Custom Badge styling */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-top: 6px;
        box-shadow: 0 0 12px rgba(0, 240, 255, 0.2);
    }
    .badge-fast { background-color: rgba(0, 240, 255, 0.2); color: #00f0ff !important; border: 1.5px solid #00f0ff; }
    .badge-std { background-color: rgba(56, 189, 248, 0.2); color: #38bdf8 !important; border: 1.5px solid #38bdf8; }
    .badge-slow { background-color: rgba(99, 102, 241, 0.25); color: #a5b4fc !important; border: 1.5px solid #818cf8; }

    /* Titles & Subheadings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1e293b;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Preprocessor
# ---------------------------------------------------------
@st.cache_resource
def load_prediction_model():
    try:
        model_path = os.path.join('final_model', 'model.pkl')
        preprocessor_path = os.path.join('final_model', 'preprocessor.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            st.error("⚠️ Model files not found in `final_model/`. Please ensure `model.pkl` and `preprocessor.pkl` exist.")
            st.stop()
            
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        return TimePredictionModel(preprocessor=preprocessor, model=model)
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e, sys)

final_model = load_prediction_model()

# ---------------------------------------------------------
# Header Banner (Neo Blue Cyber Theme)
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>⚡ Food Delivery Time Predictor</h1>
    <p>AI-Powered Logistics Engine • Cybernetic ETA Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Presets & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1526367790999-0150786686a2?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### ⚡ Quick Order Presets")
    st.caption("Select a scenario to pre-fill parameters:")
    
    preset = st.radio(
        "Choose Preset:",
        ["⚡ Quick Snack Run", "🌧️ Rainy Night Delivery", "🌇 Rush Hour Feast", "⚙️ Custom Order"],
        index=3
    )

    # Defaults based on preset selection
    if preset == "⚡ Quick Snack Run":
        def_distance, def_weather, def_traffic, def_tod, def_vehicle, def_prep = 2.5, "Clear", "Low", "Afternoon", "Bike", 10
    elif preset == "🌧️ Rainy Night Delivery":
        def_distance, def_weather, def_traffic, def_tod, def_vehicle, def_prep = 12.0, "Rainy", "Medium", "Night", "Car", 25
    elif preset == "🌇 Rush Hour Feast":
        def_distance, def_weather, def_traffic, def_tod, def_vehicle, def_prep = 8.5, "Clear", "High", "Evening", "Scooter", 20
    else:
        def_distance, def_weather, def_traffic, def_tod, def_vehicle, def_prep = 5.0, "Clear", "Medium", "Evening", "Scooter", 15

    st.markdown("---")
    st.markdown("### 🤖 Engine Specs")
    st.markdown("""
    - **Architecture:** CatBoost Regressor
    - **Transformation:** Pipeline Scaler + OHE
    - **Theme:** Neo Blue High-Contrast
    - **Status:** Active 🟢
    """)

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab_predict, tab_analytics, tab_info = st.tabs([
    "🔮 ETA Predictor",
    "📊 Neo Analytics",
    "ℹ️ Engine Architecture"
])

# ---------------------------------------------------------
# TAB 1: ETA Predictor
# ---------------------------------------------------------
with tab_predict:
    st.markdown("### 📝 Input Order Parameters")
    
    with st.container():
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("#### 📍 Route Parameters")
            distance = st.number_input(
                "Distance (km)",
                min_value=0.1, max_value=50.0,
                value=float(def_distance), step=0.5,
                help="Distance from kitchen to delivery destination"
            )
            
            traffic_options = ["Low", "Medium", "High"]
            traffic = st.selectbox(
                "Traffic Congestion",
                traffic_options,
                index=traffic_options.index(def_traffic) if def_traffic in traffic_options else 1,
                help="Current road traffic density"
            )

        with col_b:
            st.markdown("#### 🌤️ Environmental")
            weather_options = ["Clear", "Rainy", "Snowy", "Foggy", "Windy"]
            weather = st.selectbox(
                "Weather State",
                weather_options,
                index=weather_options.index(def_weather) if def_weather in weather_options else 0,
                help="Atmospheric condition during delivery"
            )
            
            tod_options = ["Morning", "Afternoon", "Evening", "Night"]
            time_of_day = st.selectbox(
                "Time Window",
                tod_options,
                index=tod_options.index(def_tod) if def_tod in tod_options else 2,
                help="Dispatch time window"
            )

        with col_c:
            st.markdown("#### 📦 Fleet & Preparation")
            preparation_time = st.number_input(
                "Prep Time (mins)",
                min_value=1, max_value=60,
                value=int(def_prep), step=1,
                help="Time needed for cooking and packaging"
            )
            
            vehicle_options = ["Bike", "Scooter", "Car"]
            vehicle = st.selectbox(
                "Vehicle Mode",
                vehicle_options,
                index=vehicle_options.index(def_vehicle) if def_vehicle in vehicle_options else 1,
                help="Courier vehicle type"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Trigger Button
    predict_btn = st.button("⚡ Compute Delivery ETA", type="primary", use_container_width=True)
    
    if predict_btn or preset != "⚙️ Custom Order":
        try:
            # Input DataFrame Construction
            input_df = pd.DataFrame([[
                distance,
                weather,
                traffic,
                time_of_day,
                vehicle,
                preparation_time
            ]], columns=[
                "Distance_km",
                "Weather",
                "Traffic_Level",
                "Time_of_Day",
                "Vehicle_Type",
                "Preparation_Time_min"
            ])
            
            # Predict ETA
            pred_raw = final_model.predict(input_df)
            eta_mins = max(1.0, float(pred_raw[0]))
            eta_rounded = round(eta_mins, 1)
            
            # Calculate arrival timestamp
            now = datetime.datetime.now()
            eta_timestamp = now + datetime.timedelta(minutes=eta_mins)
            formatted_eta_time = eta_timestamp.strftime("%I:%M %p")
            
            st.markdown("---")
            
            res_col1, res_col2 = st.columns([1, 1.2])
            
            with res_col1:
                # Speed Badge Category (Neo Cyan Theme)
                if eta_rounded < 25:
                    badge_html = '<span class="badge badge-fast">⚡ Neo Express Delivery</span>'
                elif eta_rounded <= 45:
                    badge_html = '<span class="badge badge-std">🚙 Standard Delivery Window</span>'
                else:
                    badge_html = '<span class="badge badge-slow">⚠️ High Latency Overhead</span>'
                    
                metric_card_html = f"""
                <div class="metric-box">
                    <div class="metric-label">Predicted Delivery ETA</div>
                    <div class="metric-value">{eta_rounded} <span style="font-size:1.5rem; color:#00f0ff;">mins</span></div>
                    <div>{badge_html}</div>
                    <p style="color:#cbd5e1; margin-top:14px; font-size:0.95rem;">📅 Estimated Arrival around <b style="color:#00f0ff;">{formatted_eta_time}</b></p>
                </div>
                """
                st.markdown(metric_card_html, unsafe_allow_html=True)

            with res_col2:
                # Time Component Bar Breakdown Chart (Neo Cyan Colors)
                base_prep = preparation_time
                est_transit = max(0.5, eta_rounded - base_prep)
                
                fig_breakdown = go.Figure(go.Bar(
                    x=[base_prep, est_transit],
                    y=['Kitchen Prep', 'Transit Time'],
                    orientation='h',
                    marker=dict(
                        color=['#00f0ff', '#3b82f6'],
                        line=dict(color='#0284c7', width=1.5)
                    ),
                    text=[f"{base_prep} mins", f"{round(est_transit, 1)} mins"],
                    textposition='auto'
                ))
                
                fig_breakdown.update_layout(
                    title="⏱️ Time Distribution Breakdown",
                    xaxis_title="Minutes",
                    height=220,
                    margin=dict(l=10, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#ffffff")
                )
                fig_breakdown.update_xaxes(showgrid=True, gridcolor="#1e293b")
                st.plotly_chart(fig_breakdown, use_container_width=True)

            # ---------------------------------------------------------
            # Vehicle Efficiency Comparison Matrix (Neo Cyan Styling)
            # ---------------------------------------------------------
            st.markdown("### 🏎️ Neo Fleet Vehicle Matrix")
            st.caption("Comparative ETA performance across vehicle options under current route conditions:")
            
            v_cols = st.columns(3)
            vehicle_types = ["Bike", "Scooter", "Car"]
            icons = {"Bike": "🚴", "Scooter": "🛵", "Car": "🚗"}
            
            for i, v_type in enumerate(vehicle_types):
                temp_df = input_df.copy()
                temp_df["Vehicle_Type"] = v_type
                v_pred = round(max(1.0, float(final_model.predict(temp_df)[0])), 1)
                
                diff = round(v_pred - eta_rounded, 1)
                diff_str = "Selected" if v_type == vehicle else f"{'+' if diff > 0 else ''}{diff} mins"
                
                with v_cols[i]:
                    is_current = (v_type == vehicle)
                    card_border = "#00f0ff" if is_current else "#1e293b"
                    bg_color = "rgba(0, 240, 255, 0.08)" if is_current else "#0b1120"
                    box_glow = "0 0 16px rgba(0, 240, 255, 0.25)" if is_current else "none"
                    
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; border: 2px solid {card_border}; border-radius: 14px; padding: 18px; text-align: center; box-shadow: {box_glow};">
                        <h3 style="margin:0; font-size:1.4rem; color:#ffffff;">{icons[v_type]} {v_type}</h3>
                        <h2 style="color: #00f0ff; margin: 10px 0 4px 0; text-shadow: 0 0 10px rgba(0,240,255,0.4);">{v_pred} <span style="font-size:1rem;">mins</span></h2>
                        <span style="font-size: 0.85rem; color: {'#00f0ff' if diff <= 0 else '#a5b4fc'}; font-weight:700;">{diff_str}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error predicting delivery time: {str(e)}")

# ---------------------------------------------------------
# TAB 2: Neo Analytics
# ---------------------------------------------------------
with tab_analytics:
    st.markdown("### 📊 Neo Analytics & Latency Curves")
    st.caption("Evaluating distance scaling and weather impacts")
    
    col_an1, col_an2 = st.columns(2)
    
    # Distance Curve Simulation
    dist_range = np.linspace(1, 30, 20)
    sim_data = []
    for d in dist_range:
        for v in ["Bike", "Scooter", "Car"]:
            test_row = pd.DataFrame([[d, "Clear", "Medium", "Evening", v, 15]], columns=[
                "Distance_km", "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type", "Preparation_Time_min"
            ])
            sim_pred = max(1.0, float(final_model.predict(test_row)[0]))
            sim_data.append({"Distance (km)": d, "Estimated Time (min)": sim_pred, "Vehicle": v})
            
    sim_df = pd.DataFrame(sim_data)
    
    with col_an1:
        fig_curve = px.line(
            sim_df, x="Distance (km)", y="Estimated Time (min)", color="Vehicle",
            title="📈 ETA Scaling by Distance (Neo Spectrum)",
            color_discrete_map={"Bike": "#00f0ff", "Scooter": "#38bdf8", "Car": "#818cf8"}
        )
        fig_curve.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#ffffff")
        )
        fig_curve.update_xaxes(showgrid=True, gridcolor="#1e293b")
        fig_curve.update_yaxes(showgrid=True, gridcolor="#1e293b")
        st.plotly_chart(fig_curve, use_container_width=True)

    # Weather Impact Bar Chart
    sim_weather_data = []
    for w in ["Clear", "Windy", "Foggy", "Rainy", "Snowy"]:
        test_row = pd.DataFrame([[8.0, w, "Medium", "Evening", "Scooter", 15]], columns=[
            "Distance_km", "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type", "Preparation_Time_min"
        ])
        w_pred = max(1.0, float(final_model.predict(test_row)[0]))
        sim_weather_data.append({"Weather": w, "Estimated Time (min)": round(w_pred, 1)})
        
    sim_w_df = pd.DataFrame(sim_weather_data)
    
    with col_an2:
        fig_w = px.bar(
            sim_w_df, x="Weather", y="Estimated Time (min)", color="Weather",
            title="🌦️ Weather Overhead (8km route)",
            text="Estimated Time (min)",
            color_discrete_sequence=["#00f0ff", "#38bdf8", "#60a5fa", "#818cf8", "#a5b4fc"]
        )
        fig_w.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#ffffff")
        )
        fig_w.update_yaxes(showgrid=True, gridcolor="#1e293b")
        st.plotly_chart(fig_w, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Engine Architecture
# ---------------------------------------------------------
with tab_info:
    st.markdown("### 🛠️ Machine Learning Model Specification")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        #### 📥 Input Feature Vector
        - **Distance_km** *(Numerical)*: Order travel distance in kilometers.
        - **Preparation_Time_min** *(Numerical)*: Kitchen cooking and packing time.
        - **Weather** *(Categorical)*: Clear, Rainy, Snowy, Foggy, Windy.
        - **Traffic_Level** *(Ordinal)*: Low < Medium < High.
        - **Time_of_Day** *(Ordinal)*: Morning, Afternoon, Evening, Night.
        - **Vehicle_Type** *(Categorical)*: Bike, Scooter, Car.
        """)
        
    with col_info2:
        st.markdown("""
        #### ⚙️ Data Preprocessing & Estimator
        - **Imputation:** `SimpleImputer` (Mean / Most Frequent)
        - **Scaling:** `StandardScaler` for continuous numerical features
        - **Categorical Encoders:** `OneHotEncoder` & `OrdinalEncoder`
        - **Machine Learning Regressor:** CatBoost / Gradient Boosted Trees
        """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer-text">
    ⚡ SpeedyBites Delivery Prediction Engine • High Contrast Cyber Matrix Edition
</div>
""", unsafe_allow_html=True)