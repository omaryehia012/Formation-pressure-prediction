import streamlit as st
import joblib 
import pandas as pd
import xgboost
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Kick Detection System - ML Prediction",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load pre-trained scaler and model
@st.cache_resource
def load_model():
        scaler = joblib.load("models/scaler_All.h5")
        model = joblib.load("models/model_All.h5")
        return scaler, model

scaler, Model = load_model()

# Define input features
Inputs=['TVD(ft)', 'BITSIZE(in)', 'NPHI(%)',
        'Corrected Bulk Density(gm/cc)', 'Deep Resistivity (Ohm)', 'ROP(M/hr)',
        'WOB(KLb)', 'RPM', 'Torque(lb.F)', 'Stand Pipe Pressure(Psi)',
        'Flow In(GPM)','Temp - Out', 'Total Gas(PPM)']

# Function to make prediction
def predict(TVD_FT,BITSIZE,NPHI,Corrected_Bulk_Density,Deep_Resistivity ,ROP,WOB,RPM,Torque,Stand_Pipe_Pressure,
                  Flow_In,Temp_Out,Total_Gas):
    test_df = pd.DataFrame(columns = Inputs,index=[0])
    test_df.at[0,"TVD(ft)"] = TVD_FT
    test_df.at[0,"BITSIZE(in)"] = BITSIZE
    test_df.at[0,"NPHI(%)"] = NPHI
    test_df.at[0,"Corrected Bulk Density(gm/cc)"] = Corrected_Bulk_Density
    test_df.at[0,"Deep Resistivity (Ohm)"] = Deep_Resistivity
    test_df.at[0,"ROP(M/hr)"] = ROP
    test_df.at[0,"WOB(KLb)"] = WOB
    test_df.at[0,"RPM"] = RPM
    test_df.at[0,"Torque(lb.F)"] = Torque
    test_df.at[0,"Stand Pipe Pressure(Psi)"] = Stand_Pipe_Pressure
    test_df.at[0,"Flow In(GPM)"] = Flow_In
    test_df.at[0,"Temp - Out"] = Temp_Out
    test_df.at[0,"Total Gas(PPM)"] = Total_Gas

    result = Model.predict(scaler.transform(test_df))[0]
    return result

# Custom CSS for enhanced styling
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Set background image for the entire app */
    .stApp {
        background-image: url('R.jfif');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    
    /* Force background image on main container */
    .main .block-container {
        background-image: url('R.jfif');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    
    /* Additional background enforcement */
    .stApp > div:first-child {
        background-image: url('R.jfif') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }
    
    /* Body background enforcement */
    body {
        background-image: url('R.jfif') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(25, 50, 100, 0.95) 0%, rgba(35, 70, 130, 0.95) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        border: 2px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(15px);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        margin: 0;
        text-align: center;
        text-shadow: 0 4px 8px rgba(0,0,0,0.6);
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 35px rgba(0,0,0,0.4);
        border: 2px solid rgba(255,255,255,0.25);
        margin: 1.5rem 0;
        backdrop-filter: blur(15px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(0,0,0,0.5);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.95;
        margin: 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.95) 0%, rgba(245, 87, 108, 0.95) 100%);
        padding: 2.5rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0,0,0,0.4);
        border: 2px solid rgba(255,255,255,0.3);
        margin: 2.5rem 0;
        backdrop-filter: blur(15px);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .result-card h2 {
        font-size: 2.2rem;
        margin: 0 0 1.5rem 0;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    
    .result-card h3 {
        font-size: 1.6rem;
        margin: 0.5rem 0;
        font-weight: 500;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95) 0%, rgba(52, 73, 94, 0.95) 100%);
        padding: 2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .sidebar-header h3 {
        margin: 0;
        font-weight: 600;
        font-size: 1.4rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    
    .parameter-section {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.8rem;
        border-radius: 18px;
        margin: 1.8rem 0;
        border: 2px solid rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(15px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .parameter-section:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .parameter-section h4 {
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0 0 1.2rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-align: center;
        justify-content: center;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar input containers */
    .stNumberInput > div > div > input {
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.4);
        padding: 1rem;
        font-size: 1.1rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.3rem rgba(102, 126, 234, 0.3);
        background: rgba(255, 255, 255, 1);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: rgba(255, 255, 255, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 5px 18px rgba(0,0,0,0.15);
    }
    
    /* Sidebar button styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        color: white;
        border: none;
        border-radius: 35px;
        padding: 1.2rem 2.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255,255,255,0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 2rem 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.5);
        background: linear-gradient(135deg, rgba(118, 75, 162, 0.95) 0%, rgba(102, 126, 234, 0.95) 100%);
    }
    
    /* Sidebar labels styling */
    .stNumberInput > div > div > label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5) !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sidebar help text styling */
    .stNumberInput > div > div > div[data-testid="stFormSubmitButton"] {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.9rem !important;
        font-style: italic !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.95) 0%, rgba(0, 242, 254, 0.95) 100%);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 10px 35px rgba(0,0,0,0.4);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255,255,255,0.25);
    }
    
    .info-box h4 {
        margin: 0 0 1rem 0;
        font-weight: 600;
        font-size: 1.3rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    
    .info-box p {
        margin: 0.5rem 0;
        opacity: 0.95;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    /* Enhanced text readability */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: rgba(255, 255, 255, 0.95);
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    .stMarkdown strong {
        color: rgba(255, 255, 255, 0.95);
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Dataframe styling for better visibility */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .footer {
        text-align: center;
        padding: 2.5rem;
        color: rgba(255, 255, 255, 0.95);
        font-size: 1rem;
        margin-top: 4rem;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 20px;
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(241, 241, 241, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.8);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(90, 111, 216, 0.9);
    }
    
    /* Plotly chart container styling */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 18px;
        padding: 1.5rem;
        backdrop-filter: blur(15px);
        box-shadow: 0 10px 35px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    /* Parameter labels styling */
    .parameter-label {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Help text styling */
    .help-text {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        font-style: italic;
        margin-top: 0.25rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# Function to create pressure comparison chart
def create_pressure_chart(formation_pressure, hydrostatic_pressure):
    fig = go.Figure()
    
    # Add formation pressure bar
    fig.add_trace(go.Bar(
        name='Formation Pressure',
        x=['Formation Pressure'],
        y=[formation_pressure],
        marker_color='#e74c3c',
        text=[f'{formation_pressure:.2f}'],
        textposition='auto',
    ))
    
    # Add hydrostatic pressure bar
    fig.add_trace(go.Bar(
        name='Hydrostatic Pressure',
        x=['Hydrostatic Pressure'],
        y=[hydrostatic_pressure],
        marker_color='#3498db',
        text=[f'{hydrostatic_pressure:.2f}'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Pressure Comparison Analysis',
        xaxis_title='Pressure Type',
        yaxis_title='Pressure (PSI)',
        template='plotly_white',
        height=400,
        showlegend=True,
        font=dict(family="Inter", size=12)
    )
    
    return fig

# Main function to define the app layout and functionality
def main():
    load_css()
    
    # Add background image container with enhanced styling
    st.markdown("""
    <div style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('R.jfif');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        z-index: -1;
        opacity: 0.85;
    "></div>
    """, unsafe_allow_html=True)
    
    # Check if model loaded successfully
    if scaler is None or Model is None:
        st.error("## 🚨 Application Error")
        st.error("The machine learning model could not be loaded. Please check the error message above and ensure all dependencies are properly installed.")
        st.stop()
    
    # Header section
    st.markdown("""
    <div class="main-header">
        <h1>🛢️ Kick Detection System</h1>
        <p>Advanced Machine Learning Prediction for Formation Pressure Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for all input parameters
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>📊 Input Parameters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Well Parameters Section
        st.markdown("""
        <div class="parameter-section">
            <h4>📍 Well Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        TVD_FT = st.number_input('TVD (ft)', min_value=0.0, value=5000.0, step=100.0, help="True Vertical Depth in feet")
        BITSIZE = st.number_input('Bit Size (in)', min_value=0.0, value=8.5, step=0.1, help="Drill bit size in inches")
        NPHI = st.number_input('NPHI (%)', min_value=0.0, max_value=100.0, value=15.0, step=0.1, help="Neutron Porosity percentage")
        
        # Formation Properties Section
        st.markdown("""
        <div class="parameter-section">
            <h4>🔬 Formation Properties</h4>
        </div>
        """, unsafe_allow_html=True)
        Corrected_Bulk_Density = st.number_input('Corrected Bulk Density (gm/cc)', min_value=0.0, value=2.65, step=0.01, help="Corrected bulk density in grams per cubic centimeter")
        Deep_Resistivity = st.number_input('Deep Resistivity (Ohm)', min_value=0.0, value=50.0, step=0.1, help="Deep resistivity in Ohms")
        
        # Drilling Parameters Section
        st.markdown("""
        <div class="parameter-section">
            <h4>⚙️ Drilling Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        ROP = st.number_input('ROP (M/hr)', min_value=0.0, value=30.0, step=1.0, help="Rate of Penetration in meters per hour")
        WOB = st.number_input('WOB (KLb)', min_value=0.0, value=20.0, step=1.0, help="Weight on Bit in Kilo-pounds")
        RPM = st.number_input('RPM', min_value=0, value=120, step=10, help="Rotations per minute")
        Torque = st.number_input('Torque (lb.F)', min_value=0.0, value=2000.0, step=100.0, help="Torque in pound-feet")
        
        # Mud & Flow Parameters Section
        st.markdown("""
        <div class="parameter-section">
            <h4>🌊 Mud & Flow Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        Stand_Pipe_Pressure = st.number_input('Stand Pipe Pressure (PSI)', min_value=0.0, value=2500.0, step=100.0, help="Stand pipe pressure in PSI")
        Flow_In = st.number_input('Flow In (GPM)', min_value=0.0, value=800.0, step=50.0, help="Flow rate in gallons per minute")
        Temp_Out = st.number_input('Temperature Out (°F)', min_value=0.0, value=150.0, step=5.0, help="Temperature out in Fahrenheit")
        Total_Gas = st.number_input('Total Gas (PPM)', min_value=0.0, value=100.0, step=10.0, help="Total gas concentration in parts per million")
        Mud_Density = st.number_input('Mud Density (PPG)', min_value=0.0, value=10.0, step=0.1, help="Mud density in pounds per gallon")
        
        # Prediction button
        if st.button("🚀 Predict Formation Pressure", use_container_width=True):
            st.session_state.prediction_made = True
            st.session_state.result = predict(TVD_FT, BITSIZE, NPHI, Corrected_Bulk_Density, 
                                           Deep_Resistivity, ROP, WOB, RPM, Torque, Stand_Pipe_Pressure,
                                           Flow_In, Temp_Out, Total_Gas)
            st.session_state.hydrostatic_pressure = 0.052 * TVD_FT * Mud_Density
            st.session_state.inputs = {
                'TVD': TVD_FT, 'Bit Size': BITSIZE, 'NPHI': NPHI,
                'Bulk Density': Corrected_Bulk_Density, 'Resistivity': Deep_Resistivity,
                'ROP': ROP, 'WOB': WOB, 'RPM': RPM, 'Torque': Torque,
                'Stand Pipe Pressure': Stand_Pipe_Pressure, 'Flow In': Flow_In,
                'Temp Out': Temp_Out, 'Total Gas': Total_Gas, 'Mud Density': Mud_Density
            }
        
        # Add some spacing at the bottom of sidebar
        st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Main content area (full width now)
    if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
        # Display results
        formation_pressure = st.session_state.result
        hydrostatic_pressure = st.session_state.hydrostatic_pressure
        
        # Determine kick detection
        if hydrostatic_pressure > formation_pressure:
            pred = "✅ No Kick Detected - Safe to Continue"
            color = "#27ae60"
            icon = "🟢"
        else:
            pred = "⚠️ KICK DETECTED - Take Immediate Action!"
            color = "#e74c3c"
            icon = "🔴"
        
        # Results display
        st.markdown(f"""
        <div class="result-card">
            <h2>{icon} Kick Detection Result</h2>
            <h3 style="color: {color};">{pred}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Pressure metrics in a row
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Formation Pressure</div>
                <div class="metric-value">{formation_pressure:.2f}</div>
                <div class="metric-label">PSI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Hydrostatic Pressure</div>
                <div class="metric-value">{hydrostatic_pressure:.2f}</div>
                <div class="metric-label">PSI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            safety_margin = abs(hydrostatic_pressure - formation_pressure)
            safety_percentage = (safety_margin / formation_pressure) * 100 if formation_pressure > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Safety Margin</div>
                <div class="metric-value">{safety_margin:.2f}</div>
                <div class="metric-label">PSI</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create two columns for charts and analysis
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Pressure comparison chart
            st.markdown("### 📊 Pressure Analysis Chart")
            st.plotly_chart(create_pressure_chart(formation_pressure, hydrostatic_pressure), use_container_width=True)
        
        with col2:
            # Safety Analysis
            if hydrostatic_pressure > formation_pressure:
                safety_level = "🟢 SAFE"
                safety_color = "#27ae60"
                recommendation = "Maintain current drilling parameters - formation is well controlled"
            else:
                safety_level = "🔴 CRITICAL"
                safety_color = "#e74c3c"
                recommendation = "IMMEDIATE ACTION REQUIRED - Increase mud weight or reduce TVD"
            
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Safety Analysis</h4>
                <div style="text-align: center; margin: 1rem 0;">
                    <h3 style="color: {safety_color}; margin: 0;">{safety_level}</h3>
                </div>
                <p><strong>🔍 Safety Margin:</strong> <span style="color: {safety_color}; font-weight: 600;">{safety_margin:.2f} PSI</span></p>
                <p><strong>📈 Safety Percentage:</strong> <span style="color: {safety_color}; font-weight: 600;">{safety_percentage:.1f}%</span></p>
                <p><strong>💡 Recommendation:</strong> {recommendation}</p>
                <p><strong>⚠️ Risk Level:</strong> {'Low' if hydrostatic_pressure > formation_pressure else 'High - Immediate intervention required'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Input summary in full width with enhanced styling
        st.markdown("### 📋 Input Parameters Summary")
        inputs_df = pd.DataFrame(list(st.session_state.inputs.items()), columns=['Parameter', 'Value'])
        st.dataframe(inputs_df, use_container_width=True, hide_index=True)
        
    else:
        # Welcome message and features in full width
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🎯 Welcome to the Kick Detection System</h4>
                <p>This advanced machine learning application predicts formation pressure based on drilling parameters and geological data. 
                Enter your parameters in the sidebar and click predict to analyze kick potential.</p>
                <p><strong>Perfect for:</strong> Petroleum Engineers, Drilling Operations, Research & Development, Academic Studies</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🔬 Key Features & Capabilities</h4>
                <p>🚀 <strong>Real-time Formation Pressure Prediction</strong> - Instant ML-powered analysis</p>
                <p>🤖 <strong>Advanced XGBoost Algorithms</strong> - State-of-the-art machine learning</p>
                <p>🛡️ <strong>Comprehensive Safety Analysis</strong> - Risk assessment and recommendations</p>
                <p>📊 <strong>Professional Visualization</strong> - Interactive charts and metrics</p>
                <p>⚡ <strong>High-Performance Processing</strong> - Optimized for real-world applications</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional information in full width
        st.markdown("""
        <div class="info-box">
            <h4>📚 How to Use</h4>
            <p>1. <strong>Input Parameters:</strong> Enter your drilling and formation parameters in the sidebar</p>
            <p>2. <strong>Click Predict:</strong> Use the prediction button to analyze kick potential</p>
            <p>3. <strong>Review Results:</strong> Analyze the safety assessment and recommendations</p>
            <p>4. <strong>Take Action:</strong> Follow the safety recommendations based on the analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical specifications
        st.markdown("""
        <div class="info-box">
            <h4>⚙️ Technical Specifications</h4>
            <p><strong>ML Model:</strong> XGBoost with advanced feature engineering</p>
            <p><strong>Input Features:</strong> 13 comprehensive drilling and formation parameters</p>
            <p><strong>Output:</strong> Formation pressure prediction with safety analysis</p>
            <p><strong>Accuracy:</strong> Industry-standard machine learning performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h4 style="margin: 0 0 1rem 0; color: rgba(255, 255, 255, 0.95);">🛢️ Kick Detection System</h4>
        <p style="margin: 0.5rem 0; font-weight: 500;">Advanced Machine Learning Prediction Platform</p>
        <p style="margin: 0.5rem 0; opacity: 0.9;">Master's Research Project | Petroleum Engineering</p>
        <p style="margin: 1rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">Built with Streamlit, XGBoost, and Professional Engineering Standards</p>
        <p style="margin: 0.5rem 0; font-size: 0.8rem; opacity: 0.7;">© 2024 - Advanced Formation Pressure Analysis System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__=='__main__':
    main()
