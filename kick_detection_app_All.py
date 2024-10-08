import streamlit as st
import joblib 
import pandas as pd
import xgboost

# Load pre-trained scaler and model
scaler = joblib.load("models/scaler_All.h5")
Model = joblib.load("models/model_All.h5")

# Define input features
Inputs=['TVD(ft)', 'BITSIZE(in)', 'Porosity(%)',
        'Corrected Bulk Density(gm/cc)', 'Deep Resistivity (Ohm)', 'ROP(M/hr)',
        'WOB(KLb)', 'RPM', 'Torque(lb.F)', 'Stand Pipe Pressure(Psi)',
        'Flow In(GPM)','Temp - Out', 'Total Gas(PPM)']

# Function to make prediction
def predict(TVD_FT,BITSIZE,Porosity,Corrected_Bulk_Density,Deep_Resistivity ,ROP,WOB,RPM,Torque,Stand_Pipe_Pressure,
                  Flow_In,Temp_Out,Total_Gas):
    test_df = pd.DataFrame(columns = Inputs,index=[0])
    test_df.at[0,"TVD(ft)"] = TVD_FT
    test_df.at[0,"BITSIZE(in)"] = BITSIZE
    test_df.at[0,"Porosity(%)"] = Porosity
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

# Function to style header
def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)    

# Main function to define the app layout and functionality
def main(): 

    page_element="""
        <style>
        [data-testid="stAppViewContainer"]{
          background-image: url("https://pluspng.com/img-png/oil-rig-png-hd--2880.jpg");
          background-size: cover;
        }
        </style>
        """
    st.markdown(page_element, unsafe_allow_html=True) 
        
    # Front-end elements of the web page 
    st.markdown("""
    <style>
        body {
            background-image: url('https://pluspng.com/img-png/oil-rig-png-hd--2880.jpg');
            background-size: cover;
        }
        .header {
            background-color: rgba(0, 0, 0, 0.5);
            padding: 20px;
            color: white;
            text-align: center;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        .container {
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.8); /* Adjust the opacity as needed */
            border-radius: 10px;
        }
        .card {
            background-color: rgba(51, 170, 51, 0.1); /* Adjust the color and opacity as needed */
            padding: 20px;
            border-radius: 10px;
            color: black; /* Adjust the text color as needed */
        }
    </style>
    <div class="header">
        <h1>Kick Detection System ML Prediction App</h1>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Sidebar for user input
    st.sidebar.title("Choose your Features")
    TVD_FT = st.sidebar.number_input('TVD(ft)')
    BITSIZE = st.sidebar.number_input('BITSIZE(in)')
    Porosity = st.sidebar.number_input('Porosity(%)')
    Corrected_Bulk_Density = st.sidebar.number_input('Corrected Bulk Density(gm/cc)')
    Deep_Resistivity = st.sidebar.number_input('Deep Resistivity (Ohm)')
    ROP = st.sidebar.number_input('ROP(M/hr)')
    WOB = st.sidebar.number_input('WOB(KLb)')
    RPM = st.sidebar.number_input('RPM')
    Torque = st.sidebar.number_input('Torque(lb.F)')
    Stand_Pipe_Pressure = st.sidebar.number_input('Stand Pipe Pressure(Psi)')
    Flow_In = st.sidebar.number_input('Flow In(GPM)')
    Temp_Out = st.sidebar.number_input('Temp - Out')
    Total_Gas = st.sidebar.number_input('Total Gas(PPM)')
    Mud_Density= st.sidebar.number_input('Mud Density(PPG)')   
    
    # Prediction button
    if st.sidebar.button("Predict"): 
        result = predict(TVD_FT,BITSIZE,Porosity,Corrected_Bulk_Density,Deep_Resistivity ,ROP,WOB,RPM,Torque,Stand_Pipe_Pressure,
                  Flow_In,Temp_Out,Total_Gas)

        # clculate Hydrostatic pressure
        PH=.052* TVD_FT * Mud_Density

        # Determine kick detection
        if PH > result:
                Pred = "No Kick detected.. Keep going"
                color = "green"
        else:
                Pred = "Kick! Take necessary precautions"
                color = "red"
        
        # Display result
        result_display = f"""
        <div style="background-color: rgba(6,71,180,0.3); padding:2px; border-radius:10px;">
            <h1 style="color:black; font-size:40px; text-align:center;">Formation Pressure: <span style="color:blue;">{result}</span></h1>
            <h1 style="color:black; font-size:40px; text-align:center;">Hydrostatic Pressure: {PH}</h1>
            <h1 style="color:white; font-size:40px; text-align:center;"><span style="color:{color};">{Pred}</span></h1>
        </div>
        """
        st.markdown(result_display, unsafe_allow_html=True)
        

# Show image
    #st.image('R.jfif', use_column_width=True) 
    
if __name__=='__main__': 
    main()
