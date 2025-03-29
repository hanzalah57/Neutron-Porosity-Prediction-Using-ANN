import streamlit as st
import numpy as np 
import pandas as pd
import pickle
import tensorflow as tf


model = tf.keras.models.load_model('NEUT_model.keras')

with open('Scaler_NEUT.pkl','rb') as file:
    Scaler = pickle.load(file)

st.title("Neutron Porosity Predction Using Artificial-Neural-Network")

df = pd.read_csv('processed_logs.csv')

st.sidebar.header("🔍 Input Features")
SP = st.sidebar.slider("SP(mV)", float(df['SP'].min()), float(df['SP'].max()), key="SP(mV)")
GR = st.sidebar.slider("GR(API)", float(df['GR'].min()), float(df['GR'].max()), key="GR")
DTC = st.sidebar.slider("DTC(us/ft)", float(df['DTC'].min()), float(df['DTC'].max()), key="DTC")
DENS = st.sidebar.slider("DENS(g/cc)", float(df['DENS'].min()), float(df['DENS'].max()), key="DENS")
DENC= st.sidebar.slider("DENC(g/cc)", float(df['DENC'].min()), float(df['DENC'].max()), key="DENC")

input_data = np.array([[SP, GR, DTC, DENS, DENC]])
input_data_scaled = Scaler.transform(input_data)  

NEUT_Pred = model.predict(input_data_scaled)[0]

 # Display Prediction
st.markdown(
f"""
<div style="
    padding: 20px; 
    border-radius: 12px; 
    background: linear-gradient(135deg, #3A7BD5, #00D2FF);
    color: white; 
    text-align: center; 
    font-size: 27px; 
    font-weight: bold;
    box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);
    border: 2px solid #A5F3FC;
    max-width: 500px;
    margin: auto;
">
    <span style="font-size: 32px; font-weight: bold;">{float(NEUT_Pred):.4f} v/v</span>
</div>
""",
unsafe_allow_html=True
)

st.write("### Data Statistics")
st.dataframe(df.describe())


st.sidebar.info("""This app predicts Neutron Porosity on the basis of other well log readings
""")
st.sidebar.markdown("---")  
st.sidebar.markdown("👤 **Author: Mr. Hanzalah Bin Sohail**  \n🌏*Geophysicist*")  
