import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Cutoff Score Predictor", layout="centered")

st.title("🎓 Cutoff Score Predictor App")
st.write("Enter details below to estimate cutoff values based on our ML pipeline.")

# Load the trained model pipeline
@st.cache_resource
def load_model():
    with open("best_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

try:
    pipeline = load_model()
    
    # Dynamic input options from trained feature dataset
    university = st.selectbox("University", ['Banaras Hindu University', 'Guru Gobind Singh Indraprastha University', 'University of Delhi'])
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
    programme = st.text_input("Programme (e.g. B.Tech, MBA)", "B.Tech")
    variant = st.text_input("Variant (e.g. Regular, Self-Financed)", "Regular")
    round_num = st.selectbox("Admission Round", [1, 2, 3, 4, 5])
    category = st.selectbox("Category", ['UR', 'EWS', 'OBC-NCL', 'SC', 'ST', 'PwBD'])
    seat_type = st.text_input("Seat Type", "All-India")
    figure_kind = st.selectbox("Figure Kind", ['score', 'rank'])
    basis = st.text_input("Basis", "CUET")

    if st.button("Predict Cutoff"):
        input_df = pd.DataFrame([{
            'university': university,
            'year': year,
            'programme': programme,
            'variant': variant,
            'round': round_num,
            'category': category,
            'seat_type': seat_type,
            'figure_kind': figure_kind,
            'basis': basis
        }])
        
        # Standardize columns to match training feature set
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                input_df[col] = input_df[col].astype(str).str.strip()
                
        prediction = pipeline.predict(input_df)[0]
        st.success(f"🎯 Predicted Cutoff Value: {prediction:.2f}")

except Exception as e:
    st.error(f"Error loading the application/model: {e}")