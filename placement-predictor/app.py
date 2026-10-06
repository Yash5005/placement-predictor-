import joblib
import pandas as pd
import streamlit as st

st.title("🎓 Student Placement Predictor")
st.write("Enter your details to see the predicted chance of placement.")

model = joblib.load("model.pkl")

cgpa = st.slider("CGPA", 4.0, 10.0, 7.0, 0.1)
projects = st.number_input("Number of projects", 0, 10, 2)
internships = st.number_input("Number of internships", 0, 5, 1)
coding_score = st.slider("Coding test score", 0, 100, 60)
communication = st.slider("Communication skill (1-10)", 1, 10, 6)

if st.button("Predict"):
    data = pd.DataFrame([{
        "cgpa": cgpa,
        "projects": projects,
        "internships": internships,
        "coding_score": coding_score,
        "communication": communication,
    }])
    prob = model.predict_proba(data)[0][1]
    st.metric("Placement chance", f"{prob * 100:.1f}%")
    st.progress(float(prob))