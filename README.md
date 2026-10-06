# placement-predictor-
ML web app that predicts a student's placement chance from CGPA, projects, internships, coding score and communication skill. Built with Python, scikit-learn and Streamlit.
# 🎓 Student Placement Predictor

A machine learning web app that predicts a student's chance of getting placed,
based on CGPA, number of projects, internships, coding test score and
communication skill.

## Tech Stack
Python, pandas, scikit-learn, Streamlit

## How it works
1. Data is loaded and split into training and test sets.
2. Logistic Regression and Random Forest are trained and compared by accuracy.
3. The best model is saved with joblib.
4. A Streamlit app loads the model and shows the placement probability.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Note
The dataset is synthetic, created for learning and demonstration.

## Live Demo
(add your Streamlit link here)
