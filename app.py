import streamlit as st
import pandas as pd
import joblib

model =joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_selector = joblib.load('feature_selector.pkl')

st.set_page_config(page_title="Sports man performance prediction",page_icon='⚽', layout="wide")

st.title("Player Performance Prediction App ⚽")
st.markdown("Enter the player's statistics to predict their performance score.")

with st.form("Performance predition form"):
    col1,col2 = st.columns(2)
    with col1:
        goals = st.number_input("goals",min_value=0,max_value=100)
        age = st.number_input("age",min_value=10,max_value=70)
        win_contribution = st.slider("Win contribution (%)",0,100,25)
        fitness = st.slider("Fitness level",0,10,3)
        diert_quality = st.slider("Diet quality",0,10,3)
        injury_history = st.slider("Injury history score",0,100,25)
        
    with col2:
        assist  = st.number_input("assists",min_value=0,max_value=100)
        played = st.number_input("matches played",min_value =0,max_value=100)
        accuracy = st.slider("Accuracy (%)",0,100,25)
        focus = st.slider("Mental focus score",0,10,3)
        teamwork = st.slider("Teamwork score",0,10,3)
        practice_hours = st.slider("Practice hours per week",0,20,10)
        submitted = st.form_submit_button("Predict Performance Score")
    if submitted:
        input_data = pd.DataFrame({
            'Age': [age],
            'Matches_Played':[played],
            'Goals_Scored': [goals],
            'Accuracy': [accuracy],
            'Win_Contribution':[win_contribution],
            'Diet_Quality':[diert_quality],
            'Practice_Hours':[practice_hours],
            'Fitness_Level':[fitness],
            'Teamwork_Score':[teamwork],
            'Mental_Focus_Score':[focus],
            'Injury_Days':[injury_history],


        })
        scaled_input = scaler.transform(input_data)
        input= feature_selector.transform(scaled_input)
        prediction = model.predict(input)
        st.success(f"Predicted performance score:{prediction[0]:.2f}")

       