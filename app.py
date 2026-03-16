import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders
model = joblib.load('churn_model_balanced.pkl')
encoders = joblib.load('label_encoders.pkl')

st.title("Netflix User Churn Predictor 🎬")
st.write("Adjust user behavior metrics to see the probability of churn.")

# Create input widgets in a sidebar
st.sidebar.header("User Behavior Metrics")
age = st.sidebar.slider("Age", 18, 80, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
country = st.sidebar.selectbox("Country", ["USA", "India", "UK", "Canada", "Germany", "Brazil", "France", "Australia", "Spain", "Japan"])
sub_type = st.sidebar.selectbox("Subscription", ["Basic", "Standard", "Premium"])
watch_time = st.sidebar.number_input("Avg Watch Time (min/day)", 0, 500, 120)
completion = st.sidebar.slider("Completion Rate (%)", 0, 100, 50)
last_login = st.sidebar.slider("Days Since Last Login", 0, 90, 5)

# Build the feature dictionary (include all columns used in training)
input_data = {
    'age': age, 'gender': gender, 'country': country,
    'account_age_months': 12, 'subscription_type': sub_type, 
    'monthly_fee': 12.99, 'payment_method': 'PayPal', 
    'primary_device': 'Laptop', 'devices_used': 2, 
    'favorite_genre': 'Sci-Fi', 'avg_watch_time_minutes': watch_time,
    'watch_sessions_per_week': 10, 'binge_watch_sessions': 5,
    'completion_rate': completion, 'rating_given': 4.0,
    'content_interactions': 20, 'recommendation_click_rate': 50,
    'days_since_last_login': last_login
}

# Process inputs
input_df = pd.DataFrame([input_data])
for col, le in encoders.items():
    input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict Churn Status"):
    prob = model.predict_proba(input_df)[0][1]
    st.subheader(f"Churn Probability: {prob:.2%}")
    
    if prob > 0.5:
        st.error("Warning: High Risk of Churn!")
    else:
        st.success("Safe: This user is likely to stay.")