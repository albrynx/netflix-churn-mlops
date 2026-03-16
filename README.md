# 🎬 Netflix User Churn Prediction & MLOps Deployment

This project demonstrates an end-to-end Machine Learning pipeline—from data engineering and model training to a production-ready containerized deployment. It predicts whether a user is likely to "churn" (cancel their subscription) based on behavioral metrics like watch time, engagement, and account age.

![Low probability of churn](images/low-prob.png)

**Career Goal Focus:** This project showcases **AI Engineering** and **MLOps** skills by moving beyond a local Jupyter Notebook into robust environment management, synthetic data handling, and Dockerized deployment.

[Image of Machine Learning MLOps pipeline architecture]

## 🚀 Project Overview
* **Problem:** High churn rates impact revenue in subscription-based streaming services.
* **Solution:** A Random Forest classifier that identifies high-risk users, deployed via an interactive web dashboard.
* **The "AI Engineering" Edge:** * **Signal Injection:** Modified the synthetic dataset to inject realistic business logic (e.g., users with low completion rates and high days since last login have a higher churn probability).
  * **Environment Parity:** Strictly pinned `scikit-learn==1.8.0` and `python:3.11` across local and Docker environments to prevent unpickling and vocabulary mismatch errors in production.
  * **Containerization:** Fully Dockerized the application for seamless cross-platform deployment.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Machine Learning:** Scikit-Learn 1.8.0, Pandas, Joblib
* **Web Framework:** Streamlit
* **DevOps:** Docker

## 📂 Project Structure
```
├── app.py                          # Streamlit Dashboard UI and prediction logic
├── fix_data.py                     # Data engineering script to inject realistic churn signals
├── churn_model_balanced.pkl        # Trained Random Forest model
├── label_encoders.pkl              # Encoders for categorical features (Country, Subscription, etc.)
├── Dockerfile                      # Containerization instructions
├── requirements.txt                # Strictly pinned project dependencies
└── dataset/
    └── netflix_user_behavior.csv   # Raw dataset
└── images                          # Screenshot of Streamlit app

```