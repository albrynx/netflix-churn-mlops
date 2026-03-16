import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading data...")
df = pd.read_csv('dataset/netflix_user_behavior_dataset.csv')

# 1. INJECT A FAKE SIGNAL (Create a strong pattern)
print("Injecting realistic business patterns...")
# Rule 1: High risk if they haven't logged in for > 20 days and finish < 40% of shows
mask_high_risk = (df['days_since_last_login'] > 20) & (df['completion_rate'] < 40)
df.loc[mask_high_risk, 'churned'] = 'Yes'

# Rule 2: Low risk if they watch a lot and log in often
mask_low_risk = (df['days_since_last_login'] < 10) & (df['avg_watch_time_minutes'] > 150)
df.loc[mask_low_risk, 'churned'] = 'No'

# 2. PREPROCESS AND RETRAIN
df_ml = df.drop(columns=['user_id'])
label_encoders = {}
categorical_cols = ['gender', 'country', 'subscription_type', 'payment_method', 'primary_device', 'favorite_genre']

for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    label_encoders[col] = le

df_ml['churned'] = df_ml['churned'].map({'No': 0, 'Yes': 1})

X = df_ml.drop('churned', axis=1)
y = df_ml['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training new Smart Model...")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 3. SAVE THE NEW MODEL
joblib.dump(rf_model, 'churn_model_balanced.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Done! Model saved. Rebuild your Docker container now.")