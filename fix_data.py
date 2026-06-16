import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import joblib

print("Loading data...")
df = pd.read_csv('dataset/netflix_user_behavior_dataset.csv')

# 1. Inject signal before splitting (on full dataset)
print("Injecting business logic labels...")

# Reset to neutral first to avoid partial overwrtites
df['churned'] = 'No'

mask_high = (
    (df['days_since_last_login'] > 20) & (df['completion_rate'] < 40)
) | (
    (df['avg_watch_time_minutes'] < 60) & (df['watch_sessions_per_week'] < 3)
) | (
    (df['account_age_months'] < 24) & (df['rating_given'] < 2.0)
)
df.loc[mask_high, 'churned'] = 'Yes'

# Low risk override
mask_low = (
    (df['days_since_last_login'] < 7) & 
    (df['avg_watch_time_minutes'] > 150) &
    (df['completion_rate'] > 60)
)
df.loc[mask_low, 'churned'] = 'No'

print("Adding label noise to simulate real-world messiness...")
rng = pd.Series(range(len(df)), index=df.index)
noise_idx = df.sample(frac=0.15, random_state=42).index
df.loc[noise_idx, 'churned'] = df.loc[noise_idx, 'churned'].map({'Yes': 'No', 'No': 'Yes'})

print(f"Churn distribution after noise:\n{df['churned'].value_counts()}\n")

# 2. Encode categoricals
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

# 3. Stratified split to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE inside pipeline (prevents leakage into CV folds)
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ))
])

# 6. Cross-validate before final fit
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ∓ {cv_scores.std():.4f}")

# 7. Fit on full traininf data, evaluate on unseen test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(f"\nTest ROC_AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred))

# 8. Save model
joblib.dump(pipeline, 'churn_model_balanced.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Model and encoders saved successfully.")