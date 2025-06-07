import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load data
data = pd.read_csv('Heart Prediction Quantum Dataset.csv')

# Define disease level assignment function
def assign_disease_level(row):
    if row['HeartDisease'] == 0:
        return 0  # No disease
    high_risk_age = 55 if row['Gender'] == 1 else 65
    risk_factors = 0
    if row['BloodPressure'] >= 160: risk_factors += 2
    elif row['BloodPressure'] >= 140: risk_factors += 1
    if row['Cholesterol'] >= 240: risk_factors += 1
    if row['Age'] >= high_risk_age: risk_factors += 1
    if risk_factors <= 1: return 1  # Mild
    elif risk_factors == 2: return 2  # Moderate
    else: return 3  # Severe

data['DiseaseLevel'] = data.apply(assign_disease_level, axis=1)

# Feature engineering
data['BP_Cholesterol'] = data['BloodPressure'] * data['Cholesterol']
data['Age_BP'] = data['Age'] * data['BloodPressure']

# Features and target
X = data.drop(['HeartDisease', 'DiseaseLevel'], axis=1)
y = data['DiseaseLevel']

# Feature selection
selector = SelectKBest(score_func=f_classif, k=7)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Balance data with SMOTE
target_samples = 600
smote = SMOTE(random_state=42, sampling_strategy={0: target_samples, 1: target_samples, 2: target_samples, 3: target_samples})
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Base models with reduced n_estimators for faster training
base_models = [
    ('lgbm', LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbose=-1)),
    ('catboost', CatBoostClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbose=0))
]

# Meta model
meta_model = LogisticRegression(max_iter=1000, C=10.0)

# Stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3, passthrough=True)
stacking_clf.fit(X_train, y_train)

# Save model and preprocessing objects
joblib.dump(stacking_clf, 'stacking_model.pkl')
joblib.dump(selector, 'feature_selector.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

print("Model and preprocessing objects saved.")
