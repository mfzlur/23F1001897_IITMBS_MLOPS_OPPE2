import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    # Basic cleaning based on provided snippet
    df['target'] = df['target'].map({'yes': 1, 'no': 0})
    
    # Feature Engineering / Definition
    categorical_features = ['gender', 'cp', 'restecg', 'slope', 'thal']
    numeric_features = ['age', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca']
    
    # Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return df, preprocessor

def train_and_audit():
    df, preprocessor = load_and_preprocess('data.csv')
    X = df.drop(columns=['target', 'sno'])
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    
    clf.fit(X_train, y_train)
    
    # --- FAIRNESS AUDIT (1 Mark) ---
    # Bucket age into 20-year bins
    X_test['age_bucket'] = pd.cut(X_test['age'], bins=range(0, 120, 20), right=False)
    
    # Compute Metrics using Fairlearn
    y_pred = clf.predict(X_test)
    metrics = {
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test['age_bucket']
    )
    
    print("### Fairness Audit Results (Age Buckets) ###")
    print(metric_frame.by_group)
    
    # Save artifacts
    joblib.dump(clf, 'model.joblib')
    # Save training data for drift detection later
    X_train.to_csv('reference_data.csv', index=False)
    print("Model and reference data saved.")

if __name__ == "__main__":
    train_and_audit()