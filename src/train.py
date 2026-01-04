import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['target'] = df['target'].map({'yes': 1, 'no': 0})
    
    categorical_features = ['gender', 'cp', 'restecg', 'slope', 'thal', 'fbs', 'exang']
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Handle unknown categories to prevent crashes in production
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
    
    # 1. Stratified Split to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Define Model with Balanced Weights
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', rf)])
    
    # 3. Hyperparameter  Space
    # We search for the optimal theta (\theta) to maximize F1-score
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    #  search with Cross-Validation
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=20, 
        cv=5, 
        scoring='f1', 
        n_jobs=-1, 
        random_state=42
    )
    
    print("Starting Hyperparameter Tuning...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    print(f"Best Parameters: {search.best_params_}")
    
    # --- FAIRNESS AUDIT ---
    sensitive_feature = X_test['age']
    age_buckets = pd.cut(sensitive_feature, bins=range(0, 120, 20), right=False)
    
    y_pred = best_model.predict(X_test)
    
    # Metrics
    metrics = {
        'selection_rate': selection_rate,
        'fpr': false_positive_rate,
        'precision': precision_score,
        'recall': recall_score
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=age_buckets
    )
    
    print("\n### Fairness Audit Results (Age Buckets) ###")
    print(metric_frame.by_group)
    
    # Save best model
    joblib.dump(best_model, 'model.joblib')
    print("model saved.")

if __name__ == "__main__":
    train_and_audit()