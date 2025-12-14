import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

def generate_explanation():
    # 1. Load Data and Model
    print("Loading data and model...")
    df = pd.read_csv('data.csv')
    model = joblib.load('model.joblib')

    # Define features (Must match training order)
    X = df.drop(columns=['target', 'sno'])
    
    # Map target just to be safe, though we only need X for SHAP
    if df['target'].dtype == 'O':
        y = df['target'].map({'yes': 1, 'no': 0})
    else:
        y = df['target']

    # 2. Prepare Data for SHAP
    # We must access the internal steps of the pipeline
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    # Transform the FULL dataset
    print("Transforming data...")
    X_transformed = preprocessor.transform(X)

    # 3. Get Feature Names
    # Recover names from the one-hot encoder
    try:
        num_features = ['age', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca']
        cat_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(['gender', 'cp', 'restecg', 'slope', 'thal'])
        feature_names = num_features + list(cat_features)
    except Exception as e:
        print(f"Warning: Could not extract specific feature names ({e}). Using generic names.")
        feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]

    # 4. Filter for "Heart Disease Not Predicted" (Class 0)
    print("Predicting and filtering for 'No Disease' samples...")
    predictions = model.predict(X)
    
    # Get indices where prediction is 0 (No Disease)
    # The error happened here previously because of shape mismatch
    no_disease_indices = np.where(predictions == 0)[0]
    
    if len(no_disease_indices) == 0:
        print("Error: No samples found where heart disease was NOT predicted. Cannot generate explanation.")
        return

    # Filter the input matrix to only these rows
    X_no_disease = X_transformed[no_disease_indices]
    
    # 5. Calculate SHAP Values specifically for these samples
    # Optimization: Only calculate SHAP for the rows we care about
    print(f"Calculating SHAP values for {len(no_disease_indices)} samples...")
    explainer = shap.TreeExplainer(classifier)
    
    # Calculate SHAP only for the filtered subset
    shap_values = explainer.shap_values(X_no_disease)

    # 6. Handle SHAP Output Format (Binary Classification)
    # Random Forest Classifier usually returns a list of [Class 0 Values, Class 1 Values]
    if isinstance(shap_values, list):
        # We want to explain why they are Class 0, so we look at the contributions to Class 0
        shap_values_target = shap_values[0] 
    else:
        # Some versions return a single array
        shap_values_target = shap_values

    # 7. Generate Plot
    print("Generating summary plot...")
    plt.figure()
    shap.summary_plot(shap_values_target, X_no_disease, feature_names=feature_names, show=False)
    plt.title("Why did the model predict 'No Disease'?")
    
    output_file = "shap_explanation.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Success! Plot saved as '{output_file}'.")
    print("Check the plot to see which features (top of the list) are most important.")

if __name__ == "__main__":
    generate_explanation()