import pandas as pd
import numpy as np
import requests
import json
from scipy.stats import ks_2samp

# 1. Generate 100-row randomly generated data (2 Marks part A)
def generate_random_data(n=100):
    np.random.seed(42)
    data = {
        'age': np.random.randint(29, 77, n),
        'gender': np.random.choice(['male', 'female'], n),
        'cp': np.random.randint(0, 4, n),
        'trestbps': np.random.uniform(94, 200, n),
        'chol': np.random.uniform(126, 564, n),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.uniform(71, 202, n),
        'exang': np.random.uniform(0, 1, n), # Simplified for random gen
        'oldpeak': np.random.uniform(0, 6.2, n),
        'slope': np.random.uniform(0, 3, n),
        'ca': np.random.randint(0, 5, n),
        'thal': np.random.randint(0, 4, n)
    }
    return pd.DataFrame(data)

def test_api_and_monitor():
    df_random = generate_random_data(100)
    url = "http://<EXTERNAL-IP>/predict" # Replace with actual LoadBalancer IP
    
    print("--- Starting Predictions (Observability) ---")
    predictions = []
    
    # Send requests
    for _, row in df_random.iterrows():
        payload = row.to_dict()
        try:
            resp = requests.post(url, json=payload, timeout=5)
            predictions.append(resp.json())
        except Exception as e:
            print(f"Request failed: {e}")

    print(f"Completed {len(predictions)} requests.")
    
    # 2. Compute Input Drift (1 Mark)
    print("\n--- Computing Input Drift ---")
    # Load reference data (saved during training)
    try:
        ref_data = pd.read_csv("reference_data.csv")
        
        # We compare 'chol' (Cholesterol) as a sample continuous feature
        stat, p_value = ks_2samp(ref_data['chol'], df_random['chol'])
        
        print(f"Drift Analysis for 'chol':")
        print(f"KS Statistic: {stat:.4f}")
        print(f"P-Value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("ALERT: Significant drift detected in 'chol' distribution!")
        else:
            print("No significant drift detected.")
            
    except FileNotFoundError:
        print("Reference data not found. Ensure training script has run.")

if __name__ == "__main__":
    test_api_and_monitor()