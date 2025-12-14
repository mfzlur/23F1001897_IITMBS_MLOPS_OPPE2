# Heart Disease Prediction MLOps Pipeline

A production-ready, explainable, observable, and scalable MLOps deployment for predicting heart disease. This project demonstrates an end-to-end pipeline using **FastAPI**, **Docker**, **Google Kubernetes Engine (GKE)**, **GitHub Actions**, and **Fairlearn**.

---

## 📂 Project Structure

```text
.
├── .github/workflows/
│   └── deploy.yaml         # CI/CD Pipeline (Build -> Push -> Deploy)
├── k8s/
│   └── deployment.yaml     # Kubernetes Manifests (Deployment, Service, HPA)
├── src/
│   ├── app.py              # FastAPI Inference Service
│   ├── train.py            # Model Training & Fairness Audit
│   └── explainability.py   # SHAP Explainability Analysis
├── tests/
│   ├── monitor_drift_load.py # Client script for Observability & Drift Detection
│   └── post.lua            # Lua script for wrk load testing
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── data.csv                # Dataset
└── README.md               # Documentation
````

-----

## 🚀 Key Features & Deliverables

| Feature | Implementation Details |
| :--- | :--- |
| **Model** | Random Forest Classifier trained on heart disease data. |
| **Fairness** | **Fairlearn** analysis on "Age" (bucketed into 20-year bins) to ensure objective performance across groups. |
| **Deployment** | **GCP GKE** with Horizontal Pod Autoscaling (Max 3 pods). |
| **CI/CD** | **GitHub Actions** triggers automated training, container building, and deployment on push. |
| **Observability** | Per-sample prediction logging using structured JSON logs in FastAPI. |
| **Monitoring** | **Data Drift Detection** (Kolmogorov-Smirnov test) comparing reference vs. live data. |
| **Performance** | High-concurrency stress testing (\>2000 users) using **wrk**. |
| **Explainability** | **SHAP** analysis to explain factors driving "No Disease" predictions. |

-----

## 🛠️ Setup & Installation

### Prerequisites

  * Python 3.9+
  * Docker
  * Google Cloud SDK (`gcloud`)
  * `kubectl`
  * `wrk` (for load testing)

### 1\. Local Setup

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

-----

## 🧠 Training & Fairness Audit

Train the model and generate the fairness report. This script preprocesses data, trains the Random Forest, and uses **Fairlearn** to audit the model against Age buckets.

```bash
python src/train.py
```

  * **Output:** Saves `model.joblib` and `reference_data.csv`.
  * **Fairness:** Prints Selection Rate and False Positive Rate for different age groups in the terminal.

-----

## ☁️ Deployment (CI/CD)

The deployment is fully automated using GitHub Actions.

1.  **Push to Main:**
    ```bash
    git push origin main
    ```
2.  **Pipeline Steps:**
      * Sets up Python & GCloud Auth.
      * Retrains the model to ensure freshness.
      * Builds Docker Image.
      * Pushes to Google Artifact Registry.
      * Deploys to GKE cluster (`mlops-iitm-oppe2`).

### Kubernetes Configuration

  * **Replicas:** Autoscaling enabled (min 1, max 3).
  * **Resources:** Limits set to 0.5 CPU / 512Mi Memory.
  * **Service:** LoadBalancer exposing port 80.

-----

## 📊 Monitoring & Testing

Once deployed, obtain the external IP:

```bash
kubectl get service heart-disease-service
```

### 1\. Observability & Drift Detection

Run the client simulation script. It generates 100 random samples, hits the API, and checks for input drift (KS Test) on cholesterol levels.

```bash
# Update the URL in the script with your External IP before running
python tests/monitor_drift_load.py
```

  * **Observability:** Check server logs (`kubectl logs -f deployment/heart-disease-model`) to see JSON logs for every request.
  * **Drift Alert:** If the random data differs significantly from training data, the script outputs an **ALERT**.

### 2\. High-Concurrency Stress Test

Use `wrk` to simulate 2000 concurrent users.

```bash
wrk -t12 -c2000 -d30s -s tests/post.lua http://<EXTERNAL-IP>/predict
```

  * **Goal:** Demonstrate scalability limits and analyze timeouts/latency under load.

### 3\. Model Explainability

Generate a SHAP summary plot to understand feature importance for specific predictions.

```bash
python src/explainability.py
```

  * **Output:** Generates `shap_explanation.png`.

-----

## 🔗 API Reference

**POST** `/predict`

**Payload:**

```json
{
  "age": 63,
  "gender": "male",
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Response:**

```json
{
  "prediction": "yes",
  "probability": 0.85
}
```

```