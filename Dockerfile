FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY model.joblib .
# Copy reference data if drift detection is internal, otherwise stored externally
COPY reference_data.csv . 

EXPOSE 8000

CMD ["python", "src/app.py"]