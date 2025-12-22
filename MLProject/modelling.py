import pandas as pd
import numpy as np
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi DagsHub
DAGSHUB_URI = "https://dagshub.com/Pramartha/Eksperimen_SML_Kadek-Pramartha-Mahottama.mlflow" 
DAGSHUB_USER = "Pramartha" 
DAGSHUB_TOKEN = "023e010fe8eb3de70c9874a84943112c9a8e05fe" 

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(DAGSHUB_URI)

def train():
    # 1. Menangkap Parameter dari MLProject
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"[INFO] Training dimulai. Params: n_estimators={n_estimators}, max_depth={max_depth}")

    # 2. Load Data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'telco_customer_churn_preprocessing')
    
    train_path = os.path.join(data_dir, "train_clean.csv")
    test_path = os.path.join(data_dir, "test_clean.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Data tidak ditemukan di: {train_path}")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    X_train = train.drop(columns=['Churn'])
    y_train = train['Churn']
    X_test = test.drop(columns=['Churn'])
    y_test = test['Churn']

    # 3. Training & Logging
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")
        
        # Log Metrics & Params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        
        # Log Model
        mlflow.sklearn.log_model(model, "model")
        
        # 4. SIMPAN RUN ID
        run_id = run.info.run_id
        print(f"Run ID disimpan: {run_id}")
        # Simpan file txt di folder yg sama
        with open("run_id.txt", "w") as f:
            f.write(run_id)

if __name__ == "__main__":
    train()