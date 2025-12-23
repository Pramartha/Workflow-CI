import mlflow
import mlflow.sklearn
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Setup Path
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "telco_customer_churn_preprocessing", "train_clean.csv")
test_path = os.path.join(base_dir, "telco_customer_churn_preprocessing", "test_clean.csv")

# Load Data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Pisahkan Fitur (X) dan Target (y)
X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]

X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

# Training dengan Autolog
with mlflow.start_run() as run:
    
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    # Autolog
    mlflow.sklearn.autolog()

    # Training Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc}")
    print(f"F1-score: {f1}")