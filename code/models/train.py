from pathlib import Path
import json
import joblib
import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"
MODEL_FILE = ROOT / "models" / "model.joblib"
METRICS_FILE = ROOT / "metrics" / "metrics.json"
MLFLOW_DB = ROOT / "mlflow.db"


def main():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train.drop(columns=["quality"])
    y_train = train["quality"]

    X_test = test.drop(columns=["quality"])
    y_test = test["quality"]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("wine-quality")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
        },
        MODEL_FILE,
    )

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_FILE, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "f1": f1,
            },
            f,
            indent=2,
        )

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()