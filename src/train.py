import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from src.config import DATA_PATH, MODEL_PATH

def train_model():
    df = pd.read_csv(DATA_PATH)

    # Drop timestamp only
    X = df.drop(columns=["label", "timestamp"])
    y = df["label"]

    # IMPORTANT: shuffle + realistic split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Reduce overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,          #  limit depth
        min_samples_split=5,  #  avoid overfit
        min_samples_leaf=3
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Model Evaluation:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print("✅ Model saved!")

if __name__ == "__main__":
    train_model()