import time
import joblib
import random
import pandas as pd

from src.config import MODEL_PATH, LOG_PATH
from src.utils import generate_sample, get_label_name

# Load trained model
model = joblib.load(MODEL_PATH)

# -----------------------------
# 🔥 FOMO SCORE FUNCTION
# -----------------------------
def calculate_fomo(alpha, beta, attention, scroll):
    return (
        (beta / (alpha + 1)) * 0.4 +
        (attention / 100) * 0.3 +
        (scroll / 6) * 0.3
    )

# -----------------------------
# 🔥 SIMULATION FUNCTION
# -----------------------------
def run_simulation(duration=300):

    print("🚀 Running simulation...\n")

    logs = []

    threshold = 0.75
    time_window = 5
    fomo_counter = 0

    start = time.time()

    while time.time() - start < duration:

        # Generate random state
        real_label = random.choice([0, 1, 2, 3])

        # Generate EEG sample
        sample = generate_sample(real_label)

        # Extract features
        alpha, beta, theta, gamma, attention, scroll = sample

        # 🔥 Calculate FOMO score
        fomo_score = round(calculate_fomo(alpha, beta, attention, scroll), 3)

        # 🔥 CREATE DATAFRAME (MATCH TRAINING EXACTLY)
        sample_df = pd.DataFrame([[
            alpha, beta, theta, gamma,
            attention, scroll,
            fomo_score
        ]], columns=[
            "alpha",
            "beta",
            "theta",
            "gamma",
            "attention",
            "scroll_speed",
            "fomo_score"
        ])

        # Prediction
        pred = model.predict(sample_df)[0]

        # Confidence
        proba = model.predict_proba(sample_df)
        confidence = round(max(proba[0]), 3)

        # Labels
        real_name = get_label_name(real_label)
        pred_name = get_label_name(pred)

        print(f"Real: {real_name} | Pred: {pred_name} | Confidence: {confidence}")
        print(f"FOMO Score: {fomo_score}")

        # -----------------------------
        # 🔥 TEMPORAL LOGIC
        # -----------------------------
        if fomo_score > threshold:
            fomo_counter += 1
        else:
            fomo_counter = 0

        if fomo_counter >= time_window:
            print("🚨 ALERT: Sustained FOMO DETECTED!\n")
            fomo_counter = 0

        # Save logs
        logs.append([
            alpha, beta, theta, gamma,
            attention, scroll,
            fomo_score,
            real_label,
            pred,
            confidence
        ])

        time.sleep(1)

    # Save to CSV
    df = pd.DataFrame(logs, columns=[
        "alpha", "beta", "theta", "gamma",
        "attention", "scroll_speed",
        "fomo_score",
        "real_label",
        "predicted",
        "confidence"
    ])

    df.to_csv(LOG_PATH, index=False)

    print("\n✅ Simulation completed!")
    print("📁 Log saved at:", LOG_PATH)


if __name__ == "__main__":
    run_simulation()