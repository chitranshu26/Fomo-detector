import math
import random
import threading
import time
from pathlib import Path

import joblib
import pandas as pd

from fomo_detector.config import FEATURE_COLUMNS, LOG_PATH, MODEL_PATH
from fomo_detector.utils import generate_sample, get_label_name


LOG_COLUMNS = (
    "alpha",
    "beta",
    "theta",
    "gamma",
    "attention",
    "scroll_speed",
    "fomo_score",
    "real_label",
    "predicted",
    "confidence",
)


def calculate_fomo(alpha, beta, attention, scroll):
    return (
        (beta / (alpha + 1)) * 0.4
        + (attention / 100) * 0.3
        + (scroll / 6) * 0.3
    )


def run_simulation(
    duration=300,
    *,
    model_path=MODEL_PATH,
    log_path=LOG_PATH,
    on_sample=None,
    stop_event=None,
):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Simulation duration must be a finite number greater than zero.")

    model_path = Path(model_path)
    log_path = Path(log_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train the model before simulating."
        )

    model = joblib.load(model_path)
    expected_features = tuple(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    if expected_features != FEATURE_COLUMNS:
        raise ValueError("Model feature columns do not match the current application schema.")

    print("Running synthetic simulation.\n")
    logs = []
    stop_event = stop_event or threading.Event()
    fomo_counter = 0
    threshold = 0.75
    time_window = 5
    start = time.monotonic()

    while time.monotonic() - start < duration and not stop_event.is_set():
        real_label = random.choice((0, 1, 2, 3))
        sample = generate_sample(real_label)
        alpha, beta, theta, gamma, attention, scroll = sample
        fomo_score = round(calculate_fomo(alpha, beta, attention, scroll), 3)
        sample_frame = pd.DataFrame(
            [[alpha, beta, theta, gamma, attention, scroll, fomo_score]],
            columns=FEATURE_COLUMNS,
        )

        prediction = model.predict(sample_frame)[0]
        probabilities = model.predict_proba(sample_frame)[0]
        confidence = round(float(max(probabilities)), 3)
        print(
            f"Actual: {get_label_name(real_label)} | "
            f"Predicted: {get_label_name(prediction)} | Confidence: {confidence}"
        )
        print(f"FOMO score: {fomo_score}")

        if fomo_score > threshold:
            fomo_counter += 1
        else:
            fomo_counter = 0
        if fomo_counter >= time_window:
            print("Alert: sustained high FOMO score detected.\n")
            fomo_counter = 0

        sample_log = {
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "gamma": gamma,
            "attention": attention,
            "scroll_speed": scroll,
            "fomo_score": fomo_score,
            "real_label": real_label,
            "predicted": int(prediction),
            "confidence": confidence,
        }
        logs.append([sample_log[column] for column in LOG_COLUMNS])
        if on_sample is not None:
            on_sample(sample_log)
        remaining = duration - (time.monotonic() - start)
        if remaining > 0:
            stop_event.wait(min(1.0, remaining))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(logs, columns=LOG_COLUMNS)
    output.to_csv(
        log_path,
        index=False,
        mode="a",
        header=not log_path.exists() or log_path.stat().st_size == 0,
    )
    print(f"Simulation completed. Log appended to {log_path}")
    return output