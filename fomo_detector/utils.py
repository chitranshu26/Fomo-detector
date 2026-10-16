import random


LABELS_MAP = {
    0: "Normal",
    1: "FOMO",
    2: "Relaxed",
    3: "Focused",
}


def get_label_name(label):
    return LABELS_MAP.get(label, "Unknown")


def generate_sample(label):
    """Generate a synthetic EEG feature sample for a known class."""
    if label not in LABELS_MAP:
        raise ValueError(f"Unknown label: {label}")

    noise = random.uniform(-1, 1)

    if label == 1:
        return [
            random.uniform(6, 12) + noise,
            random.uniform(18, 30) + noise,
            random.uniform(4, 9) + noise,
            random.uniform(4, 9) + noise,
            random.uniform(60, 95) + noise * 2,
            random.uniform(2, 6) + noise,
        ]

    if label == 2:
        return [
            random.uniform(10, 20) + noise,
            random.uniform(5, 12) + noise,
            random.uniform(6, 10) + noise,
            random.uniform(2, 5) + noise,
            random.uniform(20, 50) + noise * 2,
            random.uniform(0, 1.5) + noise,
        ]

    if label == 3:
        return [
            random.uniform(8, 14) + noise,
            random.uniform(14, 22) + noise,
            random.uniform(5, 9) + noise,
            random.uniform(4, 8) + noise,
            random.uniform(60, 85) + noise * 2,
            random.uniform(0, 2) + noise,
        ]

    return [
        random.uniform(8, 13) + noise,
        random.uniform(10, 18) + noise,
        random.uniform(5, 8) + noise,
        random.uniform(3, 6) + noise,
        random.uniform(40, 70) + noise * 2,
        random.uniform(1, 3) + noise,
    ]