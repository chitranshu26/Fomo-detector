import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fomo_detector.config import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH
from fomo_detector.simulate import LOG_COLUMNS, calculate_fomo, run_simulation
from fomo_detector.train import load_training_data, train_model
from fomo_detector.utils import generate_sample, get_label_name


class TrainingTests(unittest.TestCase):
    def test_default_dataset_is_valid(self):
        features, labels = load_training_data()

        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(set(labels.unique()), {0, 1, 2, 3})
        self.assertGreater(len(features), 0)

    def test_training_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "invalid.csv"
            pd.DataFrame({"label": [0], "timestamp": ["now"]}).to_csv(
                data_path, index=False
            )

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_training_data(data_path)

    def test_training_saves_to_requested_path(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "test_model.pkl"
            with contextlib.redirect_stdout(io.StringIO()):
                model = train_model(DATA_PATH, model_path)

            self.assertTrue(model_path.is_file())
            self.assertEqual(tuple(model.feature_names_in_), FEATURE_COLUMNS)


class SimulationTests(unittest.TestCase):
    def test_score_and_sample_generation(self):
        self.assertAlmostEqual(calculate_fomo(9, 10, 50, 3), 0.7)
        self.assertEqual(len(generate_sample(1)), 6)
        self.assertEqual(get_label_name(3), "Focused")

    def test_simulation_appends_rows_without_replacing_existing_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "simulation.csv"
            existing_log = pd.DataFrame([[0] * len(LOG_COLUMNS)], columns=LOG_COLUMNS)
            existing_log.to_csv(log_path, index=False)

            with contextlib.redirect_stdout(io.StringIO()):
                result = run_simulation(
                    duration=0.01,
                    model_path=MODEL_PATH,
                    log_path=log_path,
                )

            contents = log_path.read_text(encoding="utf-8")
            self.assertGreaterEqual(len(result), 1)
            self.assertEqual(contents.count("alpha,beta,theta"), 1)
            self.assertEqual(len(pd.read_csv(log_path)), len(existing_log) + len(result))

    def test_simulation_rejects_non_positive_duration(self):
        with self.assertRaisesRegex(ValueError, "finite number greater than zero"):
            run_simulation(duration=0, model_path=MODEL_PATH)


if __name__ == "__main__":
    unittest.main()