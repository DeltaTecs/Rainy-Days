import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

with mock.patch.dict(sys.modules, {"torch": mock.MagicMock(), "lightgbm": mock.MagicMock()}):
    from ml.modis_cloud_seeding.train import (
        FeatureStandardizer,
        _to_serialisable,
        apply_calibration,
        compute_class_weights,
        fit_isotonic,
        split_frame,
    )


class TrainUtilsTests(unittest.TestCase):
    def test_feature_standardizer_zero_mean_unit_variance(self) -> None:
        data = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        standardizer = FeatureStandardizer.from_array(data)
        transformed = standardizer.transform(data)
        self.assertTrue(np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6))
        self.assertTrue(np.allclose(transformed.std(axis=0), 1.0, atol=1e-6))

    def test_compute_class_weights_inverse_frequency(self) -> None:
        labels = np.array([0, 0, 1, 2, 2, 2])
        weights = compute_class_weights(labels, num_classes=3)
        expected = np.array([0.5, 1.0, 1.0 / 3.0], dtype=np.float32)
        expected = expected / expected.sum() * 3
        self.assertTrue(np.allclose(weights, expected))

    def test_apply_calibration_preserves_seedable_mass(self) -> None:
        probs = np.array(
            [
                [0.55, 0.30, 0.15],
                [0.20, 0.45, 0.35],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 1], dtype=int)
        calibrator = fit_isotonic(probs[:, 1] + probs[:, 2], labels)
        calibrated = apply_calibration(probs, calibrator)
        self.assertEqual(calibrated.shape, probs.shape)
        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6))
        predicted_seedable = calibrator.predict(probs[:, 1] + probs[:, 2])
        self.assertTrue(np.allclose(calibrated[:, 1] + calibrated[:, 2], predicted_seedable, atol=1e-6))

    def test_split_frame_prevents_granule_leakage(self) -> None:
        frame = pd.DataFrame(
            {
                "split": ["train", "val"],
                "granule": ["g1", "g1"],
                "feature": [0.1, 0.2],
                "label": [0, 1],
            }
        )
        with self.assertRaises(ValueError):
            split_frame(
                frame,
                split_column="split",
                split_values={"train": ["train"], "val": ["val"]},
                granule_column="granule",
            )

    def test_to_serialisable_converts_numpy_types(self) -> None:
        payload = {
            "values": np.array([1.0, 2.0], dtype=np.float32),
            "score": np.float64(1.234),
            "count": np.int64(5),
        }
        converted = _to_serialisable(payload)
        self.assertIsInstance(converted["values"], list)
        self.assertIsInstance(converted["score"], float)
        self.assertIsInstance(converted["count"], int)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
