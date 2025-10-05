import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ml.modis_cloud_seeding import preprocess


class _FakeDataset:
    def __init__(self, data, attrs):
        self._data = data
        self._attrs = dict(attrs)
        self.closed = False

    def get(self):
        return self._data

    def attributes(self):
        return dict(self._attrs)

    def endaccess(self):
        self.closed = True


class _FakeSD:
    last_instance = None

    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.closed = False
        self.dataset = _FakeDataset(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            {},
        )
        self._datasets = {"Cloud_Optical_Thickness": self.dataset}
        _FakeSD.last_instance = self

    def datasets(self):
        return self._datasets

    def select(self, name):
        return self._datasets[name]

    def end(self):
        self.closed = True


class PreprocessHDF4Tests(unittest.TestCase):
    def test_read_hdf4_dataset_masks_and_scales(self):
        raw = np.array([110.0, -9999.0, 205.0], dtype=np.float32)
        attrs = {
            "_FillValue": -9999,
            "valid_range": [0, 200],
            "scale_factor": 0.01,
            "add_offset": 100.0,
        }
        dataset = _FakeDataset(raw, attrs)

        processed = preprocess._read_hdf4_dataset(dataset)
        expected = np.array([0.1, np.nan, np.nan], dtype=np.float32)

        np.testing.assert_allclose(processed, expected, equal_nan=True)

    def test_load_granule_falls_back_to_pyhdf(self):
        fake_sdc = type("FakeSDC", (), {"READ": 0})
        with mock.patch.multiple(
            preprocess,
            xr=None,
            SD=_FakeSD,
            SDC=fake_sdc,
        ):
            dataset = preprocess.load_granule(Path("dummy.hdf"))

        self.assertIn("Cloud_Optical_Thickness", dataset)
        np.testing.assert_array_equal(
            dataset["Cloud_Optical_Thickness"].values,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

        dataset.close()
        self.assertIsNone(getattr(dataset, "_sd", None))
        self.assertTrue(_FakeSD.last_instance.closed)


if __name__ == "__main__":
    unittest.main()
