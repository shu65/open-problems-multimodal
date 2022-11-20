import unittest

import numpy as np

from ss_opm.utility.row_normalize import row_normalize


class RowNormalizeTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)
        self.targets_values = np.random.normal(size=(2, 128))

    def test_normalize(self):
        normalized_targets_values = row_normalize(self.targets_values)
        np.testing.assert_allclose(
            np.mean(normalized_targets_values, axis=1),
            np.zeros((self.targets_values.shape[0])),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.std(normalized_targets_values, axis=1),
            np.ones((self.targets_values.shape[0])),
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
