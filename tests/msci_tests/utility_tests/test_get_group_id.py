import os
import unittest

import torch
from torch import utils
import numpy as np
import scipy.sparse
import pandas as pd

from ss_opm.utility.get_group_id import get_group_id

class GetGroupIDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = "/media/sf_pfn-common/msci/compressed_small_dataset"
        self.metadata_df = pd.read_parquet(os.path.join(self.data_dir, "metadata.parquet"))

    def test_get_group_id(self):
        group_ids = get_group_id(self.metadata_df)

        #print(np.unique(group_ids).tolist())
        self.assertTrue((group_ids != -1).all())
        self.assertEqual(set(np.unique(group_ids)), set([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]))


if __name__ == '__main__':
    unittest.main()
