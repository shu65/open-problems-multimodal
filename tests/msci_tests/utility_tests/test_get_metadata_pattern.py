import os
import unittest

import torch
from torch import utils
import numpy as np
import scipy.sparse
import pandas as pd

from ss_opm.utility.get_metadata_pattern import generate_metadata_patterns, get_metadata_pattern


class GenerateMetadataPatternsTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_multi(self):
        patterns = generate_metadata_patterns()
        print(patterns)
        self.assertEqual(len(patterns), 24)

    def test_cite(self):
        patterns = generate_metadata_patterns()
        print(patterns)
        self.assertEqual(len(patterns), 24)

if __name__ == '__main__':
    unittest.main()
