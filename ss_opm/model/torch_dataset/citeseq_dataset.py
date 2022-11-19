import os
import random

import numpy as np
import torch
import scipy.sparse

from ss_opm.utility.metadata_utility import MALE_DONOR_IDS, FEMALE_DONOR_IDS, CELL_TYPES
from ss_opm.utility import log_counts_helper
from ss_opm.utility.estimate_library_size import estimate_library_size
from ss_opm.utility.get_selector_with_metadata_pattern import get_selector_with_metadata_pattern

METADATA_KEYS = [
    "day",
    "nonzero_ratio",
    "nonzero_q25",
    "nonzero_q50",
    "nonzero_q75",
    "mean",
    "std",
    'cell_ratio_HSC',
    'cell_ratio_EryP',
    'cell_ratio_NeuP',
    'cell_ratio_MasP',
    'cell_ratio_MkP',
    'cell_ratio_BP',
    'cell_ratio_MoP',
    'cell_count',
    'batch_sv0',
    'batch_sv1',
    'batch_sv2',
    'batch_sv3',
    'batch_sv4',
    'batch_sv5',
    'batch_sv6',
    'batch_sv7',
]

class CITEseqDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, inputs_values, preprocessed_inputs_values, metadata, targets_values, preprocessed_targets_values, selected_metadata):
        #assert isinstance(inputs_values, scipy.sparse.csr_matrix)
        if selected_metadata is not None:
            selector = get_selector_with_metadata_pattern(metadata=metadata, metadata_pattern=selected_metadata)
            inputs_values = inputs_values[selector, :]
            preprocessed_inputs_values = preprocessed_inputs_values[selector, :]
            metadata = metadata.loc[selector, :]
            if targets_values is not None:
                targets_values = targets_values[selector, :]
                preprocessed_targets_values = preprocessed_targets_values[selector, :]
        if targets_values is not None:
            #assert isinstance(targets_values, scipy.sparse.csr_matrix)
            assert preprocessed_inputs_values.shape[0] == targets_values.shape[0]
            assert preprocessed_targets_values.shape[0] == targets_values.shape[0]
        self.preprocessed_inputs_values = preprocessed_inputs_values
        self.targets_values = targets_values
        self.preprocessed_targets_values = preprocessed_targets_values

        assert preprocessed_inputs_values.shape[0] == len(metadata)
        self.metadata = metadata
        gender_ids = np.zeros((len(self.metadata),), dtype=int)
        gender_ids[self.metadata["donor"].isin(MALE_DONOR_IDS)] = 1
        self.gender_ids = gender_ids
        #print(self.metadata.columns)
        cell_type_ids = np.zeros((len(self.metadata),), dtype=int)
        cell_type_values = self.metadata["cell_type"].values
        for i, cell_type in enumerate(CELL_TYPES):
            cell_type_ids[cell_type_values == cell_type] = i
        self.cell_type_ids = cell_type_ids
        #self.cell_type_one_hot = np.eye(len(CELL_TYPES))[self.cell_type_ids]
        self.metadata_keys = METADATA_KEYS

    def __getitem__(self, index):
        if isinstance(self.preprocessed_inputs_values, scipy.sparse.csr_matrix):
            preprocessed_inputs_values = self.preprocessed_inputs_values[index].toarray().ravel()
        else:
            preprocessed_inputs_values = self.preprocessed_inputs_values[index].ravel()
        """
        if (self.max_cutout_p > 0.0) and (np.random.rand() < 0.5):
            library_size = self.inputs_library_sizes[index]
            inputs_counts_values = log_counts_helper.inverse_normalize(inputs_values, library_size)
            cutout_p = np.random.uniform(low=0.0, high=self.max_cutout_p)
            mask = np.random.rand(inputs_counts_values.shape[0]) < cutout_p
            inputs_counts_values[mask] = 0.0
            inputs_values = log_counts_helper.normalize(inputs_counts_values)
        """
        preprocessed_inputs_tensor = torch.as_tensor(preprocessed_inputs_values, dtype=torch.float32)
        gender_id = torch.as_tensor(self.gender_ids[index], dtype=torch.int64)
        cell_type_id = torch.as_tensor(self.cell_type_ids[index], dtype=torch.int64)
        info = torch.as_tensor(
            self.metadata.iloc[index, :][self.metadata_keys].values.astype(float),
            dtype=torch.float32
        )
        if self.targets_values is not None:
            if isinstance(self.targets_values, scipy.sparse.csr_matrix):
                targets_values = self.targets_values[index].toarray().ravel()
            else:
                targets_values = self.targets_values[index].ravel()
            preprocessed_targets_values = self.preprocessed_targets_values[index].ravel()
            targets_tensor = torch.as_tensor(targets_values, dtype=torch.float32)
            preprocessed_targets_tensor = torch.as_tensor(preprocessed_targets_values, dtype=torch.float32)
            return [preprocessed_inputs_tensor, gender_id, info, targets_tensor, preprocessed_targets_tensor]
        else:
            return [preprocessed_inputs_tensor, gender_id, info, ]


    def __len__(self):
        return self.preprocessed_inputs_values.shape[0]
