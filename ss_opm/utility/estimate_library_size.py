import numpy as np


def estimate_library_size(log_normalized_read_counts):
    normalized_read_counts = np.expm1(log_normalized_read_counts)
    zero_mask = normalized_read_counts == 0.0
    max_value = normalized_read_counts.max()
    tmp_normalized_read_counts = normalized_read_counts
    tmp_normalized_read_counts[zero_mask] = max_value
    nonzero_min_values_per_cell = tmp_normalized_read_counts.min(axis=1)
    library_size = 1 / nonzero_min_values_per_cell * 1e6
    return library_size
