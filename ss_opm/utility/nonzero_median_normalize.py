import numpy as np
import scipy.sparse


def median_normalize(values, ignore_zero=True, log=False):
    tmp_values = values.copy()
    if isinstance(values, scipy.sparse.csr_matrix):
        tmp_values = tmp_values.toarray()
    if ignore_zero:
        tmp_values[tmp_values == 0] = np.nan
    nonzaero_median = np.nanquantile(tmp_values, q=0.5, axis=1).astype(values.dtype)
    # print("nonzaero_median", nonzaero_median)
    if log:
        ret = values - nonzaero_median[:, None]
    else:
        ret = values / nonzaero_median[:, None]
    return ret


def row_quantile_normalize(values, q=0.5):
    ret_data = np.zeros_like(values.data)
    for i in range(values.shape[0]):
        row_values = values.data[values.indptr[i] : values.indptr[i + 1]]
        q_value = np.quantile(row_values, q=q)
        row_values /= q_value
        ret_data[values.indptr[i] : values.indptr[i + 1]] = row_values
    return scipy.sparse.csr_matrix((ret_data, values.indices, values.indptr), values.shape)
