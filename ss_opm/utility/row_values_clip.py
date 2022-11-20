import numpy as np
import scipy.sparse


def row_values_clip(values, min_q=0.0, max_q=1.0):
    if isinstance(values, scipy.sparse.csr_matrix):
        ret_data = np.zeros_like(values.data)
        for i in range(values.shape[0]):
            row = values.data[values.indptr[i] : values.indptr[i + 1]].copy()
            q_values = np.quantile(row, [min_q, max_q])
            row[row < q_values[0]] = q_values[0]
            row[row > q_values[1]] = q_values[1]
            ret_data[values.indptr[i] : values.indptr[i + 1]] = row
        return scipy.sparse.csr_matrix((ret_data, values.indices, values.indices), values.shape)
    else:
        ret = np.zeros_like(values)
        for i in range(values.shape[0]):
            row = values[i, :].copy()
            q_values = np.quantile(row, [min_q, max_q])
            row[row < q_values[0]] = q_values[0]
            row[row > q_values[1]] = q_values[1]
            ret[i, :] = row
        return ret
