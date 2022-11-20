import numpy as np
import pandas as pd
import scipy.sparse


def correlation_score(y_true, y_pred):
    if type(y_true) == pd.DataFrame:
        y_true = y_true.values
    if type(y_pred) == pd.DataFrame:
        y_pred = y_pred.values
    if isinstance(y_true, scipy.sparse.csr_matrix):
        y_true = y_true.toarray()
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes are different.")

    corr_sum = 0
    for i in range(len(y_true)):
        corr_sum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corr_sum / len(y_true)
