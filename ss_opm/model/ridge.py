import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge as _Ridge
import scipy.sparse
import numpy as np

class Ridge(object):

    def __init__(self, params):
        self.params = params
        n_jobs = int(os.getenv("OMP_NUM_THREADS", 1))
        self.model = MultiOutputRegressor(_Ridge(**params), n_jobs=n_jobs)

    @staticmethod
    def get_params(task_type, device="cpu", trial=None, debug=False):
        params = {
            "copy_X": True,
            "alpha": 1.0,
        }
        if trial is not None:
            params['alpha'] = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
        return params

    def fit(self, x, y, preprocessed_x, preprocessed_y, metadata, pre_post_process):
        if isinstance(preprocessed_y, scipy.sparse.csr_matrix):
            preprocessed_y = preprocessed_y.toarray()
        self.model.fit(preprocessed_x, preprocessed_y)

    def predict(self, x, preprocessed_x, metadata):
        chunk_size = 1024
        start_idx = 0
        y_pred_list = []
        while True:
            if start_idx == x.shape[0]:
                break

            end_idx = start_idx + chunk_size
            if x.shape[0] < end_idx:
                end_idx = x.shape[0]
            sub_preprocessed_x = preprocessed_x[start_idx:end_idx, :]
            sub_y_pred = self.model.predict(sub_preprocessed_x)
            y_pred_list.append(sub_y_pred)
            start_idx = end_idx
        y_pred = np.vstack(y_pred_list)
        assert len(y_pred) == x.shape[0]
        return y_pred
