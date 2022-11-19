import os
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import scipy.sparse
import numpy as np

class LGBM(object):

    def __init__(self, params):
        self.params = params
        n_jobs = int(os.getenv("OMP_NUM_THREADS", 1))
        self.model = MultiOutputRegressor(lgb.LGBMRegressor(**params), n_jobs=n_jobs)

    @staticmethod
    def get_params(task_type, device="cpu", trial=None, debug=False):
        params = {
            'objective':'mse',
            'metric': 'mse',
            'random_state': 42,
            'n_estimators': 400,
            "n_jobs": 1,
        }
        if trial is not None:
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
            params['colsample_bytree'] = trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0])
            params['subsample'] = trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 0.99, log=True),
            params['max_depth'] = trial.suggest_int('max_depth', 5, 20),
            params['num_leaves'] = trial.suggest_int('num_leaves', 1, 1000),
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 300),
            params['cat_smooth'] = trial.suggest_int('min_data_per_groups', 1, 100)
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
            sub_x = preprocessed_x[start_idx:end_idx, :]
            sub_y_pred = self.model.predict(sub_x)
            y_pred_list.append(sub_y_pred)
            start_idx = end_idx
        y_pred = np.vstack(y_pred_list)
        assert len(y_pred) == x.shape[0]
        return y_pred
