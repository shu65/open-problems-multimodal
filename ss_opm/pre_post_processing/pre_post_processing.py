import os

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD

from ss_opm.utility.iterative_svd_imputator import IterativeSVDImputator
from ss_opm.utility.nonzero_median_normalize import median_normalize, row_quantile_normalize
from ss_opm.utility.row_normalize import row_normalize
from ss_opm.utility.row_values_clip import row_values_clip


def _get_decomposer_method(method):
    if method == "svd":
        return TruncatedSVD
    elif method == "pca":
        return PCA
    elif method == "none":
        return None
    else:
        raise RuntimeError


class PrePostProcessing(object):
    @staticmethod
    def get_params(task_type, data_dir, device="cpu", trial=None, debug=False, seed=42):
        params = {
            "task_type": task_type,
            "inputs_decomposer_method": "svd",
            "use_test_inputs": True,
            "use_inputs_decomposer": True,
            "binary_inputs_decomposer": {
                "n_components": 128,
                "random_state": seed,
                "n_oversamples": 20,
                "n_iter": 7,
            },
            "inputs_decomposer": {
                "n_components": 128,
                "random_state": seed,
                "n_oversamples": 20,
                "n_iter": 7,
            },
            "use_inputs_scaler": False,
            "targets_decomposer_method": "svd",
            # "targets_decomposer_method": "pca",
            "use_targets_decomposer": True,
            "use_targets_normalization": True,
            "targets_decomposer": {
                "n_components": 128,
                "random_state": seed,
                "n_oversamples": 20,
                "n_iter": 7,
            },
            "postprocess_clip": False,
            "use_targets_scaler": False,
            "data_dir": data_dir,
        }
        if trial is not None:
            # params['inputs_decomposer_method'] = trial.suggest_categorical('inputs_decomposer_method', ["svd", "pca",])
            # params['targets_decomposer_method'] = trial.suggest_categorical('targets_decomposer_method', ["svd", "pca",])
            params["inputs_decomposer"]["n_components"] = trial.suggest_int("inputs_decomposer_n_components", 16, 1024)
            params["use_targets_decomposer"] = trial.suggest_categorical("use_targets_decomposer", [True, False])
            if params["use_targets_decomposer"]:
                if task_type == "multi":
                    params["targets_decomposer"]["n_components"] = trial.suggest_int("targets_decomposer_n_components", 16, 1024)
                elif task_type == "cite":
                    params["targets_decomposer"]["n_components"] = trial.suggest_int("targets_decomposer_n_components", 16, 128)
                else:
                    raise RuntimeError()

        return params

    def __init__(self, params):
        self.preprocesses = {}
        self.params = params
        self.is_fitting = False

    def _preprocess(self, inputs_values, targets_values, metadata, test_inputs_values, test_metadata, fitting):
        transformed_inputs_values = inputs_values
        # if isinstance(transformed_inputs_values, scipy.sparse.csr_matrix):
        #    transformed_inputs_values = transformed_inputs_values.toarray()
        targets_metadata = metadata
        if fitting and self.params["use_test_inputs"]:
            transformed_inputs_values = scipy.sparse.vstack((inputs_values, test_inputs_values))
            metadata = pd.concat([metadata, test_metadata])
            assert len(metadata) == transformed_inputs_values.shape[0]
            print("use test_inputs_values. total size:", transformed_inputs_values.shape[0])
        transformed_targets_values = targets_values
        if isinstance(transformed_targets_values, scipy.sparse.csr_matrix):
            transformed_targets_values = transformed_targets_values.toarray()
        if transformed_targets_values is not None:
            if self.params["task_type"] == "multi":
                transformed_targets_values = row_values_clip(transformed_targets_values, min_q=0.0, max_q=0.99)
                transformed_targets_values = np.log1p(median_normalize(np.expm1(transformed_targets_values)))
                if fitting:
                    self.preprocesses["targets_imputator"] = IterativeSVDImputator(iters=1)
                    self.preprocesses["targets_imputator"].fit(transformed_targets_values)
                transformed_targets_values = self.preprocesses["targets_imputator"].transform(transformed_targets_values)
                if self.params["use_targets_normalization"]:
                    transformed_targets_values = row_normalize(transformed_targets_values)
                if fitting:
                    unique_group_ids = targets_metadata["group"].unique()
                    targets_batch_medians = {}
                    for group_id in unique_group_ids:
                        s_group = targets_metadata["group"] == group_id
                        transformed_targets_values_batch = transformed_targets_values[s_group]
                        tmp = transformed_targets_values_batch.copy()
                        targets_batch_median = np.median(tmp, axis=0)
                        targets_batch_medians[group_id] = targets_batch_median
                    self.preprocesses["targets_batch_medians"] = targets_batch_medians
                    targets_batch_medians_list = []
                    for group_id in unique_group_ids:
                        targets_batch_medians_list.append(targets_batch_medians[group_id][None, :])
                    targets_global_median = np.vstack(targets_batch_medians_list)
                    targets_global_median = np.mean(targets_global_median, axis=0)
                    self.preprocesses["targets_global_median"] = targets_global_median
                transformed_targets_values = transformed_targets_values - self.preprocesses["targets_global_median"][None, :]
            elif self.params["task_type"] == "cite":
                transformed_targets_values = row_values_clip(transformed_targets_values, min_q=0.01, max_q=0.99)
                transformed_targets_values = median_normalize(transformed_targets_values, ignore_zero=False)
                if self.params["use_targets_normalization"]:
                    transformed_targets_values = row_normalize(transformed_targets_values)
                if fitting:
                    unique_group_ids = targets_metadata["group"].unique()
                    targets_batch_medians = {}
                    for group_id in unique_group_ids:
                        s_group = targets_metadata["group"] == group_id
                        transformed_targets_values_batch = transformed_targets_values[s_group]
                        tmp = transformed_targets_values_batch.copy()
                        targets_batch_median = np.median(tmp, axis=0)
                        targets_batch_medians[group_id] = targets_batch_median
                    self.preprocesses["targets_batch_medians"] = targets_batch_medians
                    targets_batch_medians_list = []
                    for group_id in unique_group_ids:
                        targets_batch_medians_list.append(targets_batch_medians[group_id][None, :])
                    targets_global_median = np.vstack(targets_batch_medians_list)
                    targets_global_median = np.mean(targets_global_median, axis=0)
                    self.preprocesses["targets_global_median"] = targets_global_median
                    assert self.preprocesses["targets_global_median"].shape[0] == transformed_targets_values.shape[1]
                unique_group_ids = targets_metadata["group"].unique()
                for group_id in unique_group_ids:
                    s_group = targets_metadata["group"] == group_id
                    transformed_targets_values[s_group, :] -= self.preprocesses["targets_batch_medians"][group_id][None, :]
            else:
                raise RuntimeError
            targets_decomposer_class = _get_decomposer_method(self.params["targets_decomposer_method"])
            if targets_decomposer_class is not None:
                if fitting:
                    targets_decomposer_params = self.params["targets_decomposer"]
                    if targets_decomposer_class == PCA and targets_decomposer_params["n_components"] >= targets_values.shape[0]:
                        targets_decomposer_params["n_components"] = targets_values.shape[0]
                    self.preprocesses["targets_decomposer"] = targets_decomposer_class(
                        **targets_decomposer_params,
                    )
                    self.preprocesses["targets_decomposer"].fit(transformed_targets_values)
                transformed_targets_values = self.preprocesses["targets_decomposer"].transform(transformed_targets_values)
            if self.params["use_targets_scaler"]:
                self.preprocesses["targets_scaler"] = sklearn.preprocessing.StandardScaler()
                self.preprocesses["targets_scaler"].fit(transformed_targets_values)
                transformed_targets_values = self.preprocesses["targets_scaler"].transform(transformed_targets_values)

        # inputs preprocess
        if self.params["task_type"] == "multi":
            transformed_inputs_values = row_quantile_normalize(transformed_inputs_values)
        elif self.params["task_type"] == "cite":
            transformed_inputs_values = np.log1p(median_normalize(np.expm1(transformed_inputs_values.toarray())))
            if fitting:
                inputs_targets_pair = np.load(
                    os.path.join(self.params["data_dir"], "cite_inputs_targets_pair3g.npz"), allow_pickle=True
                )["mask"]
                inputs_mask = inputs_targets_pair.max(axis=1)
                inputs_mask |= np.load(os.path.join(self.params["data_dir"], "cite_inputs_mask2.npz"), allow_pickle=True)["mask"]
                assert inputs_mask.shape[0] == transformed_inputs_values.shape[1]
                self.preprocesses["inputs_mask"] = inputs_mask
            selected_transformed_inputs_values = transformed_inputs_values[:, self.preprocesses["inputs_mask"]]
            if fitting:
                self.preprocesses["inputs_imputator"] = IterativeSVDImputator(iters=1)
                self.preprocesses["inputs_imputator"].fit(transformed_inputs_values)
            transformed_inputs_values = self.preprocesses["inputs_imputator"].transform(transformed_inputs_values)
            if fitting:
                self.preprocesses["inputs_medians"] = np.median(transformed_inputs_values, axis=0)
                assert self.preprocesses["inputs_medians"].shape[0] == transformed_inputs_values.shape[1]
            transformed_inputs_values = transformed_inputs_values - self.preprocesses["inputs_medians"]
        else:
            raise RuntimeError
        inputs_decomposer_class = _get_decomposer_method(self.params["inputs_decomposer_method"])
        if inputs_decomposer_class is not None:
            binary_transformed_inputs_values = transformed_inputs_values > 0
            if fitting:
                if self.params["task_type"] == "multi":
                    binary_inputs_decomposer_params = self.params["binary_inputs_decomposer"]
                    assert inputs_decomposer_class != PCA
                    # if inputs_decomposer_class == PCA and inputs_decomposer_params["n_components"] >= inputs_values.shape[0]:
                    #    inputs_decomposer_params["n_components"] = inputs_values.shape[0]
                    self.preprocesses["binary_inputs_decomposer"] = inputs_decomposer_class(
                        **binary_inputs_decomposer_params,
                    )
                    self.preprocesses["binary_inputs_decomposer"].fit(binary_transformed_inputs_values)
                inputs_decomposer_params = self.params["inputs_decomposer"]
                self.preprocesses["inputs_decomposer"] = inputs_decomposer_class(
                    **inputs_decomposer_params,
                )
                self.preprocesses["inputs_decomposer"].fit(transformed_inputs_values)
            transformed_inputs_values = self.preprocesses["inputs_decomposer"].transform(transformed_inputs_values)
            if self.params["task_type"] == "multi":
                binary_transformed_inputs_values = self.preprocesses["binary_inputs_decomposer"].transform(
                    binary_transformed_inputs_values
                )
                transformed_inputs_values = np.hstack((transformed_inputs_values, binary_transformed_inputs_values))
            elif self.params["task_type"] == "cite":
                transformed_inputs_values = np.hstack((transformed_inputs_values, selected_transformed_inputs_values))
        if self.params["use_inputs_scaler"]:
            if fitting:
                self.preprocesses["inputs_scaler"] = sklearn.preprocessing.StandardScaler()
                self.preprocesses["inputs_scaler"].fit(transformed_inputs_values)
            transformed_inputs_values = self.preprocesses["inputs_scaler"].transform(transformed_inputs_values)
        return transformed_inputs_values, transformed_targets_values

    def fit_preprocess(self, inputs_values, targets_values, metadata, test_inputs_values, test_metadata):
        self._preprocess(
            inputs_values=inputs_values,
            targets_values=targets_values,
            metadata=metadata,
            test_inputs_values=test_inputs_values,
            test_metadata=test_metadata,
            fitting=True,
        )
        self.is_fitting = True
        return self

    def preprocess(self, inputs_values, targets_values, metadata):
        return self._preprocess(
            inputs_values=inputs_values,
            targets_values=targets_values,
            metadata=metadata,
            test_inputs_values=None,
            test_metadata=None,
            fitting=False,
        )

    def postprocess(self, targets_values_pred):
        postprocessed_targets_values_pred = targets_values_pred
        """
        if "targets_scaler" in self.preprocesses:
            postprocessed_targets_values_pred = self.preprocesses["targets_scaler"].inverse_transform(postprocessed_targets_values_pred)
        if "targets_decomposer" in self.preprocesses:
            postprocessed_targets_values_pred = self.preprocesses["targets_decomposer"].inverse_transform(postprocessed_targets_values_pred)
        if self.params["task_type"] == "multi" and self.params["postprocess_clip"]:
            postprocessed_targets_values_pred[postprocessed_targets_values_pred < 0.0] = 0.0
        """
        return postprocessed_targets_values_pred
