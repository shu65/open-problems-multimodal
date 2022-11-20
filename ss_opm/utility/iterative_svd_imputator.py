from sklearn.decomposition import TruncatedSVD

_default_svd_params = {
    "n_components": 128,
    "random_state": 42,
    "n_oversamples": 20,
    "n_iter": 7,
}


class IterativeSVDImputator(object):
    def __init__(self, svd_params=_default_svd_params, iters=2):
        self.missing_values = 0.0
        self.svd_params = svd_params
        self.iters = iters
        self.svd_decomposers = [None for _ in range(self.iters)]

    def fit(self, X):
        mask = X == self.missing_values
        transformed_X = X.copy()
        for i in range(self.iters):
            self.svd_decomposers[i] = TruncatedSVD(**self.svd_params)
            self.svd_decomposers[i].fit(transformed_X)
            new_X = self.svd_decomposers[i].inverse_transform(self.svd_decomposers[i].transform(transformed_X))
            transformed_X[mask] = new_X[mask]

    def transform(self, X):
        mask = X == self.missing_values
        transformed_X = X.copy()
        for i in range(self.iters):
            new_X = self.svd_decomposers[i].inverse_transform(self.svd_decomposers[i].transform(transformed_X))
            transformed_X[mask] = new_X[mask]
        return transformed_X
