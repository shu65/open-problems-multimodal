
class DummyPrePostProcessing(object):

    def __init__(self, task_type):
        self.task_type = task_type
        self.preprocesses = {}

    def _preprocess(self, inputs_values, targets_values, fitting):
        return inputs_values, targets_values

    def fit_preprocess(self, inputs_values, targets_values):
        self._preprocess(inputs_values=inputs_values, targets_values=targets_values, fitting=True)
        return self

    def preprocess(self, inputs_values, targets_values):
        return self._preprocess(inputs_values=inputs_values, targets_values=targets_values, fitting=False)

    def postprocess(self, targets_values_pred):
        return targets_values_pred