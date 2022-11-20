import os

import numpy as np
import pandas as pd


def make_submission(data_dir, y_multi_pred, y_cite_pred):
    eval_ids = pd.read_parquet(os.path.join(data_dir, "evaluation_ids.parquet"))

    eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
    eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())

    submission = pd.Series(name="target", index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float32)

    y_columns = np.load(os.path.join(data_dir, "train_multi_targets_idxcol.npz"), allow_pickle=True)["columns"]
    test_index = np.load(os.path.join(data_dir, "test_multi_inputs_idxcol.npz"), allow_pickle=True)["index"]

    cell_dict = dict((k, v) for v, k in enumerate(test_index))
    assert len(cell_dict) == len(test_index)

    gene_dict = dict((k, v) for v, k in enumerate(y_columns))
    assert len(gene_dict) == len(y_columns)

    eval_ids_cell_num = eval_ids.cell_id.apply(lambda x: cell_dict.get(x, -1))
    eval_ids_gene_num = eval_ids.gene_id.apply(lambda x: gene_dict.get(x, -1))

    valid_multi_rows = (eval_ids_gene_num != -1) & (eval_ids_cell_num != -1)
    submission.iloc[valid_multi_rows] = y_multi_pred[
        eval_ids_cell_num[valid_multi_rows].to_numpy(), eval_ids_gene_num[valid_multi_rows].to_numpy()
    ]
    submission.reset_index(drop=True, inplace=True)
    submission.index.name = "row_id"

    submission.iloc[: len(y_cite_pred.ravel())] = y_cite_pred.ravel()
    assert len(submission) == 65744180
    assert not submission.isna().any()
    return submission
