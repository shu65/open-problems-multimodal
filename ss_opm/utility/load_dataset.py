import gc
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse

from ss_opm.utility.get_group_id import get_group_id


def load_dataset(data_dir, task_type, cell_type="all", split="train"):
    if (task_type == "multi") and (split == "train"):
        inputs_path = "train_multi_inputs_values.sparse.npz"
        inputs_idxcol_path = "train_multi_inputs_idxcol.npz"
        targets_path = "train_multi_targets_values.sparse.npz"
        cell_statistics_path = "normalized_multi_cell_statistics.parquet"
        batch_statistics_path = "normalized_multi_batch_statistics.parquet"
        batch_inputs_path = None
    elif (task_type == "multi") and (split == "test"):
        inputs_path = "test_multi_inputs_values.sparse.npz"
        inputs_idxcol_path = "test_multi_inputs_idxcol.npz"
        targets_path = None
        cell_statistics_path = "normalized_multi_cell_statistics.parquet"
        batch_statistics_path = "normalized_multi_batch_statistics.parquet"
        batch_inputs_path = None
    elif (task_type == "cite") and (split == "train"):
        inputs_path = "train_cite_inputs_values.sparse.npz"
        inputs_idxcol_path = "train_cite_inputs_idxcol.npz"
        targets_path = "train_cite_targets_values.sparse.npz"
        cell_statistics_path = "normalized_cite_cell_statistics.parquet"
        batch_statistics_path = "normalized_cite_batch_statistics.parquet"
        batch_inputs_path = "normalized_cite_batch_inputs.parquet"
    elif (task_type == "cite") and (split == "test"):
        inputs_path = "test_cite_inputs_values.sparse.npz"
        inputs_idxcol_path = "test_cite_inputs_idxcol.npz"
        targets_path = None
        cell_statistics_path = "normalized_cite_cell_statistics.parquet"
        batch_statistics_path = "normalized_cite_batch_statistics.parquet"
        batch_inputs_path = "normalized_cite_batch_inputs.parquet"
    else:
        assert task_type == "multi"
        assert split == "train"
        raise ValueError(f"invalid task type or split. {task_type}, {split}")
    input_index = np.load(os.path.join(data_dir, inputs_idxcol_path), allow_pickle=True)["index"]
    metadata_df = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
    metadata_df = metadata_df.set_index("cell_id")
    metadata_df = metadata_df.loc[input_index, :]
    cell_statistics_df = pd.read_parquet(os.path.join(data_dir, cell_statistics_path))
    metadata_df = pd.merge(metadata_df, cell_statistics_df, left_index=True, right_index=True)
    group_ids = get_group_id(metadata_df)
    metadata_df["group"] = group_ids
    # print("before", metadata_df["group"].unique())
    if batch_statistics_path is not None:
        batch_statistics_df = pd.read_parquet(os.path.join(data_dir, batch_statistics_path))
        metadata_df = pd.merge(metadata_df, batch_statistics_df, left_on="group", right_index=True)
    # print("after", metadata_df["group"].unique())
    if batch_inputs_path is not None:
        batch_inputs_df = pd.read_parquet(os.path.join(data_dir, batch_inputs_path))
        metadata_df = pd.merge(metadata_df, batch_inputs_df, left_on="group", right_index=True)
    assert len(metadata_df) == len(input_index)
    print("load input values")
    start_time = time.time()
    train_inputs = scipy.sparse.load_npz(os.path.join(data_dir, inputs_path))
    elapsed_time = time.time() - start_time
    print(f"completed loading input values. elapsed time:{elapsed_time: .1f}")
    if targets_path is not None:
        print("load targets values")
        start_time = time.time()
        train_target = scipy.sparse.load_npz(os.path.join(data_dir, targets_path))
        elapsed_time = time.time() - start_time
        print(f"completed loading targets values. elapsed time:{elapsed_time: .1f}")
    else:
        train_target = None
    gc.collect()
    if cell_type != "all":
        s = metadata_df["cell_type"] == cell_type
        train_inputs = train_inputs[s]
        train_target = train_target[s]
        metadata_df = metadata_df[s]
    return train_inputs, metadata_df, train_target
