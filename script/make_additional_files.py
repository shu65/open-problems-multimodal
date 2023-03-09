import argparse
import os

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from sklearn.decomposition import TruncatedSVD

from ss_opm.utility.get_group_id import get_group_id
from ss_opm.utility.nonzero_median_normalize import median_normalize


def make_cite_cell_statistics(data_dir, output_data_dir):
    cite_train_values = scipy.sparse.load_npz(os.path.join(data_dir, "train_cite_inputs_values.sparse.npz"))
    cite_train_input_index = np.load(os.path.join(data_dir, "train_cite_inputs_idxcol.npz"), allow_pickle=True)["index"]
    cite_test_values = scipy.sparse.load_npz(os.path.join(data_dir, "test_cite_inputs_values.sparse.npz"))
    cite_test_input_index = np.load(os.path.join(data_dir, "test_cite_inputs_idxcol.npz"), allow_pickle=True)["index"]
    cite_values = scipy.sparse.vstack((cite_train_values, cite_test_values))
    cite_input_index = np.hstack((cite_train_input_index, cite_test_input_index))

    nonzero_ratios = np.empty(cite_input_index.shape[0], dtype=float)
    nonzero_q25 = np.empty(cite_input_index.shape[0], dtype=float)
    nonzero_q50 = np.empty(cite_input_index.shape[0], dtype=float)
    nonzero_q75 = np.empty(cite_input_index.shape[0], dtype=float)
    means = np.empty(cite_input_index.shape[0], dtype=float)
    stds = np.empty(cite_input_index.shape[0], dtype=float)

    for i in range(len(cite_input_index)):
        row_nonzero_values = cite_values.data[cite_values.indptr[i] : cite_values.indptr[i + 1]]
        nonzero_ratios[i] = len(row_nonzero_values) / cite_values.shape[1]
        q_values = np.quantile(row_nonzero_values, q=[0.25, 0.5, 0.75])
        nonzero_q25[i] = q_values[0]
        nonzero_q50[i] = q_values[1]
        nonzero_q75[i] = q_values[2]
        row_values = cite_values[i, :].toarray()
        means[i] = np.mean(row_values)
        stds[i] = np.std(row_values)
        # break

    cell_statistics = pd.DataFrame(
        {
            "nonzero_ratio": nonzero_ratios,
            "nonzero_q25": nonzero_q25,
            "nonzero_q50": nonzero_q50,
            "nonzero_q75": nonzero_q75,
            "mean": means,
            "std": stds,
        },
        index=cite_input_index,
    )

    normalized_cell_statistics = cell_statistics.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    out_filename = os.path.join(output_data_dir, "cite_cell_statistics.parquet")
    cell_statistics.to_parquet(out_filename)

    out_filename = os.path.join(output_data_dir, "normalized_cite_cell_statistics.parquet")
    normalized_cell_statistics.to_parquet(out_filename)


def make_multi_cell_statistics(data_dir, output_data_dir):
    multi_train_values = scipy.sparse.load_npz(os.path.join(data_dir, "train_multi_inputs_values.sparse.npz"))
    multi_train_input_index_column = np.load(os.path.join(data_dir, "train_multi_inputs_idxcol.npz"), allow_pickle=True)
    multi_train_input_index = multi_train_input_index_column["index"]
    multi_input_columns = multi_train_input_index_column["columns"]

    multi_test_values = scipy.sparse.load_npz(os.path.join(data_dir, "test_multi_inputs_values.sparse.npz"))
    multi_test_input_index = np.load(os.path.join(data_dir, "test_multi_inputs_idxcol.npz"), allow_pickle=True)["index"]
    multi_values = scipy.sparse.vstack((multi_train_values, multi_test_values))
    multi_input_index = np.hstack((multi_train_input_index, multi_test_input_index))

    ch_masks = {}

    for i in range(len(multi_input_columns)):
        ch_name = multi_input_columns[i][0 : multi_input_columns[i].find(":")]
        if ch_name not in ch_masks:
            ch_masks[ch_name] = np.zeros(len(multi_input_columns), dtype=bool)
        ch_masks[ch_name][i] = True
    np.unique(ch_masks.keys())

    nonzero_ratios = np.empty(multi_input_index.shape[0], dtype=float)
    nonzero_q25 = np.empty(multi_input_index.shape[0], dtype=float)
    nonzero_q50 = np.empty(multi_input_index.shape[0], dtype=float)
    nonzero_q75 = np.empty(multi_input_index.shape[0], dtype=float)
    means = np.empty(multi_input_index.shape[0], dtype=float)
    stds = np.empty(multi_input_index.shape[0], dtype=float)

    ch_stats = {}
    for ch_name in ch_masks.keys():
        ch_stats["ch_nonzero_ratio_" + ch_name] = np.empty(multi_input_index.shape[0], dtype=float)

    for i in range(len(multi_input_index)):
        row_nonzero_values = multi_values.data[multi_values.indptr[i] : multi_values.indptr[i + 1]]
        nonzero_ratios[i] = np.log1p(len(row_nonzero_values) / multi_values.shape[1])
        q_values = np.quantile(row_nonzero_values, q=[0.25, 0.5, 0.75])
        nonzero_q25[i] = q_values[0]
        nonzero_q50[i] = q_values[1]
        nonzero_q75[i] = q_values[2]
        row_values = multi_values[i, :].toarray().ravel()
        means[i] = np.mean(row_values)
        stds[i] = np.std(row_values)
        for ch_name, ch_mask in ch_masks.items():
            nonzero_counts_in_ch = (row_values[ch_mask] > 0).sum()
            ch_counts = ch_mask.sum()
            ch_stats["ch_nonzero_ratio_" + ch_name][i] = np.log1p(nonzero_counts_in_ch / ch_counts)
        # break

    cell_statistics_dict = {
        "nonzero_ratio": nonzero_ratios,
        "nonzero_q25": nonzero_q25,
        "nonzero_q50": nonzero_q50,
        "nonzero_q75": nonzero_q75,
        "mean": means,
        "std": stds,
    }

    for k, v in ch_stats.items():
        cell_statistics_dict[k] = v

    cell_statistics = pd.DataFrame(cell_statistics_dict, index=multi_input_index)
    normalized_cell_statistics = cell_statistics.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    out_filename = os.path.join(data_dir, "multi_cell_statistics.parquet")
    cell_statistics.to_parquet(out_filename)

    out_filename = os.path.join(data_dir, "normalized_multi_cell_statistics.parquet")
    normalized_cell_statistics.to_parquet(out_filename)


def make_multi_batch_statistics(metadata, output_data_dir):
    metadata = metadata[metadata["technology"] == "multiome"]
    unique_group_ids = metadata["group"].unique()

    df_list = []
    for group_id in unique_group_ids:
        selector = metadata["group"] == group_id
        cell_type_counts = metadata[selector]["cell_type"].value_counts()
        sum_cell_type_counts = cell_type_counts.sum()
        row = cell_type_counts / sum_cell_type_counts
        row["cell_count"] = sum_cell_type_counts
        df_list.append(row)

    batch_statistics = pd.DataFrame(df_list, index=unique_group_ids)
    batch_statistics[batch_statistics.isnull()] = 0.0
    batch_statistics.index.name = "group"
    columns = []
    for c in batch_statistics.columns:
        if c in ["cell_count"]:
            columns.append("cell_count")
        else:
            columns.append("cell_ratio_" + c)
    batch_statistics.columns = columns
    normalized_batch_statistics = batch_statistics.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    out_filename = os.path.join(output_data_dir, "multi_batch_statistics.parquet")
    batch_statistics.to_parquet(out_filename)

    out_filename = os.path.join(output_data_dir, "normalized_multi_batch_statistics.parquet")
    normalized_batch_statistics.to_parquet(out_filename)


def make_cite_batch_statistics(metadata, output_data_dir):
    metadata = metadata[metadata["technology"] == "citeseq"]
    unique_group_ids = metadata["group"].unique()

    df_list = []
    for group_id in unique_group_ids:
        selector = metadata["group"] == group_id
        cell_type_counts = metadata[selector]["cell_type"].value_counts()
        sum_cell_type_counts = cell_type_counts.sum()
        row = cell_type_counts / sum_cell_type_counts
        row["cell_count"] = sum_cell_type_counts
        df_list.append(row)

    batch_statistics = pd.DataFrame(df_list, index=unique_group_ids)
    batch_statistics[batch_statistics.isnull()] = 0.0
    batch_statistics.index.name = "group"
    columns = []
    for c in batch_statistics.columns:
        if c in ["cell_count"]:
            continue
        columns.append("cell_ratio_" + c)

    columns.append("cell_count")
    batch_statistics.columns = columns

    normalized_batch_statistics = batch_statistics.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    out_filename = os.path.join(output_data_dir, "cite_batch_statistics.parquet")
    batch_statistics.to_parquet(out_filename)

    out_filename = os.path.join(output_data_dir, "normalized_cite_batch_statistics.parquet")
    normalized_batch_statistics.to_parquet(out_filename)


def make_cite_batch_inputs_median(data_dir, metadata, output_data_dir):
    train_values = scipy.sparse.load_npz(os.path.join(data_dir, "train_cite_inputs_values.sparse.npz")).toarray()
    test_values = scipy.sparse.load_npz(os.path.join(data_dir, "test_cite_inputs_values.sparse.npz")).toarray()

    values = np.vstack((train_values, test_values))
    median_norm_values = np.log1p(median_normalize(np.expm1(values)))

    train_inputs_index = np.load(os.path.join(data_dir, "train_cite_inputs_idxcol.npz"), allow_pickle=True)["index"]
    test_inputs_index = np.load(os.path.join(data_dir, "test_cite_inputs_idxcol.npz"), allow_pickle=True)["index"]
    inputs_index = np.hstack((train_inputs_index, test_inputs_index))

    metadata = metadata.loc[inputs_index, :]
    group_ids = metadata["group"].values
    unique_group_ids = np.unique(group_ids)

    median_norm_values_batch = []

    for group_id in unique_group_ids:
        selected_index = metadata["group"] == group_id
        selected_values = median_norm_values[selected_index, :].copy()
        selected_values[selected_values == 0.0] = np.nan
        selected_values = np.nanmedian(selected_values, axis=0)
        median_norm_values_batch.append(selected_values)
        # break

    median_norm_values_batch = np.vstack(median_norm_values_batch)
    median_norm_values_batch[np.isnan(median_norm_values_batch)] = 0.0

    params = {
        "n_components": 8,
        "random_state": 42,
    }
    decomposer = TruncatedSVD(**params)
    decomposer.fit(median_norm_values_batch)
    transformed_values = decomposer.transform(median_norm_values_batch)
    columns = [f"batch_sv{i}" for i in range(transformed_values.shape[1])]
    df = pd.DataFrame(transformed_values, index=unique_group_ids, columns=columns)
    df.index.name = "group"
    normalized_df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    out_filename = os.path.join(output_data_dir, "cite_batch_inputs.parquet")
    df.to_parquet(out_filename)

    out_filename = os.path.join(output_data_dir, "normalized_cite_batch_inputs.parquet")
    normalized_df.to_parquet(out_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", metavar="PATH")
    parser.add_argument("--output_data_dir", metavar="PATH")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_data_dir = args.output_data_dir
    if output_data_dir is None:
        output_data_dir = data_dir

    os.makedirs(output_data_dir, exist_ok=True)
    metadata = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
    metadata = metadata.set_index("cell_id")
    group_ids = get_group_id(metadata)
    metadata["group"] = group_ids

    make_multi_cell_statistics(data_dir=data_dir, output_data_dir=output_data_dir)
    make_cite_cell_statistics(data_dir=data_dir, output_data_dir=output_data_dir)

    make_multi_batch_statistics(metadata=metadata, output_data_dir=output_data_dir)
    make_cite_batch_statistics(metadata=metadata, output_data_dir=output_data_dir)

    make_cite_batch_inputs_median(data_dir=data_dir, metadata=metadata, output_data_dir=output_data_dir)


if __name__ == "__main__":
    main()
