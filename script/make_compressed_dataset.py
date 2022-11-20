# original notebook is https://www.kaggle.com/code/fabiencrom/multimodal-single-cell-creating-sparse-data/
import argparse
import os

import numpy as np
import pandas as pd
import scipy
import scipy.sparse


def convert_to_parquet(filename, out_filename):
    df = pd.read_csv(filename)
    df.to_parquet(out_filename + ".parquet")


def convert_h5_to_sparse_csr(filename, out_filename, chunksize=2500):
    start = 0
    total_rows = 0

    sparse_chunks_data_list = []
    chunks_index_list = []
    columns_name = None
    while True:
        df_chunk = pd.read_hdf(filename, start=start, stop=start + chunksize)
        if len(df_chunk) == 0:
            break
        chunk_data_as_sparse = scipy.sparse.csr_matrix(df_chunk.to_numpy())
        sparse_chunks_data_list.append(chunk_data_as_sparse)
        chunks_index_list.append(df_chunk.index.to_numpy())

        if columns_name is None:
            columns_name = df_chunk.columns.to_numpy()
        else:
            assert np.all(columns_name == df_chunk.columns.to_numpy())

        total_rows += len(df_chunk)
        print(total_rows)
        if len(df_chunk) < chunksize:
            del df_chunk
            break
        del df_chunk
        start += chunksize

    all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
    del sparse_chunks_data_list

    all_indices = np.hstack(chunks_index_list)

    scipy.sparse.save_npz(out_filename + "_values.sparse", all_data_sparse)
    np.savez(out_filename + "_idxcol.npz", index=all_indices, columns=columns_name)


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
    file_prefixes = ["evaluation_ids", "metadata", "sample_submission"]
    for file_prefix in file_prefixes:
        convert_to_parquet(os.path.join(data_dir, f"{file_prefix}.csv"), os.path.join(output_data_dir, file_prefix))
    file_prefixes = [
        "test_cite_inputs",
        "test_multi_inputs",
        "train_cite_inputs",
        "train_cite_targets",
        "train_multi_inputs",
        "train_multi_targets",
    ]
    for file_prefix in file_prefixes:
        convert_h5_to_sparse_csr(os.path.join(data_dir, f"{file_prefix}.h5"), os.path.join(output_data_dir, file_prefix))


if __name__ == "__main__":
    main()
