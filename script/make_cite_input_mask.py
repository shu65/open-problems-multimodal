import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import scipy.stats

from ss_opm.utility.get_group_id import get_group_id
from ss_opm.utility.nonzero_median_normalize import median_normalize
from ss_opm.utility.row_normalize import row_normalize


def read_reactome_gmt(file_path):
    data_dict_list = []
    with open(file_path) as f:
        for line in f:
            values = line.strip().split("\t")
            st_id = values[1]
            genes = values[3:]
            for gene in genes:
                data_dict_list.append({"st_id": st_id, "gene": gene})
    df = pd.DataFrame(data_dict_list)
    return df


def make_inputs_gene_names(inputs_columns):
    inputs_gene_names = []
    for c in inputs_columns:
        gene_name = c[c.find("_") + 1 :]
        inputs_gene_names.append(gene_name)
    inputs_gene_names = np.array(inputs_gene_names)
    return inputs_gene_names


def make_targets_gene2idx(targets_columns, hgnc_data):
    alias_symbols = {}
    for _, row in hgnc_data.iterrows():
        symbol = row["symbol"].upper()
        values = []
        if not pd.isnull(row["alias_symbol"]):
            values.append(row["alias_symbol"].upper())
        if not pd.isnull(row["prev_symbol"]):
            values.append(row["prev_symbol"].upper())
        for value in values:
            end = value.find("|")
            while end > 0:
                # print(symbol, value[:end])
                alias_symbol = value[:end]
                if alias_symbol not in alias_symbols:
                    alias_symbols[alias_symbol] = []
                alias_symbols[alias_symbol].append(symbol)
                value = value[end + 1 :]
                end = value.find("|")
            # print(symbol, value[:])
            alias_symbol = value[:]
            if alias_symbol not in alias_symbols:
                alias_symbols[alias_symbol] = []
            alias_symbols[alias_symbol].append(symbol)

    alias_symbols["CD3"] = ["CD3D", "CD3E", "CD3G"]
    alias_symbols["HLA-A-B-C"] = ["HLA-A", "HLA-B", "HLA-C"]
    alias_symbols["CD45RA"] = ["PTPRC"]
    alias_symbols["CD45RO"] = ["PTPRC"]
    alias_symbols["PODOPLANIN"] = ["PDPN"]
    alias_symbols["HLA-DR"] = [
        "HLA-DRA",
        "HLA-DRB1",
        "HLA-DRB2",
        "HLA-DRB3",
        "HLA-DRB4",
        "HLA-DRB5",
        "HLA-DRB6",
        "HLA-DRB7",
        "HLA-DRB8",
        "HLA-DRB9",
    ]
    alias_symbols["INTEGRINB7"] = ["ITGB7"]
    alias_symbols["CD158"] = ["CD158A"]
    alias_symbols["CD158B"] = ["CD158B1", "CD158B2"]

    targets_gene2idx = {}
    for target_i, targets_column in enumerate(targets_columns):
        targets_column = targets_column.upper()
        targets_gene2idx[targets_column] = target_i
        if targets_column in alias_symbols:
            for symbol in alias_symbols[targets_column]:
                targets_gene2idx[symbol] = target_i
    return targets_gene2idx


def make_cite_inputs_targets_pair(inputs_values, targets_values, inputs_gene_names, metadata, targets_gene2idx, out_data_dir):

    unique_group_ids = metadata["group"].unique()
    out = np.zeros((len(unique_group_ids), inputs_values.shape[1], targets_values.shape[1]), dtype=float)

    for group_i, group_id in enumerate(unique_group_ids):
        group_selector = metadata["group"] == group_id
        selected_inputs_values = inputs_values[group_selector, :]
        selected_targets_values = targets_values[group_selector, :]
        print("group", group_i, group_id)
        for input_index in range(inputs_values.shape[1]):
            inputs_gene_name = inputs_gene_names[input_index].upper()
            if inputs_gene_name in targets_gene2idx:
                target_index = targets_gene2idx[inputs_gene_name]
                v1 = selected_inputs_values[:, input_index]
                v2 = selected_targets_values[:, target_index]
                s = v1 > 0.0
                v1 = v1[s]
                v2 = v2[s]
                corr, pvalue = scipy.stats.spearmanr(v1, v2)
                if (np.abs(corr) > 0.10) and (pvalue < 1e-2):
                    # if (np.abs(corr) > 0.10) and (pvalue < 1e-3):
                    # if (np.abs(corr) > 0.10) and (pvalue < 1e-10):
                    # print(inputs_columns[input_index], targets_columns[target_index], "corr", corr, "pvalue", pvalue)
                    out[group_i, input_index, target_index] = np.abs(corr)

    all_selected = (out > 0).sum(axis=0) > 0.6 * len(unique_group_ids)
    median_corrs = np.median(out, axis=0)
    median_corrs[~all_selected] = 0.0
    mask = np.zeros((inputs_values.shape[1], targets_values.shape[1]), dtype=bool)

    for target_index in range(targets_values.shape[1]):
        median_corr = np.max(median_corrs[:, target_index])
        input_index = np.argmax(median_corrs[:, target_index])
        if median_corr > 0.0:
            mask[input_index, target_index] = True

    np.savez(os.path.join(out_data_dir, "cite_inputs_targets_pair3g"), mask=mask)


def make_cite_inputs_targets_pair_pathway(
    inputs_values, targets_values, inputs_gene_names, metadata, targets_gene2idx, pathway_gene_df, out_data_dir
):
    st_ids = pathway_gene_df["st_id"].unique()
    # pathway_genes = pathway_gene_df["gene"].unique()

    pathway_genes = {}
    for st_id in st_ids:
        s = pathway_gene_df["st_id"] == st_id
        genes = pathway_gene_df[s]["gene"]
        genes = [g.upper() for g in genes]
        selected_genes = []
        for g in genes:
            if g in targets_gene2idx:
                selected_genes.append(g)
        if len(selected_genes) == 0:
            continue
        for g in genes:
            if g not in pathway_genes:
                pathway_genes[g] = []
            pathway_genes[g].extend(selected_genes)
        # break
    unique_pathway_genes = {}
    for k, v in pathway_genes.items():
        unique_pathway_genes[k] = np.array(list(set(v)))
    unique_group_ids = metadata["group"].unique()

    mask = np.zeros((len(unique_group_ids), inputs_values.shape[1], targets_values.shape[1]), dtype=float)

    start_time = time.time()
    for i, group_id in enumerate(unique_group_ids):
        group_selector = metadata["group"] == group_id
        selected_inputs_values = inputs_values[group_selector, :]
        selected_targets_values = targets_values[group_selector, :]
        for input_index in range(inputs_values.shape[1]):
            if (input_index % 1000) == 0:
                print(
                    f"{i} {input_index} / {inputs_values.shape[1]} "
                    f"({input_index/(inputs_values.shape[1]): .4f}) "
                    f"completed. elapsed time: {time.time() - start_time}"
                )
            inputs_gene_name = inputs_gene_names[input_index].upper()
            if inputs_gene_name in unique_pathway_genes:
                pathway_genes = unique_pathway_genes[inputs_gene_name]
                # print("pathway_genes", pathway_genes)
                for pathway_gene in pathway_genes:
                    target_index = targets_gene2idx[pathway_gene]
                    # print(input_index, target_index)
                    v1 = selected_inputs_values[:, input_index]
                    v2 = selected_targets_values[:, target_index]
                    s = v1 > 0.0
                    v1 = v1[s]
                    v2 = v2[s]
                    corr, pvalue = scipy.stats.spearmanr(v1, v2)
                    if (np.abs(corr) > 0.20) and (pvalue < 1e-3):
                        # print(inputs_columns[input_index], pathway_gene, targets_columns[target_index], "corr", corr, "pvalue", pvalue)
                        mask[i, input_index, target_index] = np.abs(corr)

    all_selected = (mask > 0).sum(axis=0) > 0.6 * len(unique_group_ids)
    median_pvalues = np.median(mask, axis=0)
    median_pvalues[~all_selected] = 0.0
    inputs_mask = np.zeros(inputs_values.shape[1], dtype=bool)

    select_n = 3
    for target_index in range(targets_values.shape[1]):
        input_indexes = np.argsort(-median_pvalues[:, target_index])[:select_n]
        selected_median_pvalues = median_pvalues[input_indexes, target_index]
        for i in range(len(input_indexes)):
            input_index = input_indexes[i]
            median_pvalue = selected_median_pvalues[i]
            if median_pvalue > 0.0:
                inputs_mask[input_index] = True
    np.savez(os.path.join(out_data_dir, "cite_inputs_mask2"), mask=inputs_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", metavar="PATH")
    parser.add_argument("--output_data_dir", metavar="PATH")
    parser.add_argument("--hgnc_complete_set_path", metavar="PATH", help="path of hgnc_complete_set.txt")
    parser.add_argument("--reactome_pathways_path", metavar="PATH", help="path of ReactomePathways.gmt")

    args = parser.parse_args()

    data_dir = args.data_dir
    output_data_dir = args.output_data_dir
    if output_data_dir is None:
        output_data_dir = data_dir

    inputs_idxcol_path = "train_cite_inputs_idxcol.npz"
    targets_idxcol_path = "train_cite_targets_idxcol.npz"

    inputs_columns = np.load(os.path.join(data_dir, inputs_idxcol_path), allow_pickle=True)["columns"]
    targets_columns = np.load(os.path.join(data_dir, targets_idxcol_path), allow_pickle=True)["columns"]
    inputs_values = scipy.sparse.load_npz(os.path.join(data_dir, "train_cite_inputs_values.sparse.npz"))
    inputs_values = np.log1p(median_normalize(np.expm1(inputs_values.toarray())))

    targets_values = scipy.sparse.load_npz(os.path.join(data_dir, "train_cite_targets_values.sparse.npz"))
    targets_values = targets_values.toarray()
    targets_values = row_normalize(targets_values)

    inputs_index = np.load(os.path.join(data_dir, inputs_idxcol_path), allow_pickle=True)["index"]

    metadata = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
    metadata = metadata.set_index("cell_id")
    metadata = metadata.loc[inputs_index, :]
    group_ids = get_group_id(metadata)
    metadata["group"] = group_ids
    inputs_gene_names = make_inputs_gene_names(inputs_columns=inputs_columns)
    hgnc_data = pd.read_table(args.hgnc_complete_set_path)
    targets_gene2idx = make_targets_gene2idx(targets_columns=targets_columns, hgnc_data=hgnc_data)
    pathway_gene_df = read_reactome_gmt(args.reactome_pathways_path)
    os.makedirs(output_data_dir, exist_ok=True)
    make_cite_inputs_targets_pair(
        inputs_values=inputs_values,
        targets_values=targets_values,
        inputs_gene_names=inputs_gene_names,
        targets_gene2idx=targets_gene2idx,
        metadata=metadata,
        out_data_dir=output_data_dir,
    )
    make_cite_inputs_targets_pair_pathway(
        inputs_values=inputs_values,
        targets_values=targets_values,
        inputs_gene_names=inputs_gene_names,
        targets_gene2idx=targets_gene2idx,
        metadata=metadata,
        pathway_gene_df=pathway_gene_df,
        out_data_dir=output_data_dir,
    )


if __name__ == "__main__":
    main()
