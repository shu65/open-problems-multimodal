# open-problems-multimodal
1st place solution of Kaggle Open Problems - Multimodal Single-Cell Integration

## Preparation
Install the solution code.
```shell
pip3 install -e .
```

In addtion, download the following data
1. Open Problems - Multimodal Single-Cell Integration data set from Kaggle
2. tab separated hgnc_complete_set file from https://www.genenames.org/download/archive/
3. Reactome Pathways Gene Set from https://reactome.org/download-data

## Compress data and make addtitional data
compress kaggle dataset and make addtional data to use in training
```shell
export DATA_DIR=/path/to/kaggle/dataset/Directory
python3 script/make_compressed_dataset.py --data_dir ${DATA_DIR}
python3 script/make_additional_files.py --data_dir ${DATA_DIR}
python3 script/make_cite_input_mask.py --data_dir ${DATA_DIR} --hgnc_complete_set_path /path/to/hgnc_complete_set --reactome_pathways_path /path/to/reactome_pathways
```

## Training
### Multi
```shell
python3 scripts/train_mode.py --data_dir ${DATA_DIR} --task_type multi 
```

### Cite
```shell
python3 scripts/train_mode.py --data_dir ${DATA_DIR} --task_type cite 
```