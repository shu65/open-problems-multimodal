import argparse
import functools
import gc
import json
import os
import pickle
import time

import git
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, ShuffleSplit

from ss_opm.metric.correlation_score import correlation_score
from ss_opm.model.encoder_decoder.encoder_decoder import EncoderDecoder
from ss_opm.pre_post_processing.pre_post_processing import PrePostProcessing
from ss_opm.utility.get_group_id import get_group_id
from ss_opm.utility.get_metadata_pattern import get_metadata_pattern
from ss_opm.utility.get_selector_with_metadata_pattern import get_selector_with_metadata_pattern
from ss_opm.utility.load_dataset import load_dataset
from ss_opm.utility.row_normalize import row_normalize
from ss_opm.utility.set_seed import set_seed


def get_params_default(trial):
    params = {}
    return params


def _build_model_default(params):
    model = Ridge(**params)
    return model


def _build_pre_post_process_default(params):
    pre_post_process = PrePostProcessing(params)
    return pre_post_process


class CrossVaridation(object):
    def __init__(self):
        pass

    def compute_score(
        self,
        x,
        y,
        metadata,
        x_test,
        metadata_test,
        params,
        build_model=_build_model_default,
        build_pre_post_process=_build_pre_post_process_default,
        dump=False,
        dump_dir="./",
        n_splits=3,
        n_bagging=0,
        bagging_ratio=1.0,
        use_batch_group=True,
    ):
        groups = None
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        if use_batch_group:
            batch_groups = get_group_id(metadata)
            n_groups = len(np.unique(batch_groups))
            if n_groups > n_splits:
                print(f"use batch group. # groups is {n_groups}")
                groups = batch_groups
                kf = GroupKFold(n_splits=n_splits)
        pre_post_processes = []
        models = []
        cell_types = metadata["cell_type"].unique()
        scores_dict = {
            "mse_train": [],
            "corrscore_train": [],
            "mse_val": [],
            "corrscore_val": [],
        }

        for cell_type_name in cell_types:
            scores_dict[f"{cell_type_name}_mse_train"] = []
            scores_dict[f"{cell_type_name}_corrscore_train"] = []
            scores_dict[f"{cell_type_name}_mse_val"] = []
            scores_dict[f"{cell_type_name}_corrscore_val"] = []
        print("kfold type", type(kf), "group", groups)
        for fold, (idx_tr, idx_va) in enumerate(kf.split(x, y, groups=groups)):
            if groups is not None:
                print("train group:", np.unique(groups[idx_tr]))
                print("val group:", np.unique(groups[idx_va]))
            start_time = time.time()
            model = None
            score = {}
            gc.collect()
            _n_bagging = n_bagging
            if _n_bagging == 0:
                _n_bagging = 1
            y_train_pred = None
            y_val_pred = None
            metadata_train = metadata.iloc[idx_tr, :]
            if "selected_metadata" in params["model"]:
                selector = get_selector_with_metadata_pattern(
                    metadata=metadata_train, metadata_pattern=params["model"]["selected_metadata"]
                )
                if np.sum(selector) == 0:
                    print("skip!")
                    continue
            x_train = x[idx_tr]  # creates a copy, https://numpy.org/doc/stable/user/basics.copies.html
            y_train = y[idx_tr].toarray()
            # We validate the model
            x_val = x[idx_va]
            y_val = y[idx_va].toarray()
            metadata_val = metadata.iloc[idx_va, :]
            for bagging_i in range(_n_bagging):
                gc.collect()
                if n_bagging > 0:
                    print("bagging_i", bagging_i, flush=True)
                    n_bagging_size = int(x_train.shape[0] * bagging_ratio)
                    bagging_idx = np.random.permutation(x_train.shape[0])[:n_bagging_size]
                    x_train_bagging = x_train[bagging_idx]
                    y_train_bagging = y_train[bagging_idx]
                    metadata_train_bagging = metadata_train.iloc[bagging_idx, :]
                else:
                    x_train_bagging = x_train
                    y_train_bagging = y_train
                    metadata_train_bagging = metadata_train

                pre_post_process = build_pre_post_process(params=params["pre_post_process"])
                if not pre_post_process.is_fitting:
                    pre_post_process.fit_preprocess(
                        inputs_values=x_train_bagging,
                        targets_values=y_train_bagging,
                        metadata=metadata_train_bagging,
                    )
                else:
                    print("skip pre_post_process fit")
                pre_post_processes.append(pre_post_process)
                preprocessed_x_train, preprocessed_y_train = pre_post_process.preprocess(
                    inputs_values=x_train_bagging,
                    targets_values=y_train_bagging,
                    metadata=metadata_train_bagging,
                )
                model = build_model(params=params["model"])
                print(f"model input shape X:{preprocessed_x_train.shape} Y:{preprocessed_y_train.shape}")
                model.fit(
                    x=x_train_bagging,
                    y=y_train_bagging,
                    preprocessed_x=preprocessed_x_train,
                    preprocessed_y=preprocessed_y_train,
                    metadata=metadata_train_bagging,
                    pre_post_process=pre_post_process,
                )
                preprocessed_y_train_pred = model.predict(x=x_train, preprocessed_x=preprocessed_x_train, metadata=metadata_train)
                new_y_train_pred = pre_post_process.postprocess(preprocessed_y_train_pred)
                preprocessed_x_val, _ = pre_post_process.preprocess(
                    inputs_values=x_val,
                    targets_values=None,
                    metadata=metadata_val,
                )
                preprocessed_y_val_pred = model.predict(x=x_val, preprocessed_x=preprocessed_x_val, metadata=metadata_val)
                new_y_val_pred = pre_post_process.postprocess(preprocessed_y_val_pred)
                models.append(model)

                mse_train = mean_squared_error(y_train, new_y_train_pred)
                corrscore_train = correlation_score(y_train, new_y_train_pred)
                mse_val = mean_squared_error(y_val, new_y_val_pred)
                corrscore_val = correlation_score(y_val, new_y_val_pred)
                print(
                    f"Fold {fold} bagging {bagging_i} "
                    f"mse_train: {mse_train: .5f} "
                    f"corrscore_train: {corrscore_train: .5f} "
                    f"mse_val: {mse_val: .5f} "
                    f"corrscore_val: {corrscore_val: .5f} "
                )

                if y_train_pred is None:
                    y_train_pred = new_y_train_pred
                else:
                    y_train_pred += new_y_train_pred
                if y_val_pred is None:
                    y_val_pred = new_y_val_pred
                else:
                    y_val_pred += new_y_val_pred

            y_train_pred /= _n_bagging
            y_val_pred /= _n_bagging

            mse_train = mean_squared_error(y_train, y_train_pred)
            corrscore_train = correlation_score(y_train, y_train_pred)
            score["mse_train"] = mse_train
            score["corrscore_train"] = corrscore_train
            cell_type_train = metadata_train["cell_type"].values
            for cell_type_name in cell_types:
                s = cell_type_train == cell_type_name
                if s.sum() > 10:
                    mse_train = mean_squared_error(y_train[s], y_train_pred[s])
                    corrscore_train = correlation_score(y_train[s], y_train_pred[s])
                    score[f"{cell_type_name}_mse_train"] = mse_train
                    score[f"{cell_type_name}_corrscore_train"] = corrscore_train
                else:
                    score[f"{cell_type_name}_mse_train"] = 0.0
                    score[f"{cell_type_name}_corrscore_train"] = 1.0
            gc.collect()

            mse_val = mean_squared_error(y_val, y_val_pred)
            corrscore_val = correlation_score(y_val, y_val_pred)
            score["mse_val"] = mse_val
            score["corrscore_val"] = corrscore_val
            cell_type_val = metadata_val["cell_type"].values
            for cell_type_name in cell_types:
                s = cell_type_val == cell_type_name
                if s.sum() > 10:
                    mse_val = mean_squared_error(y_val[s], y_val_pred[s])
                    corrscore_val = correlation_score(y_val[s], y_val_pred[s])
                    score[f"{cell_type_name}_mse_val"] = mse_val
                    score[f"{cell_type_name}_corrscore_val"] = corrscore_val
                else:
                    score[f"{cell_type_name}_mse_val"] = 0.0
                    score[f"{cell_type_name}_corrscore_val"] = 1.0
            if dump:
                np.save(os.path.join(dump_dir, f"k{fold}_y_val.npy"), y_val)
                np.save(os.path.join(dump_dir, f"k{fold}_y_val_pred.npy"), y_val_pred)

            del x_train, y_train, y_train_pred, x_val, y_val_pred, y_val
            print(f"Fold {fold}: score:{score} elapsed time = {time.time() - start_time: .3f}")
            for k, v in score.items():
                scores_dict[k].append(v)

        # Show overall score
        result_df = pd.DataFrame(scores_dict)
        return result_df, models, pre_post_processes


class Objective(object):
    def __init__(
        self,
        x,
        y,
        metadata,
        x_test,
        metadata_test,
        test_ratio=0.2,
        get_params=get_params_default,
        build_model=_build_model_default,
        build_pre_post_process=_build_pre_post_process_default,
    ):
        self.test_ratio = test_ratio
        splitter = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        train_index, val_index = next(splitter.split(x))

        self.x_train = x[train_index, :]
        self.x_val = x[val_index, :]
        self.y_train = y[train_index, :]
        self.y_val = y[val_index, :].toarray()
        self.metadata_train = metadata.iloc[train_index, :]
        self.metadata_val = metadata.iloc[val_index, :]
        self.get_params = get_params
        self.build_model = build_model
        self.build_pre_post_process = build_pre_post_process

    def __call__(self, trial):
        gc.collect()
        params = self.get_params(trial=trial)
        pre_post_process = self.build_pre_post_process(params=params["pre_post_process"])
        model = self.build_model(params=params["model"])
        if not pre_post_process.is_fitting:
            pre_post_process.fit_preprocess(inputs_values=self.x_train, targets_values=self.y_train)
        else:
            print("skip pre_post_process fit")
        preprocessed_inputs_values, preprocessed_targets_values = pre_post_process.preprocess(
            inputs_values=self.x_train,
            targets_values=self.y_train,
            metadata=self.metadata_train,
        )

        print(f"model input shape X:{preprocessed_inputs_values.shape} Y:{preprocessed_targets_values.shape}")
        model.fit(
            preprocessed_x=preprocessed_inputs_values,
            preprocessed_y=preprocessed_targets_values,
            x=self.x_train,
            y=self.y_train,
            metadata=self.metadata_train,
            pre_post_process=pre_post_process,
        )
        preprocessed_inputs_values, _ = pre_post_process.preprocess(
            inputs_values=self.x_val,
            targets_values=None,
            metadata=self.metadata_val,
        )
        preprocessed_y_pred_val = model.predict(
            preprocessed_x=preprocessed_inputs_values, x=self.x_val, metadata=self.metadata_val
        )
        y_val_pred = pre_post_process.postprocess(preprocessed_y_pred_val)
        # print("y_test_pred", y_test_pred)
        corrscore = correlation_score(self.y_val, y_val_pred)

        return corrscore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", metavar="PATH")
    parser.add_argument("--task_type", default="multi", choices=["multi", "cite"])
    parser.add_argument("--cell_type", default="all", choices=["all", "hsc", "eryp", "neup", "masp", "mkp", "bp", "mop"])
    parser.add_argument("--n_trials", type=int, default=0)
    parser.add_argument("--skip_test_prediction", action="store_true")
    parser.add_argument("--n_model_train_samples", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--distributed_study_name")
    parser.add_argument("--model", default="ead")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--param_path", metavar="PATH")
    parser.add_argument("--out_dir", metavar="PATH", default="result")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cv_dump", action="store_true")
    parser.add_argument("--cv_n_bagging", type=int, default=0)
    parser.add_argument("--use_k_fold_models", action="store_true")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--pre_post_process_tuning", action="store_true")
    parser.add_argument("--check_load_model", action="store_true")
    parser.add_argument("--metadata_pattern_id", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        git_hexsha = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True).head.commit.hexsha
    except Exception:
        git_hexsha = "Failed to get git"
    print("git_hexsha", git_hexsha)
    if args.metadata_pattern_id is not None:
        print("metadata_pattern_id", args.metadata_pattern_id)
        print(get_metadata_pattern(metadata_pattern_id=args.metadata_pattern_id))

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    cell_type_names = {
        "all": "all",
        "hsc": "HSC",
        "eryp": "EryP",
        "neup": "NeuP",
        "masp": "MasP",
        "mkp": "MkP",
        "bp": "BP",
        "mop": "MoP",
    }
    data_dir = args.data_dir
    train_inputs, train_metadata, train_target = load_dataset(
        data_dir=data_dir, task_type=args.task_type, split="train", cell_type=cell_type_names[args.cell_type]
    )
    test_inputs, test_metadata, _ = load_dataset(data_dir=data_dir, task_type=args.task_type, split="test")

    if args.model == "ead":
        model_class = EncoderDecoder
        pre_post_process_class = PrePostProcessing
    else:
        raise ValueError

    loaded_params = None
    if args.param_path is not None:
        with open(os.path.join(args.param_path)) as f:
            loaded_params = json.load(f)

    def get_params_core(trial, pre_post_process_tuning=False):
        if loaded_params is not None:
            return loaded_params
        model_params = model_class.get_params(
            task_type=args.task_type,
            device=args.device,
            trial=trial,
            debug=args.debug,
            snapshot=args.snapshot,
            metadata_pattern_id=args.metadata_pattern_id,
        )
        if pre_post_process_tuning:
            pre_post_process_params = pre_post_process_class.get_params(
                task_type=args.task_type,
                trial=trial,
                debug=args.debug,
                seed=args.seed,
            )
        else:
            pre_post_process_params = pre_post_process_class.get_params(
                task_type=args.task_type,
                data_dir=args.data_dir,
                trial=None,
                debug=args.debug,
                seed=args.seed,
            )
        return {"model": model_params, "pre_post_process": pre_post_process_params}

    get_params = functools.partial(get_params_core, pre_post_process_tuning=args.pre_post_process_tuning)

    params = get_params(trial=None)
    pre_post_process_default = None
    if not args.pre_post_process_tuning:
        pre_post_process_default = pre_post_process_class(params["pre_post_process"])
        pre_post_process_default.fit_preprocess(
            inputs_values=train_inputs,
            targets_values=train_target,
            metadata=train_metadata,
            test_inputs_values=test_inputs,
            test_metadata=test_metadata,
        )

    def build_pre_post_process(params):
        if pre_post_process_default is not None:
            return pre_post_process_default
        else:
            pre_post_process = pre_post_process_class(params)
            return pre_post_process

    def build_model(params):
        model = model_class(params)
        return model

    if args.n_model_train_samples > 0:
        np.random.seed(42)
        all_row_indices = np.arange(train_inputs.shape[0])
        np.random.shuffle(all_row_indices)
        n_model_train_samples = args.n_model_train_samples
        if n_model_train_samples > train_inputs.shape[0]:
            n_model_train_samples = train_inputs.shape[0]
        selected_rows_indices = all_row_indices[:n_model_train_samples]
        train_inputs = train_inputs[selected_rows_indices]
        train_metadata = train_metadata.iloc[selected_rows_indices]
        train_target = train_target[selected_rows_indices]
    print("train sample size:", train_inputs.shape[0])

    if args.n_trials > 0:
        study_name = None
        storage = "sqlite:///{}/optuna.db".format(args.out_dir)
        if args.distributed_study_name is not None:
            study_name = args.distributed_study_name
            storage = optuna.storages.RDBStorage(os.environ["OPTUNA_STORAGE"], {"pool_pre_ping": True})
        print("study_name", study_name)
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True, constant_liar=True),
            direction="maximize",
            # pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        objective = Objective(
            x=train_inputs,
            y=train_target,
            metadata=train_metadata,
            x_test=test_inputs,
            metadata_test=test_metadata,
            test_ratio=0.2,
            get_params=get_params,
            build_model=build_model,
            build_pre_post_process=build_pre_post_process,
        )
        while len(study.get_trials()) < args.n_trials:
            study.optimize(objective, n_trials=1)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        best_params = get_params(study.best_trial)
        print(json.dumps(best_params, indent=2))
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))
        params = best_params
        del objective

    print("dump params")
    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    cv = CrossVaridation()
    result_df, k_fold_models, k_fold_pre_post_processes = cv.compute_score(
        x=train_inputs,
        y=train_target,
        metadata=train_metadata,
        x_test=test_inputs,
        metadata_test=test_metadata,
        build_model=build_model,
        build_pre_post_process=build_pre_post_process,
        params=params,
        n_splits=args.n_splits,
        dump=args.cv_dump,
        dump_dir=args.out_dir,
        n_bagging=args.cv_n_bagging,
    )
    print("Average:", result_df.mean(), flush=True)
    del cv
    gc.collect()
    if not args.skip_test_prediction:
        if args.use_k_fold_models:
            print("use k fold models for test preds")
            start_time = time.time()
            y_test_pred = None
            print("pridict with test data", flush=True)
            for _, (model, pre_post_process) in enumerate(zip(k_fold_models, k_fold_pre_post_processes)):
                preprocessed_test_inputs, _ = pre_post_process.preprocess(
                    inputs_values=test_inputs, targets_values=None, metadata=test_metadata
                )
                preprocessed_y_test_pred = model.predict(
                    x=test_inputs, preprocessed_x=preprocessed_test_inputs, metadata=test_metadata
                )
                new_y_test_pred = pre_post_process.postprocess(preprocessed_y_test_pred)
                if y_test_pred is None:
                    y_test_pred = row_normalize(new_y_test_pred)
                else:
                    y_test_pred += row_normalize(new_y_test_pred)
                y_test_pred /= len(k_fold_models)
            print(f"elapsed time = {time.time() - start_time: .3f}")
        else:
            print("train model to predict with test data", flush=True)
            start_time = time.time()
            pre_post_process = build_pre_post_process(params["pre_post_process"])
            if not pre_post_process.is_fitting:
                pre_post_process.fit_preprocess(inputs_values=train_inputs, targets_values=train_target, metadata=train_metadata)
            else:
                print("skip pre_post_process fit")
            preprocessed_inputs_values, preprocessed_targets_values = pre_post_process.preprocess(
                inputs_values=train_inputs, targets_values=train_target, metadata=train_metadata
            )
            preprocessed_test_inputs, _ = pre_post_process.preprocess(
                inputs_values=test_inputs, targets_values=None, metadata=test_metadata
            )
            model = build_model(params=params["model"])
            model.fit(
                x=train_inputs,
                y=train_target,
                preprocessed_x=preprocessed_inputs_values,
                preprocessed_y=preprocessed_targets_values,
                metadata=train_metadata,
                pre_post_process=pre_post_process,
            )
            print(f"elapsed time = {time.time() - start_time: .3f}")
            print("pridict with test data", flush=True)
            start_time = time.time()
            preprocessed_y_test_pred = model.predict(
                x=test_inputs, preprocessed_x=preprocessed_test_inputs, metadata=test_metadata
            )
            y_test_pred = pre_post_process.postprocess(preprocessed_y_test_pred)
            print(f"elapsed time = {time.time() - start_time: .3f}")
            print("dump preprocess and model")
            model_dir = os.path.join(args.out_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "pre_post_process.pickle"), "wb") as f:
                pickle.dump(pre_post_process, f)
            model.save(model_dir)
            if args.check_load_model:
                with open(os.path.join(model_dir, "pre_post_process.pickle"), "rb") as f:
                    _ = pickle.load(f)
                loaded_model = build_model(params=None)
                loaded_model.load(model_dir)
        if args.cell_type != "all":
            s = test_metadata["cell_type"] == cell_type_names[args.cell_type]
            y_test_pred[~s] = np.nan
        print("save results")
        if args.task_type == "multi":
            pred_file_path = "multimodal_pred.pickle"
        elif args.task_type == "cite":
            pred_file_path = "citeseq_pred.pickle"
        else:
            raise ValueError
        with open(os.path.join(args.out_dir, pred_file_path), "wb") as f:
            pickle.dump(y_test_pred, f)
    print("completed !")


if __name__ == "__main__":
    main()
