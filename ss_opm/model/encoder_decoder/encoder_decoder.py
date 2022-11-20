import gc
import json
import os
import pickle
import time

import numpy as np
import torch

from ss_opm.model.encoder_decoder.cite_encoder_decoder_module import CiteEncoderDecoderModule
from ss_opm.model.encoder_decoder.mlp_module import HierarchicalMLPBModule, MLPBModule
from ss_opm.model.encoder_decoder.multi_encoder_decoder_module import MultiEncoderDecoderModule
from ss_opm.model.torch_dataset.citeseq_dataset import CITEseqDataset
from ss_opm.model.torch_dataset.multiome_dataset import MultiomeDataset
from ss_opm.model.torch_helper.set_weight_decay import set_weight_decay
from ss_opm.utility.get_metadata_pattern import get_metadata_pattern
from ss_opm.utility.summeary_torch_model_parameters import summeary_torch_model_parameters


class EncoderDecoder(object):
    @staticmethod
    def get_params(task_type, device="cpu", trial=None, debug=False, snapshot=None, metadata_pattern_id=None):
        params = {
            "device": device,
            "snapshot": snapshot,
            "train_batch_size": 64,
            "test_batch_size": 16,
            "task_type": task_type,
            "lr": 1e-3,
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "epoch": 40,
            "pct_start": 0.3,
            "burnin_length_epoch": 10,
            "backbone": "mlp",
            "max_inputs_values_noisze_sigma": 0.0,
            "max_cutout_p": 0.0,
        }
        if params["backbone"] == "mlp":
            backbone_params = {
                "encoder_h_dim": 2048,  # 128,
                "decoder_h_dim": 2048,  # 128,
                "encoder_dropout_p": 0.0,
                "decoder_dropout_p": 0.0,
                "n_encoder_block": 1,
                "n_decoder_block": 5,
                "norm": "layer_nome",
                "activation": "gelu",  # relu, "gelu"
                # "norm": "batch_norm",
                "skip": False,
            }
        else:
            raise RuntimeError
        params.update(backbone_params)

        task_specific_params = {}
        if task_type == "multi":
            task_specific_params["lr"] = 9.97545796487608e-05
            task_specific_params["eps"] = 1.8042413185663546e-09
            task_specific_params["weight_decay"] = 1.7173609280566294e-07
            task_specific_params["encoder_dropout_p"] = 0.4195254015709299
            task_specific_params["decoder_dropout_p"] = 0.30449413021670935
        elif task_type == "cite":
            task_specific_params["lr"] = 0.00012520653814999459
            task_specific_params["eps"] = 7.257005721594269e-08
            task_specific_params["weight_decay"] = 2.576638574613591e-06
            task_specific_params["encoder_dropout_p"] = 0.5952997562668841
            task_specific_params["decoder_dropout_p"] = 0.31846059114042935
        params.update(task_specific_params)
        if snapshot is not None:
            specific_params = {
                "lr": 5e-5,
                "epoch": 25,
            }
            params.update(specific_params)

        if metadata_pattern_id is not None:
            params["selected_metadata"] = get_metadata_pattern(metadata_pattern_id=metadata_pattern_id)
        if trial is not None:
            # params['train_batch_size'] = trial.suggest_categorical('train_batch_size', [64,])
            # params['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
            # params['eps'] = trial.suggest_float('eps', 1e-9, 1e-5, log=True)
            # params['weight_decay'] = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)
            # params['max_inputs_values_noisze_sigma'] = trial.suggest_float('max_inputs_values_noisze_sigma', 1e-3, 1e-0, log=True)
            # params['max_cutout_p'] = trial.suggest_float('max_cutout_p', 0.05, 0.3)
            if params["backbone"] == "mlp":
                # params['encoder_h_dim'] = trial.suggest_int('encoder_h_dim', 1024, 2048)
                # params['decoder_h_dim'] = trial.suggest_int('decoder_h_dim', 128, 2048)
                # params['encoder_dropout_p'] = trial.suggest_float('encoder_dropout_p', 0.3, 0.7)
                # params['decoder_dropout_p'] = trial.suggest_float('decoder_dropout_p', 0.1, 0.4)
                # params['n_encoder_block'] = trial.suggest_int('n_encoder_block', 1, 3)
                # params['n_decoder_block'] = trial.suggest_int('n_decoder_block', 1, 5)
                # params['skip'] = trial.suggest_categorical('skip', [False, True])
                pass
            else:
                raise RuntimeError()
        if debug:
            params["epoch"] = 10
            params["encoder_h_dim"] = 128
            params["decoder_h_dim"] = 128
            # pass
        return params

    def __init__(self, params):
        self.params = params
        self.inputs_info = {}
        self.model = None

    def _build_model(self):
        if self.params["snapshot"] is not None:
            print(f"load model from {self.params['snapshot']}")
            model = torch.load(os.path.join(self.params["snapshot"], "model.pt"))
            return model
        x_dim = self.inputs_info["x_dim"]
        y_dim = self.inputs_info["y_dim"]
        inputs_decomposer_components = torch.tensor(self.inputs_info["inputs_decomposer_components"])
        targets_decomposer_components = torch.tensor(self.inputs_info["targets_decomposer_components"])
        y_statistic = {}
        for k, v in self.inputs_info["y_statistic"].items():
            y_statistic[k] = torch.tensor(v)
        if self.params["backbone"] == "mlp":
            encoder = MLPBModule(
                # input_dim=x_dim,
                input_dim=None,
                output_dim=self.params["encoder_h_dim"],
                n_block=self.params["n_encoder_block"],
                h_dim=self.params["encoder_h_dim"],
                skip=self.params["skip"],
                dropout_p=self.params["encoder_dropout_p"],
                activation=self.params["activation"],
                norm=self.params["norm"],
            )

            decoder = HierarchicalMLPBModule(
                input_dim=self.params["encoder_h_dim"],
                # output_dim=y_dim,
                # output_dim=y_dim,
                output_dim=None,
                n_block=self.params["n_decoder_block"],
                h_dim=self.params["decoder_h_dim"],
                skip=self.params["skip"],
                dropout_p=self.params["decoder_dropout_p"],
                activation=self.params["activation"],
                norm=self.params["norm"],
            )
        else:
            raise RuntimeError

        if self.params["task_type"] == "multi":
            model = MultiEncoderDecoderModule(
                x_dim=x_dim,
                y_dim=y_dim,
                y_statistic=y_statistic,
                encoder_h_dim=self.params["encoder_h_dim"],
                decoder_h_dim=self.params["decoder_h_dim"],
                n_decoder_block=self.params["n_decoder_block"],
                encoder=encoder,
                decoder=decoder,
                inputs_decomposer_components=inputs_decomposer_components,
                targets_decomposer_components=targets_decomposer_components,
            )
        elif self.params["task_type"] == "cite":
            model = CiteEncoderDecoderModule(
                x_dim=x_dim,
                y_dim=y_dim,
                y_statistic=y_statistic,
                encoder_h_dim=self.params["encoder_h_dim"],
                decoder_h_dim=self.params["decoder_h_dim"],
                n_decoder_block=self.params["n_decoder_block"],
                encoder=encoder,
                decoder=decoder,
                inputs_decomposer_components=inputs_decomposer_components,
                targets_decomposer_components=targets_decomposer_components,
            )
        else:
            raise ValueError
        return model

    def _batch_to_device(self, batch):
        return tuple(batch[i].to(self.params["device"]) for i in range(len(batch)))

    def _train_step_forward(self, batch, training_length_ratio):
        loss = self.model.loss(*batch, training_length_ratio=training_length_ratio)
        return loss

    def fit(self, x, preprocessed_x, y, preprocessed_y, metadata, pre_post_process):
        if self.params["device"] != "cpu":
            gc.collect()
            torch.cuda.empty_cache()
        self.inputs_info["x_dim"] = preprocessed_x.shape[1]
        self.inputs_info["y_dim"] = preprocessed_y.shape[1]

        dataset = self._build_dataset(
            x=x, preprocessed_x=preprocessed_x, metadata=metadata, y=y, preprocessed_y=preprocessed_y, eval=False
        )
        print("dataset size", len(dataset))
        assert len(dataset) > 0

        self.inputs_info["inputs_decomposer_components"] = pre_post_process.preprocesses["inputs_decomposer"].components_
        self.inputs_info["targets_decomposer_components"] = pre_post_process.preprocesses["targets_decomposer"].components_

        y_statistic = {
            "y_loc": np.mean(preprocessed_y, axis=0),
            "y_scale": np.std(preprocessed_y, axis=0),
        }
        if "targets_global_median" in pre_post_process.preprocesses:
            y_statistic["targets_global_median"] = pre_post_process.preprocesses["targets_global_median"]
        self.inputs_info["y_statistic"] = y_statistic
        batch_size = self.params["train_batch_size"]
        if batch_size > len(dataset):
            batch_size = len(dataset)
        num_workers = int(os.getenv("OMP_NUM_THREADS", 1))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        self.model = self._build_model()
        self.model.to(device=self.params["device"])
        dummy_batch = next(iter(data_loader))
        dummy_batch = self._batch_to_device(dummy_batch)
        self._train_step_forward(dummy_batch, 1.0)

        lr = self.params["lr"]
        eps = self.params["eps"]
        weight_decay = self.params["weight_decay"]
        model_parameters = set_weight_decay(module=self.model, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model_parameters, lr=lr, eps=eps, weight_decay=weight_decay)
        n_epochs = self.params["epoch"]

        pct_start = self.params["pct_start"]
        total_steps = n_epochs * (len(dataset) // batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=lr, total_steps=total_steps, pct_start=pct_start
        )

        print("start to train")
        start_time = time.time()
        self.model.train()
        for epoch in range(n_epochs):
            gc.collect()
            epoch_start_time = time.time()
            if epoch < self.params["burnin_length_epoch"]:
                training_length_ratio = 0.0
            else:
                training_length_ratio = (epoch - self.params["burnin_length_epoch"]) / (
                    n_epochs - self.params["burnin_length_epoch"]
                )
            for _, batch in enumerate(data_loader):
                batch = self._batch_to_device(batch)
                optimizer.zero_grad()
                losses = self._train_step_forward(batch, training_length_ratio)
                losses["loss"].backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clipping)
                optimizer.step()

                scheduler.step()
            end_time = time.time()
            if self.params["task_type"] == "multi":
                loss = losses["loss"]
                loss_corr = losses["loss_corr"]
                loss_mse = losses["loss_mse"]
                loss_res_mse = losses["loss_res_mse"]
                loss_total_corr = losses["loss_total_corr"]
                print(
                    f"epoch: {epoch} total time: {end_time - start_time:.1f}, epoch time: {end_time - epoch_start_time:.1f}, loss:{loss: .3f} "
                    f"loss_corr:{loss_corr: .3f} "
                    f"loss_mse:{loss_mse: .3f} "
                    f"loss_res_mse:{loss_res_mse: .3f} "
                    f"loss_total_corr:{loss_total_corr: .3f} ",
                    flush=True,
                )
            elif self.params["task_type"] == "cite":
                loss = losses["loss"]
                loss_corr = losses["loss_corr"]
                loss_mae = losses["loss_mae"]
                print(
                    f"epoch: {epoch} total time: {end_time - start_time:.1f}, epoch time: {end_time - epoch_start_time:.1f}, loss:{loss: .3f} "
                    f"loss_corr:{loss_corr: .3f} "
                    f"loss_mse:{loss_mae: .3f} ",
                    flush=True,
                )
            else:
                raise RuntimeError
        print("completed training", flush=True)
        summeary_torch_model_parameters(self.model)
        self.model.to("cpu")
        return self

    def _build_dataset(self, x, preprocessed_x, metadata, y, preprocessed_y, eval=True):
        selected_metadata = None
        if not eval:
            if "selected_metadata" in self.params:
                selected_metadata = self.params["selected_metadata"]
        if self.params["task_type"] == "multi":
            dataset = MultiomeDataset(
                inputs_values=x,
                preprocessed_inputs_values=preprocessed_x,
                metadata=metadata,
                targets_values=y,
                preprocessed_targets_values=preprocessed_y,
                selected_metadata=selected_metadata,
            )
        elif self.params["task_type"] == "cite":
            dataset = CITEseqDataset(
                inputs_values=x,
                preprocessed_inputs_values=preprocessed_x,
                metadata=metadata,
                targets_values=y,
                preprocessed_targets_values=preprocessed_y,
                selected_metadata=selected_metadata,
            )
        else:
            raise ValueError

        return dataset

    def predict(self, x, preprocessed_x, metadata):
        if self.params["device"] != "cpu":
            gc.collect()
            torch.cuda.empty_cache()
        self.model = self.model.to(self.params["device"])
        self.model.eval()
        dataset = self._build_dataset(
            x=x, preprocessed_x=preprocessed_x, metadata=metadata, y=None, preprocessed_y=None, eval=True
        )
        test_batch_size = self.params["test_batch_size"]
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, num_workers=0)
        y_pred = []
        with torch.no_grad():
            for batch in data_loader:
                batch = self._batch_to_device(batch)
                y_batch_pred = self.model.predict(*batch[0:3])
                y_batch_pred = y_batch_pred.to("cpu").detach().numpy()
                y_pred.append(y_batch_pred)
        y_pred = np.vstack(y_pred)
        self.model.to("cpu")
        return y_pred

    def save(self, model_dir):
        with open(os.path.join(model_dir, "params.json"), "w") as f:
            json.dump(self.params, f, indent=2)
        with open(os.path.join(model_dir, "inputs_info.pickle"), "wb") as f:
            pickle.dump(self.inputs_info, f)
        self.model.to(device="cpu")
        torch.save(self.model, os.path.join(model_dir, "model.pt"))

    def load(self, model_dir):
        with open(os.path.join(model_dir, "params.json")) as f:
            self.params = json.load(f)
        with open(os.path.join(model_dir, "inputs_info.pickle"), "rb") as f:
            self.inputs_info = pickle.load(f)
        self.model = torch.load(os.path.join(model_dir, "model.pt"))
