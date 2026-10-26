import dataclasses
import json
import os
import sys

import numpy as np
import torch
import tyro
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "fourier_neural_operator")))

from config import TrainRunConfig, FNOConfig, TransformerConfig
from transformer import Transformer
from neuralop import FNO


class FNO1D(nn.Module):
    # neuralop's FNO behind the Transformer's interface: [batch, N] -> [batch, N].
    # neuralop's FNO takes [batch, channels, N], so this only adds and removes the channel dim.
    def __init__(self, **fno_kwargs):
        super().__init__()
        self.fno = FNO(**fno_kwargs)

    def forward(self, x):
        return self.fno(x.unsqueeze(1)).squeeze(1)  # [batch, N] -> [batch, 1, N] -> [batch, N]


# model config type -> model it builds; every model maps [batch, N] -> [batch, N].
# To add an architecture, give it a config in config.py, add it to TrainRunConfig.model, and add it here.
MODELS = {FNOConfig: FNO1D, TransformerConfig: Transformer}


def count_params(model):
    # complex weights (the FNO's spectral weights) hold two real numbers each
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


class TrainBurgers():

    def __init__(self, cfg: TrainRunConfig):
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train
        self.data_dir = cfg.data_dir
        self.model_name = type(self.model_cfg).__name__.removesuffix("Config").lower()  # "fno" or "transformer"
        run_name = cfg.run_name or f"{os.path.basename(os.path.normpath(self.data_dir))}_{self.model_name}"
        self.run_dir = os.path.join(self.train_cfg.results_root, run_name)
        self.checkpoint_path = os.path.join(self.run_dir, f"burgers_{self.model_name}.pt")
        self.config_path = os.path.join(self.run_dir, "config.json")
        self.losses_path = os.path.join(self.run_dir, "losses.npz")
        torch.manual_seed(self.train_cfg.torch_seed)
        self.model = MODELS[type(self.model_cfg)](**dataclasses.asdict(self.model_cfg))
        self.n_params = count_params(self.model)
        print(f"{self.model_name}: {self.n_params} parameters")

    def import_data(self):
        data = np.load(os.path.join(self.data_dir, "burgers.npz"))
        self.data_config = json.loads(str(data["config"]))  # solver/data settings the dataset was made with
        # float32 to match the model's weights, whatever precision the dataset was saved in.
        # Saved as (trajectories, pairs, grid); flattened to one pair per row
        tensor = lambda d, k: torch.from_numpy(d[k].reshape(-1, d[k].shape[-1]).astype(np.float32))
        train = TensorDataset(tensor(data, "X_train"), tensor(data, "y_train"))
        test_data = data
        if self.cfg.test_data_dir is not None:
            test_data = np.load(os.path.join(self.cfg.test_data_dir, "burgers.npz"))
            self.test_data_config = json.loads(str(test_data["config"]))
            solver = lambda c: {k: c[k] for k in ("grid_size", "viscosity", "safety_margin")}
            if solver(self.test_data_config) != solver(self.data_config):
                raise ValueError("test_data_dir was made with different solver settings than data_dir")
        test = TensorDataset(tensor(test_data, "X_test"), tensor(test_data, "y_test"))
        self.train_loader = DataLoader(train, batch_size=self.train_cfg.batch_size, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=self.train_cfg.batch_size)
        # rms of (y - X): normalizes each loss so 1.0 == "predicted no change". Test uses train's
        # scale unless it's another dataset, whose increments can be a different size
        increment_rms = lambda d: (d.tensors[1] - d.tensors[0]).pow(2).mean().sqrt().item()
        self.increment_scale = increment_rms(train)
        self.test_increment_scale = increment_rms(test) if self.cfg.test_data_dir is not None else self.increment_scale

    def train_loops(self):
        m, t = self.model_cfg, self.train_cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=t.learning_rate, epochs=t.n_epochs, steps_per_epoch=len(self.train_loader))

        def loss_fn(pred, y, scale=self.increment_scale):
            return ((pred - y) ** 2).mean() / scale ** 2

        train_losses, test_losses = [], []
        for epoch in range(t.n_epochs):
            model.train()
            train_loss = 0.0
            for X, y in self.train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(X), y)
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip_norm)
                optimizer.step()
                scheduler.step()

            model.eval()
            with torch.no_grad():
                test_loss = sum(loss_fn(model(X.to(device)), y.to(device), self.test_increment_scale).item()
                                for X, y in self.test_loader)
            train_losses.append(train_loss / len(self.train_loader))
            test_losses.append(test_loss / len(self.test_loader))
            print(f"epoch {epoch}: normalized increment MSE: train {train_losses[-1]:.6f} | test {test_losses[-1]:.6f}")
            self.save_weights(model)
            self.save_losses(train_losses, test_losses)
        return model, train_losses, test_losses

    def save_config(self):
        # training settings plus the dataset's own recorded settings, so the run is self-describing
        os.makedirs(self.run_dir, exist_ok=True)
        config = {"train_run": dataclasses.asdict(self.cfg), "model": self.model_name,
                  "n_params": self.n_params, "dataset": self.data_config,
                  "increment_scale": self.increment_scale, "test_increment_scale": self.test_increment_scale}
        if self.cfg.test_data_dir is not None:
            config["test_dataset"] = self.test_data_config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"saved config to {self.config_path}")

    def save_weights(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"saved weights to {self.checkpoint_path}")

    def save_losses(self, train_losses, test_losses):
        # per-epoch normalized increment MSE
        np.savez(self.losses_path, train_losses=np.array(train_losses), test_losses=np.array(test_losses))
        print(f"saved losses to {self.losses_path}")

if __name__ == "__main__":
    trainer = TrainBurgers(tyro.cli(TrainRunConfig))
    trainer.import_data()
    trainer.save_config()
    trainer.train_loops()