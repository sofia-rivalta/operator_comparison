import os, sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

@dataclass
class BurgersDataGenerationConfig():

    #initial condition settings
    correlation_length_min: float = 1
    correlation_length_max: float = 5
    smoothness_min: float = 0.7  # keep > 0.5
    smoothness_max: float = 3
    ICs: int = 200
    seed: int = 0

    #rollout settings
    rollout_len: int = 10000
    sample_interval: int = 10  # sets the pairs kept per rollout; also rollout_generation's save interval
    tunable_parameter: float = 1.0  # keep 1 > lambda > 0 where 1 is highly weighted and 0 is uniform

    #solver settings
    grid_size: int = 500
    viscosity: float = 0.01 #for single viscosity
    safety_margin: float = 0.2
    
    #additional settings
    train_split: float = 0.8  # percentage of trajectories used for training
    data_dir: str = DATA_DIR
    run_name: str = f"burgers_{viscosity}_viscosity"  # dataset is saved to data_dir/run_name/burgers.npz
    precision: str = "float32"  # dtype of the saved pairs

@dataclass
class FNOConfig():

    #architecture settings, names match neuralop's FNO so FNO(**dataclasses.asdict(cfg)) works
    n_modes: tuple = (64,)  # 1D; real data keeps n//2 + 1 modes, so (64,) learns wavenumbers 0-32
    in_channels: int = 1  # u only (single viscosity)
    out_channels: int = 1  # predicted u
    hidden_channels: int = 32  # width; spectral weights scale with hidden_channels**2 * modes * n_layers
    n_layers: int = 4  # number of Fourier layers
    positional_embedding: Optional[str] = None  # "grid" appends x in [0, 1]; off since periodic Burgers is translation invariant
    domain_padding: Optional[float] = None  # off, data is periodic so the FFT wraparound is already correct
    residual: bool = True  # output = u + FNO(u), starting at the identity like the Transformer (local addition to fno.py)


@dataclass
class TransformerConfig():
    #names match Transformer's arguments so Transformer(**dataclasses.asdict(cfg)) works
    n_layer: int = 2
    n_embd: int = 128  # residual stream width
    n_head: int = 2  # must divide n_embd, with n_embd / n_head even for RoPE
    mlp_dim_ratio: int = 4  # MLP hidden width as a multiple of n_embd


@dataclass

class TrainConfig():
    torch_seed: int = 0  # weight init and batch shuffling
    batch_size: int = 100
    # Peak Adam LR. OneCycleLR ramps up to it from learning_rate/25 over the first 30% of
    # training, then cosine-anneals to ~0 over the rest. Stepped per batch.
    learning_rate: float = 1e-3
    n_epochs: int = 65
    grad_clip_norm: int = 100

    results_root: str = RESULTS_DIR


@dataclass
class TrainRunConfig():
    # the model config's type picks the architecture; on the command line choose it with the
    # model:fno-config or model:transformer-config subcommand
    model: Union[FNOConfig, TransformerConfig] = field(default_factory=FNOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_dir: str = os.path.join(DATA_DIR, BurgersDataGenerationConfig.run_name)  # folder holding burgers.npz
    test_data_dir: Optional[str] = None  # test on another dataset's test split (must share solver settings)
    run_name: Optional[str] = None  # results go to results_root/run_name; defaults to <dataset>_<model>
