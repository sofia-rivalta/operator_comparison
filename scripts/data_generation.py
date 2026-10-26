import os, sys, json, dataclasses
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from solvers import Periodic1DBurgersEquation
from config import BurgersDataGenerationConfig
from initializers import GaussianRandomField1D

class DataGeneration():
    def __init__(self, cfg: BurgersDataGenerationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(self.cfg.seed)

    def IC_gen(self):
        c = self.cfg
        ICs = []
        for _ in range(c.ICs):
            correlation_length = self.rng.uniform(c.correlation_length_min, c.correlation_length_max)
            smoothness = self.rng.uniform(c.smoothness_min, c.smoothness_max)
            ICs.append(GaussianRandomField1D(c.grid_size, correlation_length, smoothness, self.rng))
        return np.stack(ICs)  # (ICs, grid_size)
    
    def rollout(self, u, n_steps):
        c = self.cfg
        dx = 1 / c.grid_size  # domain length 1, as in GaussianRandomField1D
        solver = Periodic1DBurgersEquation(dx, c.viscosity, c.safety_margin)
        traj = [u.astype(np.float64)]
        for _ in range(n_steps):
            traj.append(solver.step(traj[-1]))
        return np.stack(traj)  # (n_steps + 1, grid_size)

    def dot_product_subsample(self, traj, n_pairs, tunable_param):
        # n_pairs pairs at equal intervals of time rescaled by 1 - cos(u_t, u_{t+1})
        X, y = traj[:-1], traj[1:]
        cos = (X * y).sum(axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(y, axis=1))
        weights = np.clip(1 - cos, 0, None) ** tunable_param # clip float round-off below 0
        tau = np.cumsum(weights)
        # side="right": target in [tau[i-1], tau[i]) -> step i, so uniform weights with
        # n_pairs == len(X) give every step once ("left" sent a target equal to tau[i] to step i)
        t = np.searchsorted(tau, np.linspace(0, tau[-1], n_pairs, endpoint=False), side="right")
        return X[t], y[t]


    def generate(self):
        d = self.cfg
        n_pairs = d.rollout_len // d.sample_interval
        rollout_list = []
        for ic in self.IC_gen():
            rollout = self.rollout(ic, d.rollout_len)
            subsampled_rollout = self.dot_product_subsample(rollout, n_pairs, d.tunable_parameter)
            rollout_list.append(subsampled_rollout)
        
        self.rng.shuffle(rollout_list)
        split_idx = int(len(rollout_list) * d.train_split) #where to split
        train, test = np.split(rollout_list, [split_idx]) # Split the array at that index
        return train, test


    def save(self, train, test):
        d = self.cfg
        dataset_dir = os.path.join(d.data_dir, d.run_name)
        os.makedirs(dataset_dir, exist_ok=True)
        path = os.path.join(dataset_dir, "burgers.npz")
        # train/test: (n_trajectories, 2, n_pairs, grid_size), axis 1 is (input, target)
        np.savez(path,
                 X_train=train[:, 0].astype(d.precision), y_train=train[:, 1].astype(d.precision),
                 X_test=test[:, 0].astype(d.precision), y_test=test[:, 1].astype(d.precision),
                 config=json.dumps(dataclasses.asdict(d)))
        return path


if __name__ == "__main__":
    generator = DataGeneration(BurgersDataGenerationConfig())
    train, test = generator.generate()
    print("saved to", generator.save(train, test))