import warnings
from circuitree.models import SimpleNetworkTree
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Iterable
from uuid import uuid4

from tf_network import TFNetworkModel


class OscillationTree(SimpleNetworkTree):
    def __init__(
        self,
        time_points: Optional[np.ndarray[np.float64]] = None,
        Q_threshold: float = 0.01,
        ACF_threshold: float = 0.4,
        init_mean: float = 10.0,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        batch_size: int = 1,
        max_iter_per_timestep: int = 100_000_000,
        save_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time_points = time_points
        self.ACF_threshold = ACF_threshold
        self.Q_threshold = Q_threshold

        self.dt = dt
        self.nt = nt
        self.init_mean = init_mean
        self.batch_size = batch_size

        self.max_iter_per_timestep = max_iter_per_timestep

        self.save_dir = save_dir

    def is_success(self, state: str) -> bool:
        reward = self.graph.nodes[state]["reward"]
        visits = self.graph.nodes[state]["visits"]
        return visits > 0 and reward / visits >= self.Q_threshold

    def is_terminal(self, state):
        """Return True if the state is a terminal state."""
        return state.startswith("*")

    def get_reward(
        self,
        state: str,
        batch_size: Optional[int] = None,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        maxiter: Optional[int] = None,
    ) -> float:
        dt = dt if dt is not None else self.dt
        nt = nt if nt is not None else self.nt
        maxiter = maxiter if maxiter is not None else self.max_iter_per_timestep
        batch_size = batch_size if batch_size is not None else self.batch_size

        model = TFNetworkModel(
            state,
            initialize=True,
            dt=dt,
            nt=nt,
            max_iter_per_timestep=maxiter,
        )
        y_t, pop0s, param_sets, rewards = model.run_batch_job(batch_size, abs=True)

        nan_rewards = np.isnan(rewards)
        if nan_rewards.all():
            self.save_nan_data(
                rewards, model=model, y_t=y_t, pop0s=pop0s, param_sets=param_sets
            )
            reward = self.get_reward(state)
        elif nan_rewards.any():
            self.save_nan_data(
                rewards, model=model, y_t=y_t, pop0s=pop0s, param_sets=param_sets
            )
            reward = (rewards[~nan_rewards] > self.ACF_threshold).mean()
        else:
            reward = np.mean(rewards > self.ACF_threshold)
        return reward

    def save_nan_data(
        self,
        rewards: list[float],
        model: TFNetworkModel,
        pop0s,
        param_sets,
        seed: int | Iterable[int] | None = None,
        **kwargs,
    ) -> None:
        if self.save_dir is None:
            warnings.warn("No save directory specified, not saving nan data")
            return

        where_nans = np.isnan(rewards)
        if not where_nans.any():
            return

        warnings.warn(f"Saving data for {where_nans.sum()} runs with nan rewards")

        # Save the nan data to HDF
        state = model.genotype
        dt = model.dt
        nt = model.nt
        maxiter = model.max_iter_per_timestep
        if seed is None:
            save_seed = model.seed
        elif isinstance(seed, Iterable):
            save_seed = seed[where_nans]
        else:
            save_seed = seed

        fname = f"nans_state_{state}_uid_{uuid4()}.hdf5"
        fpath = Path(self.save_dir) / fname
        with h5py.File(fpath, "w") as f:
            f.create_dataset("seed", data=save_seed)
            f.create_dataset("pop0s", data=pop0s[where_nans])
            f.create_dataset("param_sets", data=param_sets[where_nans])
            f.attrs["state"] = state
            f.attrs["dt"] = dt
            f.attrs["nt"] = nt
            f.attrs["max_iter_per_timestep"] = maxiter
