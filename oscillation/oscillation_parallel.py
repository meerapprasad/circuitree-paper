from circuitree import ParallelTree
import h5py
from functools import partial
import numpy as np
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
from uuid import uuid4
import warnings

from tf_network import TFNetworkModel
from oscillation import OscillationTree

__all__ = ["OscillationTreeParallel"]


def run_ssa(
    seed: int,
    prots0: Iterable[int],
    params: Iterable[float],
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
):
    start = perf_counter()
    model = TFNetworkModel(
        state,
        dt=dt,
        nt=nt,
        max_iter_per_timestep=max_iter_per_timestep,
        initialize=True,
    )
    prots_t, reward = model.run_with_params_and_get_acf_minimum(
        prots0=prots0,
        params=params,
        seed=seed,
        maxiter_ok=True,
        abs=True,
    )
    end = perf_counter()
    sim_time = end - start
    return reward, prots_t, sim_time


def save_simulation_results_to_hdf(
    model: TFNetworkModel,
    visits: int,
    rewards: list[float],
    y_t: np.ndarray,
    autocorr_threshold: float,
    save_dir: Path,
    sim_time: float = -1.0,
    prefix: str = "",
    **kwargs,
) -> None:
    state = model.genotype
    fname = f"{prefix}state_{state}_uid_{uuid4()}.hdf5"
    fpath = Path(save_dir) / fname
    with h5py.File(fpath, "w") as f:
        # f.create_dataset("seed", data=seed)
        f.create_dataset("visits", data=visits)
        f.create_dataset("rewards", data=rewards)
        f.create_dataset("y_t", data=y_t)
        f.attrs["state"] = state
        f.attrs["dt"] = model.dt
        f.attrs["nt"] = model.nt
        f.attrs["max_iter_per_timestep"] = model.max_iter_per_timestep
        f.attrs["autocorr_threshold"] = autocorr_threshold
        f.attrs["sim_time"] = sim_time


class OscillationTreeParallel(OscillationTree, ParallelTree):
    """Searches the space of TF networks for oscillatory topologies.
    Each step of the search takes the average of multiple draws.
    Uses a transposition table to access and store reward values. If desired
    results are not present in the table, they will be computed in parallel.
    Random seeds, parameter sets, and initial conditions are selected from the
    parameter table `self.param_table`.

    The `simulate_visits` and `save_results` methods must be implemented in a
    subclass.

    An invalid simulation result (e.g. a timeout) should be represented by a
    NaN reward value. If all rewards in a batch are NaNs, a new batch will be
    drawn from the transposition table. Otherwise, any NaN rewards will be
    ignored and the mean of the remaining rewards will be returned as the
    reward for the batch.
    """

    def __init__(
        self,
        *,
        bootstrap: bool = False,
        warn_if_nan: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bootstrap = bootstrap
        self.warn_if_nan = warn_if_nan

    def save_results(self, data: dict[str, Any]) -> None:
        # Add rewards to the transposition table. NaN reward values are recorded and can
        # be retrieved to find visits that need to be simulated again
        model: TFNetworkModel = data["model"]
        visits = data["visits"]
        rewards = data["rewards"]

        # Add rewards to the transposition table
        self.ttable[model.genotype].extend(list(rewards))

        successes = np.array(rewards) > self.ACF_threshold
        if successes.any():
            # Save data for successful visits
            self.save_simulation_data(
                model=model,
                visits=visits[successes],
                rewards=rewards[successes],
                y_t=data["y_t"][successes],
                prefix="oscillation_",
            )

        nans = np.isnan(rewards)
        if np.any(nans):
            # Save data for NaN rewards
            self.save_simulation_data(
                model=model,
                visits=visits[nans],
                rewards=rewards[nans],
                y_t=data["y_t"][nans],
                prefix="nans_",
            )

    def _draw_bootstrap_reward(self, state, maxiter=100):
        indices, rewards = self.ttable.draw_bootstrap_reward(
            state=state, size=self.batch_size, rg=self.rg
        )

        # Replace any nans with new draws
        for _ in range(maxiter):
            where_nan = np.isnan(rewards)
            if not where_nan.any():
                break
            indices, new_rewards = self.ttable.draw_bootstrap_reward(
                state=state, size=where_nan.sum(), rg=self.rg
            )
            rewards[where_nan] = new_rewards
        else:
            # If maxiter was reached, (probably) all rewards for a state are NaN
            if where_nan.any():
                raise RuntimeError(
                    f"Could not resolve NaN rewards in {maxiter} iterations of "
                    f"bootstrap sampling. Perhaps all rewards for state {state} "
                    "are NaN?"
                )

        return rewards.mean()

    def get_reward(self, state, maxiter=100):
        if self.bootstrap:
            return self._draw_bootstrap_reward(state, maxiter=maxiter)

        visit = self.visit_counter[state]
        n_recorded_visits = self.ttable.n_visits(state)
        n_to_read = np.clip(n_recorded_visits - visit, 0, self.batch_size)
        n_to_simulate = self.batch_size - n_to_read

        rewards = self.ttable[state, visit : visit + n_to_read]
        if n_to_simulate > 0:
            sim_visits = visit + n_to_read + np.arange(n_to_simulate)
            sim_rewards, sim_data = self.simulate_visits(state, sim_visits)
            self.save_results(sim_data)
            rewards.extend(sim_rewards)

        self.visit_counter[state] += self.batch_size

        # Handle NaN rewards
        rewards = np.array(rewards)
        nan_rewards = np.isnan(rewards)
        if nan_rewards.all():
            if self.warn_if_nan:
                warnings.warn(f"All rewards in batch are NaNs. Skipping this batch.")
            reward = self.get_reward(state)
        elif nan_rewards.any():
            if self.warn_if_nan:
                warnings.warn(f"Found NaN rewards in batch.")
            reward = np.mean(rewards[~nan_rewards] > self.ACF_threshold)
        else:
            reward = np.mean(rewards > self.ACF_threshold)

        self._done_callback(
            self, state, list(range(visit, visit + self.batch_size)), rewards.tolist()
        )
        return reward

    def save_simulation_data(
        self,
        model: TFNetworkModel,
        visits: int,
        rewards: list[float],
        y_t: np.ndarray,
        prefix: str = "",
        **kwargs,
    ) -> None:
        if self.save_dir is None:
            raise FileNotFoundError("No save directory specified")

        save_simulation_results_to_hdf(
            model=model,
            visits=visits,
            rewards=rewards,
            y_t=y_t,
            autocorr_threshold=self.ACF_threshold,
            save_dir=self.save_dir,
            prefix=prefix,
            **kwargs,
        )

    @staticmethod
    def _done_callback(self_obj, state, visits, rewards):
        pass

    def add_done_callback(self, callback, *args, **kwargs):
        self._done_callback = partial(callback, *args, **kwargs)

    def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
        raise NotImplementedError

    def save_results(self, data: dict[str, Any]) -> None:
        raise NotImplementedError
