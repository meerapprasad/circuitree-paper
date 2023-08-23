from circuitree.parallel import ParallelTree
import h5py
from functools import partial
from multiprocessing.pool import Pool
import numpy as np
from pathlib import Path
from time import perf_counter
from typing import Any, Optional
from uuid import uuid4
import warnings

from tf_network import TFNetworkModel
from oscillation import OscillationTree

__all__ = ["OscillationTreeParallel"]


def run_ssa(visit, prots0, params, model: TFNetworkModel):
    start = perf_counter()
    prots_t, reward = model.run_with_params_and_get_acf_minimum(
        prots0=prots0,
        params=params,
        seed=visit,
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

        successes = np.array(rewards) > self.autocorr_threshold
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
        nan_rewards = np.isnan(rewards)
        if nan_rewards.all():
            if self.warn_if_nan:
                warnings.warn(f"All rewards in batch are NaNs. Skipping this batch.")
            reward = self.get_reward(state)
        elif nan_rewards.any():
            if self.warn_if_nan:
                warnings.warn(f"Found NaN rewards in batch.")
            reward = np.mean(rewards[~nan_rewards] > self.autocorr_threshold)
        else:
            reward = np.mean(rewards > self.autocorr_threshold)
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
            autocorr_threshold=self.autocorr_threshold,
            save_dir=self.save_dir,
            prefix=prefix,
            **kwargs,
        )

    def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
        raise NotImplementedError

    def save_results(self, data: dict[str, Any]) -> None:
        raise NotImplementedError


class OscillationTreeMP(OscillationTreeParallel):
    def __init__(
        self,
        *,
        pool: Optional[Pool] = None,
        nprocs: int = 1,
        **kwargs,
    ):
        raise NotImplementedError

        super().__init__(**kwargs)
        self._pool = Pool(nprocs) if pool is None else pool
        self.n_procs = self.pool._processes

        # Specify any attributes that should not be serialized when dumping to file
        self._non_serializable_attrs.append("_pool")

    @property
    def pool(self) -> Pool:
        return self._pool

    def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
        """Should return a list of reward values and a dictionary of any data
        to be analyzed. Takes a state and a list of which visits to simulate.
        Random seeds, parameter sets, and initial conditions are selected from
        the parameter table `self.param_table`."""
        model = TFNetworkModel(
            state,
            dt=self.dt,
            nt=self.nt,
            max_iter_per_timestep=self.max_iter_per_timestep,
            initialize=True,
        )
        sample_state = partial(run_ssa, model=model)
        input_args = [self.param_table[v] for v in visits]
        rewards, prots_t, sim_times = self.pool.starmap(sample_state, input_args)
        data = {
            "model": model,
            "visits": visits,
            "rewards": rewards,
            "y_t": prots_t,
            "sim_times": sim_times,
        }
        return rewards, data
