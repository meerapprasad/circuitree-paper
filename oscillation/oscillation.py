from functools import cache, cached_property
from itertools import permutations
from typing import Optional, Iterable
from numba import stencil, njit
import numpy as np
from scipy.signal import correlate

from circuitree import SimpleNetworkTree

try:
    from models.oscillation.gillespie import (
        GillespieSSA,
        make_matrices_for_ssa,
        SAMPLING_RANGES,
        DEFAULT_PARAMS,
    )
except ModuleNotFoundError:
    from gillespie import (
        GillespieSSA,
        make_matrices_for_ssa,
        SAMPLING_RANGES,
        DEFAULT_PARAMS,
    )


class TFNetworkModel:
    """
    Model of a transcription factor network.
    =============

    10 parameters
        TF-promoter binding rates:
            k_on
            k_off_1
            k_off_2

            \\ NOTE: k_off_2 less than k_off_1 indicates cooperative binding)

        Transcription rates:
            km_unbound
            km_act
            km_rep
            km_act_rep

        Translation rate:
            kp

        Degradation rates:
            gamma_m
            gamma_p

    """

    def __init__(
        self,
        genotype: str,
        initialize: bool = False,
        seed: Optional[int] = None,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        max_iter_per_timestep: int = 100_000_000,
        **kwargs,
    ):
        self.genotype = genotype

        (
            self.components,
            self.activations,
            self.inhibitions,
        ) = SimpleNetworkTree.parse_genotype(genotype)
        self.m = len(self.components)
        self.a = len(self.activations)
        self.r = len(self.inhibitions)

        # self._params_dict = _default_parameters
        # self.update_params(params or param_updates or {})

        # self.population = self.get_initial_population(
        #     init_method=init_method,
        #     init_params=init_params,
        #     **kwargs,
        # )

        self.t: Optional[Iterable[float]] = None

        self.seed = seed
        self.dt = dt
        self.nt = nt
        self.max_iter_per_timestep = max_iter_per_timestep
        
        self.ssa: GillespieSSA | None = None
        if initialize:
            if any(arg is None for arg in (seed, dt, nt)):
                raise ValueError("seed, dt and nt must be specified if initialize=True")
            self.initialize_ssa(seed, dt, nt)

    def initialize_ssa(
        self,
        seed: int,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        init_mean: float = 10.0,
        max_iter_per_timestep: Optional[int] = None,
        **kwargs,
    ):
        seed = seed or self.seed
        dt = dt or self.dt
        nt = nt or self.nt
        max_iter_per_timestep = max_iter_per_timestep or self.max_iter_per_timestep
        t = dt * np.arange(nt)
        self.t = t

        Am, Rm, U = make_matrices_for_ssa(self.m, self.activations, self.inhibitions)
        self.ssa = GillespieSSA(
            seed,
            self.m,
            Am,
            Rm,
            U,
            dt,
            nt,
            init_mean,
            SAMPLING_RANGES,
            DEFAULT_PARAMS,
            max_iter_per_timestep,
        )

    def run_ssa_with_params(self, pop0, params):
        return self.ssa.run_with_params(pop0, params)

    def run_batch_with_params(self, pop0, params, n):
        pop0 = np.asarray(pop0)
        params = np.asarray(params)
        is_vectorized = pop0.ndim == 2 and params.ndim == 2
        if is_vectorized:
            return self.ssa.run_batch_with_params_vector(pop0, params)
        else:
            return self.ssa.run_batch_with_params(pop0, params, n)

    def run_ssa_random_params(self):
        pop0, params, y_t = self.ssa.run_random_sample()
        return pop0, params, y_t

    def run_batch_random(self, n_samples):
        return self.ssa.run_batch(n_samples)

    def run_job(self, abs: bool = False, **kwargs):
        """
        Run the simulation with random parameters and default time-stepping.
        For convenience, this returns the genotype ("state") and visit number in addition
        to simulation results.
        """
        y_t, pop0, params, reward = self.run_ssa_and_get_acf_minima(
            self.dt, self.nt, size=1, freqs=False, indices=False, abs=abs, **kwargs
        )
        return reward, pop0, params

    def run_batch_job(self, batch_size: int, abs: bool = False, **kwargs):
        """
        Run the simulation with random parameters and default time-stepping.
        For convenience, this returns the genotype ("state") and visit number in addition
        to simulation results.
        """
        y_t, pop0s, param_sets, rewards = self.run_ssa_and_get_acf_minima(
            self.dt,
            self.nt,
            size=batch_size,
            freqs=False,
            indices=False,
            abs=abs,
            **kwargs,
        )
        return y_t, pop0s, param_sets, rewards

    @staticmethod
    def get_autocorrelation(y_t: np.ndarray[np.float64 | np.int64]):
        filtered = filter_ndarray_binomial9(y_t.astype(np.float64))[..., 4:-4, :]
        acorrs = autocorrelate_vectorized(filtered)
        return acorrs

    @staticmethod
    def get_acf_minima(
        y_t: np.ndarray[np.float64 | np.int64], abs: bool = False
    ) -> np.ndarray[np.float64]:
        filtered = filter_ndarray_binomial9(y_t.astype(np.float64))[..., 4:-4, :]
        acorrs = autocorrelate_vectorized(filtered)
        where_minima, minima = compute_lowest_minima(acorrs)
        if abs:
            return np.abs(minima)
        else:
            return minima

    @staticmethod
    def get_acf_minima_and_results(
        t: np.ndarray,
        pop_t: np.ndarray,
        freqs: bool = True,
        indices: bool = False,
        abs: bool = False,
    ):
        """
        Get the location and height of the largest extremum of the autocorrelation
        function, excluding the bounds.
        """

        # Filter out high-frequency (salt-and-pepper) noise
        filtered = filter_ndarray_binomial9(pop_t.astype(np.float64))[..., 4:-4, :]

        # Compute autocorrelation
        acorrs = autocorrelate_vectorized(filtered)

        # Compute the location and size of the largest interior extremum over
        # all species
        where_minima, minima = compute_lowest_minima(acorrs)
        if abs:
            minima = np.abs(minima)

        tdiff = t - t[0]
        if freqs:
            minima_freqs = np.where(where_minima > 0, 1 / tdiff[where_minima], 0.0)

        squeeze = where_minima.size == 1
        if squeeze:
            where_minima = where_minima.flat[0]
            minima = minima.flat[0]
            if freqs:
                minima_freqs = minima_freqs.flat[0]

        if freqs:
            if indices:
                return where_minima, minima_freqs, minima
            else:
                return minima_freqs, minima
        elif indices:
            return where_minima, minima
        else:
            return minima

    def run_ssa_and_get_acf_minima(
        self,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        seed: Optional[int] = None,
        size: int = 1,
        freqs: bool = False,
        indices: bool = False,
        init_mean: float = 10.0,
        abs: bool = False,
        pop0: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Run the stochastic simulation algorithm for the system and get the
        autocorrelation-based reward.
        """

        if all(arg is not None for arg in (seed, dt, nt)):
            self.initialize_ssa(seed, dt, nt, init_mean)
            t = self.t

        if (params is None) != (pop0 is None):
            raise ValueError("Either both or neither of params and pop0 must be given.")

        elif (params is None) and (pop0 is None):
            if size > 1:
                pop0, params, y_t = self.run_batch_random(size)
            else:
                pop0, params, y_t = self.run_ssa_random_params()

        else:
            size = np.atleast_2d(params).shape[0]
            if size > 1:
                y_t = self.run_batch_with_params(pop0, params, size)
            else:
                y_t = self.run_ssa_with_params(pop0.flatten(), params.flatten())

        mask_axes = y_t.ndim - 2, y_t.ndim - 1
        not_nan_mask = np.logical_not(np.isnan(y_t).any(axis=mask_axes))

        # Get autocorrelation results for all simulations. Catch nans that may arise if
        # the simulation takes too long
        if not (freqs or indices):
            results = np.full(y_t.shape[:-2], np.nan)
            results[not_nan_mask] = self.get_acf_minima(y_t[not_nan_mask], abs=abs)
        else:
            nan_result = (np.nan,) * (1 + freqs + indices)
            not_nan_results = self.get_acf_minima_and_results(
                t, y_t, freqs=freqs, indices=indices, abs=abs
            )
            results = [nan_result] * size
            for i, result in zip(not_nan_mask.nonzero()[0], not_nan_results):
                results[i] = result

        prots0 = pop0[..., self.m : self.m * 2]
        prots_t = y_t[..., self.m : self.m * 2]

        return prots_t, prots0, params, results


def autocorrelate_mean0(arr1d_norm: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    "Autocorrelation of an array with mean 0"
    return correlate(arr1d_norm, arr1d_norm, mode="same")[len(arr1d_norm) // 2 :]


def autocorrelate(data1d: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    arr = data1d - data1d.mean()
    arr = autocorrelate_mean0(arr)
    arr /= arr.max()
    return arr


def autocorrelate_vectorized(
    data: np.ndarray[np.float_], axis=-2
) -> np.ndarray[np.float_]:
    """Compute autocorrelation of 1d signals arranged in an nd array, where `axis` is the
    time axis."""
    ndarr = data - data.mean(axis=axis, keepdims=True)
    ndarr = np.apply_along_axis(autocorrelate_mean0, axis, ndarr)
    arrmax = ndarr.max(axis=axis, keepdims=True)

    # avoid division by zero when signal is flat
    arrmax = np.where(arrmax == 0, 1, arrmax)

    ndarr /= arrmax
    return ndarr


@stencil
def binomial3_kernel(a):
    """Basic 3-point binomial filter."""
    return (a[-1] + a[0] + a[0] + a[1]) / 4


@stencil
def binomial5_kernel(a):
    """Basic 5-point binomial filter."""
    return (a[-2] + 4 * a[-1] + 6 * a[0] + 4 * a[1] + a[2]) / 16


@stencil
def binomial7_kernel(a):
    """Basic 7-point binomial filter."""
    return (
        a[-3] + 6 * a[-2] + 15 * a[-1] + 20 * a[0] + 15 * a[1] + 6 * a[2] + a[3]
    ) / 64


@stencil
def binomial9_kernel(a):
    """9-point binomial filter."""
    return (
        a[-4]
        + 8 * a[-3]
        + 28 * a[-2]
        + 56 * a[-1]
        + 70 * a[0]
        + 56 * a[1]
        + 28 * a[2]
        + 8 * a[3]
        + a[4]
    ) / 256


@njit
def filter_ndarray_binomial9(ndarr: np.ndarray) -> float:
    """Apply a binomial filter to 1d signals arranged in an nd array, where the time axis
    is the second to last axis (``axis = -2``).
    """
    ndarr_shape = ndarr.shape
    leading_shape = ndarr_shape[:-2]
    n = ndarr_shape[-1]
    filtered = np.zeros_like(ndarr)
    for leading_index in np.ndindex(leading_shape):
        for i in range(n):
            arr1d = ndarr[leading_index][:, i]
            filt1d = binomial9_kernel(arr1d)
            filtered[leading_index][:, i] = filt1d
    return filtered


@stencil(cval=False)
def minimum_kernel(a):
    """
    Returns a 1D mask that is True at local minima, excluding the bounds.
    Computes when the finite difference changes sign from - to + (or is zero).
    """
    return (a[0] - a[-1] <= 0) and (a[1] - a[0] >= 0)


@njit
def compute_lowest_minimum(seq: np.ndarray[np.float_]) -> tuple[int, float]:
    """
    Find the minimum in a sequence of values with the greatest absolute
    value, excluding the bounds.
    """
    minimum_mask = minimum_kernel(seq)
    if not minimum_mask.any():
        return -1, 0.0
    else:
        minima = np.where(minimum_mask)[0]
        where_lowest_minimum = minima[np.argmin(seq[minima])]
        return where_lowest_minimum, seq[where_lowest_minimum]


@njit
def compute_lowest_minima(ndarr: np.ndarray) -> float:
    """
    Get the lowest interior minimum of a batch of n 1d arrays, each of length m.
    Vectorizes over arbitrary leading axes. For an input of shape (k, l, m, n),
    k x l x n minima are calculated, and the min-of-minima is taken over the last axis
    to return an array of shape (k, l).
    Also returns the index of the minimum if it exists, otherwise -1.
    """
    nd_shape = ndarr.shape[:-2]
    where_largest_minima = np.zeros(nd_shape, dtype=np.int64)
    largest_minima = np.zeros(nd_shape, dtype=np.float64)
    for leading_index in np.ndindex(nd_shape):
        argmin = -1
        minval = 0.0
        for a in ndarr[leading_index].T:
            where_minimum, minimum = compute_lowest_minimum(a)
            if minimum < minval:
                minval = minimum
                argmin = where_minimum
        where_largest_minima[leading_index] = argmin
        largest_minima[leading_index] = minval
    return where_largest_minima, largest_minima


class OscillationTree(SimpleNetworkTree):
    def __init__(
        self,
        time_points: Optional[np.ndarray[np.float64]] = None,
        success_threshold: float = 0.005,
        autocorr_threshold: float = 0.4,
        init_mean: float = 10.0,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        batch_size: int = 1,
        max_iter_per_timestep: int = 100_000_000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time_points = time_points
        self.autocorr_threshold = autocorr_threshold
        self.success_threshold = success_threshold

        self.dt = dt
        self.nt = nt
        self.init_mean = init_mean
        self.batch_size = batch_size

    @cached_property
    def _recolor(self):
        return [dict(zip(self.components, p)) for p in permutations(self.components)]

    @staticmethod
    def _recolor_string(mapping, string):
        return "".join([mapping.get(c, c) for c in string])

    @cache
    def get_interaction_recolorings(self, genotype: str) -> list[str]:
        if "::" in genotype:
            components, interactions = genotype.split("::")
        else:
            interactions = genotype

        interaction_recolorings = []
        for mapping in self._recolor:
            recolored_interactions = sorted(
                [self._recolor_string(mapping, ixn) for ixn in interactions.split("_")]
            )
            interaction_recolorings.append("_".join(recolored_interactions).strip("_"))

        return interaction_recolorings

    @cache
    def get_component_recolorings(self, genotype: str) -> list[str]:
        if "::" in genotype:
            components, interactions = genotype.split("::")
        else:
            components = genotype

        component_recolorings = []
        for mapping in self._recolor:
            recolored_components = "".join(
                sorted(self._recolor_string(mapping, components))
            )
            component_recolorings.append(recolored_components)

        return component_recolorings

    def get_recolorings(self, genotype: str) -> Iterable[str]:
        ris = self.get_interaction_recolorings(genotype)
        rcs = self.get_component_recolorings(genotype)
        recolorings = ["::".join([rc, ri]) for rc, ri in zip(rcs, ris)]

        return recolorings

    def get_unique_state(self, genotype: str) -> str:
        return min(self.get_recolorings(genotype))

    def has_motif(self, state, motif):
        if ("::" in motif) or ("*" in motif):
            raise ValueError(
                "Motif code should only contain interactions, no components"
            )
        if "::" not in state:
            raise ValueError(
                "State code should contain both components and interactions"
            )

        interaction_code = state.split("::")[1]
        if not interaction_code:
            return False
        state_interactions = set(interaction_code.split("_"))

        for recoloring in self.get_interaction_recolorings(motif):
            motif_interactions = set(recoloring.split("_"))
            if motif_interactions.issubset(state_interactions):
                return True
        return False

    def is_success(self, state: str) -> bool:
        payout = self.graph.nodes[state]["reward"]
        visits = self.graph.nodes[state]["visits"]
        return visits > 0 and payout / visits > self.success_threshold

    def get_reward(
        self,
        state: str,
        batch_size: Optional[int] = None,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        max_iter_per_timestep: Optional[int] = None,
    ) -> float:
        dt = dt if dt is not None else self.dt
        nt = nt if nt is not None else self.nt
        max_iter_per_timestep = (
            max_iter_per_timestep
            if max_iter_per_timestep is not None
            else self.max_iter_per_timestep
        )
        batch_size = batch_size if batch_size is not None else self.batch_size

        model = TFNetworkModel(
            state,
            initialize=True,
            dt=dt,
            nt=nt,
            max_iter_per_timestep=max_iter_per_timestep,
        )
        y_t, pop0s, param_sets, rewards = model.run_batch_job(batch_size, abs=True)

        raise NotImplementedError

        ...

        # Need to insert logic here to deal with nans in rewards!!
        # Should take a parameter to decide what a nan means
        #   - remove nan from batch and take mean of rest
        #       - catch case of all nans and return 0 ?
        #   - nan means re-run entire batch
        #   - nan means re-run just the nan

        # *** Save nan runs' details for later inspection ***

        return np.mean(rewards > self.autocorr_threshold)
