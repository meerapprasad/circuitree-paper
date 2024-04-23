import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tqdm

from oscillation import TFNetworkModel


def main(
    code: str = "*ABC::ABi_BCi_CAi",
    n_threads: int = 4,
    t: np.ndarray = np.linspace(0, 80000, 4001),
    prows: int = 2,
    pcols: int = 2,
    save_dir=None,
    save=True,
    fmt="png",
    dpi=300,
):
    dt = t[1] - t[0]
    t_mins = t / 60
    t_hrs = t_mins / 60
    half_nt = len(t) // 2
    model = TFNetworkModel(code, seed=0)
    model.initialize_ode(seed=0, dt=dt, nt=len(t))
    print('model initialized')

    # pop0s = []
    # param_sets = []
    # # for _ in range(100000):
    # for _ in tqdm.tqdm(range(100000)):
    #     pop0, params = model.ssa.draw_random_initial_and_params()
    #     if params[0] > 3.0:
    #         pop0s.append(pop0)
    #         param_sets.append(params)
    #         if len(param_sets) == n_threads:
    #             break

    # y_t = model.run_batch_with_params(pop0s, param_sets, n_threads)
    # already initializes random params
    y_t = model.ode.run_ode_with_params()
    y_t = y_t[..., 3:6]
    acorr = model.get_autocorrelation(y_t)
    nt_acorr = acorr.shape[1]
    indices, freqs, peaks = model.get_acf_minima_and_results(
        t, y_t, freqs=True, indices=True
    )

    fig1 = plt.figure(figsize=(4, 3))
    for i in range(n_threads):
        ax = fig1.add_subplot(prows, pcols, i + 1)
        ax.plot(t_hrs, y_t[i, :, 0], label="TF A", lw=0.5)
        ax.plot(t_hrs, y_t[i, :, 1], label="TF B", lw=0.5)
        ax.plot(t_hrs, y_t[i, :, 2], label="TF C", lw=0.5)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Copy number")
    if save:
        fpath = Path(save_dir).joinpath(f"repressilator_traces").with_suffix(f".{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        fig1.savefig(str(fpath), dpi=dpi, bbox_inches="tight")

    fig2 = plt.figure(figsize=(4, 3))
    for i in range(n_threads):
        ax = fig2.add_subplot(prows, pcols, i + 1)
        ax.plot(t_hrs[:nt_acorr], acorr[i, :, 0], label="TF A", lw=0.5)
        ax.plot(t_hrs[:nt_acorr], acorr[i, :, 1], label="TF B", lw=0.5)
        ax.plot(t_hrs[:nt_acorr], acorr[i, :, 2], label="TF C", lw=0.5)

        # Point out second peak of the autocorrelation function
        ax.scatter(t_hrs[indices[i]], peaks[i], marker="x", s=50, c="gray", zorder=100)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Autocorrelation")

    if save:
        fpath = (
            Path(save_dir)
            .joinpath(f"repressilator_autocorr_newranges")
            .with_suffix(f".{fmt}")
        )
        print(f"Writing to: {fpath.resolve().absolute()}")
        fig2.savefig(str(fpath), dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    # save_dir = Path("./figures/oscillation")
    save_dir = Path("./figures/oscillation/newranges")
    save_dir.mkdir(exist_ok=True)

    main(
        save_dir=save_dir,
    )
