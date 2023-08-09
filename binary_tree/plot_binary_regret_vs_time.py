import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

from circuitree import vround


def main(
    data_csv: Path = Path("./data/analysis/binary_tree/binary_tree_search.csv"),
    figsize=(2.5, 2),
    nt_plot: int = 100,
    n_codes_plot: int = 100,
    min_modularity: float = 0.0,
    max_modularity: float = 1.0,
    fmt: str = "png",
    dpi: int = 300,
    save: bool = False,
    save_dir: Path = Path("./figures/binary_tree"),
):
    df = pd.read_csv(
        Path(data_csv), dtype={"method": "string", "outcome_codes": "string"}
    )

    # Sub-sample outcomes by modularity
    min_modularity = max(df["modularity"].min(), min_modularity)
    max_modularity = min(df["modularity"].max(), max_modularity)
    subset_outcome_idx = [
        (df["modularity"] - m).abs().argmin()
        for m in np.linspace(min_modularity, max_modularity, n_codes_plot)
    ]
    subset_outcome_codes = df["outcome_codes"].iloc[subset_outcome_idx].unique()

    # Include all MCTS runs and one sequential run (all sequential runs yield the same regret)
    mcts_plot_mask = df["outcome_codes"].isin(subset_outcome_codes) & (
        df["method"] == "mcts"
    )
    seq_plot_mask = (df["outcome_codes"] == df["outcome_codes"].min()) & (
        df["method"] == "sequential"
    )

    # Sample time-points evenly on a log scale
    t_plot = np.unique(vround(10 ** np.linspace(0, np.log10(df["N"].max()), nt_plot)))

    # Mask data for plotting
    mask = (mcts_plot_mask | seq_plot_mask) & df.t.isin(t_plot)
    plot_data = df.loc[mask].copy()
    ymax = plot_data.loc[plot_data["method"] == "mcts", "regret"].max()

    # labels = plot_data[["method", "Tree Modularity"]].apply(tuple, axis=1)
    fig = plt.figure(figsize=figsize)
    sns.lineplot(
        data=plot_data,
        x="t",
        y="regret",
        hue="Tree Modularity",
        style="Tree Modularity",
        palette="tab10",
        lw=0.5,
    )

    plt.xlabel("Samples")
    plt.xticks([10, 100, 1000, 10000])
    plt.xscale("log")

    plt.ylabel("Regret")
    plt.ylim(0, ymax)

    # plt.legend(
    #     bbox_to_anchor=(1.02, 0.9),
    #     loc="upper left",
    #     borderaxespad=0,
    #     title=r"$\mathcal{M}_T$",
    # )

    plt.legend(
        bbox_to_anchor=(0.05, 1.00),
        loc="upper left",
        borderaxespad=0,
        title=r"$\mathcal{M}_T$",
        ncol=1,
        fontsize=6.5,
    )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.plot(t_plot, t_plot / 2, "k--", lw=0.5, label="Naive")

    if save:
        save_dir = Path(save_dir)
        fname = save_dir.joinpath(f"regret_MCTS_vs_time").with_suffix(f".{fmt}")
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(str(fname), dpi=dpi, bbox_inches="tight")

    # # Plot outputs of different seeds for a specific case
    # plot_data2 = plot_data.loc[plot_data["outcome_codes"] == "00000000000000010111111111111111"].copy()

    # fig2 = plt.figure(figsize=figsize)
    # sns.lineplot(
    #     data=plot_data2,
    #     x="t",
    #     y="regret",
    #     hue="seed",
    #     palette="tab10",
    # )
    # plt.xscale("log")
    # plt.ylim(0, ymax)

    # if save:
    #     save_dir = Path(save_dir)
    #     fname = save_dir.joinpath(f"regret_MCTS_MT=68%_vs_seed").with_suffix(f".{fmt}")
    #     print(f"Writing to: {fname.resolve().absolute()}")
    #     plt.savefig(str(fname), dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    main(
        min_modularity=0.2,
        max_modularity=0.6,
        nt_plot=250,
        n_codes_plot=9,
        save=True,
    )
