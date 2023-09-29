from datetime import date
from typing import Optional
from circuitree.viz import plot_network
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

from oscillation import OscillationGrammar


def main(
    results_csv: Path,
    state_column: str = "state",
    p_osc_column: str = "p_oscillation",
    figsize: Optional[tuple[float, float]] = None,
    plot_motifs: tuple = ("AI", "AAI", "III"),
    plot_shape: tuple = (4, 5),
    save_dir: Path = Path("figures/oscillation"),
    bar_xshrink: float = 2 / 7,
    bar_yshrink: float = 1.0,
    bar_label_every: int = 1,
    n_bar: Optional[int] = None,
    save: bool = False,
    fmt: str = "png",
    dpi: int = 300,
    tree_kwargs: dict = {},
    text_dy=0.15,
    textscale=1.0,
    xlim=(-1.95, 1.65),
    ylim=(-1.4, 1.9),
    suptitle: bool = True,
    **kwargs,
):
    plot_rows, plot_cols = plot_shape
    n_plot = plot_rows * plot_cols

    # Load in table of topologies with oscillation probabilities
    df = pd.read_csv(results_csv, index_col=state_column)
    df = df.sort_values(p_osc_column, ascending=False)
    df["rank"] = pd.Categorical(range(1, df.shape[0] + 1))
    df_plot = df.iloc[:n_plot]

    _kw = dict(components=("A", "B", "C"), interactions=("activates", "inhibits"))
    _kw |= tree_kwargs
    grammar = OscillationGrammar(**_kw)
    motifs = {}
    for motif in plot_motifs:
        if motif in df_plot.columns:
            motifs[motif] = df_plot[motif]
        else:
            has_motif = [grammar.has_motif(s, motif) for s in df_plot.index]
            mseries = pd.Series(has_motif, index=df_plot.index, name=motif)
            motifs[motif] = mseries

    figsize = figsize or (plot_cols, plot_rows * 0.8)
    fig = plt.figure(figsize=figsize)

    title_font = dict(
        ha="center",
        va="top",
        size=textscale * 6,
        color="darkred",
        # weight="bold",
    )
    text_kw = dict(
        ha="left",
        va="top",
        size=textscale * 6,
        color="gray",
        weight="bold",
        # style="italic",
    )
    num_kw = dict(
        ha="center",
        va="center",
        size=textscale * 5,
        color="white",
        weight="bold",
    )

    num_x = 0.125
    num_y = 0.85
    text_x = 0.90
    for i, state in enumerate(df_plot.index):
        ax = fig.add_subplot(plot_rows, plot_cols, i + 1)
        plot_network(*grammar.parse_genotype(state), ax=ax, **kwargs)
        p_osc = df_plot[p_osc_column][state]
        # ax.set_title(rf"$Q=${p_osc:.4f}", **title_font)
        # ax.text(
        #     x=(xlim[0] + xlim[1]) / 2,
        #     y=ylim[0],
        #     s=rf"$Q=${p_osc:.4f}",
        #     **title_font,
        # )
        ax.scatter(
            num_x,
            num_y,
            s=130,
            c="gray",
            zorder=0,
            lw=0,
            marker="s",
            transform=ax.transAxes,
        )
        ax.text(num_x, num_y, str(i + 1), transform=ax.transAxes, **num_kw)

        text_y = 0.95
        if plot_motifs:
            state_has_motifs = [motifs[m][state] for m in plot_motifs]
            for m, has_m in zip(plot_motifs, state_has_motifs):
                if has_m:
                    ax.text(text_x, text_y, m, transform=ax.transAxes, **text_kw)
                    text_y -= text_dy
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if suptitle:
        plt.suptitle(
            "Topologies with highest oscillation probability", size=10 * textscale
        )
    plt.tight_layout()

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_oscillating_topologies.{fmt}"
        fpath = Path(save_dir).joinpath(fname).resolve().absolute()
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi)

    n_bar = n_bar or n_plot
    df_bars = df.iloc[:n_bar].copy()
    df_bars["rank"] = df_bars["rank"].cat.remove_unused_categories()

    # Skip every nth bar label if labels are cluttered
    yticklabels = [""] * n_bar
    yticklabels[::bar_label_every] = [
        f"{i}" for i in range(1, n_bar + 1, bar_label_every)
    ]
    figsize = (figsize[0] * bar_xshrink, figsize[1] * bar_yshrink)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    bars = sns.barplot(data=df_bars, x=p_osc_column, y="rank", color="gray", ax=ax)
    ax.set_ylabel(None)
    ax.set_yticks(range(n_bar))
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(r"$Q$")
    ax.set_xticks([0, 0.4, 0.8])
    sns.despine()
    plt.tight_layout()

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_oscillating_topologies_barchart.{fmt}"
        fpath = Path(save_dir).joinpath(fname).resolve().absolute()
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    results_csv = Path("data/oscillation/230717_motifs.csv")
    main(
        results_csv=results_csv,
        save=True,
        textscale=1.4,
        text_dy=0.25,
        suptitle=False,
        fontsize=6,
        padding=0.5,
        lw=1,
        node_shrink=0.7,
        offset=0.8,
        auto_shrink=0.9,
        width=0.005,
        plot_labels=False,
        fmt="eps",
        plot_shape=(5, 2),
        bar_xshrink=0.6,
        bar_yshrink=0.75,
        bar_label_every=5,
        n_bar=25,
    )
