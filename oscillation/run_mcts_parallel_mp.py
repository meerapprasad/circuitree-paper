from functools import partial
import h5py
from itertools import cycle, chain, repeat
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from time import perf_counter
from typing import Optional
from uuid import uuid4

from tf_network import TFNetworkModel
from oscillation import OscillationTree


def run_batch_and_save(
    seed_visit_genotype,
    batch_size,
    dt,
    nt,
    save_dir,
    init_columns,
    param_names,
    oscillation_thresh,
    log_dir,
    task_id=None,
):
    start_time = perf_counter()

    start_logging(log_dir, task_id)

    seed, visit_num, genotype = seed_visit_genotype
    logging.info(f"Seed: {seed}")
    logging.info(f"Visit no.: {visit_num}")
    logging.info(f"Genotype: {genotype}")
    logging.info(f"Batch_size: {batch_size}")

    start_sim_time = perf_counter()
    model = TFNetworkModel(genotype)
    y_t, pop0s, param_sets, rewards = model.run_ssa_and_get_acf_minima(
        size=batch_size,
        freqs=False,
        indices=False,
        seed=seed,
        dt=dt,
        nt=nt,
        abs=True,
    )
    end_sim_time = perf_counter()
    sim_time = end_sim_time - start_sim_time
    run_results = visit_num, genotype, y_t, pop0s, param_sets, rewards, sim_time

    logging.info(f"Simulation took {sim_time:.2f}s ")

    save_run(
        run_results=run_results,
        save_dir=save_dir,
        init_columns=init_columns,
        param_names=param_names,
        oscillation_thresh=oscillation_thresh,
    )

    return seed, visit_num, genotype, sim_time


def save_run(
    run_results,
    save_dir,
    init_columns,
    param_names,
    oscillation_thresh,
):
    visit_num, genotype, y_t, pop0s, param_sets, rewards = run_results
    state_dir = Path(save_dir).joinpath(f"state_{genotype.strip('*')}")
    state_dir.mkdir(exist_ok=True)

    save_results(
        state_dir,
        visit_num,
        genotype,
        rewards,
        pop0s,
        param_sets,
        init_columns,
        param_names,
    )

    if np.any(rewards > oscillation_thresh):
        pop_data_dir = Path(save_dir).joinpath(f"extras")
        pop_data_dir.mkdir(exist_ok=True)
        save_pop_data(
            pop_data_dir,
            genotype,
            visit_num,
            y_t,
            pop0s,
            param_sets,
            rewards,
            init_columns,
            param_names,
            oscillation_thresh,
        )

    return genotype


def save_results(
    state_dir,
    visit_num,
    genotype,
    rewards,
    pop0s,
    param_sets,
    init_columns,
    param_names,
    ext="parquet",
):
    data = (
        dict(state=genotype, visit=visit_num, reward=rewards)
        | dict(zip(init_columns, np.atleast_2d(pop0s).T))
        | dict(zip(param_names, np.atleast_2d(param_sets).T))
    )
    df = pd.DataFrame(data)
    df["state"] = df["state"].astype("category")

    fname = state_dir.joinpath(f"{uuid4()}.{ext}").resolve().absolute()

    logging.info(f"Writing results to: {fname}")

    if ext == "csv":
        df.to_csv(fname, index=False)
    elif ext == "parquet":
        df.to_parquet(fname, index=False)


def save_pop_data(
    pop_data_dir,
    genotype,
    visit_num,
    y_t,
    pop0s,
    param_sets,
    rewards,
    init_columns,
    param_names,
    thresh,
):
    save_idx = np.where(rewards > thresh)[0]
    data = (
        dict(state=genotype, visit=visit_num, reward=rewards[save_idx])
        | dict(zip(init_columns, np.atleast_2d(pop0s[save_idx]).T))
        | dict(zip(param_names, np.atleast_2d(param_sets[save_idx]).T))
    )
    df = pd.DataFrame(data)

    state_no_asterisk = genotype.strip("*")
    fname = pop_data_dir.joinpath(f"state_{state_no_asterisk}_ID#{uuid4()}.hdf5")
    fname = fname.resolve().absolute()

    logging.info(f"\tWriting all data for {len(save_idx)} runs to: {fname}")

    with h5py.File(fname, "w") as f:
        f.create_dataset("y_t", data=y_t[save_idx])
    df.to_hdf(fname, key="metadata", mode="a", format="table")


def start_logging(log_dir, task_id=None):
    if task_id is None:
        task_id = uuid4().hex
    logger, logfile = _init_logger(task_id, log_dir)
    logger.info(f"Initialized logger for task {task_id}")
    logger.info(f"Logging to {logfile}")


def _init_logging(level=logging.INFO, mode="a"):
    fmt = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s --- %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    global _log_meta
    _log_meta = {"mode": mode, "fmt": fmt}


def _init_logger(task_id, log_dir: Path):
    logger = logging.getLogger()
    logger.handlers = []  # remove all handlers
    logfile = Path(log_dir).joinpath(f"task_{task_id}.log")
    fh = logging.FileHandler(logfile, mode=_log_meta["mode"])
    fh.setFormatter(_log_meta["fmt"])
    logger.addHandler(fh)
    return logger, logfile


def prompt_before_wiping_logs(log_dir):
    while True:
        decision = input(f"Delete all files in log directory?\n\t{log_dir}\n[Y/n]: ")
        if decision.lower() in ("", "y", "yes"):
            import shutil

            shutil.rmtree(log_dir)
            log_dir.mkdir()
            break
        elif decision.lower() in ("n", "no"):
            import sys

            print("Exiting...")
            sys.exit(0)
        else:
            print(f"Invalid input: {decision}")


def main(
    log_dir: Path,
    save_dir: Path,
    batch_size: int = 100,
    nt: int = 2000,
    dt_seconds: float = 20.0,
    n_samples: Optional[int] = None,
    n_workers: Optional[int] = None,
    print_every: int = 50,
    oscillation_thresh: float = 0.35,
    seed_start: int = 0,
    shuffle_seed: Optional[int] = None,
):
    init_columns = ["A_0", "B_0", "C_0"]
    param_names = [
        "k_on",
        "k_off_1",
        "k_off_2",
        "km_unbound",
        "km_act",
        "km_rep",
        "km_act_rep",
        "kp",
        "gamma_m",
        "gamma_p",
    ]
    components = ["A", "B", "C"]
    interactions = ["activates", "inhibits"]
    root = "ABC::"
    tree = OscillationTree(
        components=components,
        interactions=interactions,
        root=root,
        dt=dt_seconds,
        nt=nt,
    )
    tree.grow_tree()

    # Split up the BFS into rounds
    # In each round, we will run n_samples simulations for each genotype.
    # Samples will be run and results saved in batches of size ``save_every``.
    # This ensures that workers with shorter simulations can steal work periodically.
    if n_workers is None:
        n_workers = cpu_count(logical=True)

    # Cycle through nodes in BFS order, taking n_batches_per_cycle batches of samples
    # from each node. n_cycles is set by balancing two factors. More cycles (fewer
    # samples per cycle) allows us to gradually accumulate data on all genotypes, rather
    # than one-by-one. However, it also means that for every cycle, we will end up JIT-
    # compiling the models again.
    bfs_arr = np.array([n for n in tree.bfs_iterator() if tree.is_terminal(n)])
    if shuffle_seed is not None:
        rg = np.random.default_rng(shuffle_seed)
        rg.shuffle(bfs_arr)

    # Start logging
    logging.basicConfig(filename=Path(log_dir).joinpath("main.log"), level=logging.INFO)

    # Run indefinitely or until a fixed number of samples
    if n_samples is None:
        bfs = cycle(bfs_arr.tolist())
        n_batches = None
        iter_seeds_and_genotypes = enumerate(cycle(bfs), start=seed_start)
        _msg = (
            f"Using {n_workers} workers to sample each genotype in batches of size "
            f"{batch_size}. Sampling will continue indefinitely."
        )
    else:
        n_batches, mod = divmod(n_samples, batch_size)
        if mod != 0:
            raise ValueError(
                f"n_samples ({n_samples}) must be divisible by batch_size ({batch_size})"
            )
        bfs = chain.from_iterable(repeat(bfs_arr.tolist(), n_batches))
        iter_seeds_and_genotypes = enumerate(
            chain.from_iterable(repeat(bfs, n_batches)), start=seed_start
        )
        _msg = (
            f"Using {n_workers} workers to make {n_samples} samples for each genotype "
            f"({n_batches} batches of size {batch_size})."
        )

    logging.info(_msg)
    print(_msg)

    run_batch_job = partial(
        run_batch_and_save,
        batch_size=batch_size,
        dt=dt_seconds,
        nt=nt,
        save_dir=save_dir,
        init_columns=init_columns,
        param_names=param_names,
        oscillation_thresh=oscillation_thresh,
        log_dir=log_dir,
    )

    with Pool(n_workers, initializer=_init_logging, initargs=(logging.DEBUG,)) as pool:
        k = 0
        for seed, genotype, simulation_time, total_time in pool.imap_unordered(
            run_batch_job, iter_seeds_and_genotypes
        ):
            if k % print_every == 0:
                print(f"Finished {k} batches")
                logging.info(f"Finished {k} batches")

            logging.info(
                f"Batch {seed - seed_start} took {total_time:.2f}s "
                f"({simulation_time:.2f}s SSA) for {genotype}"
            )
            k += 1


if __name__ == "__main__":
    save_dir = Path("data/oscillation/bfs")
    save_dir.mkdir(exist_ok=True)

    log_dir = Path("logs/oscillation/bfs")
    log_dir.mkdir(exist_ok=True)
    if any(log_dir.iterdir()):
        prompt_before_wiping_logs(log_dir)

    main(
        save_dir=save_dir,
        log_dir=log_dir,
        # n_samples=10000,
        batch_size=2,
        nt=2000,
        dt_seconds=20.0,
        n_workers=4,
        print_every=50,
        # oscillation_thresh=0.35,
        shuffle_seed=2023,
    )
