from typing import Optional
from circuitree.parallel import search_mcts_in_thread
from datetime import datetime
from functools import partial
from pathlib import Path

import gevent

from oscillation_multithreaded import (
    BatchedOscillationTree,
    MultithreadedOscillationTree,
    progress_and_backup_in_thread,
    # progress_backup_and_sync,
)


def main(
    save_dir: Path,
    backup_dir: Optional[Path] = None,
    threads: int = 16,
    batch_size: int = 1,
    n_steps_per_thread: int = 10_000,
    max_interactions: int = 12,
    exploration_constant: Optional[float] = None,
    callback_every: int = 1,
    progress_every: int = 10,
    backup_every: int = 3600,
    n_tree_backups: int = 1000,
    cache_maxsize: int = 2**20,
    logger=None,
    dry_run: bool = False,
    seed: Optional[int] = None,
):
    if threads == 0:
        run_in_main_thread = True
        threads = 1
    else:
        run_in_main_thread = False

    mtree_kw = dict(
        root="ABCDE::",
        components=["A", "B", "C", "D", "E"],
        interactions=["activates", "inhibits"],
        max_interactions=max_interactions,
        exploration_constant=exploration_constant,
        dt=20.0,  # seconds
        nt=2000,
        save_dir=save_dir,
        threads=threads,
        logger=logger,
        batch_size=batch_size,
        cache_maxsize=cache_maxsize,
    )
    if seed is not None:
        mtree_kw["seed"] = seed
    print('trying to build')
    mtree = BatchedOscillationTree(**mtree_kw)

    print(f"Making a search tree with {threads} search threads.")
    print(f"Saving results to {save_dir}")
    print()

    if run_in_main_thread:
        callback = partial(
            # progress_backup_and_sync,
            progress_and_backup_in_thread,
            backup_dir=backup_dir,
            progress_every=1,
            backup_every=1,
            n_tree_backups=n_tree_backups,
            backup_results=True,
            dry_run=dry_run,
        )
        run_search = partial(
            search_mcts_in_thread,
            mtree=mtree,
            n_steps=n_steps_per_thread,
            callback=callback,
            callback_every=1,
            return_metrics=False,
        )
        start_msg = "Running search in main thread...\n"
        print(start_msg)
        mtree.logger.info(start_msg)
        run_search(0)

        finish_msg = (
            f"Finished searches in {threads} threads. Saving results to {save_dir}"
        )
        print(finish_msg)
        mtree.logger.info(finish_msg)

    else:
        callback = partial(
            # progress_backup_and_sync,
            progress_and_backup_in_thread,
            progress_every=progress_every,
            backup_dir=backup_dir,
            backup_every=backup_every,
            n_tree_backups=n_tree_backups,
            backup_results=True,
            dry_run=dry_run,
        )

        run_search = partial(
            search_mcts_in_thread,
            mtree=mtree,
            n_steps=n_steps_per_thread,
            callback=callback,
            callback_every=callback_every,
            return_metrics=False,
        )
        start_msg = f"Starting searches in {threads} threads...\n"
        print(start_msg)
        mtree.logger.info(start_msg)
        gthreads = [gevent.spawn(run_search, i) for i in range(threads)]
        gevent.joinall(gthreads)

        finish_msg = (
            f"Finished searches in {threads} threads. Saving results to {save_dir}"
        )
        print(finish_msg)
        mtree.logger.info(finish_msg)
        callback(mtree=mtree, iteration=-1, force_backup=True, skip_sync=True)


if __name__ == "__main__":
    from datetime import datetime
    import logging

    batch_size = 1

    # Default is sqrt(2)
    exploration_constant = 2.0
    max_interactions = 15

    now = datetime.now().strftime("%y%m%d-%H-%M-%S")
    # save_dir = Path(
    #     f"~/git/circuitree-paper/data/oscillation/mcts/"
    #     f"{now}_5tf_exhaustion_mutationrate0.5_batch{batch_size}"
    #     f"_max_interactions{max_interactions}"
    #     f"_exploration{exploration_constant:.3f}"
    # ).expanduser()

    save_dir = Path(
        f"data/oscillation/mcts/"
        f"{now}_5tf_exhaustion_mutationrate0.5_batch{batch_size}"
        f"_max_interactions{max_interactions}"
        f"_exploration{exploration_constant:.3f}"
    ).expanduser()

    save_dir.mkdir()
    backup_dir = save_dir.joinpath("backups")
    backup_dir.mkdir()

    log_file = Path(f"data/logs/worker-logs/main.log").expanduser()
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=str(log_file),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        # level=logging.DEBUG,
    )
    logger = logging.getLogger(__name__)
    logger.info("Running main() program.")
    print('pre-main')
    main(
        logger=logger,
        save_dir=save_dir,
        max_interactions=max_interactions,
        backup_dir=backup_dir,
        batch_size=batch_size,
        exploration_constant=exploration_constant,
        # cache_maxsize=1,
        # threads=0,
        # threads=3,
        # threads=30,
        threads=1_000,
        n_steps_per_thread=2_000,
        # dry_run=True,
        # progress_every=1,
        progress_every=10,
        # backup_every=60,
        # backup_every=3600,
        backup_every=7200,
        # callback_every=1,
        callback_every=10,
        # callback_every=20,
        seed=2026,
    )
