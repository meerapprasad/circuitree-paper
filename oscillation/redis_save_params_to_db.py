from itertools import islice
import json
import redis
import pandas as pd
from pathlib import Path

from oscillation_app import app


def _split_seq_into_chunks(seq, chunk_sizes):
    it = iter(seq)
    return [list(islice(it, size)) for size in chunk_sizes]


def main(
    param_sets_csv: str | Path,
    n_init_columns: int,
    n_parameters: int = 10,
    database_url: str | Path = None,
    index_column: str = "param_index",
    table_name: str = "parameter_table",
    extra_columns: list[str] = None,
):
    if extra_columns is None:
        extra_columns = []
    n_extra_columns = len(extra_columns)

    print("Loading parameter sets")
    params_df = pd.read_csv(param_sets_csv, index_col=index_column)

    columns = params_df.columns
    ncol = len(columns)
    ncol_expected = 1 + n_init_columns + n_parameters + n_extra_columns
    if ncol != ncol_expected:
        raise ValueError(
            f"Expected dataframe with {ncol_expected} columns, not {ncol}."
        )
    params_df = params_df.sort_index()

    print("Connecting to database")
    database_url = database_url or app.conf["broker_url"]
    database: redis.Redis = redis.Redis.from_url(database_url)

    # JSON-serialize each row as a [seed, prots0, params[, extra_cols]] tuple
    chunk_sizes = [1, n_init_columns, n_parameters]
    params_mapping = {}
    if n_extra_columns > 0:
        chunk_sizes.append(n_extra_columns)
        for pidx, *row in params_df.itertuples():
            (seed,), prots0, params, extra_data = _split_seq_into_chunks(
                row, chunk_sizes
            )
            params_mapping[str(pidx)] = json.dumps([seed, prots0, params, extra_data])
    else:
        for pidx, *row in params_df.itertuples():
            (seed,), prots0, params, *extra_data = _split_seq_into_chunks(
                row, chunk_sizes
            )
            params_mapping[str(pidx)] = json.dumps([seed, prots0, params])

    # Store in database
    clear_before_storing = (
        input("Delete parameter table in database before storing? [y/N] ").lower()
        == "y"
    )
    if clear_before_storing:
        print("Deleting parameter table...")
        _ = database.delete(table_name)
    else:
        print("Not clearing database.")
    print("Storing parameter sets in database...")
    _ = database.hset(table_name, mapping=params_mapping)

    print("Done.")


if __name__ == "__main__":
    param_sets_csv = Path(
        # "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000.csv"
        # "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000_5tf.csv"
        # "~/git/circuitree-paper/data/oscillation/231020_param_sets_10000_5tf.csv"
        # "~/git/circuitree-paper/data/oscillation"
        # "/231104_param_sets_10000_5tf_pmutation0.5.csv"
        "/home/mprasad/PycharmProjects/HomeostasisCircuits_v1/circuitree-paper/oscillation/data/"
        "oscillation/240415_param_sets_10000_5tf_pmutation0.5.csv"
    ).expanduser()

    main(
        param_sets_csv=param_sets_csv,
        n_init_columns=5,
        n_parameters=10,
        extra_columns=["component_mutation"],
    )
