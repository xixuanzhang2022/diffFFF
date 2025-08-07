import pandas as pd
import pickle
from pathlib import Path


def load_csv(path, **kwargs):
    """Load CSV with default UTF-8 encoding."""
    return pd.read_csv(path, encoding='utf-8', **kwargs)


def load_pickle(path):
    """Load a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_all_data(base_dir):
    """
    Load all required data files and return as dictionary.
    Paths are relative to the given base directory.
    """
    base = Path(base_dir)

    data = {
        "tweets": load_csv(base / "tweet_df_ordered.csv", dtype={'ref': "string"}),
        "umap": load_csv(base / "tweet_umap.csv", dtype={'ref': "string"}),
        "users": load_csv(base / "users_df.csv", header=None),
        "name_dict": load_pickle(base / "dict_name.pkl"),
        "diff_gephi": load_csv(base / "diff_gephi.csv"),
        "diff_sto": load_csv(base / "diff_gephi_sto.csv"),
        "diff_sto_cross": load_csv(base / "diff_gephi_sto_cross.csv"),
        "classified": load_csv(base / "tweet_df_classified14.csv"),
        "user_modularity": load_csv(base / "gephi_mod1.1_380_439+des.csv"),
        "centrality": load_csv(base / "diff_gephi_centrality.csv"),
        "diff_gephi_mod": load_csv(base / "diff_gephi_mod.csv"),
    }

    return data
