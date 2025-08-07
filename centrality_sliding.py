import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from pathlib import Path


def parse_iso(timestamp):
    """Parse ISO string without the 'Z' suffix."""
    return datetime.fromisoformat(timestamp.rstrip("Z"))


def compute_sliding_centrality(
    edge_df,
    time_col="time",
    window_days=7,
    step_days=1,
    time_format='iso',
):
    """
    Computes node-level betweenness, in-degree, and out-degree centralities
    over sliding windows.

    Parameters:
        edge_df (pd.DataFrame): Edgelist with time column.
        time_col (str): Name of the time column.
        window_days (int): Width of sliding window (in days).
        step_days (int): Step size between windows.
        time_format (str): 'iso' (default) assumes ISO timestamps.

    Returns:
        pd.DataFrame: with columns [Id, time, indegree, outdegree, betweenness]
    """
    df = edge_df.copy()
    df[time_col] = df[time_col].map(parse_iso)
    df = df.sort_values(by=time_col).reset_index(drop=True)

    start = df[time_col].min()
    end = df[time_col].max()

    results = []

    i = 0
    while start + timedelta(days=window_days) <= end:
        slide_end = start + timedelta(days=window_days)
        print(f"### Window {i}: {start.date()} to {slide_end.date()} ###")

        temp = df[(df[time_col] >= start) & (df[time_col] < slide_end)]

        if temp.empty:
            print("No edges in this window.")
            start += timedelta(days=step_days)
            i += 1
            continue

        users = pd.unique(temp[["Source", "Target"]].values.ravel())
        G = nx.from_pandas_edgelist(temp, "Source", "Target", create_using=nx.DiGraph())
        G.add_nodes_from(users)

        try:
            bt = nx.betweenness_centrality(G, normalized=True)
        except Exception:
            bt = {}

        try:
            in_deg = nx.in_degree_centrality(G)
            out_deg = nx.out_degree_centrality(G)
        except Exception:
            in_deg, out_deg = {}, {}

        for node in users:
            results.append({
                "Id": node,
                "time": i,
                "indegree": in_deg.get(node, 0),
                "outdegree": out_deg.get(node, 0),
                "betweenness": bt.get(node, 0)
            })

        start += timedelta(days=step_days)
        i += 1

    return pd.DataFrame(results)
