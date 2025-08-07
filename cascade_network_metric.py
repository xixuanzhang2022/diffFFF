import pandas as pd
import networkx as nx
import collections
from pathlib import Path

# File paths
BASE = Path("/Users/xixuan/Desktop/twitter_test/fff_api_alltweets")
EDGE_FILE = BASE / "diff_gephi.csv"
TWEET_FILE = BASE / "tweet_df_ordered_real.csv"
LABEL_FILE = BASE / "tweet_df_classified14.csv"
OUTPUT_FILE = BASE / "network_metrics.csv"

# Load data
edges = pd.read_csv(EDGE_FILE, encoding="utf-8")
tweets = pd.read_csv(TWEET_FILE, encoding="utf-8")

# Map ref to refu (original tweet)
ref_map = tweets.groupby("ref")["refu"].first().to_dict()
edges["refu"] = edges["ref"].map(ref_map)
edges = edges.sort_values("nr").reset_index(drop=True)

# Initialize results
records = []

# Process one cascade (network) per unique 'nr'
for nr, group in edges.groupby("nr"):
    ref_user = group["refu"].iloc[0]
    users = pd.unique(group[["Source", "Target"]].values.ravel())
    user_count = len(users)

    G = nx.from_pandas_edgelist(group, "Source", "Target", create_using=nx.DiGraph())
    G.add_nodes_from(users)
    UG = G.to_undirected()

    # Depth: longest shortest-path from the root (if reachable)
    try:
        bfs_lengths = nx.single_source_shortest_path_length(UG, ref_user)
        depth = max(bfs_lengths.values())
        breadth = collections.Counter(bfs_lengths.values()).most_common(1)[0][1]
    except Exception:
        depth = 0
        breadth = 0

    # Virality: average pairwise shortest path
    try:
        total_dist = sum(sum(nx.single_source_shortest_path_length(UG, u).values()) for u in users)
        virality = total_dist / (user_count * (user_count - 1)) if user_count > 1 else 0
    except Exception:
        virality = 0

    records.append({
        "Origin": ref_user,
        "nr": nr,
        "contents": group["ref"].iloc[0],
        "size": user_count,
        "depth": depth,
        "max_breadth": breadth,
        "virality": virality
    })

# Store metrics
metrics_df = pd.DataFrame(records)
metrics_df.to_csv(OUTPUT_FILE, sep=";", index=False)
print(f"Saved network metrics to: {OUTPUT_FILE}")
