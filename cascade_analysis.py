import pandas as pd
import collections


def label_retweets_by_content(cascade_df, classification_df):
    """Attach manual labels to cascades based on the `ref` (original post)."""
    label_map = classification_df.groupby("post")["label_manual"].first().to_dict()
    cascade_df["label"] = cascade_df["ref"].map(label_map)
    return cascade_df


def assign_modularity_groups(cascade_df, user_modularity_df):
    """Attach modularity class info to Source, Target, Refuser (O) users."""
    mod_map = user_modularity_df.groupby("Id")["modularity_class"].first().to_dict()

    for col in ["Source", "Target", "refu"]:
        key = f"{col[0]}_modularity"
        cascade_df[key] = cascade_df[col].map(lambda u: mod_map.get(u, "999"))

    return cascade_df


def mark_top_users(cascade_df, centrality_df, top_k=0.01):
    """Mark top-k% users by centrality as 'top' in Source/Target columns."""
    top_n = int(len(centrality_df) * top_k)
    top_users = set(centrality_df.sort_values(by='eigencentrality', ascending=False).head(top_n)["Id"])

    cascade_df["S_top"] = cascade_df["Source"].isin(top_users).astype(int)
    cascade_df["T_top"] = cascade_df["Target"].isin(top_users).astype(int)

    return cascade_df


def count_group_distribution(cascade_df, group_col):
    """
    Count number of rows per group in a specified column.
    Returns a DataFrame of proportions.
    """
    counter = collections.Counter(cascade_df[group_col])
    count_df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
    count_df["percent"] = count_df["count"] / len(cascade_df)
    return count_df.reset_index(names=group_col)


def compute_in_group_sharing_rate(cascade_df):
    """
    Compute the percentage of in-group retweets (same modularity between source and target).
    """
    same_group = cascade_df[cascade_df["S_modularity"] == cascade_df["T_modularity"]]
    return len(same_group) / len(cascade_df)


def compute_direct_exposure_rate(cascade_df):
    """Percentage of tweets seen without any intermediary (sawgu = 'direct')."""
    return len(cascade_df[cascade_df["sawgu"] == "direct"]) / len(cascade_df)


def compute_indirect_exposure_rate(cascade_df):
    """Percentage of users exposed indirectly through intermediaries."""
    indirect = cascade_df[cascade_df["sawgu"] != "direct"]
    indirect_via_intermediary = indirect[indirect["level"] == 0]
    return len(indirect_via_intermediary) / len(cascade_df)


def groupwise_top_exposure(cascade_df, group_col="S_modularity", saw_col="sawgu", top_n=5):
    """
    For each group, get top N exposure sources (from 'sawgu' column).
    Returns a DataFrame with relative proportions.
    """
    results = []

    for group in cascade_df[group_col].unique():
        group_df = cascade_df[cascade_df[group_col] == group]
        counter = collections.Counter(group_df[saw_col])
        total = sum(counter.values())
        top_items = counter.most_common(top_n)

        for i, (source, count) in enumerate(top_items):
            results.append({
                "group": group,
                "rank": i + 1,
                "source": source,
                "count": count,
                "percent": count / total
            })

    return pd.DataFrame(results)
