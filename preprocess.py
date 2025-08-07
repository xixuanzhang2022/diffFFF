import pandas as pd


def filter_retweets(tweet_df):
    """Filter for retweeted tweets only."""
    return tweet_df[tweet_df["type"].astype(str) == "retweeted"]


def build_ref_to_retweets(retweet_df):
    """
    Group retweets by reference post ID.
    Returns a dict mapping ref_id → list of retweet post_ids.
    """
    return retweet_df.groupby("ref")["post"].apply(list).to_dict()


def filter_valid_refs(ref_dict, valid_refs):
    """
    Remove entries from ref_dict whose keys are not in valid_refs.
    """
    invalid_keys = set(map(int, ref_dict.keys())) - set(valid_refs)
    for key in invalid_keys:
        ref_dict.pop(str(key), None)
    return ref_dict


def flatten_dict_values(d):
    """Flatten nested lists from dictionary values into a single list."""
    return [item for sublist in d.values() for item in sublist]


def clean_usernames(users_df):
    """
    Lowercase usernames, drop duplicates on ID (column 1), and
    return a mapping from user ID to lowercase username.
    """
    users_df[2] = users_df[2].astype(str).str.lower()
    users_df = users_df.drop_duplicates(subset=[1])
    return users_df.groupby(1)[2].apply(list).to_dict()


def update_usernames(retweet_df, ref_user_map, user_map):
    """
    Add resolved usernames to 'refu' (retweeted-from user)
    and 'user' (current user) columns.
    """
    retweet_df['refu'] = retweet_df['ref'].astype(int).map(lambda ref: ref_user_map.get(ref, [None])[0])
    retweet_df['refu'] = retweet_df['refu'].map(lambda uid: user_map.get(uid, [None])[0])
    retweet_df['user'] = retweet_df['user'].astype(int).map(lambda uid: user_map.get(uid, [None])[0])
    return retweet_df


def finalize_retweet_columns(retweet_df):
    """Rename columns and return final cleaned DataFrame."""
    retweet_df.columns = ['date', 'post', 'author', 'content', 'type', 'ref', 'refu', 'user']
    return retweet_df


def preprocess_retweets(tweet_df, umap_df, users_df):
    """Main preprocessing pipeline."""
    retweet_df = filter_retweets(tweet_df)

    # Build ref -> [retweet_post_ids]
    ref_to_retweets = build_ref_to_retweets(retweet_df)

    # Remove ref entries that don't exist in umap
    valid_posts = umap_df["post"].tolist()
    ref_to_retweets = filter_valid_refs(ref_to_retweets, valid_posts)

    # Flatten valid retweet IDs
    flattened_retweets = flatten_dict_values(ref_to_retweets)

    # Filter retweet_df to valid entries only
    retweet_df = retweet_df[retweet_df["post"].isin(flattened_retweets)]

    # Build post -> user map from umap
    ref_user_map = umap_df.groupby("post")["user"].apply(list).to_dict()

    # Build user ID -> lowercase username map
    user_map = clean_usernames(users_df)

    # Add username columns
    retweet_df = update_usernames(retweet_df, ref_user_map, user_map)

    # Rename columns to final format
    retweet_df = finalize_retweet_columns(retweet_df)

    return retweet_df
