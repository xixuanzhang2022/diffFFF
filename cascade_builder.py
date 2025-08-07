import pandas as pd


def lower_list(lst):
    """Lowercase all elements of a list."""
    return [str(x).lower() for x in lst]


def intersection_ordered(a, b):
    """Ordered intersection: keep elements in `a` that are in `b`, preserving `a`'s order."""
    return [x for x in a if x in b]


def get_earlybird_list(user_list, ref_id):
    """
    Create earlybird exposure list (user saw ref and prior retweeters).
    Returns a list of lists.
    """
    earlybirds = []
    for i in range(len(user_list)):
        earlybirds.append([ref_id] + user_list[:i + 1])
    return earlybirds


def get_ordered_following(users, following_dict):
    """
    Returns ordered lowercase following list per user, reversed for recency ordering.
    """
    return [lower_list(following_dict.get(user, []))[::-1] for user in users]


def get_order_saw(earlybirds, following, ref_id):
    """
    For each user: intersect their following with earlybird exposure list.
    If ref_id appears in list, and was followed later than someone else, trim the list.
    """
    order_po = [intersection_ordered(eb, f) for eb, f in zip(earlybirds, following)]
    order_saw = []
    for i in range(len(order_po)):
        saw_list = order_po[i]
        if ref_id in saw_list:
            sep = saw_list.index(ref_id)
            saw_list = saw_list[sep + 1:]  # only what was seen after ref
        order_saw.append(saw_list)
    return order_po, order_saw


def get_last_diff(saw_lists, ref_id):
    """Get the last user in the saw list or default to ref_id."""
    return [saw[-1] if saw else ref_id for saw in saw_lists]


def build_diffusion_trees(contents_posted, dict_refu, dictAll):
    """
    Build exposure trees for each original tweet.
    """
    all_sources = []
    all_targets = []
    all_refs = []
    all_post_ids = []
    all_times = []
    all_retweet_ids = []
    all_saw_lists = []

    for i, row in contents_posted.iterrows():
        content_id = row["contents"]
        users = row["users"]
        timestamps = row["time"]
        retweet_ids = row["retweets"]
        ref_user = dict_refu[content_id][0]

        df = pd.DataFrame({
            "user": users,
        })

        df["content"] = content_id
        df["following"] = get_ordered_following(users, dictAll)

        df["earlybird"] = get_earlybird_list(users, ref_user)
        df["order_fo"] = [intersection_ordered(f, e) for f, e in zip(df["following"], df["earlybird"])]
        order_po, order_saw = get_order_saw(df["earlybird"], df["following"], ref_user)
        df["order_po"] = order_po
        df["order_saw"] = order_saw
        df["diff_last"] = get_last_diff(order_po, ref_user)

        all_sources += df["user"].tolist()
        all_targets += df["diff_last"].tolist()
        all_refs += [content_id] * len(df)
        all_post_ids += [i] * len(df)
        all_times += timestamps
        all_retweet_ids += retweet_ids
        all_saw_lists += df["order_saw"].tolist()

        print(f"Processed cascade {i}/{len(contents_posted)}")

    diffusion_df = pd.DataFrame({
        "Source": all_sources,
        "Target": all_targets,
        "time": all_times,
        "ref": all_refs,
        "nr": all_post_ids,
        "retweet": all_retweet_ids,
        "saw": all_saw_lists
    })

    return diffusion_df
