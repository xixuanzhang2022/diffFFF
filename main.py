import os
from pathlib import Path
from load_data import load_all_data
from preprocess import preprocess_retweets
from cascade_builder import build_diffusion_trees
from analysis import (
    label_retweets_by_content,
    assign_modularity_groups,
    mark_top_users,
    count_group_distribution,
    compute_in_group_sharing_rate,
    compute_direct_exposure_rate,
    compute_indirect_exposure_rate,
    groupwise_top_exposure
)
from visualization import (
    plot_ccdf,
    plot_sliding_window,
    plot_reinforcement
)

# ========== CONFIGURATION ==========
BASE_DIR = Path("/Users/xixuan/Desktop/twitter_test/fff_api_alltweets")
PIC_DIR = BASE_DIR / "pic"
PIC_DIR.mkdir(exist_ok=True)

TOP_K_PERCENT = 0.01  # Top centrality users

# ========== STEP 1: Load data ==========
print("Loading data...")
data = load_all_data(BASE_DIR)

# ========== STEP 2: Preprocess retweets ==========
print("Preprocessing retweets...")
retweet_df = preprocess_retweets(
    tweet_df=data["tweets"],
    umap_df=data["umap"],
    users_df=data["users"]
)

# ========== STEP 3: Build cascades ==========
print("Building diffusion trees...")
# Contents_posted grouped data structure
contents_posted = retweet_df.groupby("ref").agg({
    "user": list,
    "date": list,
    "refu": list,
    "post": list
}).reset_index().rename(columns={"ref": "contents", "user": "users", "date": "time", "refu": "refu", "post": "retweets"})

dict_refu = retweet_df.groupby("ref")["refu"].apply(list).to_dict()
diffusion_df = build_diffusion_trees(contents_posted, dict_refu, data["name_dict"])

# ========== STEP 4: Label + Community Info ==========
print("Attaching labels and modularity groups...")
diffusion_df = label_retweets_by_content(diffusion_df, data["classified"])
diffusion_df = assign_modularity_groups(diffusion_df, data["user_modularity"])
diffusion_df = mark_top_users(diffusion_df, data["centrality"], top_k=TOP_K_PERCENT)

# ========== STEP 5: Descriptive Analysis ==========
print("Running analysis...")
group_dist = count_group_distribution(diffusion_df, group_col="S_modularity")
ingroup_rate = compute_in_group_sharing_rate(diffusion_df)
direct_rate = compute_direct_exposure_rate(diffusion_df)
indirect_rate = compute_indirect_exposure_rate(diffusion_df)
top_exposures = groupwise_top_exposure(diffusion_df)

print("\n=== Summary Statistics ===")
print("In-group sharing rate:", round(ingroup_rate, 3))
print("Direct exposure rate:", round(direct_rate, 3))
print("Indirect exposure via intermediaries:", round(indirect_rate, 3))
print("\nTop exposure types by group:")
print(top_exposures)

# ========== STEP 6: Visualization ==========
print("Generating plots...")

# CCDF of exposure count by community
plot_ccdf(
    df=diffusion_df,
    var="sawl",
    group_col="S_modularity",
    top_labels=["right", "fff", "liberalleft"],
    save_path=PIC_DIR / "ccdf_exposure_by_community.png"
)

# Timeline of topic spread
plot_sliding_window(
    df=diffusion_df,
    time_col="time",
    label_col="label",
    window_days=7,
    step_days=1,
    save_path=PIC_DIR / "label_timeline.png"
)

# Reinforcement plot
plot_reinforcement(
    df=diffusion_df,
    group_col="S_modularity",
    var="sawl",
    time_col="time",
    save_path=PIC_DIR / "reinforcement.png"
)

print("Done.")
