import pandas as pd
import numpy as np
from pathlib import Path
import umap
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from collections import Counter

# Global paths
BASE_PATH = Path("/Users/xixuan/Desktop/twitter_test/fff_api_alltweets")
PERCENT = '01'
VAR = 'cased'
RANDOM_STATE = 42
CLUSTER_COUNT = 2
SAMPLE_LIMIT = 2000  # limit to speed up; set to None for full

def load_data(percent: str, var: str, tweet_df_path: Path, base_path: Path):
    data = pd.read_pickle(base_path / f"data_preprocessed_combined{percent}.pkl")
    embeddings = np.load(base_path / f"embeddings_combined_{var}{percent}.npy")

    # Drop empty cleaned rows
    data['cleaned'] = data['cleaned'].str.strip()
    data = data[data['cleaned'].str.len() > 0]
    embeddings = embeddings[data.index]
    data = data.reset_index(drop=True)

    # Merge tweet metadata
    tweet_df = pd.read_csv(tweet_df_path, encoding="utf-8")
    tweet_df = tweet_df.loc[data["id"].values].reset_index(drop=True)

    data["user"] = tweet_df["user"]
    data["time"] = tweet_df["date"]
    data["ref"] = tweet_df["ref"]

    return data, embeddings


def reduce_embeddings(embeddings, n_components=96, random_state=42):
    reducer = umap.UMAP(
        n_components=n_components,
        metric='cosine',
        learning_rate=0.5,
        init='spectral',
        random_state=random_state,
        force_approximation_algorithm=True,
        unique=True
    )
    return reducer.fit_transform(embeddings)


def optimize_kmedoids_clusters(embeddings, min_k, max_k, step=1, random_state=42):
    results = []
    max_k = min(max_k, len(embeddings))
    for k in range(min_k, max_k, step):
        model = KMedoids(n_clusters=k, metric='euclidean', init='k-medoids++',
                         max_iter=200, random_state=random_state).fit(embeddings)
        score = silhouette_score(embeddings, model.labels_)
        print(f"n_clusters: {k} → silhouette: {score:.4f}")
        results.append({'n_clusters': k, 'silhouette_avg': score})
    return pd.DataFrame(results)


def apply_kmedoids(embeddings, k, random_state=42):
    model = KMedoids(n_clusters=k, metric='cosine', init='k-medoids++',
                     max_iter=200, random_state=random_state).fit(embeddings)
    return model


def refine_clusters(data, embeddings_umap, min_cluster_size, random_state=42):
    current_labels = set(data["label"])
    for parent_label in current_labels:
        indices = data.index[data["label"] == parent_label].tolist()
        if len(indices) >= min_cluster_size:
            emb_subset = embeddings_umap[indices, :]
            sub_results = optimize_kmedoids_clusters(
                emb_subset, min_k=2, max_k=max(3, len(indices)//10), step=1,
                random_state=random_state
            )
            best_k = sub_results.loc[sub_results['silhouette_avg'].idxmax(), 'n_clusters']
            model = apply_kmedoids(emb_subset, best_k, random_state)

            label_prefix = str(parent_label)
            for i, idx in enumerate(indices):
                data.loc[idx, "label"] = f"{label_prefix}{model.labels_[i]}_"
            medoid_indices = [indices[i] for i in model.medoid_indices_]
            data.loc[medoid_indices, "medoid"] = 1
    return len(set(data["label"]))


def hierarchical_clustering(data, embeddings_umap, min_cluster_size, random_state=42):
    current_clusters = len(set(data["label"]))
    while True:
        updated_clusters = refine_clusters(data, embeddings_umap, min_cluster_size, random_state)
        if updated_clusters == current_clusters:
            break
        current_clusters = updated_clusters
    return data


def main():
    print("Loading tweet metadata...")
    tweet_df_path = BASE_PATH / "tweet_df_ordered.csv"
    print("Loading text + embeddings...")
    data, embeddings = load_data(PERCENT, VAR, tweet_df_path, BASE_PATH)

    if SAMPLE_LIMIT:
        data = data[:SAMPLE_LIMIT]
        embeddings = embeddings[:SAMPLE_LIMIT]

    print("Reducing dimensions with UMAP...")
    embeddings_umap = reduce_embeddings(embeddings, n_components=96, random_state=RANDOM_STATE)

    print(f"Applying initial KMedoids with k={CLUSTER_COUNT}...")
    model = apply_kmedoids(embeddings_umap, CLUSTER_COUNT, RANDOM_STATE)
    data["label"] = [f"{l}_" for l in model.labels_]
    data["medoid"] = 0
    data.loc[model.medoid_indices_, "medoid"] = 1

    print("Refining clusters recursively...")
    min_cluster_size = max(len(data) // 100, 10)
    data = hierarchical_clustering(data, embeddings_umap, min_cluster_size, RANDOM_STATE)

    print(f"Final cluster distribution: {Counter(data['label'])}")
    output_path = BASE_PATH / f"test_data{PERCENT}_clustered.pkl"
    data.to_pickle(output_path)
    print(f"Clustered data saved to {output_path}")


if __name__ == "__main__":
    main()
