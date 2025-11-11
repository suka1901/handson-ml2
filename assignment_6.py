import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# Load and prepare dataset
# -----------------------------
CSV_PATH = "datasets/housing/housing.csv" 
df = pd.read_csv(CSV_PATH)

features = df[["longitude", "latitude", "median_income"]].dropna()

# inspect first rows
print("Data sample:")
print(features.head())

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(features)  # scaled features for clustering

original = scaler.inverse_transform(X)
original_df = pd.DataFrame(original, columns=features.columns)
print(original_df.head())

print(f"\nScaled data shape: {X.shape}")

# -----------------------------
# Utility plotting helpers
# -----------------------------
def plot_elbow_and_silhouette(K_range, inertias, sil_scores, out_prefix="kmeans_eval"):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(K_range, inertias, 'o-', linewidth=1.5)
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow curve")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(K_range, sil_scores, 'o-', color='C1', linewidth=1.5)
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette vs k")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=150)
    plt.show()

def plot_clusters_scatter(df_plot, label_col, title, out_fname=None, legend=True, s=8):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="longitude", y="latitude", hue=label_col, data=df_plot,
                    palette="tab10", s=s, linewidth=0, legend=legend)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(title)
    if out_fname:
        plt.savefig(out_fname, dpi=150)
    plt.show()

# -----------------------------
#  K-Means++: search for best k
# -----------------------------
K_range = range(2, 11)
inertias = []
sil_scores = []

for k in K_range:
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        algorithm="lloyd",
        random_state=42
    )
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, labels)  # silhouette defined for k >= 2
    sil_scores.append(sil)
    print(f"k={k} → inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

plot_elbow_and_silhouette(list(K_range), inertias, sil_scores, out_prefix="kmeans_eval")

# Choose k by silhouette
best_k = int(K_range[np.nanargmax(sil_scores)])
print(f"\nRecommended k by silhouette = {best_k}")

# Fit final K-Means++ with recommended k
#k_value = best_k # Best K according to S_score
k_value = 4 # manual value
kmeans = KMeans(
    n_clusters=k_value,
    init="k-means++",
    n_init=20,
    algorithm="lloyd",
    random_state=42
)
k_labels = kmeans.fit_predict(X)
features["kmeans_cluster"] = k_labels

# Visualise K-Means clustering
plot_clusters_scatter(features, "kmeans_cluster", f"K-Means++ clusters (k={k_value})", out_fname=f"kmeans_k{k_value}.png")

# Compute cluster centroids 
centroids_scaled = kmeans.cluster_centers_  # scaled space
#Original unscaled data 
centroids_unscaled = scaler.inverse_transform(centroids_scaled)
centroid_table = pd.DataFrame(centroids_unscaled, columns=["longitude", "latitude", "median_income"])
centroid_table["cluster"] = centroid_table.index
print("\nCluster centroids (approxmately original units):")
print(centroid_table)

# -----------------------------
# Visualise clusters for multiple k values side-by-side
# -----------------------------
def fit_and_plot_k_set(K_list, X, features_df):
    fig, axs = plt.subplots(1, len(K_list), figsize=(5*len(K_list), 4), squeeze=False)
    for i, k in enumerate(K_list):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)
        plot_df = features_df.copy()
        plot_df[f"labels_{k}"] = labels
        ax = axs[0, i]
        sns.scatterplot(x="longitude", y="latitude", hue=f"labels_{k}", data=plot_df,
                        palette="tab10", s=8, linewidth=0, ax=ax, legend=True)
        ax.set_title(f"k={k}")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig("k_values_comparison.png", dpi=100)
    plt.show()

fit_and_plot_k_set([2, 3, 4, 5], X, features)

# -----------------------------
#  DBSCAN comparison
# -----------------------------
#  doing a small grid search over eps and min_samples. For DBSCAN silhouette we must have >1 cluster;
eps_values = np.linspace(0.05, 0.5, 10)
min_samples_list = [5, 10, 15]

best_db = {"silhouette": -1, "eps": None, "min_samples": None, "labels": None, "n_clusters": 0}
for eps in eps_values:
    for min_s in min_samples_list:
        db = DBSCAN(eps=eps, min_samples=min_s)
        db_labels = db.fit_predict(X)
        # Number of clusters ignoring noise
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        if n_clusters_db < 2:
            continue
        try:
            sil = silhouette_score(X, db_labels)
        except Exception as e:
            # silhouette can still fail if some clusters have 1 sample; skip
            continue
        if sil > best_db["silhouette"]:
            best_db.update({
                "silhouette": sil,
                "eps": eps,
                "min_samples": min_s,
                "labels": db_labels,
                "n_clusters": n_clusters_db
            })

if best_db["eps"] is None:
    print("\nDBSCAN grid search did not find a suitable parameter combination that yields >=2 clusters.")
else:
    print(f"\nBest DBSCAN: eps={best_db['eps']:.3f}, min_samples={best_db['min_samples']}, "
          f"clusters={best_db['n_clusters']}, silhouette={best_db['silhouette']:.4f}")
    features["dbscan_cluster"] = best_db["labels"]
    plot_clusters_scatter(features, "dbscan_cluster",
                          f"DBSCAN (eps={best_db['eps']:.3f}, min_samples={best_db['min_samples']})",
                          out_fname="dbscan_best.png")
    # Summary for DBSCAN
    summary_db = features.groupby("dbscan_cluster").agg(
        count=("longitude", "count"),
        median_income=("median_income", "median"),
        lon_min=("longitude", "min"),
        lon_max=("longitude", "max"),
        lat_min=("latitude", "min"),
        lat_max=("latitude", "max")
    ).reset_index().sort_values("dbscan_cluster")
    print("\nDBSCAN cluster summary:")
    print(summary_db)

# -----------------------------
# Quantitative comparison & caveats
# -----------------------------
print("\n=====--- Quantitative comparison ---======")
print(f"K-Means++ (k={k_value}) silhouette: {silhouette_score(X, k_labels):.4f}")
if best_db["eps"] is not None:
    print(f"DBSCAN best silhouette: {best_db['silhouette']:.4f} (eps={best_db['eps']:.3f}, min_samples={best_db['min_samples']})")
else:
    print("Not any meaningful clustering for the tested parameter grid.")

# -----------------------------
# Save cluster
# -----------------------------
#features.to_csv("housing_with_cluster_labels.csv", index=False)
#print("\nSaved 'housing_with_cluster_labels.csv' with cluster assignments.")
