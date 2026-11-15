"""
Part 3 — Segmentation and dimensionality reduction
Uses all 800 rows and X1-X5 from the Flipkart records dataset.

Expected columns:
X1 customer_tenure_months
X2 complaint_count_90d
X3 discount_order_share
X4 avg_order_value_inr_hundreds
X5 loyalty_score
outcome: churned
value: order_value_inr_hundreds
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Locate the dataset. Prefer records.csv in the project folder.
candidates = [
    BASE_DIR / "records.csv",
    BASE_DIR / "records(2).csv",
    BASE_DIR / "records (2).csv",
]
DATA_PATH = next((p for p in candidates if p.exists()), None)
if DATA_PATH is None:
    csv_candidates = [
        p for p in BASE_DIR.glob("*.csv")
        if p.name not in {
            "part3_segments.csv", "part3_hierarchical.csv",
            "part2_losses.csv", "part2_weights.csv",
            "part2_thresholds.csv", "part2_knn_cv.csv"
        }
    ]
    if len(csv_candidates) == 1:
        DATA_PATH = csv_candidates[0]
    else:
        raise FileNotFoundError(
            "Could not identify the dataset. Put records.csv (or records(2).csv) "
            "in the same folder as part3.py."
        )

df = pd.read_csv(DATA_PATH)
if len(df) != 800:
    raise ValueError(f"Expected 800 rows, found {len(df)}.")

feature_cols = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
]
outcome_col = "churned"
value_col = "order_value_inr_hundreds"
required = feature_cols + [outcome_col, value_col]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required dataset columns: {missing}")

raw = df[feature_cols].to_numpy(dtype=float)
n, p = raw.shape

# ---------------------------------------------------------------------
# 3.1 Standardisation and K-means from scratch
# ---------------------------------------------------------------------
means = raw.mean(axis=0)
sample_sds = raw.std(axis=0, ddof=1)
if np.any(sample_sds == 0):
    raise ValueError("At least one predictor has zero sample standard deviation.")
X = (raw - means) / sample_sds

def kmeans_pinned(data, k):
    """Pinned initial rows, squared Euclidean distance, no iteration cap."""
    step = len(data) // k
    initial_indices = [j * step for j in range(k)]
    centroids = data[initial_indices].copy()
    labels = np.full(len(data), -1, dtype=int)
    passes = 0

    while True:
        # argmin returns the first (lowest-numbered) centroid on ties.
        squared_distances = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(squared_distances, axis=1)
        passes += 1
        unchanged = np.array_equal(new_labels, labels)
        labels = new_labels

        new_centroids = centroids.copy()
        for cluster_id in range(k):
            members = data[labels == cluster_id]
            if len(members) > 0:
                new_centroids[cluster_id] = members.mean(axis=0)
        centroids = new_centroids

        if unchanged:
            break

    # WCSS measured from each row to its final cluster centroid.
    residuals = data - centroids[labels]
    wcss = float(np.sum(residuals ** 2))
    return labels, centroids, wcss, passes

kmeans_results = {}
for k in range(2, 7):
    labels, centers, wcss, passes = kmeans_pinned(X, k)
    kmeans_results[k] = {
        "labels": labels, "centers": centers,
        "wcss": wcss, "passes": passes
    }

wcss = {k: kmeans_results[k]["wcss"] for k in range(2, 7)}
elbow_k = None
for k in (3, 4, 5):
    if (wcss[k] - wcss[k + 1]) < 0.5 * (wcss[k - 1] - wcss[k]):
        elbow_k = k
        break

    key=lambda cid: raw[labels3 == cid, 2].mean()
)
segment_labels = np.empty(n, dtype=object)
segment_rows = []
segment_indices = {}

for name, cid in zip(("A", "B", "C"), cluster_ids):
    mask = labels3 == cid
    segment_labels[mask] = name
    segment_indices[name] = mask
    segment_rows.append({
        "segment": name,
        "size": int(mask.sum()),
        "mean_X1": float(raw[mask, 0].mean()),
        "mean_X2": float(raw[mask, 1].mean()),
        "mean_X3": float(raw[mask, 2].mean()),
        "mean_X4": float(raw[mask, 3].mean()),
        "mean_X5": float(raw[mask, 4].mean()),
        "outcome_rate": float(df.loc[mask, outcome_col].mean()),
        "mean_value": float(df.loc[mask, value_col].mean()),
    })

segments_df = pd.DataFrame(segment_rows, columns=[
    "segment", "size", "mean_X1", "mean_X2", "mean_X3",
    "mean_X4", "mean_X5", "outcome_rate", "mean_value"
])
segments_df.to_csv(OUTPUT_DIR / "part3_segments.csv", index=False)

# ---------------------------------------------------------------------
# 3.4 Agglomerative hierarchical clustering from supplied distance table
# ---------------------------------------------------------------------
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
], dtype=float)

def cluster_distance(cluster1, cluster2, method):
    distances = [D[i, j] for i in cluster1 for j in cluster2]
    if method == "single":
        return min(distances)
    if method == "complete":
        return max(distances)
    if method == "average":
        return sum(distances) / len(distances)
    raise ValueError("Unknown linkage method")

def hierarchical(method):
    clusters = [[i] for i in range(len(names))]
    merge_rows = []
    heights = []

    for merge_no in range(1, len(names)):
        best = None
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = cluster_distance(clusters[i], clusters[j], method)
                if distance < best_distance:
                    best_distance = distance
                    best = (i, j)

        i, j = best
        ca, cb = clusters[i], clusters[j]
        # cluster_a is the one with the smaller minimum member index.
        if min(ca) > min(cb):
            ca, cb = cb, ca

        def format_cluster(c):
            return "+".join(names[idx] for idx in sorted(c))

        merge_rows.append({
            "linkage": method,
            "merge_no": merge_no,
            "cluster_a": format_cluster(ca),
            "cluster_b": format_cluster(cb),
            "height": float(best_distance),
        })
        heights.append(float(best_distance))

        merged = sorted(ca + cb)
        # Remove original indices safely, then add merged cluster.
        for idx in sorted((i, j), reverse=True):
            clusters.pop(idx)
        clusters.append(merged)
        clusters.sort(key=lambda c: min(c))

    gaps = np.diff(heights)
    largest_gap_index = int(np.argmax(gaps))
    merges_below_cut = largest_gap_index + 1
    clusters_at_cut = 6 - merges_below_cut

    return merge_rows, heights, clusters_at_cut

hierarchical_rows = []
cut_counts = {}
for method in ("single", "complete", "average"):
    rows, heights, count = hierarchical(method)
    hierarchical_rows.extend(rows)
    cut_counts[method] = count

hierarchical_df = pd.DataFrame(hierarchical_rows, columns=[
    "linkage", "merge_no", "cluster_a", "cluster_b", "height"
])
hierarchical_df.to_csv(OUTPUT_DIR / "part3_hierarchical.csv", index=False)

# Recover actual cluster memberships at the cut for reporting.
def clusters_at_largest_gap(method):
    clusters = [[i] for i in range(6)]
    rows, heights, _ = hierarchical(method)
    gaps = np.diff(heights)
    merges_to_apply = int(np.argmax(gaps)) + 1
    for row in rows[:merges_to_apply]:
        a = [names.index(x) for x in row["cluster_a"].split("+")]
        b = [names.index(x) for x in row["cluster_b"].split("+")]
        ca = next(c for c in clusters if sorted(c) == sorted(a))
        cb = next(c for c in clusters if sorted(c) == sorted(b))
        clusters.remove(ca)
        clusters.remove(cb)
        clusters.append(sorted(ca + cb))
        clusters.sort(key=lambda c: min(c))
    return ["+".join(names[i] for i in c) for c in clusters]

cut_memberships = {
    method: clusters_at_largest_gap(method)
    for method in ("single", "complete", "average")
}

# ---------------------------------------------------------------------
# 3.5 PCA by the seven required steps
# ---------------------------------------------------------------------
# Step 1: Standardised mean vector is approximately all zeros.
standardised_mean = X.mean(axis=0)
# Step 2: Centre the standardised data.
X_centered = X - X.mean(axis=0)
# Step 3: Sample covariance matrix.
S = (X_centered.T @ X_centered) / (n - 1)
# Step 4: Eigen-decomposition (the allowed library routine).
eigenvalues, eigenvectors = np.linalg.eigh(S)
# Step 5: Sort eigenvalues descending.
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
# Sign convention: largest-absolute entry in each vector is positive.
for j in range(eigenvectors.shape[1]):
    pivot = np.argmax(np.abs(eigenvectors[:, j]))
    if eigenvectors[pivot, j] < 0:
        eigenvectors[:, j] *= -1
# Step 6: Keep top two eigenvectors.
W = eigenvectors[:, :2]
# Step 7: Project all rows.
Z = X_centered @ W

eigenvalue_sum = float(eigenvalues.sum())
explained = eigenvalues / eigenvalue_sum
cumulative = np.cumsum(explained)
components_for_80pct = int(np.argmax(cumulative >= 0.8) + 1)

# ---------------------------------------------------------------------
# Write every requested key to part3_results.txt
# ---------------------------------------------------------------------
result_lines = []
def add(key, value):
    if isinstance(value, (float, np.floating)):
        result_lines.append(f"{key} = {value:.10f}")
    else:
        result_lines.append(f"{key} = {value}")

for k in range(2, 7):
    add(f"wcss_k{k}", wcss[k])
    add(f"kmeans_passes_k{k}", kmeans_results[k]["passes"])
add("elbow_k", elbow_k if elbow_k is not None else "no qualifying K")

for row in segment_rows:
    name = row["segment"]
    add(f"size_segment_{name}", row["size"])

for method in ("single", "complete", "average"):
    add(f"{method}_clusters_at_cut", "; ".join(cut_memberships[method]))

add("standardised_mean_vector", np.array2string(standardised_mean, precision=10))
add("eigenvalue_sum", eigenvalue_sum)
for i in range(5):
    add(f"eigenvalue_{i+1}", eigenvalues[i])
for i in range(5):
    add(f"explained_ratio_{i+1}", explained[i])
for i in range(5):
    add(f"cumulative_ratio_{i+1}", cumulative[i])
add("components_for_80pct", components_for_80pct)

for j in range(5):
    add(f"pc1_loading_X{j+1}", W[j, 0])
for j in range(5):
    add(f"pc2_loading_X{j+1}", W[j, 1])

for r in range(3):
    add(f"row{r+1}_pc1", Z[r, 0])
    add(f"row{r+1}_pc2", Z[r, 1])

for name in ("A", "B", "C"):
    add(f"mean_pc1_segment_{name}", Z[segment_indices[name], 0].mean())

(OUTPUT_DIR / "part3_results.txt").write_text(
    "\n".join(result_lines) + "\n", encoding="utf-8"
)

# Also print a compact completion summary.
print(f"Dataset: {DATA_PATH}")
print(f"Rows used: {n}")
print(f"X columns: {feature_cols}")
print("\nWCSS:")
for k in range(2, 7):
    print(f"  K={k}: {wcss[k]:.4f} (passes={kmeans_results[k]['passes']})")
print(f"Elbow K: {elbow_k}")
print("\nSegments:")
print(segments_df.to_string(index=False))
print("\nHierarchical clusters at cut:")
for method in ("single", "complete", "average"):
    print(f"  {method}: {cut_memberships[method]}")
print(f"\nPCA eigenvalue sum: {eigenvalue_sum:.4f}")
