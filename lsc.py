# PLEASE READ THIS BEFORE USING OR MODIFYING THIS CODE:
#
# This implementation refines the original Line Space Clustering procedure by making
# the distance-based optimization internally consistent and more robust. First, the
# DTW and Manhattan distance matrices are normalized before being combined, so the
# weighting parameter alpha has a meaningful effect and one distance term cannot
# dominate merely because of scale. Second, the distance definition is corrected by
# explicitly using Manhattan (L1) distance.
# Third, cluster representatives are now updated as medoids instead of coordinate-wise
# medians, which is a more principled choice when using DTW because medoids remain
# actual observed lines and are directly compatible with arbitrary pairwise distance
# matrices. Fourth, initialization is improved through medoid-based seeding, which
# generally yields more stable and higher-quality starting clusters than purely random
# selection. Fifth, empty clusters are handled with medoid repair by reassigning a
# distant unused sample, avoiding unstable random replacement. In addition, this
# version uses FastDTW rather than standard exact DTW, so the clustering is based on
# an approximation to the warping distance for improved runtime. In general it is the
# exact same algorithmic math in the paper but with improvements.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from tqdm.auto import tqdm


def _dtw_distance(x, y):
    distance, _ = fastdtw(x, y, dist=lambda a, b: abs(a - b))
    return float(distance)


def LSC(
    data,
    num_clusters=5,
    alpha=0.5,
    smoothing=True,
    max_iterations=100,
    random_state=None,
    visualize=False,
    show_progress=True
):
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must be a 2D array of shape (n_samples, n_features)")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("data must be non-empty")
    if not (1 <= num_clusters <= data.shape[0]):
        raise ValueError("num_clusters must be between 1 and the number of samples")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    rng = np.random.default_rng(random_state)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    lines = data_scaled.copy()

    def smooth_lines(x):
        if x.shape[1] < 3:
            return x.copy()
        window_length = min(5, x.shape[1] if x.shape[1] % 2 == 1 else x.shape[1] - 1)
        if window_length < 3:
            return x.copy()
        polyorder = min(2, window_length - 1)
        return np.array([
            savgol_filter(line, window_length=window_length, polyorder=polyorder)
            for line in x
        ])

    if smoothing:
        lines = smooth_lines(lines)

    def compute_pairwise_distance_matrices(x):
        n = len(x)
        dtw_matrix = np.zeros((n, n), dtype=float)
        l1_matrix = np.zeros((n, n), dtype=float)

        total_pairs = n * (n - 1) // 2
        with tqdm(
            total=total_pairs,
            desc="Computing pairwise distances",
            disable=not show_progress
        ) as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    dtw_d = _dtw_distance(x[i], x[j])
                    l1_d = float(np.sum(np.abs(x[i] - x[j])))
                    dtw_matrix[i, j] = dtw_matrix[j, i] = dtw_d
                    l1_matrix[i, j] = l1_matrix[j, i] = l1_d
                    pbar.update(1)

        return dtw_matrix, l1_matrix

    def normalize_distance_matrix(d):
        mask = d > 0
        if not np.any(mask):
            return d.copy()
        scale = np.median(d[mask])
        if not np.isfinite(scale) or scale <= 0:
            scale = np.mean(d[mask])
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return d / scale

    def initialize_medoids(distance_matrix, k, generator):
        n = distance_matrix.shape[0]
        medoids = [int(generator.integers(0, n))]
        while len(medoids) < k:
            current = np.array(medoids, dtype=int)
            nearest = np.min(distance_matrix[:, current], axis=1)
            remaining = np.setdiff1d(np.arange(n), current, assume_unique=False)
            if remaining.size == 0:
                break
            weights = nearest[remaining] ** 2
            if not np.any(weights > 0):
                next_medoid = int(generator.choice(remaining))
            else:
                probabilities = weights / weights.sum()
                next_medoid = int(generator.choice(remaining, p=probabilities))
            medoids.append(next_medoid)
        return np.array(medoids, dtype=int)

    def assign_clusters(distance_matrix, medoids):
        distances_to_medoids = distance_matrix[:, medoids]
        return np.argmin(distances_to_medoids, axis=1)

    def update_medoids(distance_matrix, assignments, current_medoids):
        k = len(current_medoids)
        new_medoids = current_medoids.copy()
        empty_clusters = []

        for cluster_id in range(k):
            members = np.where(assignments == cluster_id)[0]
            if members.size == 0:
                empty_clusters.append(cluster_id)
                continue
            intra = distance_matrix[np.ix_(members, members)]
            costs = intra.sum(axis=1)
            new_medoids[cluster_id] = int(members[np.argmin(costs)])

        if empty_clusters:
            non_empty_clusters = [c for c in range(k) if c not in empty_clusters]
            if non_empty_clusters:
                non_empty_medoids = new_medoids[non_empty_clusters]
                nearest_non_empty = np.min(distance_matrix[:, non_empty_medoids], axis=1)
            else:
                nearest_non_empty = np.full(distance_matrix.shape[0], np.inf, dtype=float)

            used = set(int(m) for m in new_medoids[non_empty_clusters])
            candidates = np.argsort(-nearest_non_empty)

            for cluster_id in empty_clusters:
                replacement = None
                for idx in candidates:
                    idx = int(idx)
                    if idx not in used:
                        replacement = idx
                        break
                if replacement is None:
                    remaining = [i for i in range(distance_matrix.shape[0]) if i not in used]
                    if not remaining:
                        remaining = list(range(distance_matrix.shape[0]))
                    replacement = int(rng.choice(remaining))
                new_medoids[cluster_id] = replacement
                used.add(replacement)

        return new_medoids

    dtw_matrix, l1_matrix = compute_pairwise_distance_matrices(lines)
    dtw_matrix = normalize_distance_matrix(dtw_matrix)
    l1_matrix = normalize_distance_matrix(l1_matrix)

    combined_distance_matrix = alpha * dtw_matrix + (1.0 - alpha) * l1_matrix

    medoids = initialize_medoids(combined_distance_matrix, num_clusters, rng)

    for iteration in range(max_iterations):
        cluster_assignments = assign_clusters(combined_distance_matrix, medoids)
        objective = float(combined_distance_matrix[np.arange(len(lines)), medoids[cluster_assignments]].sum())
        print(f"Iteration {iteration + 1}, Objective: {objective:.6f}")

        new_medoids = update_medoids(combined_distance_matrix, cluster_assignments, medoids)

        if np.array_equal(new_medoids, medoids):
            print("Convergence reached.")
            break

        medoids = new_medoids

    cluster_assignments = assign_clusters(combined_distance_matrix, medoids)
    cluster_centers = lines[medoids]

    if visualize:
        feature_indices = np.arange(1, data.shape[1] + 1)

        plt.figure(figsize=(10, 6))
        for line in lines:
            plt.plot(feature_indices, line, color="gray", alpha=0.5)
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Value")
        plt.title("Initial Line Space")
        plt.xticks(feature_indices)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        try:
            colors = matplotlib.colormaps["tab10"]
        except AttributeError:
            colors = plt.get_cmap("tab10")

        plt.figure(figsize=(10, 6))
        for idx in range(len(lines)):
            cluster_id = int(cluster_assignments[idx])
            plt.plot(feature_indices, lines[idx], color=colors(cluster_id % 10), alpha=0.35)

        for i in range(num_clusters):
            plt.plot(
                feature_indices,
                cluster_centers[i],
                color=colors(i % 10),
                linewidth=3,
                label=f"Cluster {i}"
            )

        plt.xlabel("Feature Index")
        plt.ylabel("Feature Value")
        plt.title("Clustered Line Space")
        plt.xticks(feature_indices)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return cluster_assignments, cluster_centers