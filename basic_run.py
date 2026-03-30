from sklearn.metrics import adjusted_rand_score
from lsc import LSC
import numpy as np

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n_per_cluster = 40
    length = 50
    t = np.linspace(0, 1, length)

    base_1 = np.sin(2 * np.pi * t)
    base_2 = 2 * t - 1
    base_3 = np.exp(-((t - 0.3) ** 2) / 0.01) - np.exp(-((t - 0.7) ** 2) / 0.01)

    cluster_1 = np.array([
        1.00 * base_1 + rng.normal(0.0, 0.10, length) + rng.normal(0.0, 0.03)
        for _ in range(n_per_cluster)
    ])
    cluster_2 = np.array([
        0.95 * base_2 + rng.normal(0.0, 0.10, length) + rng.normal(0.0, 0.03)
        for _ in range(n_per_cluster)
    ])
    cluster_3 = np.array([
        1.05 * base_3 + rng.normal(0.0, 0.10, length) + rng.normal(0.0, 0.03)
        for _ in range(n_per_cluster)
    ])

    data = np.vstack([cluster_1, cluster_2, cluster_3])
    true_labels = np.array(
        [0] * n_per_cluster +
        [1] * n_per_cluster +
        [2] * n_per_cluster
    )

    predicted_labels, cluster_centers = LSC(
        data,
        num_clusters=3,
        alpha=0.5,
        smoothing=True,
        max_iterations=100,
        random_state=42,
        visualize=True
    )

    ari = adjusted_rand_score(true_labels, predicted_labels)
    print(f"Adjusted Rand Index: {ari:.4f}")

    if ari < 0.90:
        raise RuntimeError(f"Test failed: ARI too low ({ari:.4f})")

    print("Test passed.")
    