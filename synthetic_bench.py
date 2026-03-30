import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler
from lsc import LSC


def smooth_series(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")


def make_real_world_templates(length: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length)

    def gauss(mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    def from_knots(knot_t, knot_y, smooth_window: int = 7) -> np.ndarray:
        y = np.interp(t, knot_t, knot_y)
        return smooth_series(y, window=smooth_window)

    c0 = from_knots(
        [0.00, 0.10, 0.28, 0.48, 0.70, 1.00],
        [-1.10, -0.75, 0.85, 1.10, 0.55, -0.10],
        smooth_window=9,
    )
    c0 += 0.55 * gauss(0.22, 0.055) - 0.38 * gauss(0.82, 0.08)

    c1 = from_knots(
        [0.00, 0.18, 0.40, 0.62, 0.82, 1.00],
        [0.95, 0.50, -0.85, -1.05, -0.10, 0.55],
        smooth_window=9,
    )
    c1 += -0.45 * gauss(0.38, 0.06) + 0.50 * gauss(0.90, 0.07)

    c2 = from_knots(
        [0.00, 0.18, 0.36, 0.54, 0.76, 1.00],
        [-0.55, -0.05, 0.55, 0.25, -0.25, 0.15],
        smooth_window=7,
    )
    c2 += 0.70 * np.sin(2.0 * np.pi * (1.15 * t + 0.10)) * np.exp(-0.30 * t)
    c2 += 0.35 * gauss(0.62, 0.09)

    templates = np.vstack([c0, c1, c2])
    templates = templates - templates.mean(axis=1, keepdims=True)
    templates = templates / templates.std(axis=1, keepdims=True)
    return templates


def warp_signal(
    x: np.ndarray,
    rng: np.random.Generator,
    warp_strength: float = 0.08,
) -> np.ndarray:
    n = len(x)
    t = np.linspace(0.0, 1.0, n)

    src = np.array([0.00, 0.18, 0.36, 0.54, 0.72, 0.88, 1.00])
    inner = src[1:-1] + rng.normal(0.0, warp_strength, size=len(src) - 2)
    inner = np.clip(inner, 0.03, 0.97)
    inner = np.sort(inner)
    dst = np.concatenate([[0.0], inner, [1.0]])

    source_locations = np.interp(t, dst, src)
    return np.interp(source_locations, t, x)


def shift_signal(
    x: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = 6,
) -> np.ndarray:
    n = len(x)
    shift = float(rng.integers(-max_shift, max_shift + 1)) + rng.normal(0.0, 0.35)
    idx = np.arange(n) - shift
    idx = np.clip(idx, 0, n - 1)
    return np.interp(idx, np.arange(n), x)


def smooth_random_curve(
    length: int,
    rng: np.random.Generator,
    scale: float,
    window: int,
) -> np.ndarray:
    z = rng.normal(0.0, scale, size=length)
    return smooth_series(z, window=window)


def colored_noise(
    length: int,
    rng: np.random.Generator,
    std: float,
    rho: float = 0.78,
) -> np.ndarray:
    eps = rng.normal(0.0, std, size=length)
    y = np.empty(length, dtype=float)
    y[0] = eps[0]
    for i in range(1, length):
        y[i] = rho * y[i - 1] + eps[i]
    return y


def heteroskedastic_noise(
    length: int,
    rng: np.random.Generator,
    base_std: float,
) -> np.ndarray:
    envelope = 0.75 + np.abs(
        smooth_random_curve(length, rng, scale=0.55, window=max(5, length // 8))
    )
    noise = colored_noise(length, rng, std=base_std, rho=0.72)
    return envelope * noise


def local_regime_shift(
    x: np.ndarray,
    rng: np.random.Generator,
    prob: float = 0.45,
) -> np.ndarray:
    y = x.copy()
    if rng.random() >= prob:
        return y

    n = len(y)
    start = int(rng.integers(n // 8, max(n // 8 + 1, 3 * n // 4)))
    width = int(rng.integers(max(4, n // 10), max(6, n // 4)))
    end = min(n, start + width)

    mode = rng.choice(["step", "slope"])
    magnitude = rng.normal(0.0, 0.55)

    if mode == "step":
        y[start:end] += magnitude
    else:
        ramp = np.linspace(0.0, magnitude, end - start)
        y[start:end] += ramp

    return y


def dropout_or_saturation(
    x: np.ndarray,
    rng: np.random.Generator,
    prob: float = 0.20,
) -> np.ndarray:
    y = x.copy()
    if rng.random() >= prob:
        return y

    n = len(y)
    start = int(rng.integers(0, max(1, n - max(4, n // 6))))
    width = int(rng.integers(max(3, n // 12), max(5, n // 6)))
    end = min(n, start + width)
    mode = rng.choice(["dropout", "saturation"])

    if end - start < 2:
        return y

    if mode == "dropout":
        left_val = y[start - 1] if start > 0 else y[start]
        right_val = y[end] if end < n else y[end - 1]
        y[start:end] = np.linspace(left_val, right_val, end - start)
        y[start:end] += rng.normal(0.0, 0.05, size=end - start)
    else:
        level = np.median(y[max(0, start - 3):min(n, end + 3)])
        y[start:end] = level

    return y


def add_sparse_outliers(
    x: np.ndarray,
    rng: np.random.Generator,
    outlier_prob: float = 0.50,
    outlier_scale: float = 1.35,
) -> np.ndarray:
    y = x.copy()
    if rng.random() < outlier_prob:
        count = int(rng.integers(1, 4))
        idx = rng.integers(0, len(y), size=count)
        y[idx] += rng.normal(0.0, outlier_scale, size=count)
    return y


def generate_real_world_like_dataset(
    n_per_cluster: int = 20,
    length: int = 64,
    noise_std: float = 5.0,
    max_shift: int = 6,
    warp_strength: float = 0.08,
    amplitude_jitter: float = 0.18,
    offset_std: float = 0.30,
    baseline_drift_scale: float = 0.28,
    artifact_prob: float = 0.20,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    templates = make_real_world_templates(length)

    X = []
    y = []

    effective_noise = 0.08 * noise_std
    t = np.linspace(0.0, 1.0, length)

    for cluster_id, template in enumerate(templates):
        for _ in range(n_per_cluster):
            x = template.copy()

            x = warp_signal(x, rng, warp_strength=warp_strength)
            x = shift_signal(x, rng, max_shift=max_shift)

            amp = 1.0 + rng.normal(0.0, amplitude_jitter)
            off = rng.normal(0.0, offset_std)
            x = amp * x + off

            drift = smooth_random_curve(
                length,
                rng,
                scale=baseline_drift_scale,
                window=max(7, length // 5),
            )
            x = x + drift

            x = local_regime_shift(x, rng, prob=0.45)

            seasonal = 0.12 * np.sin(
                2.0 * np.pi * (rng.uniform(0.6, 1.8) * t + rng.uniform(0.0, 1.0))
            )
            x = x + seasonal

            x = x + heteroskedastic_noise(length, rng, base_std=effective_noise)
            x = add_sparse_outliers(x, rng, outlier_prob=0.55, outlier_scale=1.20)
            x = dropout_or_saturation(x, rng, prob=artifact_prob)

            if rng.random() < 0.35:
                x = smooth_series(x, window=int(rng.choice([3, 5, 7])))

            X.append(x)
            y.append(cluster_id)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    perm = rng.permutation(len(X))
    return X[perm], y[perm], templates


def safe_silhouette(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return float("nan")
    return float(silhouette_score(X, labels))


def clustering_metrics(y_true, labels, X_for_silhouette):
    return {
        "ARI": float(adjusted_rand_score(y_true, labels)),
        "AMI": float(adjusted_mutual_info_score(y_true, labels)),
        "Homogeneity": float(homogeneity_score(y_true, labels)),
        "Completeness": float(completeness_score(y_true, labels)),
        "V-measure": float(v_measure_score(y_true, labels)),
        "Silhouette": safe_silhouette(X_for_silhouette, labels),
    }


def evaluate_one_seed(
    seed: int,
    n_per_cluster: int = 20,
    length: int = 64,
    noise_std: float = 5.0,
    alpha: float = 0.80,
    smoothing: bool = True,
    max_iterations: int = 30,
):
    X, y, _ = generate_real_world_like_dataset(
        n_per_cluster=n_per_cluster,
        length=length,
        noise_std=noise_std,
        max_shift=6,
        warp_strength=0.08,
        amplitude_jitter=0.18,
        offset_std=0.30,
        baseline_drift_scale=0.28,
        artifact_prob=0.20,
        random_state=seed,
    )

    n_clusters = len(np.unique(y))
    X_scaled = StandardScaler().fit_transform(X)

    results = []

    start = time.perf_counter()
    labels_lsc, _ = LSC(
        X,
        num_clusters=n_clusters,
        alpha=alpha,
        smoothing=smoothing,
        max_iterations=max_iterations,
        random_state=seed,
        visualize=False,
    )
    elapsed = time.perf_counter() - start
    results.append({
        "method": "LSC",
        "runtime_sec": elapsed,
        **clustering_metrics(y, labels_lsc, X_scaled),
    })

    baselines = {
        "KMeans": KMeans(n_clusters=n_clusters, n_init=20, random_state=seed),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "Spectral": SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(10, len(X_scaled) - 1),
            assign_labels="kmeans",
            random_state=seed,
        ),
    }

    for name, model in baselines.items():
        start = time.perf_counter()
        labels = model.fit_predict(X_scaled)
        elapsed = time.perf_counter() - start
        results.append({
            "method": name,
            "runtime_sec": elapsed,
            **clustering_metrics(y, labels, X_scaled),
        })

    return results


def print_results(results):
    print("\nSingle-run results")
    print("=" * 118)
    print(
        f"{'Method':<16}"
        f"{'ARI':>12}"
        f"{'AMI':>12}"
        f"{'Homog.':>12}"
        f"{'Compl.':>12}"
        f"{'V-measure':>12}"
        f"{'Silhouette':>14}"
        f"{'Runtime(s)':>14}"
    )
    print("-" * 118)

    for method in ["LSC", "KMeans", "Agglomerative", "Spectral"]:
        row = next(r for r in results if r["method"] == method)
        print(
            f"{row['method']:<16}"
            f"{row['ARI']:>12.4f}"
            f"{row['AMI']:>12.4f}"
            f"{row['Homogeneity']:>12.4f}"
            f"{row['Completeness']:>12.4f}"
            f"{row['V-measure']:>12.4f}"
            f"{row['Silhouette']:>14.4f}"
            f"{row['runtime_sec']:>14.3f}"
        )


def main():
    seed = 0

    n_per_cluster = 1000
    length = 24

    noise_std = 6.0

    alpha = 0.5
    smoothing = True
    max_iterations = 1000

    results = evaluate_one_seed(
        seed=seed,
        n_per_cluster=n_per_cluster,
        length=length,
        noise_std=noise_std,
        alpha=alpha,
        smoothing=smoothing,
        max_iterations=max_iterations,
    )

    print_results(results)


if __name__ == "__main__":
    main()