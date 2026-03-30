import argparse
import csv
import importlib
import json
import math
import time
from pathlib import Path
import lsc

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_blobs
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler


METRICS = [
    "ARI",
    "AMI",
    "Homogeneity",
    "Completeness",
    "V-measure",
    "Silhouette",
]

PAPER_RESULTS = {
    "synthetic": {
        5.0: {
            "LSC": {"ARI": 0.3939, "AMI": 0.4309, "Homogeneity": 0.4391, "Completeness": 0.4423, "V-measure": 0.4407, "Silhouette": 0.2105},
            "KM": {"ARI": 0.2954, "AMI": 0.3984, "Homogeneity": 0.4065, "Completeness": 0.4113, "V-measure": 0.4089, "Silhouette": 0.1914},
            "Agglo": {"ARI": 0.2896, "AMI": 0.4035, "Homogeneity": 0.4084, "Completeness": 0.4196, "V-measure": 0.4139, "Silhouette": 0.1600},
            "DBSCAN": {"ARI": math.nan, "AMI": math.nan, "Homogeneity": math.nan, "Completeness": math.nan, "V-measure": math.nan, "Silhouette": math.nan},
            "Spectral": {"ARI": 0.2763, "AMI": 0.4158, "Homogeneity": 0.4188, "Completeness": 0.4335, "V-measure": 0.4260, "Silhouette": 0.2129},
        },
        10.0: {
            "LSC": {"ARI": 0.1064, "AMI": 0.1592, "Homogeneity": 0.1637, "Completeness": 0.1745, "V-measure": 0.1641, "Silhouette": 0.2038},
            "KM": {"ARI": 0.0761, "AMI": 0.1351, "Homogeneity": 0.1491, "Completeness": 0.1510, "V-measure": 0.1500, "Silhouette": 0.1823},
            "Agglo": {"ARI": 0.0729, "AMI": 0.1140, "Homogeneity": 0.1274, "Completeness": 0.1318, "V-measure": 0.1295, "Silhouette": 0.1296},
            "DBSCAN": {"ARI": math.nan, "AMI": math.nan, "Homogeneity": math.nan, "Completeness": math.nan, "V-measure": math.nan, "Silhouette": math.nan},
            "Spectral": {"ARI": 0.0751, "AMI": 0.1260, "Homogeneity": 0.1398, "Completeness": 0.1426, "V-measure": 0.1412, "Silhouette": 0.1710},
        },
    },
    "real_world": {
        "iris": {
            "LSC": {"ARI": 0.6779, "AMI": 0.7045, "Homogeneity": 0.7038, "Completeness": 0.7125, "V-measure": 0.7081, "Silhouette": 0.4573},
            "KM": {"ARI": 0.4328, "AMI": 0.5838, "Homogeneity": 0.5347, "Completeness": 0.6570, "V-measure": 0.5896, "Silhouette": 0.4799},
            "Agglo": {"ARI": 0.6153, "AMI": 0.6713, "Homogeneity": 0.6579, "Completeness": 0.6940, "V-measure": 0.6755, "Silhouette": 0.4467},
            "DBSCAN": {"ARI": 0.4421, "AMI": 0.5052, "Homogeneity": 0.5005, "Completeness": 0.5228, "V-measure": 0.5114, "Silhouette": 0.3565},
            "Spectral": {"ARI": 0.6451, "AMI": 0.6856, "Homogeneity": 0.6824, "Completeness": 0.6968, "V-measure": 0.6895, "Silhouette": 0.4630},
        },
        "wine": {
            "LSC": {"ARI": 0.8961, "AMI": 0.8757, "Homogeneity": 0.8795, "Completeness": 0.8746, "V-measure": 0.8770, "Silhouette": 0.2818},
            "KM": {"ARI": 0.8975, "AMI": 0.8746, "Homogeneity": 0.8788, "Completeness": 0.8730, "V-measure": 0.8759, "Silhouette": 0.2849},
            "Agglo": {"ARI": 0.7899, "AMI": 0.7842, "Homogeneity": 0.7904, "Completeness": 0.7825, "V-measure": 0.7865, "Silhouette": 0.2774},
            "DBSCAN": {"ARI": math.nan, "AMI": math.nan, "Homogeneity": math.nan, "Completeness": math.nan, "V-measure": math.nan, "Silhouette": math.nan},
            "Spectral": {"ARI": 0.4446, "AMI": 0.5662, "Homogeneity": 0.4689, "Completeness": 0.7348, "V-measure": 0.5725, "Silhouette": 0.2486},
        },
    },
    "smoothing": {
        "with_smoothing": 0.85766,
        "without_smoothing": 0.834644,
    },
}


def load_lsc(module_name: str, func_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def format_value(x):
    if x is None:
        return "N/A"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "N/A"
    return f"{x:.4f}"


def compute_metrics(X, y_true, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    valid_cluster_labels = [x for x in unique if x != -1]
    metrics = {
        "ARI": float(adjusted_rand_score(y_true, labels)),
        "AMI": float(adjusted_mutual_info_score(y_true, labels)),
        "Homogeneity": float(homogeneity_score(y_true, labels)),
        "Completeness": float(completeness_score(y_true, labels)),
        "V-measure": float(v_measure_score(y_true, labels)),
        "Silhouette": math.nan,
    }
    if len(valid_cluster_labels) >= 2 and len(valid_cluster_labels) < len(labels):
        try:
            metrics["Silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            metrics["Silhouette"] = math.nan
    return metrics


def run_lsc(LSC, X, n_clusters, alpha, smoothing, max_iterations, random_state):
    start = time.perf_counter()
    labels, centers = LSC(
        X,
        num_clusters=n_clusters,
        alpha=alpha,
        smoothing=smoothing,
        max_iterations=max_iterations,
        random_state=random_state,
        visualize=False,
    )
    elapsed = time.perf_counter() - start
    return np.asarray(labels), centers, elapsed


def run_baseline(name, X, n_clusters, random_state):
    start = time.perf_counter()
    if name == "KM":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
    elif name == "Agglo":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
    elif name == "DBSCAN":
        model = DBSCAN()
        labels = model.fit_predict(X)
    elif name == "Spectral":
        nn = min(10, max(2, len(X) - 1))
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=nn,
            assign_labels="kmeans",
            random_state=random_state,
        )
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    elapsed = time.perf_counter() - start
    return np.asarray(labels), elapsed


def generate_paper_like_synthetic(n_samples, n_features, n_clusters, noise_std, random_state):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=n_features,
        cluster_std=noise_std,
        random_state=random_state,
    )

    base_feature_scales = np.array([
        50, 0.1, 592, 0.01, 140, 1, 52, 26, 105, 0.0001, 108, 0.0032,
        220, 108, 108, 0.01, 2.03, 112, 1008, 33, 0.8, 120, 754, 0.0004
    ], dtype=float)

    feature_scales = np.resize(base_feature_scales, n_features)

    rng = np.random.default_rng(random_state)
    X = X * feature_scales
    noise = rng.normal(0.0, feature_scales / 10.0, size=X.shape)
    X = X + noise
    return X, y


def load_real_world_datasets():
    iris = load_iris()
    wine = load_wine()
    bc = load_breast_cancer()
    return {
        "iris": (iris.data, iris.target),
        "wine": (wine.data, wine.target),
        "breast_cancer": (bc.data, bc.target),
    }


def permute_feature_order(X, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[1])
    return X[:, order], order


def evaluate_dataset(LSC, X, y, dataset_name, alpha, smoothing, max_iterations, random_state):
    X_scaled = StandardScaler().fit_transform(X)
    n_clusters = len(np.unique(y))
    rows = []

    labels, _, elapsed = run_lsc(LSC, X, n_clusters, alpha, smoothing, max_iterations, random_state)
    metrics = compute_metrics(X_scaled, y, labels)
    rows.append({"dataset": dataset_name, "method": "LSC", "runtime_sec": elapsed, **metrics})

    for method in ["KM", "Agglo", "DBSCAN", "Spectral"]:
        labels, elapsed = run_baseline(method, X_scaled, n_clusters, random_state)
        metrics = compute_metrics(X_scaled, y, labels)
        rows.append({"dataset": dataset_name, "method": method, "runtime_sec": elapsed, **metrics})

    return rows


def print_table(rows, title):
    print("\n" + title)
    print("=" * len(title))
    header = ["dataset", "method", *METRICS, "runtime_sec"]
    widths = {h: len(h) for h in header}
    for row in rows:
        for h in header:
            val = format_value(row[h]) if h in METRICS or h == "runtime_sec" else str(row[h])
            widths[h] = max(widths[h], len(val))
    print("  ".join(h.ljust(widths[h]) for h in header))
    print("  ".join("-" * widths[h] for h in header))
    for row in rows:
        rendered = []
        for h in header:
            if h in METRICS or h == "runtime_sec":
                rendered.append(format_value(row[h]).ljust(widths[h]))
            else:
                rendered.append(str(row[h]).ljust(widths[h]))
        print("  ".join(rendered))


def compare_against_paper(rows, section_key):
    print(f"\nComparison against paper results: {section_key}")
    print("-" * (34 + len(section_key)))
    for row in rows:
        dataset = row["dataset"]
        method = row["method"]
        if section_key == "synthetic":
            noise = dataset
            reference = PAPER_RESULTS[section_key].get(noise, {}).get(method)
            label = f"noise={noise}, method={method}"
        else:
            reference = PAPER_RESULTS[section_key].get(dataset, {}).get(method)
            label = f"dataset={dataset}, method={method}"
        if reference is None:
            continue
        deltas = []
        for metric in METRICS:
            current = row[metric]
            ref = reference[metric]
            if (isinstance(ref, float) and math.isnan(ref)) or (isinstance(current, float) and math.isnan(current)):
                deltas.append(f"{metric}: current={format_value(current)}, paper={format_value(ref)}")
            else:
                deltas.append(f"{metric}: Δ={current - ref:+.4f}")
        print(label)
        print("  " + " | ".join(deltas))


def smoothing_ablation(LSC, X, y, alpha, max_iterations, random_state):
    n_clusters = len(np.unique(y))
    labels_s, _, time_s = run_lsc(LSC, X, n_clusters, alpha, True, max_iterations, random_state)
    labels_ns, _, time_ns = run_lsc(LSC, X, n_clusters, alpha, False, max_iterations, random_state)
    X_scaled = StandardScaler().fit_transform(X)
    ari_s = adjusted_rand_score(y, labels_s)
    ari_ns = adjusted_rand_score(y, labels_ns)
    rows = [
        {"dataset": "synthetic_noise_10", "method": "with_smoothing", "ARI": float(ari_s), "runtime_sec": float(time_s)},
        {"dataset": "synthetic_noise_10", "method": "without_smoothing", "ARI": float(ari_ns), "runtime_sec": float(time_ns)},
    ]
    _ = X_scaled
    return rows


def runtime_sweep(LSC, alpha, smoothing, max_iterations, random_state):
    sample_grid = [10, 50, 100]
    feature_grid = [3, 10, 20]
    rows = []
    for n_samples in sample_grid:
        for n_features in feature_grid:
            X, y = generate_paper_like_synthetic(
                n_samples=n_samples,
                n_features=n_features,
                n_clusters=5,
                noise_std=5.0,
                random_state=random_state,
            )
            start = time.perf_counter()
            run_lsc(LSC, X, 5, alpha, smoothing, max_iterations, random_state)
            elapsed = time.perf_counter() - start
            rows.append({
                "dataset": f"n={n_samples},d={n_features}",
                "method": "LSC",
                "ARI": math.nan,
                "AMI": math.nan,
                "Homogeneity": math.nan,
                "Completeness": math.nan,
                "V-measure": math.nan,
                "Silhouette": math.nan,
                "runtime_sec": elapsed,
            })
    return rows


def save_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="lsc", help="Python module containing LSC")
    parser.add_argument("--func", default="LSC", help="Function name to import from the module")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--synthetic-samples", type=int, default=3000)
    parser.add_argument("--synthetic-features", type=int, default=64)
    parser.add_argument("--synthetic-clusters", type=int, default=5)
    parser.add_argument("--outdir", type=str, default=str(Path.home() / "Desktop" / "benchmark_outputs"))
    args = parser.parse_args()

    LSC = load_lsc(args.module, args.func)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("Saving outputs to:", outdir.resolve())

    all_rows = []

    synthetic_rows = []
    for noise in [5.0, 10.0]:
        X, y = generate_paper_like_synthetic(
            n_samples=args.synthetic_samples,
            n_features=args.synthetic_features,
            n_clusters=args.synthetic_clusters,
            noise_std=noise,
            random_state=args.random_state,
        )
        rows = evaluate_dataset(
            LSC,
            X,
            y,
            noise,
            args.alpha,
            True,
            args.max_iterations,
            args.random_state,
        )
        synthetic_rows.extend(rows)
        all_rows.extend(rows)
    print_table(synthetic_rows, "Synthetic benchmarks")
    compare_against_paper(synthetic_rows, "synthetic")
    save_csv(synthetic_rows, outdir / "synthetic_results.csv")

    real_rows = []
    real_datasets = load_real_world_datasets()
    for name in ["iris", "wine"]:
        X, y = real_datasets[name]
        rows = evaluate_dataset(
            LSC,
            X,
            y,
            name,
            args.alpha,
            True,
            args.max_iterations,
            args.random_state,
        )
        real_rows.extend(rows)
        all_rows.extend(rows)
    print_table(real_rows, "Real-world benchmarks from the paper")
    compare_against_paper(real_rows, "real_world")
    save_csv(real_rows, outdir / "real_world_results.csv")

    X_noise10, y_noise10 = generate_paper_like_synthetic(
        n_samples=args.synthetic_samples,
        n_features=args.synthetic_features,
        n_clusters=args.synthetic_clusters,
        noise_std=10.0,
        random_state=args.random_state,
    )
    smooth_rows = smoothing_ablation(
        LSC,
        X_noise10,
        y_noise10,
        args.alpha,
        args.max_iterations,
        args.random_state,
    )
    print("\nSmoothing ablation")
    print("==================")
    for row in smooth_rows:
        print(
            f"{row['method']}: ARI={row['ARI']:.4f}, runtime_sec={row['runtime_sec']:.4f}"
        )
    print(
        "Paper smoothing table: "
        f"with={PAPER_RESULTS['smoothing']['with_smoothing']:.6f}, "
        f"without={PAPER_RESULTS['smoothing']['without_smoothing']:.6f}"
    )
    save_csv(smooth_rows, outdir / "smoothing_ablation.csv")

    extra_rows = []
    X_bc, y_bc = real_datasets["breast_cancer"]
    extra_rows.extend(
        evaluate_dataset(
            LSC,
            X_bc,
            y_bc,
            "breast_cancer",
            args.alpha,
            True,
            args.max_iterations,
            args.random_state,
        )
    )
    X_iris_perm, order_iris = permute_feature_order(real_datasets["iris"][0], args.random_state)
    extra_rows.extend(
        evaluate_dataset(
            LSC,
            X_iris_perm,
            real_datasets["iris"][1],
            "iris_feature_permuted",
            args.alpha,
            True,
            args.max_iterations,
            args.random_state,
        )
    )
    X_wine_perm, order_wine = permute_feature_order(real_datasets["wine"][0], args.random_state)
    extra_rows.extend(
        evaluate_dataset(
            LSC,
            X_wine_perm,
            real_datasets["wine"][1],
            "wine_feature_permuted",
            args.alpha,
            True,
            args.max_iterations,
            args.random_state,
        )
    )
    print_table(extra_rows, "Additional non-sequential/tabular stress tests")
    save_csv(extra_rows, outdir / "extra_nonsequential_results.csv")

    runtime_rows = runtime_sweep(
        LSC,
        args.alpha,
        True,
        args.max_iterations,
        args.random_state,
    )
    print_table(runtime_rows, "LSC runtime sweep")
    save_csv(runtime_rows, outdir / "runtime_sweep.csv")

    manifest = {
        "paper_results_embedded": PAPER_RESULTS,
        "feature_permutation_examples": {
            "iris_order": order_iris.tolist(),
            "wine_order": order_wine.tolist(),
        }
    }
    with open(outdir / "benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    save_csv(all_rows, outdir / "all_primary_rows.csv")


if __name__ == "__main__":
    main()
