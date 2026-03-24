# Line Space Clustering (LSC)

This repository contains a revised implementation of **Line Space Clustering (LSC)** and a separate benchmark script for testing it on synthetic and real datasets.

Please view the original paper for an introduction.
The implementation clusters rows of a data matrix by treating each row as an **ordered 1D profile** over feature index and than using that in a combined distance function.

## Files

- `lsc.py` — the main LSC implementation containing the `LSC(...)` function
- `test_lsc_benchmark.py` — standalone benchmark script for evaluating LSC against baseline clustering methods
- `basic_run.py` — just a basic run of the algorithm.

If your main algorithm file is not named `lsc.py`, either rename it or pass its module name explicitly when running the benchmark.

## What this implementation does

NOTE: This new version has the same algorithm logic but has been updated to have better performance and to be more robust.
Compared with the earlier algorithm, this version makes the clustering procedure more internally consistent:

- it standardizes the input data
- it optionally smooths each row with a Savitzky–Golay filter
- it computes **exact DTW** distances between rows
- it computes **Manhattan (L1)** distances between rows
- it normalizes both distance matrices before combining them
- it uses a weighted combined distance
- it performs **k-medoids-style clustering**
- it uses medoid-based seeding for initialization
- it repairs empty clusters with medoid repair instead of random replacement

## Important assumption

Good use cases:
- time series segments
- ordered signal profiles
- spectra
- depth or position-based measurements

Poor use cases:
- ordinary tabular data with unrelated columns
- one-hot encoded features
- feature sets where column order is arbitrary

The benchmark script includes extra tests on non-sequential tabular data specifically to stress-test this limitation.

## Dependencies

Install the following Python packages:

```bash
pip install numpy scipy scikit-learn matplotlib
```

### Tested imports used by the code

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- standard library modules such as `argparse`, `csv`, `json`, `time`, and `pathlib`

## Python version

Python 3.9+ is recommended.

## Recommended project structure

```text
project_folder/
├── lsc.py
├── test_lsc_benchmark.py
└── README_LSC.md
```

## Main function

The main algorithm entry point is:

```python
cluster_labels, cluster_centers = LSC(
    data,
    num_clusters=5,
    alpha=0.5,
    smoothing=True,
    max_iterations=100,
    random_state=42,
    visualize=False
)
```

## LSC parameters

### `data`
A NumPy array of shape `(n_samples, n_features)`.

- rows = samples
- columns = ordered features

### `num_clusters`
Number of clusters to form.

### `alpha`
Weight for combining DTW and Manhattan distances:

- `alpha = 1.0` → pure DTW
- `alpha = 0.0` → pure Manhattan distance
- `0 < alpha < 1` → weighted mixture

Because both distance matrices are normalized before combination, `alpha` has a meaningful effect.

### `smoothing`
Whether to apply Savitzky–Golay smoothing row-wise before clustering.

Set this to `True` when neighboring feature positions should vary smoothly.
Set it to `False` if smoothing may distort meaningful sharp changes.

### `max_iterations`
Maximum number of medoid-update iterations.

### `random_state`
Random seed for reproducibility.

### `visualize`
If `True`, the function plots:
- the initial line space
- the clustered line space with medoid representatives

## Outputs

### `cluster_labels`
A 1D NumPy array of cluster assignments for each sample.

### `cluster_centers`
The final medoid representatives in the **scaled and optionally smoothed** line space.

These are not automatically transformed back into the original raw feature scale.

## How to run the algorithm directly

If your `lsc.py` file contains a test block such as:

```python
if __name__ == "__main__":
    ...
```

then run:

```bash
python lsc.py
```

## How to run the benchmark script

Basic usage:

```bash
python test_lsc_benchmark.py --module lsc --func LSC
```

This tells the benchmark script to import:
- module: `lsc`
- function: `LSC`

### If your file has a different name

Example: if your algorithm is in `my_lsc_impl.py`, run:

```bash
python test_lsc_benchmark.py --module my_lsc_impl --func LSC
```

## Benchmark script options

```bash
python test_lsc_benchmark.py \
  --module lsc \
  --func LSC \
  --alpha 0.5 \
  --max-iterations 100 \
  --random-state 42 \
  --synthetic-samples 300 \
  --synthetic-features 64 \
  --synthetic-clusters 5 \
  --outdir benchmark_outputs
```

### Main arguments

- `--module` — Python module containing the `LSC` function
- `--func` — function name to import
- `--alpha` — DTW/L1 weighting
- `--max-iterations` — maximum clustering iterations
- `--random-state` — seed for reproducibility
- `--synthetic-samples` — synthetic benchmark sample count
- `--synthetic-features` — synthetic benchmark feature count
- `--synthetic-clusters` — synthetic benchmark cluster count
- `--outdir` — output folder for CSV and JSON results

## What the benchmark reports

The benchmark script reports:

### Synthetic benchmarks
Noise levels:
- 1.0
- 2.0
- 3.0
- 5.0
- 10.0

### Real-world benchmarks
- Iris
- Wine

### Additional stress tests
- Breast cancer dataset
- Iris with permuted feature order
- Wine with permuted feature order

These extra tests help check how sensitive the method is to non-sequential tabular data.

### Smoothing ablation
Compares performance with smoothing enabled and disabled.

### Runtime sweep
Reports local runtime over a grid of sample counts and feature counts.

## Metrics reported

The benchmark reports the following clustering metrics:

- ARI
- AMI
- Homogeneity
- Completeness
- V-measure
- Silhouette
- Runtime in seconds

## Output files

The benchmark writes results into the directory given by `--outdir`.

Expected files include:

- `synthetic_results.csv`
- `real_world_results.csv`
- `extra_nonsequential_results.csv`
- `smoothing_ablation.csv`
- `runtime_sweep.csv`
- `all_primary_rows.csv`
- `benchmark_manifest.json`

## What to change depending on your use case

### To use only exact DTW
Set:

```python
alpha=1.0
```

### To use only Manhattan distance
Set:

```python
alpha=0.0
```

### To reduce runtime
You can lower:
- number of samples
- number of features
- number of clusters
- `max_iterations`

Exact DTW is computationally expensive because all pairwise distances are computed.

### To test whether feature order matters
Use the extra permuted-feature tests in the benchmark script.
If performance drops strongly after column permutation, that indicates the algorithm depends on meaningful feature order of your data.

## Troubleshooting

### `ModuleNotFoundError: No module named 'lsc'`
Your algorithm file is not named `lsc.py`, or you are not running the command from the correct folder.

Fix:
- rename the file to `lsc.py`, or
- pass the correct module name with `--module`, or
- run the command from the folder containing the file
