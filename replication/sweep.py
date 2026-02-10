"""Sweep script comparing HL-Projected vs HL-Gaussian across bins, sigma, lr, and int_dim.

Generates a grid of (method, n_bins, sig_ratio, int_dim, lr, seed) configurations
and trains each one, logging results to wandb and saving locally as JSON under
sweep_results/{dataset}/.

Supports multi-GPU parallelism via --worker/--num_workers, and multi-dataset
sequential sweeps with --launch.

Usage:
    # Single GPU, single dataset:
    python -m replication.sweep --dataset ctscan --gpu 0

    # 8 GPUs, 2 workers per GPU, single dataset:
    python -m replication.sweep --dataset ctscan --launch --num_workers 8 --workers_per_gpu 2

    # 8 GPUs across all four datasets (sequential):
    python -m replication.sweep --dataset ctscan bike pole songyear --launch --num_workers 8
"""

import os
import sys
import math
import json
import argparse
import itertools
import subprocess

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import wandb
from wandb.integration.keras import WandbMetricsLogger

from replication.csvdataset import CSVDataset
from experiment.models import *
from experiment.hypermodels import *
from experiment.preprocessing import *


# ── Defaults ────────────────────────────────────────────────────────────────
WANDB_ENTITY = "ut-austin-rpl-general-team"
WANDB_PROJECT = "histogram_loss"

SEEDS = [1, 2, 3]
TEST_RATIO = 0.2

N_BINS_VALUES = [10, 25, 50, 100, 200, 400]
SIG_RATIO_VALUES = [
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    12.0,
    20.0,
    50.0,
]
PADDING = 0.125

LR_VALUES = [1e-3, 1e-4]

METHODS = ["HL-Gaussian", "HL-Projected", "HL-Gibbs"]


# ── Dataset loading (reused from replication.py) ────────────────────────────
def get_dataset_by_name(base_dir, name):
    """Load a single dataset by name."""
    data_dir = os.path.join(base_dir, "data")
    if name == "ctscan":
        ds = CSVDataset(
            os.path.join(data_dir, "slice_localization_data.csv"),
            "reference",
            batch_size=256,
        )
        ds.bounds = (0.0, 100.0)
        ds.name = "ctscan"
        ds.epochs = 1000
    elif name == "bike":
        ds = CSVDataset(
            os.path.join(data_dir, "hour.csv"), "cnt", drop="dteday", batch_size=256
        )
        ds.bounds = (0.0, 1000.0)
        ds.name = "bike"
        ds.epochs = 500
    elif name == "songyear":
        ds = CSVDataset(
            os.path.join(data_dir, "YearPredictionMSD.txt"),
            0,
            header=None,
            batch_size=256,
        )
        ds.bounds = (1922.0, 2011.0)
        ds.name = "songyear"
        ds.epochs = 150
    elif name == "pole":
        ds = CSVDataset(os.path.join(data_dir, "pole.csv"), "target", batch_size=256)
        ds.bounds = (0.0, 100.0)
        ds.name = "pole"
        ds.epochs = 500
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds


def mlp_base(input_width, hidden=4, dropout=0.05, int_dim=0.5):
    """Return an MLP base model."""
    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(dropout))
    width = int(int_dim * input_width)
    for _ in range(hidden):
        model.add(
            keras.layers.Dense(
                width, activation="relu", kernel_initializer="lecun_uniform"
            )
        )
    return model


# Per-dataset MLP defaults: (input_width, default_int_dim, dropout)
DATASET_MLP_INFO = {
    "ctscan": {"input_width": 385, "int_dim": 0.5, "dropout": 0.05},
    "bike": {"input_width": 16, "int_dim": 4.0, "dropout": 0.0},
    "pole": {"input_width": 49, "int_dim": 0.5, "dropout": 0.0},
    "songyear": {"input_width": 90, "int_dim": 0.5, "dropout": 0.05},
}


def get_int_dim_values(dataset_name, n_bins):
    """Return the list of int_dim values to sweep for a (dataset, n_bins) pair.

    Three values (deduplicated and sorted):
      1. The per-dataset default int_dim.
      2. 2× the default.
      3. The smallest integer multiple of the default such that
         hidden_width = int_dim * input_width >= n_bins.
         This removes the bottleneck from hidden layers to the output bins.
    """
    info = DATASET_MLP_INFO[dataset_name]
    default = info["int_dim"]
    input_w = info["input_width"]

    hidden_default = default * input_w
    if hidden_default >= n_bins:
        k = 1
    else:
        k = math.ceil(n_bins / hidden_default)

    vals = sorted(set([default, 2 * default, k * default]))
    return vals


def base_model_for(dataset, int_dim=None):
    """Return the base model for a given dataset, optionally overriding int_dim."""
    info = DATASET_MLP_INFO[dataset.name]
    return mlp_base(
        info["input_width"],
        int_dim=int_dim if int_dim is not None else info["int_dim"],
        dropout=info["dropout"],
    )


# ── Build a single model for one sweep configuration ───────────────────────
def build_model(dataset, method, n_bins, sig_ratio, padding, int_dim=None, lr=1e-3):
    """Build a compiled Keras model for one sweep configuration."""
    y_min, y_max = 0.0, 1.0  # always scale to [0,1]
    base = lambda: base_model_for(dataset, int_dim=int_dim)
    metrics = ["mse", "mae"]

    hp = kt.HyperParameters()
    hp.Fixed("dropout", 0)
    hp.Fixed("padding", padding)
    hp.Fixed("n_bins", n_bins)
    hp.Fixed("sig_ratio", sig_ratio)
    hp.Fixed("learning_rate", lr)

    if method == "HL-Gaussian":
        hyper = HyperHLGaussian(base, y_min, y_max, metrics=metrics)
    elif method == "HL-MCGaussian":
        hyper = HyperHLMCGaussian(base, y_min, y_max, metrics=metrics)
    elif method == "HL-Projected":
        hyper = HyperHLProjected(base, y_min, y_max, metrics=metrics)
    elif method == "HL-Gibbs":
        hyper = HyperHLGibbs(base, y_min, y_max, metrics=metrics)
    else:
        raise ValueError(f"Unknown method: {method}")

    return hyper.build(hp)


# ── Callbacks ───────────────────────────────────────────────────────────────
class OrigScaleMetricsLogger(keras.callbacks.Callback):
    """Log RMSE and MAE in original (un-scaled) range each epoch."""

    def __init__(self, y_range):
        super().__init__()
        self.y_range = y_range

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        metrics = {}
        for prefix in ("", "val_"):
            mse_key = f"{prefix}mse"
            mae_key = f"{prefix}mae"
            out_prefix = "orig_train" if prefix == "" else "orig_test"
            if mse_key in logs:
                metrics[f"{out_prefix}_rmse"] = math.sqrt(logs[mse_key]) * self.y_range
            if mae_key in logs:
                metrics[f"{out_prefix}_mae"] = logs[mae_key] * self.y_range
        if metrics:
            wandb.log(metrics, commit=False)


# ── Train one configuration ────────────────────────────────────────────────
def run_one(model, dataset, train, test, epochs):
    """Train and evaluate a single model. Returns results dict."""
    y_range = dataset.bounds[1] - dataset.bounds[0]
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_mse",
        patience=50,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks = [WandbMetricsLogger(), OrigScaleMetricsLogger(y_range), early_stop]
    hist = model.fit(
        train, epochs=epochs, verbose=2, callbacks=callbacks, validation_data=test
    )
    outputs = model.evaluate(test, return_dict=True, verbose=2)

    train_rmse_s = math.sqrt(hist.history["mse"][-1])
    train_mae_s = hist.history["mae"][-1]
    test_rmse_s = math.sqrt(outputs["mse"])
    test_mae_s = outputs["mae"]

    results = {
        "train_rmse_scaled": train_rmse_s,
        "train_mae_scaled": train_mae_s,
        "test_rmse_scaled": test_rmse_s,
        "test_mae_scaled": test_mae_s,
        "train_rmse": train_rmse_s * y_range,
        "train_mae": train_mae_s * y_range,
        "test_rmse": test_rmse_s * y_range,
        "test_mae": test_mae_s * y_range,
        "best_epoch": (
            int(early_stop.best_epoch)
            if hasattr(early_stop, "best_epoch")
            else len(hist.history["mse"])
        ),
    }
    for k, v in results.items():
        wandb.summary[k] = v
    return results


# ── Grid generation ─────────────────────────────────────────────────────────
def build_grid(
    methods, n_bins_values, sig_ratio_values, lr_values, seeds, dataset_name
):
    """Build a list of sweep configurations.

    For HL-Projected and HL-Gibbs, sig_ratio is fixed to 1.0 (not used, but
    kept for consistent config shape). For HL-Gaussian, we sweep sig_ratio.
    int_dim is swept per (dataset, n_bins): default, 2× default, and the
    smallest multiple that removes the hidden→bins bottleneck.
    """
    grid = []
    for method in methods:
        for n_bins in n_bins_values:
            if method in ("HL-Gaussian", "HL-MCGaussian"):
                ratios = sig_ratio_values
            else:
                ratios = [1.0]  # sig_ratio is unused for Projected/Gibbs
            int_dims = get_int_dim_values(dataset_name, n_bins)
            for sig_ratio in ratios:
                for int_dim in int_dims:
                    for lr in lr_values:
                        for seed in seeds:
                            grid.append(
                                {
                                    "method": method,
                                    "n_bins": n_bins,
                                    "sig_ratio": sig_ratio,
                                    "int_dim": int_dim,
                                    "lr": lr,
                                    "seed": seed,
                                }
                            )
    return grid


# ── Main sweep loop ────────────────────────────────────────────────────────
def run_sweep(args):
    """Run the sweep for this worker's slice of the grid."""
    dataset = get_dataset_by_name(args.data_dir, args.dataset)

    grid = build_grid(
        methods=args.methods,
        n_bins_values=args.n_bins,
        sig_ratio_values=args.sig_ratio,
        lr_values=args.lr,
        seeds=args.seeds,
        dataset_name=args.dataset,
    )

    # Slice the grid for this worker
    my_grid = grid[args.worker :: args.num_workers]
    print(
        f"Worker {args.worker}/{args.num_workers}: {len(my_grid)}/{len(grid)} configs"
    )

    results_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    for i, cfg in enumerate(my_grid):
        method = cfg["method"]
        n_bins = cfg["n_bins"]
        sig_ratio = cfg["sig_ratio"]
        int_dim = cfg["int_dim"]
        lr = cfg["lr"]
        seed = cfg["seed"]

        run_tag = (
            f"{method}_bins{n_bins}_sig{sig_ratio}_idim{int_dim}_lr{lr}_seed{seed}"
        )
        result_file = os.path.join(results_dir, f"{run_tag}.json")

        # Skip if already completed
        if os.path.exists(result_file):
            print(f"[{i+1}/{len(my_grid)}] Skipping {run_tag} (already done)")
            continue

        print(f"[{i+1}/{len(my_grid)}] Running {run_tag}")

        # Fresh random state
        keras.utils.set_random_seed(seed)

        # Build model
        model = build_model(
            dataset, method, n_bins, sig_ratio, PADDING, int_dim=int_dim, lr=lr
        )

        # Prepare data
        train, test = dataset.get_split(TEST_RATIO, shuffle=True)
        sc = Scaler(*dataset.bounds)
        train = sc.transform(train)
        test = sc.transform(test)
        norm = Normalizer()
        norm.fit(train)
        train = norm.transform(train)
        test = norm.transform(test)

        # Wandb
        group = f"sweep/{args.dataset}/{method}"
        if args.run_prefix:
            group = f"{args.run_prefix}/{group}"

        info = DATASET_MLP_INFO[args.dataset]
        hidden_width = int(int_dim * info["input_width"])
        config = {
            "dataset": args.dataset,
            "method": method,
            "n_bins": n_bins,
            "sig_ratio": sig_ratio,
            "int_dim": int_dim,
            "hidden_width": hidden_width,
            "lr": lr,
            "seed": seed,
            "padding": PADDING,
            "learning_rate": lr,
            "epochs": dataset.epochs,
        }

        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=group,
            name=f"bins{n_bins}_sig{sig_ratio}_idim{int_dim}_lr{lr}_seed{seed}",
            config=config,
        )

        try:
            results = run_one(model, dataset, train, test, dataset.epochs)
            results.update(config)

            # Save locally
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)
            print(
                f"  -> test_rmse={results['test_rmse']:.4f}  test_mae={results['test_mae']:.4f}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
        finally:
            wandb.finish()

        # Clear keras session to free memory between runs
        keras.backend.clear_session()


# ── Launch helper ───────────────────────────────────────────────────────────
def launch_workers_for_dataset(args, dataset_name):
    """Launch parallel worker processes across GPUs for a single dataset.

    With --workers_per_gpu N, each GPU runs N processes (sharing memory via
    memory-growth), so total workers = num_gpus * workers_per_gpu.
    """
    xla_dir = None
    try:
        import nvidia.cuda_nvcc

        xla_dir = os.path.dirname(nvidia.cuda_nvcc.__file__)
    except ImportError:
        pass

    num_gpus = args.num_workers
    wpg = args.workers_per_gpu
    total_workers = num_gpus * wpg

    procs = []
    for g in range(num_gpus):
        gpu = args.gpu_start + g
        for local in range(wpg):
            w = g * wpg + local
            cmd = [
                sys.executable,
                "-m",
                "replication.sweep",
                "--data_dir",
                args.data_dir,
                "--dataset",
                dataset_name,
                "--gpu",
                str(gpu),
                "--worker",
                str(w),
                "--num_workers",
                str(total_workers),
                "--results_dir",
                args.results_dir,
                "--seeds",
                *[str(s) for s in args.seeds],
                "--n_bins",
                *[str(b) for b in args.n_bins],
                "--sig_ratio",
                *[str(r) for r in args.sig_ratio],
                "--lr",
                *[str(l) for l in args.lr],
                "--methods",
                *args.methods,
            ]
            if args.run_prefix:
                cmd.extend(["--run_prefix", args.run_prefix])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            if xla_dir:
                env["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={xla_dir}"

            print(f"[{dataset_name}] Launching worker {w}/{total_workers} on GPU {gpu}")
            p = subprocess.Popen(cmd, env=env)
            procs.append(p)

    for p in procs:
        p.wait()
    print(
        f"[{dataset_name}] All {total_workers} workers ({num_gpus} GPUs × {wpg} per GPU) finished."
    )


def launch_workers(args):
    """Launch sweeps across all requested datasets, one after another."""
    for ds in args.dataset:
        print(f"\n{'='*60}")
        print(f"  Starting sweep for dataset: {ds}")
        print(f"{'='*60}\n")
        launch_workers_for_dataset(args, ds)
    print(f"\nAll datasets complete: {args.dataset}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sweep HL-Projected vs HL-Gaussian across bins and sigma.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", default=".", help="Directory containing the data/ folder"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["ctscan"],
        help="Dataset(s) to sweep on. Multiple datasets are run sequentially in --launch mode.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index to use (single-worker mode)"
    )
    parser.add_argument("--worker", type=int, default=0, help="Worker index (0-based)")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Total number of workers"
    )
    parser.add_argument(
        "--gpu_start", type=int, default=0, help="First GPU index (for --launch mode)"
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of worker processes per GPU (memory-growth lets them share)",
    )
    parser.add_argument(
        "--results_dir",
        default="sweep_results",
        help="Local directory for JSON results",
    )
    parser.add_argument("--run_prefix", default="", help="Prefix for wandb group names")
    parser.add_argument(
        "--launch", action="store_true", help="Launch num_workers parallel processes"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n_bins", type=int, nargs="+", default=N_BINS_VALUES)
    parser.add_argument("--sig_ratio", type=float, nargs="+", default=SIG_RATIO_VALUES)
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=LR_VALUES,
        help="Learning rates to sweep over",
    )
    parser.add_argument(
        "--methods", nargs="+", default=METHODS, help="Methods to include in the sweep"
    )

    args = parser.parse_args()

    if args.launch:
        launch_workers(args)
        return

    # Single-worker mode: --dataset is a list, but workers always get exactly one
    if len(args.dataset) != 1:
        parser.error(
            "Worker mode expects exactly one --dataset. Use --launch for multiple."
        )
    args.dataset = args.dataset[0]

    # Single-worker mode: restrict to one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Allow memory to grow dynamically instead of pre-allocating the full GPU
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    run_sweep(args)


if __name__ == "__main__":
    main()
