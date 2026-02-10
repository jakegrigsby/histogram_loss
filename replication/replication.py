"""Module for running replication experiments.

Based on previous papers and code by Ehsan Imani.

Usage: python replication.py data_dir

Params:
    data_dir - the directory containing the CSV files for the datasets
"""

import sys
import os
import math
import argparse
from replication.csvdataset import CSVDataset
from tensorflow import keras
from experiment.models import *
from experiment.hypermodels import *
from experiment.preprocessing import *
import keras_tuner as kt
import json
import wandb
from wandb.integration.keras import WandbMetricsLogger

WANDB_ENTITY = "ut-austin-rpl-general-team"
WANDB_PROJECT = "histogram_loss"
RUN_PREFIX = ""

import tensorflow as tf


class GradientNormLogger(keras.callbacks.Callback):
    """Callback that computes and logs the global gradient norm at the end of each epoch."""

    def __init__(self, train_data, n_batches=10):
        super().__init__()
        self.train_data = train_data
        self.n_batches = n_batches

    def _compute_grad_norm(self, x_batch, y_batch):
        """Compute the global gradient norm for a single batch."""
        with tf.GradientTape() as tape:
            if hasattr(self.model, "get_hist") and hasattr(self.model, "transform"):
                hist = self.model.get_hist(x_batch, training=True)
                y_transformed = self.model.transform(y_batch)
                loss = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(y_transformed, hist)
                )
            else:
                y_pred = self.model(x_batch, training=True)
                loss_fn = keras.losses.get(self.model.loss)
                loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [g for g in grads if g is not None]
        if grads:
            return tf.linalg.global_norm(grads).numpy()
        return 0.0

    def on_epoch_end(self, epoch, logs=None):
        norms = []
        for x_batch, y_batch in self.train_data.take(self.n_batches):
            norms.append(self._compute_grad_norm(x_batch, y_batch))
        avg_norm = sum(norms) / len(norms)
        wandb.log({"grad_norm": float(avg_norm)}, commit=False)


class OrigScaleMetricsLogger(keras.callbacks.Callback):
    """Callback that logs RMSE and MAE in the original (un-scaled) data range each epoch."""

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


def mlp_base(input_width, hidden=4, dropout=0.05, int_dim=0.5):
    """Return an MLP base model to learn features from the data.

    Params:
        input_width - the number of input features
        hidden - the number of hidden layers
        dropout - the dropout rate to use on the input
        int_dim - the ratio of the size of the hidden layers relative to the input size

    Returns: a Keras model
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(dropout))
    width = int(int_dim * input_width)
    for i in range(hidden):
        model.add(
            keras.layers.Dense(
                width, activation="relu", kernel_initializer="lecun_uniform"
            )
        )
    return model


def base_models(dataset):
    """Return the base model for a given dataset.

    Params:
        dataset - the name of the dataset

    Returns: the MLP base model
    """
    if dataset.name == "ctscan":
        return mlp_base(385)
    elif dataset.name == "bike":
        return mlp_base(16, int_dim=4, dropout=0)
    elif dataset.name == "pole":
        return mlp_base(49, dropout=0)
    elif dataset.name == "songyear":
        return mlp_base(90)


def get_datasets(base_dir):
    """Load the datasets to use in the experiment.

    Params:
        base_dir - the directory in which the CSV files are located

    Returns: a list of dataset objects
    """
    data_dir = os.path.join(base_dir, "data")
    ctscan = CSVDataset(
        os.path.join(data_dir, "slice_localization_data.csv"),
        "reference",
        batch_size=256,
    )
    ctscan.bounds = (0.0, 100.0)
    ctscan.name = "ctscan"
    ctscan.epochs = 1000

    bikeshare = CSVDataset(
        os.path.join(data_dir, "hour.csv"), "cnt", drop="dteday", batch_size=256
    )
    bikeshare.bounds = (0.0, 1000.0)
    bikeshare.name = "bike"
    bikeshare.epochs = 500

    songyear = CSVDataset(
        os.path.join(data_dir, "YearPredictionMSD.txt"), 0, header=None, batch_size=256
    )
    songyear.bounds = (1922.0, 2011.0)
    songyear.name = "songyear"
    songyear.epochs = 150

    pole = CSVDataset(os.path.join(data_dir, "pole.csv"), "target", batch_size=256)
    pole.bounds = (0.0, 100.0)
    pole.name = "pole"
    pole.epochs = 500

    return [ctscan, bikeshare, songyear, pole]


def get_models(dataset, scale=True):
    """Get the models for a given dataset.

    Params:
        dataset - a Dataset object with the data and info
        scale - True if the y data will be scaled to [0, 1]

    Returns: a list of models to run on the dataset
    """
    if scale:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = dataset.bounds
    base = lambda: base_models(dataset)
    metrics = ["mse", "mae"]
    hp = kt.HyperParameters()
    hp.Fixed("dropout", 0)
    hp.Fixed("padding", 0.125)
    hp.Fixed("sig_ratio", 1.0)
    hp.Fixed("n_bins", 100)
    hp.Fixed("nu", 1.0)
    hp.Fixed("learning_rate", 1e-3)

    hyperhlg = HyperHLGaussian(base, y_min, y_max, metrics=metrics)
    hlg = hyperhlg.build(hp)

    hyperl2 = HyperRegression(base, loss="mse", metrics=metrics, name="L2")
    l2 = hyperl2.build(hp)

    hyperl1 = HyperRegression(base, loss="mae", metrics=metrics, name="L1")
    l1 = hyperl1.build(hp)

    hyperhl1 = HyperHLOneBin(base, y_min, y_max, metrics=metrics)
    hl1 = hyperhl1.build(hp)

    hypergibbs = HyperHLGibbs(base, y_min, y_max, metrics=metrics)
    hlgibbs = hypergibbs.build(hp)

    hypermaxent = HyperHLMaxEnt(base, y_min, y_max, metrics=metrics)
    hlmaxent = hypermaxent.build(hp)

    hypercauchy = HyperHLCauchy(base, y_min, y_max, metrics=metrics)
    hlcauchy = hypercauchy.build(hp)

    hyperproj = HyperHLProjected(base, y_min, y_max, metrics=metrics)
    hlproj = hyperproj.build(hp)

    lin_base = lambda: keras.layers.Identity()
    hyperlin = HyperRegression(lin_base, loss="mse", metrics=metrics, name="LinReg")
    lin = hyperlin.build(hp)

    return [l1, l2, hlg, hl1, hlgibbs, hlmaxent, hlcauchy, hlproj, lin]


def run_model(model, epochs, train, test, bounds):
    """Run an experiment for one model on a dataset.

    Params:
        model - the Keras model to test
        epochs - the number of epochs to train for
        train - the tf Dataset with the training split
        test - the tf Dataset with the testing split
        bounds - (y_min, y_max) tuple for the original data range

    Returns: results - a dict of the training and testing metrics
    """
    y_range = bounds[1] - bounds[0]
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_mse",
        patience=100,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks = [
        WandbMetricsLogger(),
        OrigScaleMetricsLogger(y_range),
        GradientNormLogger(train),
        early_stop,
    ]
    hist = model.fit(
        train, epochs=epochs, verbose=2, callbacks=callbacks, validation_data=test
    )
    outputs = model.evaluate(test, return_dict=True, verbose=2)

    # Scaled [0,1] metrics
    train_rmse_scaled = math.sqrt(hist.history["mse"][-1])
    train_mae_scaled = hist.history["mae"][-1]
    test_rmse_scaled = math.sqrt(outputs["mse"])
    test_mae_scaled = outputs["mae"]

    # Original scale metrics
    train_rmse_orig = train_rmse_scaled * y_range
    train_mae_orig = train_mae_scaled * y_range
    test_rmse_orig = test_rmse_scaled * y_range
    test_mae_orig = test_mae_scaled * y_range

    results = {
        "train_rmse_scaled": train_rmse_scaled,
        "train_mae_scaled": train_mae_scaled,
        "test_rmse_scaled": test_rmse_scaled,
        "test_mae_scaled": test_mae_scaled,
        "train_rmse": train_rmse_orig,
        "train_mae": train_mae_orig,
        "test_rmse": test_rmse_orig,
        "test_mae": test_mae_orig,
    }

    # Log summaries in both scales
    for k, v in results.items():
        wandb.summary[k] = v

    return results


def preprocess(train, test, bounds, scale, norm):
    """Preprocess the data by scaling and normalizing.

    Params:
        train - the tf Dataset with the train split
        test - the tf Dataset with the test split
        bounds - a tuple with the minimum and maximum y values
        scale - True if the y values will be scaled to [0, 1]; False otherwise
        norm - True if the x values will be normalized based on the training data; False otherwise

    Returns: (train, test) - the transformed train and test splits
    """
    if scale:
        sc = Scaler(*bounds)
        train = sc.transform(train)
        test = sc.transform(test)

    if norm:
        norm = Normalizer()
        norm.fit(train)
        train = norm.transform(train)
        test = norm.transform(test)
    return train, test


def run_seed(dataset, seed, test_ratio, method_filter=None, scale=True, norm=True):
    """Run the experiment on a dataset for all models with a given seed.

    Params:
        dataset - the Dataset object to run the experiment on
        seed - the seed to use for the experiment
        test_ratio - the proportion of samples held out for testing
        method_filter - optional list of model names to run (None = all)
        scale - True if the y values will be scaled to [0, 1]; False otherwise
        norm - True if the x values will be normalized based on the training data; False otherwise

    Returns: results - a dict with the results for each model
    """
    results = {}
    keras.utils.set_random_seed(seed)
    models = get_models(dataset)
    train, test = dataset.get_split(test_ratio, shuffle=True)

    train, test = preprocess(train, test, dataset.bounds, scale, norm)

    if method_filter:
        filter_lower = [m.lower() for m in method_filter]
        models = [m for m in models if m.name.lower() in filter_lower]
        if not models:
            print(
                f"WARNING: no models matched filter {method_filter}. Available: {[m.name for m in get_models(dataset)]}"
            )
            return results

    for model in models:
        group = f"{dataset.name}/{model.name}"
        if RUN_PREFIX:
            group = f"{RUN_PREFIX}/{group}"
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=group,
            name=f"seed{seed}",
            config={
                "dataset": dataset.name,
                "model": model.name,
                "seed": seed,
                "epochs": dataset.epochs,
                "n_bins": 100,
                "padding": 0.125,
                "sig_ratio": 1.0,
                "learning_rate": 1e-3,
                "run_prefix": RUN_PREFIX,
            },
        )
        results[model.name] = run_model(
            model, dataset.epochs, train, test, dataset.bounds
        )
        wandb.finish()
    return results


def run_dataset(dataset, seeds, test_ratio, method_filter=None):
    """Run an experiment on a dataset with multiple seeds.

    Params:
        dataset - the Dataset object to run the experiment on
        seeds - the list of seeds to use
        test_ratio - the proportion of samples held out for testing
        method_filter - optional list of model names to run (None = all)

    Returns: results - a dict with the results for each seed
    """
    results = {}
    for seed in seeds:
        results[seed] = run_seed(dataset, seed, test_ratio, method_filter=method_filter)
        outfile = os.path.join("temp_results", f"{dataset.name}-{seed}.json")
        save(outfile, results)
    return results


def run(seeds, datasets, test_ratio, method_filter=None):
    """Run the experiment on multiple datasets and seeds.

    Params:
        seeds - the list of seeds to use
        datasets - the list of Datasets to use
        test_ratio - the proportion of samples held out for testing
        method_filter - optional list of model names to run (None = all)

    Returns: results - a dict with the results for each dataset
    """
    results = {}
    for dataset in datasets:
        results[dataset.name] = run_dataset(
            dataset, seeds, test_ratio, method_filter=method_filter
        )
    return results


def save(outfile, results):
    """Save the results of the experiment to a JSON file.

    Params:
        outfile - the file to save the results to
        results - the dict with the results to save
    """
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as out_file:
        json.dump(results, out_file, indent=4)


ALL_METHODS = [
    "L1",
    "L2",
    "HL-Gaussian",
    "HL-OneBin",
    "HL-Gibbs",
    "HL-MaxEnt",
    "HL-Cauchy",
    "HL-Projected",
    "LinReg",
]
ALL_DATASETS = ["ctscan", "bike", "songyear", "pole"]


def main(base_dir, dataset_filter=None, method_filter=None):
    """Run the replication experiment."""
    test_ratio = 0.2
    seeds = [1, 2, 3, 4, 5]
    outfile = "replication.json"
    datasets = get_datasets(base_dir)

    if dataset_filter:
        filter_lower = [d.lower() for d in dataset_filter]
        datasets = [d for d in datasets if d.name.lower() in filter_lower]
        if not datasets:
            print(
                f"No datasets matched filter {dataset_filter}. Available: {ALL_DATASETS}"
            )
            return

    results = run(seeds, datasets, test_ratio, method_filter=method_filter)
    save(outfile, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run replication experiments.")
    parser.add_argument("data_dir", help="Directory containing the CSV data files")
    parser.add_argument(
        "--run_prefix",
        default="",
        help="Prefix for wandb group names (e.g. 'new_attempt')",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        default=None,
        help=f"Method(s) to run. Choices: {ALL_METHODS}. Default: all",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help=f"Dataset(s) to run. Choices: {ALL_DATASETS}. Default: all",
    )
    args = parser.parse_args()
    RUN_PREFIX = args.run_prefix
    main(args.data_dir, dataset_filter=args.dataset, method_filter=args.method)
