import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .classifier_pipeline import (
    load_feature_frames,
    train_classifier_run,
)
from .features import load_feature_schema


def summarize_repetitions(frame, metric_names):
    rows = []
    for metric_name in metric_names:
        values = frame[metric_name].to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": metric_name,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "percentile_2_5": float(np.percentile(values, 2.5)),
                "percentile_97_5": float(np.percentile(values, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def control_values(control_config):
    run_name = control_config["run"]
    first_seed = int(control_config["first_seed"])
    repetitions = int(control_config["repetitions"])
    keep_models = bool(control_config["keep_models"])

    if first_seed < 0:
        raise ValueError("The first control seed cannot be negative.")
    if not 2 <= repetitions <= 50:
        raise ValueError("Control repetitions must be from 2 to 50.")
    return run_name, first_seed, repetitions, keep_models


def run_seed_robustness(config, control_config, run_dir, device):
    run_name, first_seed, repetitions, keep_models = control_values(
        control_config
    )

    frames = load_feature_frames(config, run_dir)
    schema = load_feature_schema(run_dir)
    feature_columns = schema["run_columns"][run_name]
    control_dir = Path(run_dir) / "controls" / "seed_robustness"
    control_dir.mkdir(parents=True, exist_ok=True)

    results_path = control_dir / "seed_robustness_results.csv"
    rows = []
    for repetition in range(repetitions):
        seed = first_seed + repetition
        output_dir = control_dir / f"seed_{seed}"
        print(f"\nSeed robustness {repetition + 1}/{repetitions}: seed={seed}")
        summary = train_classifier_run(
            config,
            frames,
            feature_columns,
            output_dir,
            seed,
            None,
            keep_models,
            device,
        )
        row = {
            "repetition": repetition + 1,
            "seed": seed,
            "best_epoch": summary["best_epoch"],
        }
        for split_name in ["validation", "test"]:
            for metric_name in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "matthews_correlation",
                "cohen_kappa",
            ]:
                row[f"{split_name}_{metric_name}"] = summary["metrics"][
                    split_name
                ][metric_name]
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_path, index=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    metric_names = [
        column
        for column in results.columns
        if column not in {"repetition", "seed", "best_epoch"}
    ]
    aggregate = summarize_repetitions(results, metric_names)
    aggregate_path = control_dir / "seed_robustness_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    return results, aggregate


def shuffled_label_values(config, frames, seed):
    label_column = config["dataset"]["label_column"]
    random_generator = np.random.default_rng(seed)
    values = {}

    for split_name, frame in frames.items():
        labels = frame[label_column].astype(str).to_numpy(copy=True)

        if split_name in {"train", "validation"}:
            labels = random_generator.permutation(labels)

        values[split_name] = labels

    return values



def load_observed_reference_metrics(config, run_dir, run_name):
    path = Path(run_dir) / "classifiers" / run_name / "metrics_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Observed {run_name} metrics do not exist. Run script 06 first."
        )
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]["test"]


def run_shuffled_label_control(config, control_config, run_dir, device):
    run_name, first_seed, repetitions, keep_models = control_values(
        control_config
    )

    frames = load_feature_frames(config, run_dir)
    schema = load_feature_schema(run_dir)
    feature_columns = schema["run_columns"][run_name]
    observed = load_observed_reference_metrics(config, run_dir, run_name)
    control_dir = Path(run_dir) / "controls" / "shuffled_labels"
    control_dir.mkdir(parents=True, exist_ok=True)

    results_path = control_dir / "shuffled_label_results.csv"
    rows = []
    for repetition in range(repetitions):
        seed = first_seed + repetition
        shuffled = shuffled_label_values(config, frames, seed)
        output_dir = control_dir / f"shuffle_{repetition + 1:03d}"
        print(f"\nShuffled-label control {repetition + 1}/{repetitions}: seed={seed}")
        summary = train_classifier_run(
            config,
            frames,
            feature_columns,
            output_dir,
            seed,
            shuffled,
            keep_models,
            device,
        )
        test_metrics = summary["metrics"]["test"]
        rows.append(
            {
                "repetition": repetition + 1,
                "seed": seed,
                "accuracy": test_metrics["accuracy"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "matthews_correlation": test_metrics[
                    "matthews_correlation"
                ],
            }
        )
        pd.DataFrame(rows).to_csv(results_path, index=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)

    comparison_rows = []
    for metric_name in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "matthews_correlation",
    ]:
        null_values = results[metric_name].to_numpy(dtype=np.float64)
        observed_value = float(observed[metric_name])
        p_value = float(
            (1 + np.sum(null_values >= observed_value))
            / (repetitions + 1)
        )
        comparison_rows.append(
            {
                "metric": metric_name,
                "observed": observed_value,
                "shuffled_mean": float(np.mean(null_values)),
                "shuffled_standard_deviation": float(
                    np.std(null_values, ddof=1)
                ),
                "empirical_p_value": p_value,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = control_dir / "shuffled_label_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    return results, comparison

