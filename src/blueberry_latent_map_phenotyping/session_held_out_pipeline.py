import gc
import json
from pathlib import Path

import pandas as pd
import torch

from .classifier_pipeline import load_feature_frames, train_classifier_run
from .features import extract_latent_features, load_feature_schema
from .paths import project_relative_text, session_fold_paths
from .vae_pipeline import (
    evaluate_beta_vae_from_paths,
    extract_latent_maps_from_paths,
    train_beta_vae_from_paths,
)


def session_fold_names(config):
    return list(config["session_held_out"]["folds"])


def session_manifest_paths(config, project_root, fold_name):
    configured_paths = session_fold_paths(
        config,
        project_root,
        fold_name,
    )
    return {
        split_name: configured_paths[split_name]
        for split_name in ["train", "validation", "test"]
    }


def validate_session_beta_vae_splits(config, project_root):
    for fold_name in session_fold_names(config):
        configured_paths = session_manifest_paths(
            config,
            project_root,
            fold_name,
        )
        for split_name, path in configured_paths.items():
            if not path.is_file():
                relative_path = project_relative_text(project_root, path)
                raise FileNotFoundError(
                    f"Missing {fold_name} {split_name} split: {relative_path}"
                )


def release_training_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def beta_vae_summary_row(config, fold_name, summary):
    return {
        "fold": fold_name,
        "held_out_session": config["session_held_out"]["folds"][fold_name],
        "seed": int(config["vae"]["seed"]),
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "best_validation_reconstruction_loss": float(
            summary["best_validation_reconstruction_loss"]
        ),
        "checkpoint": summary["checkpoint"],
    }


def train_session_beta_vae_experiments(
    config,
    project_root,
    experiment_dir,
    device,
):
    rows = []
    summary_path = Path(experiment_dir) / "experiment_summary.csv"

    for fold_name in session_fold_names(config):
        release_training_memory()
        fold_dir = Path(experiment_dir) / fold_name
        configured_paths = session_manifest_paths(
            config,
            project_root,
            fold_name,
        )
        print(f"\nStarting independent beta-VAE run: {fold_name}")
        print(
            "Initialized a fresh model, optimizer, scheduler, and "
            "early-stopping state."
        )
        try:
            summary = train_beta_vae_from_paths(
                config,
                project_root,
                fold_dir,
                device,
                configured_paths,
            )
        finally:
            release_training_memory()

        rows.append(beta_vae_summary_row(config, fold_name, summary))
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(
            f"Completed {fold_name}: "
            f"best_epoch={summary['best_epoch']}, "
            "best_validation_reconstruction_loss="
            f"{summary['best_validation_reconstruction_loss']:.6f}"
        )

    relative_summary = project_relative_text(project_root, summary_path)
    print(f"Saved beta-VAE experiment summary: {relative_summary}")
    return rows


def evaluate_session_beta_vae_experiments(
    config,
    project_root,
    experiment_dir,
    device,
):
    rows = []
    summary_path = Path(experiment_dir) / "reconstruction_summary.csv"

    for fold_name in session_fold_names(config):
        fold_dir = Path(experiment_dir) / fold_name
        configured_paths = session_manifest_paths(
            config,
            project_root,
            fold_name,
        )
        print(f"\nEvaluating beta-VAE run: {fold_name}")
        fold_rows = evaluate_beta_vae_from_paths(
            config,
            project_root,
            fold_dir,
            device,
            configured_paths,
        )
        for row in fold_rows:
            rows.append(
                {
                    "fold": fold_name,
                    "held_out_session": config["session_held_out"]["folds"][
                        fold_name
                    ],
                    **row,
                }
            )
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    relative_summary = project_relative_text(project_root, summary_path)
    print(f"Saved reconstruction summary: {relative_summary}")
    return rows


def extract_session_beta_vae_latent_maps(
    config,
    project_root,
    experiment_dir,
    device,
):
    summaries = {}
    for fold_name in session_fold_names(config):
        fold_dir = Path(experiment_dir) / fold_name
        configured_paths = session_manifest_paths(
            config,
            project_root,
            fold_name,
        )
        print(f"\nExtracting latent maps: {fold_name}")
        summaries[fold_name] = extract_latent_maps_from_paths(
            config,
            project_root,
            fold_dir,
            device,
            configured_paths,
        )
    return summaries


def extract_session_beta_vae_features(config, project_root, experiment_dir):
    summaries = {}
    for fold_name in session_fold_names(config):
        fold_dir = Path(experiment_dir) / fold_name
        print(f"\nExtracting latent-map features: {fold_name}")
        summaries[fold_name] = extract_latent_features(
            config,
            project_root,
            fold_dir,
        )
    return summaries


def c5_summary_row(config, fold_name, summary):
    test_metrics = summary["metrics"]["test"]
    return {
        "fold": fold_name,
        "held_out_session": config["session_held_out"]["folds"][fold_name],
        "seed": int(summary["seed"]),
        "best_epoch": int(summary["best_epoch"]),
        "best_validation_loss": float(summary["best_validation_loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_samples": int(test_metrics["samples"]),
    }


def train_session_beta_vae_c5(
    config,
    project_root,
    experiment_dir,
    device,
):
    classifier_name = config["evaluation"]["reference_classifier"]
    if classifier_name != "C5":
        raise ValueError("evaluation.reference_classifier must be C5.")

    rows = []
    summary_path = Path(experiment_dir) / "c5_fold_results.csv"
    for fold_name in session_fold_names(config):
        release_training_memory()
        fold_dir = Path(experiment_dir) / fold_name
        frames = load_feature_frames(config, fold_dir)
        schema = load_feature_schema(fold_dir)
        feature_columns = schema["run_columns"][classifier_name]
        output_dir = fold_dir / "classifiers" / classifier_name
        print(f"\nTraining independent {classifier_name} run: {fold_name}")
        try:
            summary = train_classifier_run(
                config,
                frames,
                feature_columns,
                output_dir,
                int(config["classifier"]["seed"]),
                None,
                True,
                device,
            )
        finally:
            release_training_memory()

        rows.append(c5_summary_row(config, fold_name, summary))
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(
            f"Completed {fold_name}: "
            f"test_accuracy={summary['metrics']['test']['accuracy']:.4f}, "
            f"test_macro_f1={summary['metrics']['test']['macro_f1']:.4f}"
        )

    relative_summary = project_relative_text(project_root, summary_path)
    print(f"Saved C5 fold results: {relative_summary}")
    return rows


def load_c5_fold_results(config, experiment_dir):
    rows = []
    for fold_name in session_fold_names(config):
        summary_path = (
            Path(experiment_dir)
            / fold_name
            / "classifiers"
            / "C5"
            / "metrics_summary.json"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"C5 metrics summary does not exist: {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(c5_summary_row(config, fold_name, summary))
    return rows


def summarize_session_beta_vae_results(config, project_root, experiment_dir):
    rows = load_c5_fold_results(config, experiment_dir)
    fold_path = Path(experiment_dir) / "c5_fold_results.csv"
    fold_frame = pd.DataFrame(rows)
    fold_frame.to_csv(fold_path, index=False)

    metric_columns = {
        "accuracy": "test_accuracy",
        "balanced_accuracy": "test_balanced_accuracy",
        "macro_f1": "test_macro_f1",
    }
    summary_rows = []
    for metric_name, column_name in metric_columns.items():
        mean = float(fold_frame[column_name].mean())
        standard_deviation = float(fold_frame[column_name].std(ddof=1))
        summary_rows.append(
            {
                "metric": metric_name,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "mean_sd": f"{mean:.4f} ± {standard_deviation:.4f}",
            }
        )

    summary_path = Path(experiment_dir) / "c5_unweighted_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    relative_fold_path = project_relative_text(project_root, fold_path)
    relative_summary_path = project_relative_text(project_root, summary_path)
    print(f"Saved C5 fold results: {relative_fold_path}")
    print(f"Saved unweighted mean and sample SD: {relative_summary_path}")
    return rows, summary_rows

