import json
from pathlib import Path

import matplotlib

from blueberry_latent_map_phenotyping.paths import project_relative_text

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import font_manager



def project_relative_paths(project_root, paths):
    return [project_relative_text(project_root, path) for path in paths]


def require_font(font_family):
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=font_family),
            fallback_to_default=False,
        )
    except ValueError as error:
        raise RuntimeError(
            f"Required figure font is not installed: {font_family}"
        ) from error
    return font_family


def configure_article_style(config):
    family = require_font(config["figures"]["font_family"])
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return family


def save_figure(figure, output_base, formats, png_dpi):
    saved_paths = []
    for file_format in formats:
        output_path = Path(f"{output_base}.{file_format}")
        if file_format.lower() == "png":
            figure.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
        else:
            figure.savefig(output_path, bbox_inches="tight")
        saved_paths.append(str(output_path))
    plt.close(figure)
    return saved_paths


def load_classifier_summaries(config, run_dir):
    summaries = {}
    for run_name in config["classifier"]["runs"]:
        path = Path(run_dir) / "classifiers" / run_name / "metrics_summary.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Classifier metrics do not exist for {run_name}: {path}"
            )
        summaries[run_name] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def classifier_metrics_table(config, summaries):
    rows = []
    for run_name in config["classifier"]["runs"]:
        summary = summaries[run_name]
        for split_name in ["train", "validation", "test"]:
            row = {
                "run": run_name,
                "feature_count": summary["feature_count"],
                "split": split_name,
            }
            row.update(summary["metrics"][split_name])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_vae_training(config, run_dir):
    history_path = Path(run_dir) / "vae" / "metrics" / "training_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"VAE training history does not exist: {history_path}")
    history = pd.read_csv(history_path)

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    pairs = [
        ("total_loss", "Total loss"),
        ("reconstruction_loss", "Reconstruction loss"),
        ("kl_loss", "KL divergence"),
    ]
    for axis, (column_suffix, title) in zip(axes, pairs):
        axis.plot(
            history["epoch"],
            history[f"train_{column_suffix}"],
            label="Train",
            color="#1f77b4",
        )
        axis.plot(
            history["epoch"],
            history[f"validation_{column_suffix}"],
            label="Validation",
            color="#d95f02",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss per image")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.tight_layout()

    output_base = Path(run_dir) / "reports" / "vae_training_curves"
    return save_figure(
        figure,
        output_base,
        config["figures"]["formats"],
        int(config["figures"]["png_dpi"]),
    )


def plot_classifier_training(config, run_dir):
    run_name = config["evaluation"]["reference_classifier"]
    history_path = (
        Path(run_dir)
        / "classifiers"
        / run_name
        / "training_history.csv"
    )
    if not history_path.exists():
        raise FileNotFoundError(
            f"Classifier training history does not exist: {history_path}"
        )
    history = pd.read_csv(history_path)

    figure, axes = plt.subplots(1, 2, figsize=(5.8, 2.5))
    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        label="Train",
        color="#1f77b4",
    )
    axes[0].plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation",
        color="#d95f02",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title(f"{run_name} loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(
        history["epoch"],
        history["train_accuracy"],
        label="Train",
        color="#1f77b4",
    )
    axes[1].plot(
        history["epoch"],
        history["validation_accuracy"],
        label="Validation",
        color="#d95f02",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(f"{run_name} accuracy")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.tight_layout()

    output_base = Path(run_dir) / "reports" / "classifier_training_curves"
    return save_figure(
        figure,
        output_base,
        config["figures"]["formats"],
        int(config["figures"]["png_dpi"]),
    )


def plot_classifier_comparison(config, metrics_table, run_dir):
    bar_label_font_size = 6.5
    bar_label_rotation = 60
    bar_label_padding = 2
    percentage_axis_top = 130

    test_table = metrics_table.loc[
        metrics_table["split"] == "test"
    ].copy()
    run_order = list(config["classifier"]["runs"])
    test_table["run"] = pd.Categorical(
        test_table["run"],
        categories=run_order,
        ordered=True,
    )
    test_table = test_table.sort_values("run")

    metric_columns = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
    ]
    positions = np.arange(len(run_order))
    width = 0.15
    bar_gap = 0.04
    colors = ["#1f77b4", "#4daf4a", "#E69F00"]

    figure, axis = plt.subplots(figsize=(6, 2.8))

    for index, (column, label) in enumerate(metric_columns):
        offset = (index - 1) * (width + bar_gap)
        bars = axis.bar(
            positions + offset,
            test_table[column].to_numpy(),
            width,
            label=label,
            color=colors[index],
        )
        percentage_labels = [
            f"{100.0 * bar.get_height():.2f}%"
            for bar in bars
        ]
        axis.bar_label(
            bars,
            labels=percentage_labels,
            padding=bar_label_padding,
            fontsize=bar_label_font_size,
            rotation=bar_label_rotation,
        )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xticks(positions)
    axis.set_xticklabels(run_order)
    axis.set_xlabel("Latent descriptor configuration")
    axis.set_ylabel("Score (%)")

    tick_positions = np.linspace(0.0, 1.0, 6)
    axis.set_yticks(tick_positions)
    axis.set_yticklabels([f"{100.0 * value:.0f}" for value in tick_positions])
    axis.set_ylim(0.0, percentage_axis_top / 100.0)

    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()

    output_base = Path(run_dir) / "reports" / "classifier_ablation_comparison"
    return save_figure(
        figure,
        output_base,
        config["figures"]["formats"],
        int(config["figures"]["png_dpi"]),
    )


def plot_reference_classifier(config, run_dir):
    run_name = config["evaluation"]["reference_classifier"]
    classifier_dir = Path(run_dir) / "classifiers" / run_name
    matrix_path = classifier_dir / "test_confusion_matrix.csv"
    report_path = classifier_dir / "test_classification_report.csv"
    if not matrix_path.exists() or not report_path.exists():
        raise FileNotFoundError(
            f"Reference classifier outputs are incomplete in {classifier_dir}"
        )

    class_names = config["dataset"]["classes"]
    matrix = pd.read_csv(matrix_path, index_col=0).to_numpy(dtype=np.float64)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix),
        where=row_totals != 0,
    )
    report = pd.read_csv(report_path, index_col=0).loc[class_names]

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    image = axes[0].imshow(
        normalized,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title(f"{run_name} normalized confusion matrix")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=30, ha="right")
    axes[0].set_yticklabels(class_names)
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            color = "white" if value >= 0.5 else "black"
            axes[0].text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    positions = np.arange(len(class_names))
    width = 0.23
    for index, (column, label, color) in enumerate(
        [
            ("precision", "Precision", "#1f77b4"),
            ("recall", "Recall", "#4daf4a"),
            ("f1-score", "F1", "#d95f02"),
        ]
    ):
        axes[1].bar(
            positions + (index - 1) * width,
            report[column].to_numpy(dtype=np.float64),
            width,
            label=label,
            color=color,
        )
    axes[1].set_title(f"{run_name} class-wise performance")
    axes[1].set_xlabel("Canopy size class")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(class_names, rotation=30, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()

    output_base = Path(run_dir) / "reports" / "reference_classifier_results"
    return save_figure(
        figure,
        output_base,
        config["figures"]["formats"],
        int(config["figures"]["png_dpi"]),
    )


def generate_results(config, project_root, run_dir):
    font_family = configure_article_style(config)
    summaries = load_classifier_summaries(config, run_dir)
    table = classifier_metrics_table(config, summaries)
    table_path = Path(run_dir) / "reports" / "classifier_metrics_summary.csv"
    table.to_csv(table_path, index=False)

    outputs = {
        "font_family": font_family,
        "metrics_table": str(project_relative_text(project_root, table_path)),
        "vae_training_figures": project_relative_paths(project_root, plot_vae_training(config, run_dir)),
        "classifier_training_figures": project_relative_paths(project_root, plot_classifier_training(
            config, run_dir)),
        "classifier_comparison_figures": project_relative_paths(project_root, plot_classifier_comparison(
            config, table, run_dir)),
        "reference_classifier_figures": project_relative_paths(project_root, plot_reference_classifier(
            config, run_dir)),
    }
    manifest_path = Path(run_dir) / "reports" / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(outputs, indent=2),
        encoding="utf-8",
    )
    return outputs
