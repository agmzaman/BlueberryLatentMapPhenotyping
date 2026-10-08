import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config, load_run_config
from blueberry_latent_map_phenotyping.paths import project_path, project_relative_text, run_config_path



def figure_settings():
    return {
        "experiment_directory": "",
        "output_directory": "figures/efficientnet",
        "font_family": "Calibri",
        "font_size": 9,
        "axes_label_size": 9,
        "axes_title_size": 9,
        "tick_label_size": 8,
        "legend_size": 8,
        "png_dpi": 600,
        "line_width": 1.4,
        "grid_alpha": 0.20,
        "training_color": "#0072B2",
        "validation_color": "#E69F00",
        "checkpoint_color": "#666666",
        "metric_colors": {
            "Precision": "#0072B2",
            "Recall": "#009E73",
            "F1 score": "#E69F00",
        },
        "run_labels": {
            "stratified_random": "Stratified random",
            "S1": "S1",
            "S2": "S2",
            "S3": "S3",
        },
        "training_histories_size": [7.2, 8.4],
        "confusion_matrices_size": [7.2, 6.0],
        "classwise_metrics_size": [7.2, 5.4],
    }


def configure_article_style(style):
    font = font_manager.FontProperties(family=style["font_family"])
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError(f"Required figure font is not installed: {style['font_family']}") from error

    plt.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.size": style["font_size"],
            "axes.labelsize": style["axes_label_size"],
            "axes.titlesize": style["axes_title_size"],
            "xtick.labelsize": style["tick_label_size"],
            "ytick.labelsize": style["tick_label_size"],
            "legend.fontsize": style["legend_size"],
            "axes.linewidth": 0.8,
            "lines.linewidth": style["line_width"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def efficientnet_experiment_directory(config, project_root, style):
    experiment_root = project_path(project_root, config["paths"]["efficientnet_experiments_dir"]).resolve()
    selected = str(style["experiment_directory"]).strip()

    if selected:
        selected_path = Path(selected)
        if selected_path.is_absolute() or ".." in selected_path.parts:
            raise ValueError("experiment_directory must be project-relative.")
        experiment_directory = project_path(project_root, selected_path).resolve()
    else:
        candidates = sorted(path.resolve() for path in experiment_root.glob("run_*") if path.is_dir())
        if not candidates:
            relative_root = project_relative_text(project_root, experiment_root)
            raise FileNotFoundError(f"No EfficientNet experiment directory was found under {relative_root}.")
        experiment_directory = candidates[-1]

    try:
        experiment_directory.relative_to(experiment_root)
    except ValueError as error:
        raise ValueError("The selected EfficientNet experiment is outside its configured root.") from error

    if not experiment_directory.exists():
        relative_path = project_relative_text(project_root, experiment_directory)
        raise FileNotFoundError(f"EfficientNet experiment directory does not exist: {relative_path}")
    return experiment_directory


def figure_output_directory(experiment_directory, style):
    relative_path = Path(style["output_directory"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("The figure output directory must be experiment-relative.")
    output_directory = Path(experiment_directory) / relative_path
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def save_figure(fig, output_base, style):
    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    pdf_path = output_base.with_suffix(".pdf")

    fig.savefig(png_path, dpi=int(style["png_dpi"]), bbox_inches="tight")
    with plt.rc_context({"svg.fonttype": "path"}):
        fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    for output_path in [png_path, svg_path, pdf_path]:
        print(f"Saved figure: {project_relative_text(PROJECT_ROOT, output_path)}")


def require_result_file(path):
    if not path.exists():
        relative_path = project_relative_text(PROJECT_ROOT, path)
        raise FileNotFoundError(f"Required result file does not exist: {relative_path}")


def require_columns(frame, columns, source_name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {source_name}: {missing}")


def panel_title(axis, panel_label, title, font_size, x_position, y_position):
    axis.text(
        x_position,
        y_position,
        f"({panel_label}) {title}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=font_size,
    )


def class_tick_labels(class_names):
    return [class_name.replace(" ", "\n") for class_name in class_names]


def run_label(style, run_name):
    if run_name not in style["run_labels"]:
        raise KeyError(f"Missing display label for EfficientNet run: {run_name}")
    return style["run_labels"][run_name]


def load_training_history(experiment_directory, run_name):
    path = Path(experiment_directory) / run_name / "training_history.csv"
    require_result_file(path)
    history = pd.read_csv(path)
    columns = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"]
    require_columns(history, columns, f"{run_name}/training_history.csv")
    if history.empty:
        raise ValueError(f"Training history is empty for {run_name}.")
    return history


def load_metrics_summary(experiment_directory, run_name):
    path = Path(experiment_directory) / run_name / "metrics_summary.json"
    require_result_file(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if "best_epoch" not in summary:
        raise KeyError(f"Missing best_epoch in {run_name}/metrics_summary.json")
    return summary


def load_confusion_matrix(config, experiment_directory, run_name):
    path = Path(experiment_directory) / run_name / "test_confusion_matrix.csv"
    require_result_file(path)
    class_names = list(config["dataset"]["classes"])
    frame = pd.read_csv(path, index_col=0)
    missing_rows = [class_name for class_name in class_names if class_name not in frame.index]
    missing_columns = [class_name for class_name in class_names if class_name not in frame.columns]
    if missing_rows or missing_columns:
        raise KeyError(f"Confusion-matrix labels are incomplete for {run_name}.")
    return frame.loc[class_names, class_names].to_numpy(dtype=np.int64)


def load_classwise_metrics(config, experiment_directory, run_name):
    path = Path(experiment_directory) / run_name / "test_class_metrics.csv"
    require_result_file(path)
    frame = pd.read_csv(path)
    require_columns(frame, ["class", "precision", "recall", "f1", "support"], f"{run_name}/test_class_metrics.csv")
    frame = frame.set_index("class")
    class_names = list(config["dataset"]["classes"])
    missing_classes = [class_name for class_name in class_names if class_name not in frame.index]
    if missing_classes:
        raise KeyError(f"Class-wise results are incomplete for {run_name}: {missing_classes}")
    return frame.loc[class_names]


def draw_confusion_matrix(axis, matrix, class_names, annotation_font_size):
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=np.float64), where=totals != 0)
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    positions = np.arange(len(class_names))
    labels = class_tick_labels(class_names)
    axis.set_xticks(positions)
    axis.set_yticks(positions)
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted canopy-size class")
    axis.set_ylabel("True canopy-size class")
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            percentage = 100.0 * normalized[row, column]
            color = "white" if normalized[row, column] >= 0.50 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=annotation_font_size,
            )
    return image


def draw_classwise_metrics(axis, metrics, class_names, style, bar_width, bar_spacing):
    positions = np.arange(len(class_names))
    definitions = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 score")]

    for index, (column, label) in enumerate(definitions):
        axis.bar(
            positions + (index - 1) * bar_spacing,
            metrics[column].to_numpy(dtype=np.float64),
            bar_width,
            color=style["metric_colors"][label],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(class_tick_labels(class_names))
    axis.set_xlabel("Canopy-size class")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=style["grid_alpha"], linewidth=0.6)
    axis.set_axisbelow(True)


def plot_efficientnet_training_histories(config, experiment_directory, output_directory, style):
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    checkpoint_marker_size = 55
    column_gap = 0.30
    row_gap = 0.58
    left_margin = 0.09
    right_margin = 0.98
    bottom_margin = 0.07
    top_margin = 0.94

    run_names = list(config["efficientnet"]["runs"])
    panel_labels = list("abcdefgh")
    fig, axes = plt.subplots(len(run_names), 2, figsize=style["training_histories_size"], squeeze=False)

    for row, run_name in enumerate(run_names):
        history = load_training_history(experiment_directory, run_name)
        summary = load_metrics_summary(experiment_directory, run_name)
        epochs = history["epoch"].to_numpy(dtype=np.int64)
        best_epoch = int(summary["best_epoch"])
        selected = history.loc[history["epoch"] == best_epoch]
        if selected.empty:
            raise ValueError(f"The saved best epoch is absent from the training history for {run_name}.")
        selected_row = selected.iloc[0]
        maximum_epoch = max(int(epochs.max()), int(epochs.min()) + 1)
        loss_axis = axes[row, 0]
        accuracy_axis = axes[row, 1]

        loss_axis.plot(epochs, history["train_loss"], color=style["training_color"], linestyle="-")
        loss_axis.plot(epochs, history["validation_loss"], color=style["validation_color"], linestyle="--")
        accuracy_axis.plot(epochs, history["train_accuracy"], color=style["training_color"], linestyle="-")
        accuracy_axis.plot(epochs, history["validation_accuracy"], color=style["validation_color"], linestyle="--")

        for axis in [loss_axis, accuracy_axis]:
            axis.axvline(best_epoch, color=style["checkpoint_color"], linestyle=":", linewidth=1.0, zorder=1)
            axis.set_xlabel("Epoch")
            axis.set_xlim(float(epochs.min()), float(maximum_epoch))
            axis.grid(axis="y", alpha=style["grid_alpha"], linewidth=0.6)
            axis.set_axisbelow(True)

        loss_axis.scatter(
            best_epoch,
            float(selected_row["validation_loss"]),
            marker="*",
            s=checkpoint_marker_size,
            color=style["validation_color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        loss_axis.set_ylabel("Cross-entropy loss")
        loss_axis.set_ylim(bottom=0.0)
        accuracy_axis.set_ylabel("Accuracy")
        accuracy_axis.set_ylim(0.0, 1.0)

        label = run_label(style, run_name)
        panel_title(loss_axis, panel_labels[2 * row], f"{label}: loss", panel_title_font_size, panel_title_x, panel_title_y)
        panel_title(accuracy_axis, panel_labels[2 * row + 1], f"{label}: accuracy", panel_title_font_size, panel_title_x, panel_title_y)

    handles = [
        Line2D([0], [0], color=style["training_color"], linestyle="-", label="Training"),
        Line2D([0], [0], color=style["validation_color"], linestyle="--", label="Validation"),
        Line2D(
            [0],
            [0],
            color=style["checkpoint_color"],
            linestyle=":",
            marker="*",
            markerfacecolor=style["validation_color"],
            markeredgecolor="white",
            label="Selected checkpoint",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.013))
    # fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=column_gap,
        hspace=row_gap,
    )
    save_figure(fig, output_directory / "efficientnet_training_histories", style)


def plot_efficientnet_confusion_matrices(config, experiment_directory, output_directory, style):
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    annotation_font_size = 7
    colorbar_label_font_size = 8
    colorbar_tick_font_size = 7
    colorbar_fraction = 0.046
    colorbar_pad = 0.04
    column_gap = 0.32
    row_gap = 0.62
    left_margin = 0.08
    right_margin = 0.97
    bottom_margin = 0.09
    top_margin = 0.95

    run_names = list(config["efficientnet"]["runs"])
    class_names = list(config["dataset"]["classes"])
    fig, axes = plt.subplots(2, 2, figsize=style["confusion_matrices_size"], squeeze=False)
    axes = axes.ravel()

    if len(run_names) != len(axes):
        raise ValueError("The confusion-matrix layout requires exactly four EfficientNet runs.")

    for axis, panel_label, run_name in zip(axes, ["a", "b", "c", "d"], run_names):
        matrix = load_confusion_matrix(config, experiment_directory, run_name)
        image = draw_confusion_matrix(axis, matrix, class_names, annotation_font_size)
        panel_title(axis, panel_label, run_label(style, run_name), panel_title_font_size, panel_title_x, panel_title_y)
        colorbar = fig.colorbar(image, ax=axis, fraction=colorbar_fraction, pad=colorbar_pad)
        colorbar.set_label("Row-wise proportion", fontsize=colorbar_label_font_size)
        colorbar.ax.tick_params(labelsize=colorbar_tick_font_size)

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=column_gap,
        hspace=row_gap,
    )
    save_figure(fig, output_directory / "efficientnet_confusion_matrices", style)


def plot_efficientnet_classwise_metrics(config, experiment_directory, output_directory, style):
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    bar_width = 0.20
    bar_spacing = 0.24
    column_gap = 0.20
    row_gap = 0.58
    left_margin = 0.09
    right_margin = 0.98
    bottom_margin = 0.10
    top_margin = 0.91

    bar_label_font_size = 6.5
    bar_label_rotation = 60
    bar_label_padding = 2
    percentage_axis_top = 130

    run_names = list(config["efficientnet"]["runs"])
    class_names = list(config["dataset"]["classes"])
    fig, axes = plt.subplots(2, 2, figsize=style["classwise_metrics_size"], squeeze=False)
    axes = axes.ravel()

    if len(run_names) != len(axes):
        raise ValueError("The class-wise metric layout requires exactly four EfficientNet runs.")

    for axis, panel_label, run_name in zip(axes, ["a", "b", "c", "d"], run_names):
        metrics = load_classwise_metrics(config, experiment_directory, run_name)
        draw_classwise_metrics(axis, metrics, class_names, style, bar_width, bar_spacing)

        for container in axis.containers:
            percentage_labels = [
                f"{100.0 * bar.get_height():.2f}%"
                for bar in container
            ]
            axis.bar_label(
                container,
                labels=percentage_labels,
                padding=bar_label_padding,
                fontsize=bar_label_font_size,
                rotation=bar_label_rotation,
            )

        class_labels = [
            f"{label}\n(n={int(metrics.loc[class_name, 'support'])})"
            for label, class_name in zip(class_tick_labels(class_names), class_names)
        ]
        axis.set_xticklabels(class_labels)

        tick_positions = np.linspace(0.0, 1.0, 6)
        axis.set_yticks(tick_positions)
        axis.set_yticklabels([f"{100.0 * value:.0f}" for value in tick_positions])
        axis.set_ylabel("Score (%)")
        axis.set_ylim(0.0, percentage_axis_top / 100.0)

        panel_title(
            axis,
            panel_label,
            run_label(style, run_name),
            panel_title_font_size,
            panel_title_x,
            panel_title_y,
        )

    handles = [
        Patch(facecolor=style["metric_colors"][label], edgecolor="none", label=label)
        for label in ["Precision", "Recall", "F1 score"]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=column_gap,
        hspace=row_gap,
    )
    save_figure(fig, output_directory / "efficientnet_classwise_metrics", style)

def main():
    current_config = load_config(CONFIG_PATH, False)
    style = figure_settings()
    experiment_directory = efficientnet_experiment_directory(current_config, PROJECT_ROOT, style)
    config = load_run_config(run_config_path(experiment_directory), current_config, PROJECT_ROOT, False)
    configure_article_style(style)
    output_directory = figure_output_directory(experiment_directory, style)

    print(f"Experiment: {project_relative_text(PROJECT_ROOT, experiment_directory)}")

    plot_efficientnet_training_histories(config, experiment_directory, output_directory, style)
    plot_efficientnet_confusion_matrices(config, experiment_directory, output_directory, style)

    plot_efficientnet_classwise_metrics(config, experiment_directory, output_directory, style)


if __name__ == "__main__":
    print("Generate figures for the EfficientNet-B0 experiments.")
    main()

