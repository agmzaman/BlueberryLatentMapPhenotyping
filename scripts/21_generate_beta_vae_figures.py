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
from PIL import Image, ImageOps

from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config, load_run_config
from blueberry_latent_map_phenotyping.paths import current_run_directory, dataset_image_root
from blueberry_latent_map_phenotyping.paths import project_relative_text, run_config_path
from blueberry_latent_map_phenotyping.paths import current_session_beta_vae_experiment


def figure_settings():
    return {
        "output_directory": "figures/beta_vae",
        "font_family": "Calibri",
        "font_size": 9,
        "axes_label_size": 9,
        "axes_title_size": 9,
        "tick_label_size": 8,
        "legend_size": 8,
        "png_dpi": 600,
        "line_width": 1.4,
        "reference_line_width": 2.2,
        "grid_alpha": 0.20,
        "loss_scale": 1000.0,
        "loss_axis_label": "Loss per image (×10³)",
        "training_color": "#0072B2",
        "validation_color": "#D55E00",
        "warmup_color": "#D9D9D9",
        "configuration_colors": {
            "C1": "#4477AA",
            "C2": "#66CCEE",
            "C3": "#228833",
            "C4": "#EE7733",
            "C5": "#AA3377",
        },
        "metric_colors": {
            "Precision": "#0072B2",
            "Recall": "#009E73",
            "F1 score": "#E69F00",
        },
        "beta_vae_training_size": [7.2, 2.6],
        "reconstruction_size": [7.2, 4.0],
        "classifier_training_size": [7.2, 3.1],
        "reference_classifier_size": [7.2, 3.2],
        "all_confusion_matrices_size": [7.2, 5.0],
        "all_classwise_metrics_size": [7.2, 5.0],
        "reconstruction_pairs": [
            {"class_name": "Very Small", "original": "2025-08-07/IMG_9113.JPG", "reconstruction": "vae/reconstructions/IMG_9113_recon.JPG"},
            {"class_name": "Small", "original": "2025-09-29/IMG_1850.JPG", "reconstruction": "vae/reconstructions/IMG_1850_recon.JPG"},
            {"class_name": "Medium", "original": "2025-08-22/IMG_0101.JPG", "reconstruction": "vae/reconstructions/IMG_0101_recon.JPG"},
            {"class_name": "Large", "original": "2025-08-22/IMG_9870.JPG", "reconstruction": "vae/reconstructions/IMG_9870_recon.JPG"},
        ],
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


def figure_output_directory(run_directory, style):
    relative_path = Path(style["output_directory"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("The figure output directory must be run-relative.")
    output_directory = Path(run_directory) / relative_path
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


def require_result_file(path, run_directory):
    if not path.exists():
        relative_path = Path(path).relative_to(run_directory).as_posix()
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


def load_classifier_histories(config, run_directory):
    histories = {}
    columns = ["epoch", "train_loss", "validation_loss", "train_accuracy", "validation_accuracy"]
    for run_name in config["classifier"]["runs"]:
        path = Path(run_directory) / "classifiers" / run_name / "training_history.csv"
        require_result_file(path, run_directory)
        history = pd.read_csv(path)
        require_columns(history, columns, f"classifiers/{run_name}/training_history.csv")
        histories[run_name] = history
    return histories


def load_classifier_evaluation(config, run_directory, run_name):
    classifier_directory = Path(run_directory) / "classifiers" / run_name
    matrix_path = classifier_directory / "test_confusion_matrix.csv"
    report_path = classifier_directory / "test_classification_report.csv"
    require_result_file(matrix_path, run_directory)
    require_result_file(report_path, run_directory)

    class_names = config["dataset"]["classes"]
    rows = [f"true_{class_name}" for class_name in class_names]
    columns = [f"predicted_{class_name}" for class_name in class_names]
    matrix_frame = pd.read_csv(matrix_path, index_col=0)
    missing_rows = [name for name in rows if name not in matrix_frame.index]
    missing_columns = [name for name in columns if name not in matrix_frame.columns]
    if missing_rows or missing_columns:
        raise KeyError(f"Confusion-matrix labels are incomplete for {run_name}.")
    matrix = matrix_frame.loc[rows, columns].to_numpy(dtype=np.int64)

    report = pd.read_csv(report_path, index_col=0)
    missing_classes = [name for name in class_names if name not in report.index]
    if missing_classes:
        raise KeyError(f"Classification-report classes are incomplete for {run_name}: {missing_classes}")
    require_columns(report, ["precision", "recall", "f1-score"], f"{run_name} classification report")
    return matrix, report.loc[class_names]


def draw_confusion_matrix(axis, matrix, class_names):
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=totals != 0,
    )
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
                f"{matrix[row, column]}\n({percentage:.2f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    return image


def draw_classwise_metrics(axis, report, class_names, style, bar_width, bar_spacing):
    positions = np.arange(len(class_names))
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1 score")]

    for index, (column, label) in enumerate(metrics):
        axis.bar(
            positions + (index - 1) * bar_spacing,
            report[column].to_numpy(dtype=np.float64),
            bar_width,
            label=label,
            color=style["metric_colors"][label],
            edgecolor="white",
            linewidth=0.5,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(class_tick_labels(class_names))
    axis.set_xlabel("Canopy-size class")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=style["grid_alpha"], linewidth=0.6)
    axis.set_axisbelow(True)


def plot_beta_vae_training_curves(config, run_directory, output_directory, style):
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    warmup_fill_alpha = 0.60
    path = Path(run_directory) / "vae" / "metrics" / "training_history.csv"
    require_result_file(path, run_directory)
    history = pd.read_csv(path)
    columns = [
        "epoch",
        "train_total_loss",
        "validation_total_loss",
        "train_weighted_kl_loss",
        "validation_weighted_kl_loss",
        "train_reconstruction_loss",
        "validation_reconstruction_loss",
    ]
    require_columns(history, columns, "vae/metrics/training_history.csv")

    panels = [
        ("total_loss", "Total loss"),
        ("weighted_kl_loss", "Weighted KL divergence loss"),
        ("reconstruction_loss", "L1 reconstruction loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=style["beta_vae_training_size"])
    epochs = history["epoch"].to_numpy(dtype=np.int64)

    for axis, panel_label, (suffix, title) in zip(axes, ["a", "b", "c"], panels):
        training = history[f"train_{suffix}"] / style["loss_scale"]
        validation = history[f"validation_{suffix}"] / style["loss_scale"]

        axis.plot(epochs, training, color=style["training_color"], linewidth= 0.9, linestyle="-")
        axis.plot(epochs, validation, color=style["validation_color"], linewidth= 0.9, linestyle="--")

        panel_title(
            axis,
            panel_label,
            title,
            panel_title_font_size,
            panel_title_x,
            panel_title_y,
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(style["loss_axis_label"])
        axis.set_xlim(float(epochs.min()), float(epochs.max()))
        axis.grid(axis="y", alpha=style["grid_alpha"], linewidth=0.6)
        axis.set_axisbelow(True)

    warmup_epochs = min(int(config["vae"]["warmup_epochs"]), int(epochs.max()))
    warmup_mask = epochs <= warmup_epochs
    warmup_training = history["train_weighted_kl_loss"].to_numpy(dtype=np.float64) / style["loss_scale"]
    warmup_validation = history["validation_weighted_kl_loss"].to_numpy(dtype=np.float64) / style["loss_scale"]
    warmup_boundary = np.maximum(warmup_training[warmup_mask], warmup_validation[warmup_mask])
    axes[1].fill_between(
        epochs[warmup_mask],
        0.0,
        warmup_boundary,
        color=style["warmup_color"],
        alpha=warmup_fill_alpha,
        linewidth=0.0,
        zorder=0,
    )

    handles = [
        Line2D([0], [0], color=style["training_color"], label="Training"),
        Line2D([0], [0], color=style["validation_color"], label="Validation"),
        Patch(
            facecolor=style["warmup_color"],
            edgecolor="none",
            alpha=warmup_fill_alpha,
            label="KL warm-up",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.align_ylabels(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    save_figure(fig, output_directory / "beta_vae_training_curves", style)


def relative_image_path(value, field_name):
    text = str(value).strip()
    if not text:
        raise ValueError(f"Set the {field_name} path in figure_settings().")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be a relative path.")
    return path


def read_rgb_image(path, displayed_path):
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {displayed_path}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def plot_reconstruction_examples(config, run_directory, output_directory, style):
    class_title_font_size = 9
    row_label_font_size = 9
    pairs = style["reconstruction_pairs"]
    class_names = config["dataset"]["classes"]
    if [pair["class_name"] for pair in pairs] != class_names:
        raise ValueError("The reconstruction-pair classes must follow dataset.classes.")

    fig, axes = plt.subplots(2, len(pairs), figsize=style["reconstruction_size"], squeeze=False)
    original_directory = dataset_image_root(config)

    for column, pair in enumerate(pairs):
        original_relative = relative_image_path(pair["original"], f"{pair['class_name']} original")
        reconstruction_relative = relative_image_path(
            pair["reconstruction"],
            f"{pair['class_name']} reconstruction",
        )
        original = read_rgb_image(original_directory / original_relative, original_relative.as_posix())

        reconstruction = read_rgb_image(
            Path(run_directory) / reconstruction_relative,
            reconstruction_relative.as_posix(),
        )

        reconstruction_height, reconstruction_width = reconstruction.shape[:2]

        original = np.asarray(
            Image.fromarray(original).resize(
                (reconstruction_width, reconstruction_height),
                Image.Resampling.BILINEAR,
            )
        )

        axes[0, column].imshow(original)
        axes[1, column].imshow(reconstruction)
        axes[0, column].set_title(pair["class_name"], fontsize=class_title_font_size, pad=5)
        axes[0, column].set_axis_off()
        axes[1, column].set_axis_off()

        axes[0, column].set_anchor("S")
        axes[1, column].set_anchor("N")

    axes[0, 0].text(
        -0.08,
        0.5,
        "Original",
        transform=axes[0, 0].transAxes,
        rotation=90,
        ha="right",
        va="center",
        fontsize=row_label_font_size,
    )
    axes[1, 0].text(
        -0.08,
        0.5,
        "Reconstructed",
        transform=axes[1, 0].transAxes,
        rotation=90,
        ha="right",
        va="center",
        fontsize=row_label_font_size,
    )
    row_gap = 0.02
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.0)
    fig.subplots_adjust(hspace=row_gap)

    save_figure(fig, output_directory / "original_and_reconstructed_examples", style)


def reference_best_epoch(config, histories, run_directory):
    run_name = config["evaluation"]["reference_classifier"]
    path = Path(run_directory) / "classifiers" / run_name / "metrics_summary.json"
    require_result_file(path, run_directory)
    best_epoch = int(json.loads(path.read_text(encoding="utf-8"))["best_epoch"])
    selected = histories[run_name].loc[histories[run_name]["epoch"] == best_epoch]
    if selected.empty:
        raise ValueError(f"The saved best epoch is absent from {run_name} history.")
    return run_name, best_epoch, float(selected.iloc[0]["validation_loss"])


def plot_classifier_training_dynamics(config, run_directory, output_directory, style):
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    histories = load_classifier_histories(config, run_directory)
    run_names = list(config["classifier"]["runs"])
    missing_colors = [name for name in run_names if name not in style["configuration_colors"]]
    if missing_colors:
        raise KeyError(f"Missing classifier colors: {missing_colors}")

    reference_name, best_epoch, best_loss = reference_best_epoch(config, histories, run_directory)
    # fig, axes = plt.subplots(1, 2, figsize=style["classifier_training_size"])
    fig, axes = plt.subplots(1, 2, figsize=[7.2, 4.0])


    for run_name in run_names:
        history = histories[run_name]
        color = style["configuration_colors"][run_name]
        # width = style["reference_line_width"] if run_name == reference_name else style["line_width"]
        width = 0.9
        zorder = 3 if run_name == reference_name else 2
        axes[0].plot(history["epoch"], history["train_loss"], color=color, linewidth=width, zorder=zorder)
        axes[0].plot(
            history["epoch"],
            history["validation_loss"],
            color=color,
            linestyle="--",
            linewidth=width,
            zorder=zorder,
        )
        axes[1].plot(history["epoch"], history["train_accuracy"], color=color, linewidth=width, zorder=zorder)
        axes[1].plot(
            history["epoch"],
            history["validation_accuracy"],
            color=color,
            linestyle="--",
            linewidth=width,
            zorder=zorder,
        )

    for axis in axes:
        axis.axvline(best_epoch, color="#666666", linestyle=":", linewidth=0.9, zorder=1)
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", alpha=style["grid_alpha"], linewidth=0.6)
        axis.set_axisbelow(True)

    axes[0].scatter(
        best_epoch,
        best_loss,
        marker="*",
        s=75,
        color=style["configuration_colors"][reference_name],
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    axes[0].annotate(
        f"{reference_name} best epoch: {best_epoch}",
        xy=(best_epoch, best_loss),
        xytext=(3, 7.5),
        textcoords="offset points",
        color=style["configuration_colors"][reference_name],
        fontsize=style["tick_label_size"],
    )
    panel_title(axes[0], "a", "Cross-entropy loss", panel_title_font_size, panel_title_x, panel_title_y)
    panel_title(axes[1], "b", "Accuracy", panel_title_font_size, panel_title_x, panel_title_y)
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)

    configuration_handles = []
    for run_name in run_names:
        label = f"{run_name} (reference)" if run_name == reference_name else run_name
        width = style["reference_line_width"] if run_name == reference_name else style["line_width"]
        configuration_handles.append(
            Line2D([0], [0], color=style["configuration_colors"][run_name], linewidth=width, label=label)
        )
    split_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", label="Training"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Validation"),
    ]
    fig.legend(
        handles=configuration_handles,
        loc="upper center",
        ncol=len(configuration_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.legend(handles=split_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        save_figure(fig, output_directory / "classifier_training_dynamics", style)

    # save_figure(fig, output_directory / "classifier_training_dynamics", style)


def plot_reference_classifier_performance(config, run_directory, output_directory, style):
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    bar_width = 0.21
    bar_spacing = 0.24

    bar_label_font_size = 6.5
    bar_label_rotation = 60
    bar_label_padding = 2
    percentage_axis_top = 130

    run_name = config["evaluation"]["reference_classifier"]
    class_names = config["dataset"]["classes"]
    matrix, report = load_classifier_evaluation(config, run_directory, run_name)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=style["reference_classifier_size"],
        gridspec_kw={"width_ratios": [1.0, 1.18]},
    )

    image = draw_confusion_matrix(axes[0], matrix, class_names)
    panel_title(
        axes[0],
        "a",
        f"{run_name} confusion matrix",
        panel_title_font_size,
        panel_title_x,
        panel_title_y,
    )
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Row-wise proportion")

    draw_classwise_metrics(axes[1], report, class_names, style, bar_width, bar_spacing)

    for container in axes[1].containers:
        percentage_labels = [
            f"{100.0 * bar.get_height():.2f}%"
            for bar in container
        ]
        axes[1].bar_label(
            container,
            labels=percentage_labels,
            padding=bar_label_padding,
            fontsize=bar_label_font_size,
            rotation=bar_label_rotation,
        )

    class_labels = [
        f"{label}\n(n={int(report.loc[class_name, 'support'])})"
        for label, class_name in zip(class_tick_labels(class_names), class_names)
    ]
    axes[1].set_xticklabels(class_labels)

    tick_positions = np.linspace(0.0, 1.0, 6)
    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels([f"{100.0 * value:.0f}" for value in tick_positions])
    axes[1].set_ylabel("Score (%)")
    axes[1].set_ylim(0.0, percentage_axis_top / 100.0)

    panel_title(
        axes[1],
        "b",
        f"{run_name} per-class performance",
        panel_title_font_size,
        panel_title_x,
        panel_title_y,
    )
    axes[1].legend(loc="upper center", ncol=3, frameon=False)

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, output_directory / "reference_classifier_performance", style)


def plot_all_classifier_confusion_matrices(config, run_directory, output_directory, style):
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    colorbar_fraction = 0.046
    colorbar_pad = 0.04
    column_gap = 0.05
    row_gap = 0.60
    run_names = ["C1", "C2", "C3", "C4"]
    class_names = config["dataset"]["classes"]
    fig, axes = plt.subplots(2, 2, figsize=style["all_confusion_matrices_size"])
    axes = axes.ravel()

    for axis, panel_label, run_name in zip(axes, ["a", "b", "c", "d"], run_names):
        matrix, _ = load_classifier_evaluation(config, run_directory, run_name)
        image = draw_confusion_matrix(axis, matrix, class_names)
        panel_title(axis, panel_label, run_name, panel_title_font_size, panel_title_x, panel_title_y)
        colorbar = fig.colorbar(image, ax=axis, fraction=colorbar_fraction, pad=colorbar_pad)
        colorbar.set_label("Row-wise proportion")

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.94, wspace=column_gap, hspace=row_gap)
    save_figure(fig, output_directory / "classifier_c1_c4_confusion_matrices", style)


def plot_all_classifier_classwise_metrics(config, run_directory, output_directory, style):
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    bar_width = 0.18
    bar_spacing = 0.24

    bar_label_font_size = 6.5
    bar_label_rotation = 60
    bar_label_padding = 2
    percentage_axis_top = 130

    run_names = ["C1", "C2", "C3", "C4"]
    class_names = config["dataset"]["classes"]
    fig, axes = plt.subplots(2, 2, figsize=style["all_classwise_metrics_size"])
    axes = axes.ravel()

    for axis, panel_label, run_name in zip(axes, ["a", "b", "c", "d"], run_names):
        _, report = load_classifier_evaluation(config, run_directory, run_name)
        draw_classwise_metrics(axis, report, class_names, style, bar_width, bar_spacing)

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
            f"{label}\n(n={int(report.loc[class_name, 'support'])})"
            for label, class_name in zip(class_tick_labels(class_names), class_names)
        ]
        axis.set_xticklabels(class_labels)

        tick_positions = np.linspace(0.0, 1.0, 6)
        axis.set_yticks(tick_positions)
        axis.set_yticklabels([f"{100.0 * value:.0f}" for value in tick_positions])
        axis.set_ylabel("Score (%)")
        axis.set_ylim(0.0, percentage_axis_top / 100.0)

        panel_title(axis, panel_label, run_name, panel_title_font_size, panel_title_x, panel_title_y)

    handles = [
        Patch(facecolor=style["metric_colors"][label], edgecolor="none", label=label)
        for label in ["Precision", "Recall", "F1 score"]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    plt.subplots_adjust(hspace=0.7)
    save_figure(fig, output_directory / "classifier_c1_c4_classwise_metrics", style)


def plot_session_c5_training_dynamics(config, run_directory, output_directory, style):
    figure_size = (7.2, 7.0)
    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    axis_label_font_size = 9
    tick_font_size = 8
    legend_font_size = 8
    legend_y = 0.995
    line_width = 0.9
    column_gap = 0.32
    row_gap = 0.65
    left_margin = 0.10
    right_margin = 0.98
    bottom_margin = 0.08
    top_margin = 0.92

    fold_names = list(config["session_held_out"]["folds"])
    if len(fold_names) != 3:
        raise ValueError("This layout requires exactly three session-held-out folds.")

    columns = ["epoch", "train_loss", "validation_loss", "train_accuracy", "validation_accuracy"]
    histories = {}

    for fold_name in fold_names:
        path = Path(run_directory) / fold_name / "classifiers" / "C5" / "training_history.csv"
        require_result_file(path, run_directory)
        history = pd.read_csv(path)
        require_columns(history, columns, f"{fold_name}/classifiers/C5/training_history.csv")

        if history.empty:
            raise ValueError(f"Training history is empty for {fold_name}.")

        histories[fold_name] = history.sort_values("epoch")

    fig, axes = plt.subplots(3, 2, figsize=figure_size, squeeze=False)
    panel_labels = list("abcdef")
    metrics = [("loss", "Cross-entropy loss"), ("accuracy", "Accuracy")]

    for row, fold_name in enumerate(fold_names):
        history = histories[fold_name]
        epochs = history["epoch"].to_numpy(dtype=np.int64)
        maximum_epoch = max(int(epochs.max()), int(epochs.min()) + 1)

        for column, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            axis.plot(
                epochs, history[f"train_{metric}"],
                color=style["training_color"], linestyle="-", linewidth=line_width,
            )
            axis.plot(
                epochs, history[f"validation_{metric}"],
                color=style["validation_color"], linestyle="--", linewidth=line_width,
            )

            panel_title(
                axis, panel_labels[2 * row + column], fold_name,
                panel_title_font_size, panel_title_x, panel_title_y,
            )
            axis.set_xlabel("Epoch", fontsize=axis_label_font_size)
            axis.set_ylabel(ylabel, fontsize=axis_label_font_size)
            axis.tick_params(labelsize=tick_font_size)
            axis.set_xlim(float(epochs.min()), float(maximum_epoch))
            axis.set_ylim(bottom=0.0)
            axis.grid(axis="y", alpha=style["grid_alpha"], linewidth=0.6)
            axis.set_axisbelow(True)

            if metric == "accuracy":
                axis.set_ylim(0.0, 1.0)

    handles = [
        Line2D(
            [0], [0], color=style["training_color"],
            linestyle="-", linewidth=line_width, label="Training",
        ),
        Line2D(
            [0], [0], color=style["validation_color"],
            linestyle="--", linewidth=line_width, label="Validation",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=2, frameon=False,
        fontsize=legend_font_size, bbox_to_anchor=(0.5, legend_y),
    )
    fig.subplots_adjust(
        left=left_margin, right=right_margin,
        bottom=bottom_margin, top=top_margin,
        wspace=column_gap, hspace=row_gap,
    )
    save_figure(fig, output_directory / "session_held_out_c5_training_dynamics", style)


def plot_session_c5_confusion_matrices(config, run_directory, output_directory, style):
    figure_size = (9.0, 10.0)

    panel_title_font_size = 9
    panel_title_x = 0.0
    panel_title_y = 1.04
    axis_label_font_size = 9
    tick_font_size = 8
    annotation_font_size = 7

    colorbar_label_font_size = 8
    colorbar_tick_font_size = 7
    colorbar_fraction = 0.046
    colorbar_pad = 0.04

    bar_width = 0.18
    bar_spacing = 0.24
    bar_label_font_size = 6.5
    bar_label_rotation = 60
    bar_label_padding = 2
    percentage_axis_top = 130

    legend_font_size = 8
    legend_x = 0.5
    legend_y = 0.995

    column_width_ratios = [1.0, 1.18]
    column_gap = 0.30
    row_gap = 0.55
    left_margin = 0.09
    right_margin = 0.97
    bottom_margin = 0.07
    top_margin = 0.92

    fold_names = list(config["session_held_out"]["folds"])
    class_names = config["dataset"]["classes"]

    if len(fold_names) != 3:
        raise ValueError("This layout requires exactly three session-held-out folds.")

    matrices = {}
    reports = {}

    for fold_name in fold_names:
        fold_directory = Path(run_directory) / fold_name
        matrix, report = load_classifier_evaluation(config, fold_directory, "C5")
        matrices[fold_name] = matrix
        reports[fold_name] = report

    fig, axes = plt.subplots(
        3, 2,
        figsize=figure_size,
        squeeze=False,
        gridspec_kw={"width_ratios": column_width_ratios},
    )

    panel_labels = [("a", "b"), ("c", "d"), ("e", "f")]

    for row_index, fold_name in enumerate(fold_names):
        matrix_axis = axes[row_index, 0]
        metrics_axis = axes[row_index, 1]
        matrix_label, metrics_label = panel_labels[row_index]
        report = reports[fold_name]

        image = draw_confusion_matrix(matrix_axis, matrices[fold_name], class_names)

        for annotation in matrix_axis.texts:
            annotation.set_fontsize(annotation_font_size)

        panel_title(
            matrix_axis, matrix_label, f"{fold_name}: Confusion matrix",
            panel_title_font_size, panel_title_x, panel_title_y,
        )
        matrix_axis.set_xlabel("Predicted canopy-size class", fontsize=axis_label_font_size)
        matrix_axis.set_ylabel("True canopy-size class", fontsize=axis_label_font_size)
        matrix_axis.tick_params(labelsize=tick_font_size)

        colorbar = fig.colorbar(
            image, ax=matrix_axis,
            fraction=colorbar_fraction, pad=colorbar_pad,
        )
        colorbar.set_label("Row-wise proportion", fontsize=colorbar_label_font_size)
        colorbar.ax.tick_params(labelsize=colorbar_tick_font_size)

        draw_classwise_metrics(
            metrics_axis, report, class_names, style, bar_width, bar_spacing,
        )

        for container in metrics_axis.containers:
            percentage_labels = [
                f"{100.0 * bar.get_height():.2f}%"
                for bar in container
            ]
            metrics_axis.bar_label(
                container,
                labels=percentage_labels,
                padding=bar_label_padding,
                fontsize=bar_label_font_size,
                rotation=bar_label_rotation,
            )

        class_labels = [
            f"{label}\n(n={int(report.loc[class_name, 'support'])})"
            for label, class_name in zip(class_tick_labels(class_names), class_names)
        ]
        metrics_axis.set_xticks(np.arange(len(class_names)))
        metrics_axis.set_xticklabels(class_labels)

        tick_positions = np.linspace(0.0, 1.0, 6)
        metrics_axis.set_yticks(tick_positions)
        metrics_axis.set_yticklabels([
            f"{100.0 * value:.0f}"
            for value in tick_positions
        ])
        metrics_axis.set_ylim(0.0, percentage_axis_top / 100.0)
        metrics_axis.set_xlabel("Canopy-size class", fontsize=axis_label_font_size)
        metrics_axis.set_ylabel("Score (%)", fontsize=axis_label_font_size)
        metrics_axis.tick_params(labelsize=tick_font_size)

        panel_title(
            metrics_axis, metrics_label, f"{fold_name}: Class-wise performance",
            panel_title_font_size, panel_title_x, panel_title_y,
        )

    handles = [
        Patch(facecolor=style["metric_colors"][label], edgecolor="none", label=label)
        for label in ["Precision", "Recall", "F1 score"]
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=legend_font_size,
        bbox_to_anchor=(legend_x, legend_y),
    )

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=column_gap,
        hspace=row_gap,
    )
    save_figure(fig, output_directory / "session_held_out_c5_confusion_matrices", style)

def main():
    current_config = load_config(CONFIG_PATH, False)
    run_directory = current_run_directory(current_config, PROJECT_ROOT)
    config = load_run_config(run_config_path(run_directory), current_config, PROJECT_ROOT, False)
    style = figure_settings()
    configure_article_style(style)
    output_directory = figure_output_directory(run_directory, style)
    print(f"Experiment: {project_relative_text(PROJECT_ROOT, run_directory)}")

    session_directory = current_session_beta_vae_experiment(current_config, PROJECT_ROOT)
    session_config = load_run_config(run_config_path(session_directory), current_config, PROJECT_ROOT, False)
    session_output_directory = figure_output_directory(session_directory, style)
    print(f"Session experiment: {project_relative_text(PROJECT_ROOT, session_directory)}")


    plot_beta_vae_training_curves(config, run_directory, output_directory, style)
    plot_classifier_training_dynamics(config, run_directory, output_directory, style)
    plot_reference_classifier_performance(config, run_directory, output_directory, style)

    plot_all_classifier_confusion_matrices(config, run_directory, output_directory, style)
    plot_all_classifier_classwise_metrics(config, run_directory, output_directory, style)
    plot_reconstruction_examples(config, run_directory, output_directory, style)

    plot_session_c5_training_dynamics(session_config, session_directory, session_output_directory, style)
    plot_session_c5_confusion_matrices(session_config, session_directory, session_output_directory, style)



if __name__ == "__main__":
    print("Generate figures for the stratified-random beta-VAE experiment.")
    main()

