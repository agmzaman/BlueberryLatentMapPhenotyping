import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.image_descriptor_settings import (
    config_with_image_descriptor_settings,
)
from blueberry_latent_map_phenotyping.paths import (
    current_image_descriptor_experiment,
    current_run_directory,
    current_session_beta_vae_experiment,
    project_path,
    project_relative_text,
)


SETTINGS_PATH = PROJECT_ROOT / "configs" / "image_descriptor.yaml"

def figure_settings():
    return {
        "experiment_directories": {
            "stratified_beta_vae": "",
            "session_beta_vae": "",
            "efficientnet": "",
            "image_descriptors": "",
        },
        "output_directory": "figures/cross_experiment",
        "descriptor_algorithm": "rbf_svm",
        "descriptor_configuration": "morphology_color_texture",
        "font_family": "Calibri",
        "font_size": 9,
        "axes_label_size": 9,
        "axes_title_size": 9,
        "tick_label_size": 8,
        "legend_size": 7,
        "png_dpi": 600,
        "grid_alpha": 0.20,
        "method_labels": {
            "spatial_beta_vae_c5": r"$\beta$-VAE" + "+ C5",
            "efficientnet_b0": "EfficientNet-B0",
            "image_descriptors_rbf_svm": "Descriptors\n+ RBF-SVM",
        },
        "protocol_labels": {
            "stratified_random": "Stratified random",
            "S1": "S1",
            "S2": "S2",
            "S3": "S3",
            "session_held_out_mean": "Held-out mean ± SD",
        },
        "protocol_colors": {
            "stratified_random": "#333333",
            "S1": "#0072B2",
            "S2": "#009E73",
            "S3": "#E69F00",
            "session_held_out_mean": "#AA3377",
        },
        "protocol_markers": {
            "stratified_random": "D",
            "S1": "o",
            "S2": "^",
            "S3": "s",
            "session_held_out_mean": "P",
        },
        "comparison_size": [7.2, 3.3],
    }


def configure_article_style(style):
    font = font_manager.FontProperties(family=style["font_family"])
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError(
            f"Required figure font is not installed: {style['font_family']}"
        ) from error

    plt.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.size": style["font_size"],
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": style["axes_label_size"],
            "axes.titlesize": style["axes_title_size"],
            "xtick.labelsize": style["tick_label_size"],
            "ytick.labelsize": style["tick_label_size"],
            "legend.fontsize": style["legend_size"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def selected_directory(project_root, experiment_root, selected, label):
    selected_path = Path(str(selected).strip())
    if selected_path.is_absolute() or ".." in selected_path.parts:
        raise ValueError(f"{label} must be a project-relative directory.")
    directory = project_path(project_root, selected_path).resolve()
    return validate_experiment_directory(directory, experiment_root, label)


def validate_experiment_directory(directory, experiment_root, label):
    directory = Path(directory).resolve()
    experiment_root = Path(experiment_root).resolve()
    try:
        directory.relative_to(experiment_root)
    except ValueError as error:
        raise ValueError(f"{label} is outside its configured experiment root.") from error
    if not directory.is_dir():
        relative_path = project_relative_text(PROJECT_ROOT, directory)
        raise FileNotFoundError(f"{label} does not exist: {relative_path}")
    return directory


def latest_experiment_directory(experiment_root, label):
    candidates = sorted(
        path.resolve()
        for path in Path(experiment_root).glob("run_*")
        if path.is_dir()
    )
    if not candidates:
        relative_root = project_relative_text(PROJECT_ROOT, experiment_root)
        raise FileNotFoundError(f"No {label} directory was found under {relative_root}.")
    return candidates[-1]


def cross_experiment_directories(config, project_root, style):
    selections = style["experiment_directories"]
    required = [
        "stratified_beta_vae",
        "session_beta_vae",
        "efficientnet",
        "image_descriptors",
    ]
    missing = [name for name in required if name not in selections]
    if missing:
        raise KeyError(f"Missing experiment-directory settings: {missing}")

    roots = {
        "stratified_beta_vae": project_path(
            project_root,
            config["paths"]["experiments_dir"],
        ).resolve(),
        "session_beta_vae": project_path(
            project_root,
            config["paths"]["session_held_out_beta_vae_experiments_dir"],
        ).resolve(),
        "efficientnet": project_path(
            project_root,
            config["paths"]["efficientnet_experiments_dir"],
        ).resolve(),
        "image_descriptors": project_path(
            project_root,
            config["image_descriptors"]["output_directory"],
        ).resolve(),
    }

    directories = {}
    selected = str(selections["stratified_beta_vae"]).strip()
    if selected:
        directories["stratified_beta_vae"] = selected_directory(
            project_root,
            roots["stratified_beta_vae"],
            selected,
            "Stratified-random β-VAE experiment directory",
        )
    else:
        directories["stratified_beta_vae"] = validate_experiment_directory(
            current_run_directory(config, project_root),
            roots["stratified_beta_vae"],
            "Stratified-random β-VAE experiment directory",
        )

    selected = str(selections["session_beta_vae"]).strip()
    if selected:
        directories["session_beta_vae"] = selected_directory(
            project_root,
            roots["session_beta_vae"],
            selected,
            "Session-held-out β-VAE experiment directory",
        )
    else:
        directories["session_beta_vae"] = validate_experiment_directory(
            current_session_beta_vae_experiment(config, project_root),
            roots["session_beta_vae"],
            "Session-held-out β-VAE experiment directory",
        )

    selected = str(selections["efficientnet"]).strip()
    if selected:
        directories["efficientnet"] = selected_directory(
            project_root,
            roots["efficientnet"],
            selected,
            "EfficientNet-B0 experiment directory",
        )
    else:
        directories["efficientnet"] = latest_experiment_directory(
            roots["efficientnet"],
            "EfficientNet-B0 experiment",
        )

    selected = str(selections["image_descriptors"]).strip()
    if selected:
        directories["image_descriptors"] = selected_directory(
            project_root,
            roots["image_descriptors"],
            selected,
            "Image-descriptor experiment directory",
        )
    else:
        directories["image_descriptors"] = validate_experiment_directory(
            current_image_descriptor_experiment(config, project_root),
            roots["image_descriptors"],
            "Image-descriptor experiment directory",
        )
    return directories


def figure_output_directory(project_root, style):
    relative_path = Path(style["output_directory"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("The figure output directory must be project-relative.")
    output_directory = project_path(project_root, relative_path)
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
    if not path.is_file():
        relative_path = project_relative_text(PROJECT_ROOT, path)
        raise FileNotFoundError(f"Required result file does not exist: {relative_path}")


def require_columns(frame, columns, source_name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {source_name}: {missing}")


def selected_result(frame, conditions, source_name):
    selected = frame
    for column, value in conditions.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        description = ", ".join(
            f"{column}={value}"
            for column, value in conditions.items()
        )
        raise ValueError(
            f"Expected one result row in {source_name} for {description}, "
            f"but found {len(selected)}."
        )
    return selected.iloc[0]


def performance_row(method, protocol, accuracy, balanced_accuracy, macro_f1):
    return {
        "method": method,
        "protocol": protocol,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
    }


def load_stratified_beta_vae_result(experiment_directory):
    path = Path(experiment_directory) / "classifiers" / "C5" / "metrics_summary.json"
    require_result_file(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    try:
        metrics = summary["metrics"]["test"]
        return performance_row(
            "spatial_beta_vae_c5",
            "stratified_random",
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
        )
    except KeyError as error:
        relative_path = project_relative_text(PROJECT_ROOT, path)
        raise KeyError(f"Incomplete test metrics in {relative_path}: {error}") from error


def load_session_beta_vae_results(experiment_directory):
    path = Path(experiment_directory) / "c5_fold_results.csv"
    require_result_file(path)
    frame = pd.read_csv(path)
    source_name = project_relative_text(PROJECT_ROOT, path)
    require_columns(
        frame,
        [
            "fold",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
        ],
        source_name,
    )

    rows = []
    for protocol in ["S1", "S2", "S3"]:
        result = selected_result(frame, {"fold": protocol}, source_name)
        rows.append(
            performance_row(
                "spatial_beta_vae_c5",
                protocol,
                result["test_accuracy"],
                result["test_balanced_accuracy"],
                result["test_macro_f1"],
            )
        )
    return rows


def load_efficientnet_results(experiment_directory):
    path = Path(experiment_directory) / "experiment_summary.csv"
    require_result_file(path)
    frame = pd.read_csv(path)
    source_name = project_relative_text(PROJECT_ROOT, path)
    require_columns(
        frame,
        [
            "run",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
        ],
        source_name,
    )

    rows = []
    for protocol in ["stratified_random", "S1", "S2", "S3"]:
        result = selected_result(frame, {"run": protocol}, source_name)
        rows.append(
            performance_row(
                "efficientnet_b0",
                protocol,
                result["test_accuracy"],
                result["test_balanced_accuracy"],
                result["test_macro_f1"],
            )
        )
    return rows


def load_image_descriptor_results(experiment_directory, style):
    path = Path(experiment_directory) / "summaries" / "classifier_results.csv"
    require_result_file(path)
    frame = pd.read_csv(path)
    source_name = project_relative_text(PROJECT_ROOT, path)
    require_columns(
        frame,
        [
            "run",
            "algorithm",
            "configuration",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
        ],
        source_name,
    )

    rows = []
    for protocol in ["stratified_random", "S1", "S2", "S3"]:
        conditions = {
            "run": protocol,
            "algorithm": style["descriptor_algorithm"],
            "configuration": style["descriptor_configuration"],
        }
        result = selected_result(frame, conditions, source_name)
        rows.append(
            performance_row(
                "image_descriptors_rbf_svm",
                protocol,
                result["test_accuracy"],
                result["test_balanced_accuracy"],
                result["test_macro_f1"],
            )
        )
    return rows


def validate_performance_results(frame, style):
    method_names = list(style["method_labels"])
    protocol_names = ["stratified_random", "S1", "S2", "S3"]
    expected = {
        (method, protocol)
        for method in method_names
        for protocol in protocol_names
    }
    observed = set(zip(frame["method"], frame["protocol"]))
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"Cross-experiment results are incomplete. Missing={missing}; "
            f"unexpected={unexpected}."
        )

    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1"]
    values = frame[metric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Cross-experiment results contain a non-finite metric value.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Cross-experiment metric values must be within [0, 1].")


def load_cross_experiment_results(directories, style):
    rows = [
        load_stratified_beta_vae_result(directories["stratified_beta_vae"]),
    ]
    rows.extend(load_session_beta_vae_results(directories["session_beta_vae"]))
    rows.extend(load_efficientnet_results(directories["efficientnet"]))
    rows.extend(
        load_image_descriptor_results(
            directories["image_descriptors"],
            style,
        )
    )
    frame = pd.DataFrame(rows)
    validate_performance_results(frame, style)
    return frame


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


def plot_cross_experiment_performance(results, output_directory, style):
    panel_title_font_size = 7
    panel_title_x = 0.0
    panel_title_y = 1.04
    stratified_offset = -0.20
    held_out_offsets = {"S1": -0.06, "S2": 0.0, "S3": 0.06}
    held_out_mean_offset = 0.20
    point_size = 20
    stratified_point_size = 23
    mean_marker_size = 6.0
    marker_edge_width = 0.55
    error_line_width = 1.0
    error_cap_size = 2.5
    column_gap = 0.07
    left_margin = 0.08
    right_margin = 0.99
    bottom_margin = 0.28
    top_margin = 0.84
    legend_y = 0.99

    method_label_font_size = 7
    method_label_line_spacing = 1.3


    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
    ]
    method_names = list(style["method_labels"])
    method_positions = np.arange(len(method_names), dtype=np.float64)
    method_labels = [style["method_labels"][name] for name in method_names]
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=style["comparison_size"],
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for axis, panel_label, (metric_column, metric_title) in zip(
        axes,
        ["a", "b", "c"],
        metrics,
    ):
        for method_position, method_name in zip(method_positions, method_names):
            method_results = results.loc[results["method"] == method_name]
            stratified = selected_result(
                method_results,
                {"protocol": "stratified_random"},
                method_name,
            )
            axis.scatter(
                method_position + stratified_offset,
                float(stratified[metric_column]),
                s=stratified_point_size,
                marker=style["protocol_markers"]["stratified_random"],
                facecolor="white",
                edgecolor=style["protocol_colors"]["stratified_random"],
                linewidth=0.9,
                zorder=4,
            )

            axis.tick_params(axis="x", pad=3)

            held_out_values = []
            for protocol, offset in held_out_offsets.items():
                result = selected_result(
                    method_results,
                    {"protocol": protocol},
                    method_name,
                )
                value = float(result[metric_column])
                held_out_values.append(value)
                axis.scatter(
                    method_position + offset,
                    value,
                    s=point_size,
                    marker=style["protocol_markers"][protocol],
                    facecolor=style["protocol_colors"][protocol],
                    edgecolor="white",
                    linewidth=marker_edge_width,
                    zorder=4,
                )

            held_out_mean = float(np.mean(held_out_values))
            held_out_sample_sd = float(np.std(held_out_values, ddof=1))
            axis.errorbar(
                method_position + held_out_mean_offset,
                held_out_mean,
                yerr=held_out_sample_sd,
                fmt=style["protocol_markers"]["session_held_out_mean"],
                color=style["protocol_colors"]["session_held_out_mean"],
                markerfacecolor=style["protocol_colors"]["session_held_out_mean"],
                markeredgecolor="white",
                markeredgewidth=marker_edge_width,
                markersize=mean_marker_size,
                elinewidth=error_line_width,
                capsize=error_cap_size,
                zorder=5,
            )

        panel_title(
            axis,
            panel_label,
            metric_title,
            panel_title_font_size,
            panel_title_x,
            panel_title_y,
        )
        axis.set_xticks(method_positions)
        axis.set_xticklabels(
            method_labels,
            fontsize=method_label_font_size,
            linespacing=method_label_line_spacing,
        )
        axis.set_xlim(-0.45, len(method_names) - 0.55)
        axis.grid(axis="y", alpha=style["grid_alpha"], linewidth=0.6)
        tick_positions = np.arange(0.0, 1.01, 0.2)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(tick_positions)
        axis.set_yticklabels([f"{100.0 * value:.0f}" for value in tick_positions])

        axis.set_axisbelow(True)

    axes[0].set_ylabel("Score (%)")
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=style["protocol_markers"]["stratified_random"],
            markerfacecolor="white",
            markeredgecolor=style["protocol_colors"]["stratified_random"],
            markeredgewidth=0.9,
            markersize=5.5,
            label=style["protocol_labels"]["stratified_random"],
        )
    ]
    for protocol in ["S1", "S2", "S3", "session_held_out_mean"]:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=style["protocol_markers"][protocol],
                markerfacecolor=style["protocol_colors"][protocol],
                markeredgecolor="white",
                markeredgewidth=marker_edge_width,
                markersize=5.5,
                label=style["protocol_labels"][protocol],
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, legend_y),
        markerscale=0.8,
    )
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=column_gap,
    )
    save_figure(
        fig,
        output_directory / "cross_experiment_classification_performance",
        style,
    )


def print_result_sources(directories):
    print("Result sources:")
    for name, directory in directories.items():
        relative_directory = project_relative_text(PROJECT_ROOT, directory)
        print(f"  {name}: {relative_directory}")


def print_result_values(results, style):
    method_order = list(style["method_labels"])
    protocol_order = ["stratified_random", "S1", "S2", "S3"]
    displayed = results.copy()
    displayed["method"] = pd.Categorical(
        displayed["method"],
        categories=method_order,
        ordered=True,
    )
    displayed["protocol"] = pd.Categorical(
        displayed["protocol"],
        categories=protocol_order,
        ordered=True,
    )
    displayed = displayed.sort_values(["method", "protocol"])
    displayed["method"] = displayed["method"].map(style["method_labels"])
    displayed["protocol"] = displayed["protocol"].map(style["protocol_labels"])
    print("\nCross-experiment test metrics:")
    print(displayed.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main():
    config = config_with_image_descriptor_settings(
        load_config(CONFIG_PATH, False),
        SETTINGS_PATH,
    )
    style = figure_settings()
    configure_article_style(style)
    directories = cross_experiment_directories(config, PROJECT_ROOT, style)
    output_directory = figure_output_directory(PROJECT_ROOT, style)
    results = load_cross_experiment_results(directories, style)

    print_result_sources(directories)
    print_result_values(results, style)
    plot_cross_experiment_performance(
        results,
        output_directory,
        style,
    )


if __name__ == "__main__":
    print("Generate the cross-experiment classification-performance figure.")
    main()
