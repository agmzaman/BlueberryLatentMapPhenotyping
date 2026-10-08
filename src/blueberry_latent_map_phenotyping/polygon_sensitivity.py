from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import require_keys, validate_relative_path
from .paths import dataset_path, project_path, project_relative_text


def load_polygon_sensitivity_settings(settings_path):
    settings_path = Path(settings_path)
    with settings_path.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    require_keys(
        settings,
        [
            "seed",
            "repetitions",
            "noise_percentages",
            "boundary_mode",
            "clip_vertices_to_image",
            "columns",
            "output",
        ],
        "polygon_sensitivity",
    )
    require_keys(
        settings["columns"],
        [
            "polygon_points",
            "polygon_area",
            "image_width",
            "image_height",
            "image_area",
            "relative_area",
        ],
        "polygon_sensitivity.columns",
    )
    require_keys(
        settings["output"],
        [
            "directory",
            "log_file",
            "boundaries_file",
            "repetitions_file",
            "summary_file",
            "transitions_file",
            "table_file",
        ],
        "polygon_sensitivity.output",
    )

    seed = int(settings["seed"])
    repetitions = int(settings["repetitions"])
    percentages = [
        float(value)
        for value in settings["noise_percentages"]
    ]

    if seed < 0:
        raise ValueError("polygon_sensitivity.seed cannot be negative.")
    if repetitions < 2:
        raise ValueError(
            "polygon_sensitivity.repetitions must be at least 2."
        )
    if not percentages:
        raise ValueError(
            "polygon_sensitivity.noise_percentages cannot be empty."
        )
    if any(value <= 0.0 or value > 10.0 for value in percentages):
        raise ValueError(
            "Polygon noise percentages must be above 0 and at most 10."
        )
    if percentages != sorted(set(percentages)):
        raise ValueError(
            "Polygon noise percentages must be unique and increasing."
        )
    if settings["boundary_mode"] != "fixed_original_jenks":
        raise ValueError(
            "polygon_sensitivity.boundary_mode must be fixed_original_jenks."
        )
    if settings["clip_vertices_to_image"] is not True:
        raise ValueError(
            "polygon_sensitivity.clip_vertices_to_image must be true."
        )

    for name, value in settings["columns"].items():
        if not str(value).strip():
            raise ValueError(
                f"polygon_sensitivity.columns.{name} cannot be empty."
            )

    for name, value in settings["output"].items():
        validate_relative_path(
            value,
            f"polygon_sensitivity.output.{name}",
        )
    return settings


def polygon_sensitivity_paths(settings, project_root):
    output = settings["output"]
    output_directory = project_path(
        project_root,
        output["directory"],
    )
    return {
        "directory": output_directory,
        "log": output_directory / output["log_file"],
        "boundaries": output_directory / output["boundaries_file"],
        "repetitions": output_directory / output["repetitions_file"],
        "summary": output_directory / output["summary_file"],
        "transitions": output_directory / output["transitions_file"],
        "table": output_directory / output["table_file"],
    }


def parse_polygon_points(value):
    points = []
    for pair in str(value).split(";"):
        pair = pair.strip()
        if not pair:
            continue
        values = pair.split(",")
        if len(values) != 2:
            raise ValueError(f"Invalid polygon point: {pair}")
        points.append(
            [
                float(values[0].strip()),
                float(values[1].strip()),
            ]
        )
    if len(points) < 3:
        raise ValueError("A polygon must contain at least three points.")
    return np.asarray(points, dtype=np.float64)


def polygon_area(points):
    x_values = points[..., 0]
    y_values = points[..., 1]
    return 0.5 * np.abs(
        np.sum(
            x_values * np.roll(y_values, -1, axis=-1)
            - y_values * np.roll(x_values, -1, axis=-1),
            axis=-1,
        )
    )


def load_polygon_annotations(config, settings):
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]
    columns = settings["columns"]
    required_columns = [
        image_column,
        label_column,
        columns["polygon_points"],
        columns["polygon_area"],
        columns["image_width"],
        columns["image_height"],
        columns["image_area"],
        columns["relative_area"],
    ]
    annotation_path = dataset_path(
        config,
        config["dataset"]["annotation_csv"],
    )
    frame = pd.read_csv(
        annotation_path,
        usecols=required_columns,
    )

    expected_total = int(config["dataset"]["expected_total"])
    if len(frame) != expected_total:
        raise ValueError(
            f"Expected {expected_total} annotation rows but found {len(frame)}."
        )
    if frame[image_column].duplicated().any():
        raise ValueError("Duplicate image filenames were found.")
    if frame[required_columns].isna().any().any():
        raise ValueError("Polygon annotations contain missing values.")

    numeric_columns = [
        columns["polygon_area"],
        columns["image_width"],
        columns["image_height"],
        columns["image_area"],
        columns["relative_area"],
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame[numeric_columns] <= 0.0).any().any():
        raise ValueError("Polygon dimensions and areas must be positive.")

    observed_classes = set(frame[label_column].astype(str))
    configured_classes = set(config["dataset"]["classes"])
    if observed_classes != configured_classes:
        raise ValueError(
            "Polygon annotation classes do not match dataset.classes."
        )

    polygons = [
        parse_polygon_points(value)
        for value in frame[columns["polygon_points"]]
    ]
    computed_areas = np.asarray(
        [polygon_area(points) for points in polygons],
        dtype=np.float64,
    )
    recorded_areas = frame[columns["polygon_area"]].to_numpy(
        dtype=np.float64
    )
    if not np.allclose(
        computed_areas,
        recorded_areas,
        rtol=0.000001,
        atol=0.5,
    ):
        raise ValueError(
            "Polygon points do not reproduce the recorded polygon areas."
        )

    image_areas = frame[columns["image_area"]].to_numpy(
        dtype=np.float64
    )
    recorded_relative_areas = frame[
        columns["relative_area"]
    ].to_numpy(dtype=np.float64)
    computed_relative_areas = computed_areas / image_areas
    if not np.allclose(
        computed_relative_areas,
        recorded_relative_areas,
        rtol=0.00001,
        atol=0.000001,
    ):
        raise ValueError(
            "Polygon areas do not reproduce the recorded relative areas."
        )
    return frame, polygons


def fixed_jenks_boundaries(frame, config, settings):
    label_column = config["dataset"]["label_column"]
    class_names = config["dataset"]["classes"]
    relative_area_column = settings["columns"]["relative_area"]
    rows = []
    boundaries = []

    for index in range(len(class_names) - 1):
        lower_class = class_names[index]
        upper_class = class_names[index + 1]
        lower_values = frame.loc[
            frame[label_column] == lower_class,
            relative_area_column,
        ].to_numpy(dtype=np.float64)
        upper_values = frame.loc[
            frame[label_column] == upper_class,
            relative_area_column,
        ].to_numpy(dtype=np.float64)

        lower_maximum = float(np.max(lower_values))
        upper_minimum = float(np.min(upper_values))
        if lower_maximum >= upper_minimum:
            raise ValueError(
                "Original class ranges overlap, so fixed Jenks boundaries "
                "cannot be reconstructed from the labels."
            )

        boundaries.append(lower_maximum)
        rows.append(
            {
                "lower_class": lower_class,
                "upper_class": upper_class,
                "fixed_boundary_relative_area": lower_maximum,
                "next_class_minimum_relative_area": upper_minimum,
            }
        )

    boundary_values = np.asarray(boundaries, dtype=np.float64)
    original_values = frame[relative_area_column].to_numpy(
        dtype=np.float64
    )
    predicted_indices = classify_relative_areas(
        original_values,
        boundary_values,
    )
    original_indices = encode_classes(
        frame[label_column].astype(str).to_numpy(),
        class_names,
    )
    if not np.array_equal(predicted_indices, original_indices):
        raise ValueError(
            "The reconstructed fixed boundaries do not reproduce the "
            "original class labels."
        )
    return boundary_values, pd.DataFrame(rows)


def encode_classes(values, class_names):
    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }
    unknown = sorted(set(values) - set(class_to_index))
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}")
    return np.asarray(
        [class_to_index[value] for value in values],
        dtype=np.int64,
    )


def classify_relative_areas(relative_areas, boundaries):
    return np.searchsorted(
        boundaries,
        relative_areas,
        side="left",
    ).astype(np.int64)


def perturbation_predictions(
    frame,
    polygons,
    settings,
    boundaries,
    noise_percentage,
):
    columns = settings["columns"]
    repetitions = int(settings["repetitions"])
    seed = int(settings["seed"])
    level_seed = np.random.SeedSequence(
        [seed, int(round(noise_percentage * 1000.0))]
    )
    random_generator = np.random.default_rng(level_seed)
    predictions = np.empty(
        (repetitions, len(frame)),
        dtype=np.int64,
    )

    widths = frame[columns["image_width"]].to_numpy(
        dtype=np.float64
    )
    heights = frame[columns["image_height"]].to_numpy(
        dtype=np.float64
    )
    image_areas = frame[columns["image_area"]].to_numpy(
        dtype=np.float64
    )
    fraction = noise_percentage / 100.0

    for image_index, points in enumerate(polygons):
        point_count = len(points)
        x_values = points[:, 0][None, :] + random_generator.normal(
            0.0,
            fraction * widths[image_index],
            size=(repetitions, point_count),
        )
        y_values = points[:, 1][None, :] + random_generator.normal(
            0.0,
            fraction * heights[image_index],
            size=(repetitions, point_count),
        )
        x_values = np.clip(
            x_values,
            0.0,
            widths[image_index] - 1.0,
        )
        y_values = np.clip(
            y_values,
            0.0,
            heights[image_index] - 1.0,
        )
        perturbed_points = np.stack(
            [x_values, y_values],
            axis=-1,
        )
        relative_areas = (
            polygon_area(perturbed_points)
            / image_areas[image_index]
        )
        predictions[:, image_index] = classify_relative_areas(
            relative_areas,
            boundaries,
        )
    return predictions


def repetition_rows(
    predictions,
    original_indices,
    class_names,
    noise_percentage,
):
    changes = predictions != original_indices[None, :]
    rows = []
    for repetition in range(predictions.shape[0]):
        row = {
            "noise_percentage": noise_percentage,
            "repetition": repetition + 1,
            "overall_change_percent": float(
                100.0 * np.mean(changes[repetition])
            ),
        }
        for class_index, class_name in enumerate(class_names):
            class_mask = original_indices == class_index
            column_name = (
                class_name.lower().replace(" ", "_")
                + "_change_percent"
            )
            row[column_name] = float(
                100.0
                * np.mean(changes[repetition, class_mask])
            )
        rows.append(row)
    return rows


def summary_rows(
    predictions,
    original_indices,
    class_names,
    noise_percentage,
):
    changes = predictions != original_indices[None, :]
    groups = [("Overall", np.ones(len(original_indices), dtype=bool))]
    groups.extend(
        (
            class_name,
            original_indices == class_index,
        )
        for class_index, class_name in enumerate(class_names)
    )

    rows = []
    for group_name, group_mask in groups:
        repetition_rates = (
            100.0 * np.mean(changes[:, group_mask], axis=1)
        )
        changed_trials = int(np.sum(changes[:, group_mask]))
        image_count = int(np.sum(group_mask))
        rows.append(
            {
                "noise_percentage": noise_percentage,
                "group": group_name,
                "images_per_repetition": image_count,
                "total_trials": int(
                    image_count * predictions.shape[0]
                ),
                "changed_trials": changed_trials,
                "mean_change_percent": float(
                    np.mean(repetition_rates)
                ),
                "standard_deviation_percent": float(
                    np.std(repetition_rates, ddof=1)
                ),
                "percentile_2_5": float(
                    np.percentile(repetition_rates, 2.5)
                ),
                "percentile_97_5": float(
                    np.percentile(repetition_rates, 97.5)
                ),
            }
        )
    return rows


def transition_rows(
    predictions,
    original_indices,
    class_names,
    noise_percentage,
):
    rows = []
    for original_index, original_class in enumerate(class_names):
        class_predictions = predictions[
            :,
            original_indices == original_index,
        ].ravel()
        counts = np.bincount(
            class_predictions,
            minlength=len(class_names),
        )
        total = int(np.sum(counts))
        for predicted_index, predicted_class in enumerate(class_names):
            count = int(counts[predicted_index])
            rows.append(
                {
                    "noise_percentage": noise_percentage,
                    "original_class": original_class,
                    "perturbed_class": predicted_class,
                    "count": count,
                    "percentage_within_original_class": float(
                        100.0 * count / total
                    ),
                }
            )
    return rows


def article_table(summary, class_names):
    means = summary.pivot(index="noise_percentage", columns="group", values="mean_change_percent")
    standard_deviations = summary.pivot(index="noise_percentage", columns="group", values="standard_deviation_percent")
    groups = ["Overall", *class_names]
    rows = []

    for noise_percentage in sorted(summary["noise_percentage"].unique()):
        row = {"Perturbation level (%)": noise_percentage}
        for group in groups:
            mean = means.loc[noise_percentage, group]
            standard_deviation = standard_deviations.loc[noise_percentage, group]
            row[f"{group}, mean ± SD (%)"] = f"{mean:.2f} ± {standard_deviation:.2f}"
        rows.append(row)

    return pd.DataFrame(rows)



def run_polygon_sensitivity(
    config,
    settings,
    project_root,
):
    frame, polygons = load_polygon_annotations(
        config,
        settings,
    )
    boundaries, boundary_frame = fixed_jenks_boundaries(
        frame,
        config,
        settings,
    )
    label_column = config["dataset"]["label_column"]
    class_names = config["dataset"]["classes"]
    original_indices = encode_classes(
        frame[label_column].astype(str).to_numpy(),
        class_names,
    )
    paths = polygon_sensitivity_paths(
        settings,
        project_root,
    )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    boundary_frame.to_csv(paths["boundaries"], index=False)

    all_repetitions = []
    all_summaries = []
    all_transitions = []
    for noise_percentage in settings["noise_percentages"]:
        noise_percentage = float(noise_percentage)
        predictions = perturbation_predictions(
            frame,
            polygons,
            settings,
            boundaries,
            noise_percentage,
        )
        all_repetitions.extend(
            repetition_rows(
                predictions,
                original_indices,
                class_names,
                noise_percentage,
            )
        )
        all_summaries.extend(
            summary_rows(
                predictions,
                original_indices,
                class_names,
                noise_percentage,
            )
        )
        all_transitions.extend(
            transition_rows(
                predictions,
                original_indices,
                class_names,
                noise_percentage,
            )
        )

    repetitions = pd.DataFrame(all_repetitions)
    summary = pd.DataFrame(all_summaries)
    transitions = pd.DataFrame(all_transitions)
    table = article_table(summary, class_names)

    repetitions.to_csv(paths["repetitions"], index=False)
    summary.to_csv(paths["summary"], index=False)
    transitions.to_csv(paths["transitions"], index=False)
    table.to_csv(paths["table"], index=False)

    print(table.to_string(index=False))
    for name in [
        "boundaries",
        "repetitions",
        "summary",
        "transitions",
        "table",
    ]:
        relative_path = project_relative_text(
            project_root,
            paths[name],
        )
        print(f"Saved {name}: {relative_path}")
    return table, summary

