import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from .paths import project_relative_text


def safe_skew(values, epsilon):
    flattened = np.asarray(values, dtype=np.float64).ravel()
    if flattened.size == 0 or np.std(flattened) < epsilon:
        return 0.0
    value = skew(flattened, bias=False, nan_policy="omit")
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def global_statistics(values, statistic_names, epsilon):
    flattened = np.asarray(values, dtype=np.float64).ravel()
    median = float(np.median(flattened))
    p10, p90 = np.percentile(flattened, [10, 90])
    available = {
        "mean": float(np.mean(flattened)),
        "median": median,
        "std": float(np.std(flattened)),
        "mad": float(np.median(np.abs(flattened - median))),
        "p10": float(p10),
        "p90": float(p90),
        "abs_mean": float(np.mean(np.abs(flattened))),
        "positive_frac": float(np.mean(flattened > 0.0)),
        "skewness": safe_skew(flattened, epsilon),
    }
    return [available[name] for name in statistic_names]


def split_quadrants(spatial_map):
    height, width = spatial_map.shape
    half_height = height // 2
    half_width = width // 2
    return [
        spatial_map[:half_height, :half_width],
        spatial_map[:half_height, half_width:],
        spatial_map[half_height:, :half_width],
        spatial_map[half_height:, half_width:],
    ]


def quadrant_statistics(spatial_map, statistic_names):
    values = []
    for quadrant in split_quadrants(spatial_map):
        available = {
            "mean": float(np.mean(quadrant)),
            "std": float(np.std(quadrant)),
        }
        values.extend(available[name] for name in statistic_names)
    return values


def region_means(spatial_map, grid_size):
    height, width = spatial_map.shape
    row_edges = np.linspace(0, height, grid_size + 1, dtype=int)
    column_edges = np.linspace(0, width, grid_size + 1, dtype=int)
    values = []
    for row_index in range(grid_size):
        for column_index in range(grid_size):
            region = spatial_map[
                row_edges[row_index]:row_edges[row_index + 1],
                column_edges[column_index]:column_edges[column_index + 1],
            ]
            values.append(float(np.mean(region)))
    return values


def center_edge_statistics(spatial_map, center_fraction, epsilon):
    absolute_map = np.abs(np.asarray(spatial_map, dtype=np.float64))
    height, width = absolute_map.shape
    center_height = int(round(height * center_fraction))
    center_width = int(round(width * center_fraction))
    if center_height <= 0 or center_width <= 0:
        raise ValueError("The configured center region is empty.")
    if center_height >= height or center_width >= width:
        raise ValueError("The configured center region must be smaller than the map.")

    row_start = (height - center_height) // 2
    row_end = row_start + center_height
    column_start = (width - center_width) // 2
    column_end = column_start + center_width

    center = absolute_map[row_start:row_end, column_start:column_end]
    edge_mask = np.ones_like(absolute_map, dtype=bool)
    edge_mask[row_start:row_end, column_start:column_end] = False
    edge = absolute_map[edge_mask]

    center_mean = float(np.mean(center))
    edge_mean = float(np.mean(edge))
    ratio = float((center_mean + epsilon) / (edge_mean + epsilon))
    return [center_mean, edge_mean, ratio]


def kl_map(mean_map, log_variance_map, lower_bound, upper_bound):
    clipped = np.clip(log_variance_map, lower_bound, upper_bound)
    values = 0.5 * (np.exp(clipped) + mean_map ** 2 - 1.0 - clipped)
    return np.maximum(values, 0.0)


def build_feature_groups(config):
    channel_count = int(config["vae"]["latent_channels"])
    global_names = config["features"]["global_statistics"]
    quadrant_names = config["features"]["quadrant_statistics"]
    kl_names = config["features"]["kl_statistics"]
    kl_quadrant_names = config["features"]["kl_quadrant_statistics"]
    grid_size = int(config["features"]["region_grid_size"])

    groups = {
        "mu_global": [],
        "mu_quadrant": [],
        "mu_region16": [],
        "mu_center_edge": [],
        "logvar_global": [],
        "logvar_quadrant": [],
        "kl_global": [],
        "kl_quadrant": [],
    }

    for channel in range(channel_count):
        prefix = f"mu_c{channel + 1}"
        groups["mu_global"].extend(
            f"{prefix}_global_{name}" for name in global_names
        )
        groups["mu_quadrant"].extend(
            f"{prefix}_q{quadrant}_{name}"
            for quadrant in range(1, 5)
            for name in quadrant_names
        )
        groups["mu_region16"].extend(
            f"{prefix}_r{region}_mean"
            for region in range(1, grid_size * grid_size + 1)
        )
        groups["mu_center_edge"].extend(
            [
                f"{prefix}_center_abs_mean",
                f"{prefix}_edge_abs_mean",
                f"{prefix}_center_edge_abs_ratio",
            ]
        )

    for channel in range(channel_count):
        prefix = f"logvar_c{channel + 1}"
        groups["logvar_global"].extend(
            f"{prefix}_global_{name}" for name in global_names
        )
        groups["logvar_quadrant"].extend(
            f"{prefix}_q{quadrant}_{name}"
            for quadrant in range(1, 5)
            for name in quadrant_names
        )

    for channel in range(channel_count):
        prefix = f"kl_c{channel + 1}"
        groups["kl_global"].extend(
            f"{prefix}_global_{name}" for name in kl_names
        )
        groups["kl_quadrant"].extend(
            f"{prefix}_q{quadrant}_{name}"
            for quadrant in range(1, 5)
            for name in kl_quadrant_names
        )
    return groups


def all_feature_columns(groups):
    columns = []
    for group_columns in groups.values():
        columns.extend(group_columns)
    return columns


def selected_feature_columns(config, groups, run_name):
    if run_name not in config["classifier"]["runs"]:
        raise KeyError(f"Unknown classifier run: {run_name}")

    columns = []
    for group_name in config["classifier"]["runs"][run_name]:
        if group_name not in groups:
            raise KeyError(f"Unknown feature group in {run_name}: {group_name}")
        columns.extend(groups[group_name])

    expected = int(config["classifier"]["expected_feature_counts"][run_name])
    if len(columns) != expected:
        raise ValueError(
            f"{run_name} expected {expected} features but resolved {len(columns)}."
        )
    if len(columns) != len(set(columns)):
        raise ValueError(f"{run_name} contains duplicate feature columns.")
    return columns


def validate_feature_groups(config, groups):
    expected_group_counts = {
        "mu_global": 45,
        "mu_quadrant": 40,
        "mu_region16": 80,
        "mu_center_edge": 15,
        "logvar_global": 45,
        "logvar_quadrant": 40,
        "kl_global": 15,
        "kl_quadrant": 20,
    }
    actual_group_counts = {
        name: len(columns)
        for name, columns in groups.items()
    }
    if actual_group_counts != expected_group_counts:
        raise ValueError(
            f"Feature group counts changed: {actual_group_counts}"
        )

    for run_name in config["classifier"]["runs"]:
        selected_feature_columns(config, groups, run_name)


def extract_sample_features(config, mean_sample, log_variance_sample):
    feature_config = config["features"]
    global_names = feature_config["global_statistics"]
    quadrant_names = feature_config["quadrant_statistics"]
    kl_names = feature_config["kl_statistics"]
    kl_quadrant_names = feature_config["kl_quadrant_statistics"]
    epsilon = float(feature_config["safe_skew_epsilon"])
    center_fraction = float(feature_config["center_fraction"])
    center_epsilon = float(feature_config["center_edge_epsilon"])
    grid_size = int(feature_config["region_grid_size"])
    lower_bound = float(config["vae"]["log_variance_min"])
    upper_bound = float(config["vae"]["log_variance_max"])

    values = []

    for channel_map in mean_sample:
        values.extend(global_statistics(channel_map, global_names, epsilon))
    for channel_map in mean_sample:
        values.extend(quadrant_statistics(channel_map, quadrant_names))
    for channel_map in mean_sample:
        values.extend(region_means(channel_map, grid_size))
    for channel_map in mean_sample:
        values.extend(
            center_edge_statistics(
                channel_map,
                center_fraction,
                center_epsilon,
            )
        )

    for channel_map in log_variance_sample:
        values.extend(global_statistics(channel_map, global_names, epsilon))
    for channel_map in log_variance_sample:
        values.extend(quadrant_statistics(channel_map, quadrant_names))

    kl_maps = [
        kl_map(mean_map, log_variance_map, lower_bound, upper_bound)
        for mean_map, log_variance_map in zip(
            mean_sample,
            log_variance_sample,
        )
    ]
    for channel_map in kl_maps:
        available = {
            "mean": float(np.mean(channel_map)),
            "std": float(np.std(channel_map)),
            "p90": float(np.percentile(channel_map, 90)),
        }
        values.extend(available[name] for name in kl_names)
    for channel_map in kl_maps:
        for quadrant in split_quadrants(channel_map):
            available = {"mean": float(np.mean(quadrant))}
            values.extend(available[name] for name in kl_quadrant_names)

    return values


def load_latent_arrays(latent_path):
    with np.load(latent_path, allow_pickle=False) as data:
        required = {"mean", "log_variance", "label_index", "image_path"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing arrays in {latent_path}: {sorted(missing)}")
        arrays = {name: data[name] for name in required}

    if arrays["mean"].shape != arrays["log_variance"].shape:
        raise ValueError(f"Mean and log-variance shapes differ in {latent_path}.")
    sample_count = arrays["mean"].shape[0]
    if len(arrays["label_index"]) != sample_count:
        raise ValueError(f"Label count differs from latent maps in {latent_path}.")
    if len(arrays["image_path"]) != sample_count:
        raise ValueError(f"Path count differs from latent maps in {latent_path}.")
    return arrays


def extract_split_features(config, latent_path, output_path, groups):
    arrays = load_latent_arrays(latent_path)
    feature_columns = all_feature_columns(groups)
    class_names = config["dataset"]["classes"]
    rows = []

    for sample_index in range(arrays["mean"].shape[0]):
        values = extract_sample_features(
            config,
            arrays["mean"][sample_index],
            arrays["log_variance"][sample_index],
        )
        if len(values) != len(feature_columns):
            raise ValueError(
                f"Feature count mismatch at sample {sample_index}: "
                f"{len(values)} versus {len(feature_columns)}"
            )
        label_index = int(arrays["label_index"][sample_index])
        if label_index < 0 or label_index >= len(class_names):
            raise ValueError(f"Invalid class index {label_index} at sample {sample_index}.")
        row = {
            config["dataset"]["image_column"]: str(arrays["image_path"][sample_index]),
            config["dataset"]["label_column"]: class_names[label_index],
        }
        row.update(dict(zip(feature_columns, values)))
        rows.append(row)

        if (sample_index + 1) % 100 == 0:
            print(f"Extracted features for {sample_index + 1} samples.")

    frame = pd.DataFrame(rows)
    numeric_values = frame[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"Non-finite features were generated for {latent_path}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def extract_latent_features(config, project_root, run_dir):
    groups = build_feature_groups(config)
    validate_feature_groups(config, groups)
    schema = {
        "groups": groups,
        "all_feature_columns": all_feature_columns(groups),
        "run_columns": {
            run_name: selected_feature_columns(config, groups, run_name)
            for run_name in config["classifier"]["runs"]
        },
        "run_feature_counts": {
            run_name: len(selected_feature_columns(config, groups, run_name))
            for run_name in config["classifier"]["runs"]
        },
    }

    schema_path = Path(run_dir) / "features" / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    summaries = {}
    for split_name in ["train", "validation", "test"]:
        latent_path = Path(run_dir) / "latent" / f"{split_name}_latent_maps.npz"
        output_path = Path(run_dir) / "features" / f"{split_name}_features.csv"
        frame = extract_split_features(
            config,
            latent_path,
            output_path,
            groups,
        )
        summaries[split_name] = {
            "samples": len(frame),
            "feature_count": len(schema["all_feature_columns"]),
            "path": project_relative_text(project_root, output_path),
        }
        relative_output = project_relative_text(project_root, output_path)
        print(f"Saved {split_name} features -> {relative_output}")

    summary_path = Path(run_dir) / "features" / "feature_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def load_feature_schema(run_dir):
    schema_path = Path(run_dir) / "features" / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema does not exist: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))

