import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .data import normalize_dataset_relative_path
from .paths import dataset_image_root, project_relative_text, split_paths


def morphology_feature_names():
    return [
        "morphology_area_fraction",
        "morphology_width_fraction",
        "morphology_height_fraction",
        "morphology_normalized_perimeter",
        "morphology_solidity",
        "morphology_eccentricity",
    ]


def color_feature_names(config):
    settings = config["image_descriptors"]["color"]
    return [
        f"color_{channel}_{statistic}"
        for channel in settings["channels"]
        for statistic in settings["statistics"]
    ]


def glcm_feature_names(config):
    settings = config["image_descriptors"]["glcm"]
    return [
        f"glcm_{property_name}_{summary_name}"
        for property_name in settings["properties"]
        for summary_name in settings["summaries"]
    ]


def image_descriptor_groups(config):
    groups = {
        "morphology": morphology_feature_names(),
        "color": color_feature_names(config),
        "glcm": glcm_feature_names(config),
    }
    configured = {
        name: int(value)
        for name, value in config["image_descriptors"]["feature_groups"].items()
    }
    calculated = {
        name: len(columns)
        for name, columns in groups.items()
    }
    if calculated != configured:
        raise ValueError(
            f"Configured and calculated feature-group counts differ: "
            f"{configured} versus {calculated}."
        )
    return groups


def image_descriptor_columns(config):
    groups = image_descriptor_groups(config)
    columns = []
    for group_columns in groups.values():
        columns.extend(group_columns)
    expected = int(config["image_descriptors"]["expected_feature_count"])
    if len(columns) != expected:
        raise ValueError(
            f"Expected {expected} image descriptors but generated {len(columns)}."
        )
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate image-descriptor names were generated.")
    return columns


def configuration_feature_columns(config, configuration_name):
    settings = config["image_descriptors"]
    if configuration_name not in settings["configurations"]:
        raise KeyError(f"Unknown descriptor configuration: {configuration_name}")
    group_names = settings["configurations"][configuration_name]
    groups = image_descriptor_groups(config)
    columns = []
    for group_name in group_names:
        columns.extend(groups[group_name])
    return columns


def load_descriptor_image_frame(config, project_root):
    image_column = config["dataset"]["image_column"]
    configured_paths = split_paths(config, project_root)
    frames = []
    for split_name in ["train", "validation", "test"]:
        path = configured_paths[split_name]
        if not path.is_file():
            relative_path = project_relative_text(project_root, path)
            raise FileNotFoundError(f"Missing {split_name} split: {relative_path}")
        frames.append(pd.read_csv(path, usecols=[image_column]))

    frame = pd.concat(frames, ignore_index=True)
    root = dataset_image_root(config)
    frame[image_column] = frame[image_column].map(
        lambda value: normalize_dataset_relative_path(value, root)
    )
    if frame[image_column].duplicated().any():
        duplicates = frame.loc[
            frame[image_column].duplicated(),
            image_column,
        ].tolist()
        raise ValueError(f"Duplicate image paths were found: {duplicates[:5]}")
    expected = int(config["dataset"]["expected_total"])
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} images but found {len(frame)}.")

    missing = [
        path
        for path in frame[image_column]
        if not (root / Path(path)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} descriptor input images are missing. "
            f"First entries: {missing[:5]}"
        )
    return frame.sort_values(image_column, kind="stable").reset_index(drop=True)


def read_resized_rgb(path, image_size):
    height, width = [int(value) for value in image_size]
    with Image.open(path) as image_file:
        image = image_file.convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def normalized_rgb_and_excess_green(rgb):
    denominator = np.sum(rgb, axis=2, keepdims=True)
    normalized = np.zeros_like(rgb, dtype=np.float32)
    np.divide(rgb, denominator, out=normalized, where=denominator > 0.0)
    excess_green = (
        2.0 * normalized[:, :, 1]
        - normalized[:, :, 0]
        - normalized[:, :, 2]
    )
    return normalized, excess_green


def morphology_kernel(radius):
    size = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def remove_small_components(mask, minimum_size):
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for component_index in range(1, component_count):
        area = int(statistics[component_index, cv2.CC_STAT_AREA])
        if area >= minimum_size:
            cleaned[labels == component_index] = 1
    return cleaned.astype(bool)


def mask_component_summary(mask):
    count, _, statistics, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    areas = statistics[1:, cv2.CC_STAT_AREA].astype(np.float64)
    component_count = int(max(0, count - 1))
    largest_fraction = 0.0
    if areas.size and np.sum(areas) > 0.0:
        largest_fraction = float(np.max(areas) / np.sum(areas))
    return component_count, largest_fraction


def automatic_vegetation_mask(config, excess_green):
    settings = config["image_descriptors"]["mask"]
    scaled = np.clip((excess_green + 1.0) / 3.0, 0.0, 1.0)
    scaled = np.round(255.0 * scaled).astype(np.uint8)
    threshold, raw_mask = cv2.threshold(
        scaled,
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = raw_mask.astype(bool)
    before_fraction = float(np.mean(mask))

    opening_radius = int(settings["opening_radius"])
    if opening_radius > 0:
        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            morphology_kernel(opening_radius),
        ).astype(bool)
    closing_radius = int(settings["closing_radius"])
    if closing_radius > 0:
        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            morphology_kernel(closing_radius),
        ).astype(bool)

    minimum_size = max(
        1,
        int(
            math.ceil(
                float(settings["minimum_component_fraction"])
                * mask.size
            )
        ),
    )
    mask = remove_small_components(mask, minimum_size)
    fallback_used = False
    if not np.any(mask):
        mask = raw_mask.astype(bool)
        fallback_used = True
    if not np.any(mask):
        row, column = np.unravel_index(np.argmax(excess_green), excess_green.shape)
        mask[row, column] = True
        fallback_used = True

    component_count, _ = mask_component_summary(mask)
    threshold_excess_green = 3.0 * float(threshold) / 255.0 - 1.0
    diagnostics = {
        "otsu_threshold_excess_green": threshold_excess_green,
        "foreground_fraction_before_cleanup": before_fraction,
        "foreground_fraction_after_cleanup": float(np.mean(mask)),
        "component_count": component_count,
        "fallback_used": fallback_used,
    }
    return mask, diagnostics


def convex_hull_pixel_area(mask, contours):
    points = np.vstack(contours)
    hull = cv2.convexHull(points)
    hull_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    return float(np.count_nonzero(hull_mask))


def morphology_values(mask):
    height, width = mask.shape
    area = float(np.count_nonzero(mask))
    if area <= 0.0:
        raise ValueError("The vegetation mask is empty.")

    rows, columns = np.nonzero(mask)
    foreground_width = int(columns.max() - columns.min() + 1)
    foreground_height = int(rows.max() - rows.min() + 1)
    mask_uint8 = mask.astype(np.uint8)
    external_contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not external_contours:
        raise ValueError("No contour could be obtained from the vegetation mask.")

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    perimeter = float(
        np.count_nonzero(padded[1:, :] != padded[:-1, :])
        + np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    )
    hull_area = convex_hull_pixel_area(mask, external_contours)

    moments = cv2.moments(mask_uint8, binaryImage=True)
    covariance = np.asarray(
        [
            [moments["mu20"] / area, moments["mu11"] / area],
            [moments["mu11"] / area, moments["mu02"] / area],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    minor_variance, major_variance = eigenvalues
    eccentricity = 0.0
    if major_variance > 0.0:
        eccentricity = math.sqrt(
            max(0.0, 1.0 - minor_variance / major_variance)
        )

    return [
        area / float(width * height),
        foreground_width / float(width),
        foreground_height / float(height),
        perimeter / math.hypot(width, height),
        area / hull_area,
        eccentricity,
    ]


def color_values(config, rgb, excess_green, mask):
    hsv = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
    channels = {
        "exg": excess_green,
        "hsv_saturation": hsv[:, :, 1],
        "hsv_value": hsv[:, :, 2],
    }
    settings = config["image_descriptors"]["color"]
    values = []
    for channel_name in settings["channels"]:
        foreground_values = channels[channel_name][mask].astype(np.float64)
        values.extend(
            [
                float(np.mean(foreground_values)),
                float(np.std(foreground_values, ddof=0)),
            ]
        )
    return values


def shifted_slices(length, offset):
    if offset >= 0:
        return slice(0, length - offset), slice(offset, length)
    return slice(-offset, length), slice(0, length + offset)


def masked_glcm(quantized, mask, levels, row_offset, column_offset):
    source_rows, target_rows = shifted_slices(quantized.shape[0], row_offset)
    source_columns, target_columns = shifted_slices(
        quantized.shape[1],
        column_offset,
    )
    source = quantized[source_rows, source_columns]
    target = quantized[target_rows, target_columns]
    valid = mask[source_rows, source_columns] & mask[target_rows, target_columns]
    if not np.any(valid):
        return np.zeros((levels, levels), dtype=np.float64)

    pair_indices = source[valid].astype(np.int64) * levels + target[valid]
    matrix = np.bincount(
        pair_indices,
        minlength=levels * levels,
    ).reshape(levels, levels).astype(np.float64)
    matrix += matrix.T
    total = float(np.sum(matrix))
    if total > 0.0:
        matrix /= total
    return matrix


def glcm_property_values(matrix):
    levels = matrix.shape[0]
    rows, columns = np.indices((levels, levels), dtype=np.float64)
    difference = rows - columns
    row_mean = float(np.sum(rows * matrix))
    column_mean = float(np.sum(columns * matrix))
    row_variance = float(np.sum(((rows - row_mean) ** 2) * matrix))
    column_variance = float(
        np.sum(((columns - column_mean) ** 2) * matrix)
    )
    denominator = math.sqrt(row_variance * column_variance)
    correlation = 1.0
    if denominator > 0.0:
        correlation = float(
            np.sum(
                (rows - row_mean)
                * (columns - column_mean)
                * matrix
            )
            / denominator
        )
    return {
        "contrast": float(np.sum((difference ** 2) * matrix)),
        "dissimilarity": float(np.sum(np.abs(difference) * matrix)),
        "homogeneity": float(
            np.sum(matrix / (1.0 + difference ** 2))
        ),
        "energy": float(math.sqrt(np.sum(matrix ** 2))),
        "correlation": correlation,
    }


def glcm_offsets(config):
    settings = config["image_descriptors"]["glcm"]
    directions = {
        0: (0, 1),
        45: (-1, 1),
        90: (-1, 0),
        135: (-1, -1),
    }
    offsets = []
    for grid_step in settings["grid_steps"]:
        for angle in settings["angles_degrees"]:
            row_direction, column_direction = directions[int(angle)]
            offsets.append(
                (
                    int(grid_step) * row_direction,
                    int(grid_step) * column_direction,
                )
            )
    if len(offsets) != 8 or len(set(offsets)) != 8:
        raise ValueError(f"Expected eight unique GLCM offsets but found {offsets}.")
    return offsets


def grayscale_image(config, rgb):
    weights = np.asarray(
        config["image_descriptors"]["glcm"]["grayscale_weights"],
        dtype=np.float64,
    )
    return np.clip(
        np.sum(rgb * weights[None, None, :], axis=2),
        0.0,
        1.0,
    )


def glcm_values(config, rgb, mask):
    settings = config["image_descriptors"]["glcm"]
    levels = int(settings["levels"])
    grayscale = grayscale_image(config, rgb)
    quantized = np.minimum(
        np.floor(grayscale * levels).astype(np.int64),
        levels - 1,
    )
    values_by_property = {
        property_name: []
        for property_name in settings["properties"]
    }
    for row_offset, column_offset in glcm_offsets(config):
        matrix = masked_glcm(
            quantized,
            mask,
            levels,
            row_offset,
            column_offset,
        )
        properties = glcm_property_values(matrix)
        for property_name in values_by_property:
            values_by_property[property_name].append(
                properties[property_name]
            )

    values = []
    for property_name in settings["properties"]:
        property_values = np.asarray(
            values_by_property[property_name],
            dtype=np.float64,
        )
        values.extend(
            [
                float(np.mean(property_values)),
                float(np.std(property_values, ddof=0)),
            ]
        )
    return values


def extract_image_descriptor_values(config, rgb):
    _, excess_green = normalized_rgb_and_excess_green(rgb)
    mask, diagnostics = automatic_vegetation_mask(config, excess_green)
    values = []
    values.extend(morphology_values(mask))
    values.extend(color_values(config, rgb, excess_green, mask))
    values.extend(glcm_values(config, rgb, mask))

    columns = image_descriptor_columns(config)
    if len(values) != len(columns):
        raise ValueError(
            f"Generated {len(values)} values for {len(columns)} descriptor columns."
        )
    values_array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values_array).all():
        raise ValueError("A non-finite image descriptor was generated.")
    return dict(zip(columns, values_array.tolist())), diagnostics


def feature_check_table(feature_frame, feature_columns):
    rows = []
    for feature_name in feature_columns:
        values = feature_frame[feature_name].to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature": feature_name,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "population_std": float(np.std(values, ddof=0)),
                "constant": bool(np.ptp(values) == 0.0),
                "finite": bool(np.isfinite(values).all()),
            }
        )
    return pd.DataFrame(rows)


def extract_image_descriptors(config, project_root, experiment_dir, image_frame):
    image_column = config["dataset"]["image_column"]
    image_root = dataset_image_root(config)
    feature_rows = []
    diagnostic_rows = []

    for image_index, row in image_frame.iterrows():
        relative_path = row[image_column]
        rgb = read_resized_rgb(
            image_root / Path(relative_path),
            config["image_descriptors"]["image_size"],
        )
        descriptors, diagnostics = extract_image_descriptor_values(config, rgb)
        feature_rows.append({image_column: relative_path, **descriptors})
        diagnostic_rows.append({image_column: relative_path, **diagnostics})
        completed = image_index + 1
        if completed % 100 == 0 or completed == len(image_frame):
            print(f"Extracted image descriptors: {completed}/{len(image_frame)}")

    feature_frame = pd.DataFrame(feature_rows)
    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    feature_columns = image_descriptor_columns(config)
    if not np.isfinite(
        feature_frame[feature_columns].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("The image-descriptor table contains NaN or infinity.")

    feature_dir = Path(experiment_dir) / "features"
    feature_path = feature_dir / "image_descriptors.csv"
    diagnostic_path = feature_dir / "mask_check.csv"
    check_path = feature_dir / "feature_check.csv"
    schema_path = feature_dir / "feature_schema.json"
    summary_path = feature_dir / "feature_extraction_summary.json"
    feature_frame.to_csv(feature_path, index=False)
    diagnostic_frame.to_csv(diagnostic_path, index=False)
    check_frame = feature_check_table(feature_frame, feature_columns)
    check_frame.to_csv(check_path, index=False)

    groups = image_descriptor_groups(config)
    configurations = {
        name: configuration_feature_columns(config, name)
        for name in config["image_descriptors"]["configurations"]
    }
    schema = {
        "identifier_column": image_column,
        "feature_count": len(feature_columns),
        "group_counts": {
            name: len(columns)
            for name, columns in groups.items()
        },
        "feature_groups": groups,
        "feature_columns": feature_columns,
        "configuration_counts": {
            name: len(columns)
            for name, columns in configurations.items()
        },
        "configurations": configurations,
        "glcm_offsets": [
            {
                "row_offset": row_offset,
                "column_offset": column_offset,
            }
            for row_offset, column_offset in glcm_offsets(config)
        ],
        "population_standard_deviation_ddof": 0,
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    foreground = diagnostic_frame["foreground_fraction_after_cleanup"]
    summary = {
        "images": len(feature_frame),
        "feature_count": len(feature_columns),
        "group_counts": {
            name: len(columns)
            for name, columns in groups.items()
        },
        "configuration_counts": {
            name: len(columns)
            for name, columns in configurations.items()
        },
        "fallback_masks": int(diagnostic_frame["fallback_used"].sum()),
        "constant_features": check_frame.loc[
            check_frame["constant"],
            "feature",
        ].tolist(),
        "foreground_fraction_minimum": float(foreground.min()),
        "foreground_fraction_median": float(foreground.median()),
        "foreground_fraction_maximum": float(foreground.max()),
        "feature_csv": project_relative_text(project_root, feature_path),
        "mask_check_csv": project_relative_text(project_root, diagnostic_path),
        "feature_check_csv": project_relative_text(project_root, check_path),
        "feature_schema": project_relative_text(project_root, schema_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(feature_columns)} descriptors for {len(feature_frame)} images.")
    print(
        "Configuration counts: "
        + ", ".join(
            f"{name}={len(columns)}"
            for name, columns in configurations.items()
        )
    )
    print(f"Feature table: {summary['feature_csv']}")
    print(f"Feature check: {summary['feature_check_csv']}")
    print(f"Mask check: {summary['mask_check_csv']}")
    return summary


def load_image_descriptor_schema(experiment_dir):
    path = Path(experiment_dir) / "features" / "feature_schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"Image-descriptor schema does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_image_descriptor_frame(experiment_dir):
    path = Path(experiment_dir) / "features" / "image_descriptors.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Image-descriptor table does not exist: {path}")
    return pd.read_csv(path)


