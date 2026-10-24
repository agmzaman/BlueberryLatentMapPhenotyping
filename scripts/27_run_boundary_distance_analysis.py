from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import yaml


def project_path(project_root, relative_path):
    text = str(relative_path).strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or PureWindowsPath(text).drive:
        raise ValueError("Use paths relative to the project root.")
    return (project_root / path).resolve()


def read_csv(project_root, relative_path, required_columns):
    path = project_path(project_root, relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {relative_path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {relative_path}: {missing}")
    if frame.empty:
        raise ValueError(f"Input CSV is empty: {relative_path}")
    return frame.loc[:, required_columns].copy()


def read_annotations(project_root, config_relative_path):
    config_path = project_path(project_root, config_relative_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_relative_path}")
    with config_path.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError("The configuration must contain a dataset section.")
    dataset = config["dataset"]
    for key in ["root", "annotation_csv"]:
        if not isinstance(dataset.get(key), str) or not dataset[key].strip():
            raise ValueError(f"Set dataset.{key} in {config_relative_path}.")
    root_text = dataset["root"].strip().replace("\\", "/")
    dataset_root = Path(root_text)
    if not dataset_root.is_absolute():
        dataset_root = project_path(project_root, root_text)
    columns = ["image_filename", "size_class", "relative_area"]
    try:
        return read_csv(dataset_root, dataset["annotation_csv"], columns)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Annotation CSV not found. Check dataset.root and dataset.annotation_csv in {config_relative_path}."
        ) from None


def normalize_image_id(value):
    text = str(value).strip().replace("\\", "/")
    path = Path(text)
    if not text or text == "." or path.is_absolute() or PureWindowsPath(text).drive or ".." in path.parts:
        raise ValueError("Image identifiers must be non-empty relative paths.")
    return path.as_posix()


def index_images(frame, image_column, label_columns, classes, source_name):
    frame = frame.copy()
    frame[image_column] = frame[image_column].map(normalize_image_id)
    if frame[image_column].duplicated().any():
        raise ValueError(f"Duplicate image identifiers in {source_name}.")
    for column in label_columns:
        if not frame[column].isin(classes).all():
            raise ValueError(f"Unknown or blank class labels in {source_name}: {column}")
    return frame.set_index(image_column).sort_index()


def read_boundaries(project_root, paths, classes):
    columns = ["lower_class", "upper_class", "fixed_boundary_relative_area"]
    frame = read_csv(project_root, paths["boundaries"], columns)
    if frame["lower_class"].tolist() != classes[:-1] or frame["upper_class"].tolist() != classes[1:]:
        raise ValueError("Boundary rows must follow the configured class order.")
    boundaries = pd.to_numeric(frame["fixed_boundary_relative_area"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(boundaries).all() or not ((boundaries > 0) & (boundaries < 1)).all():
        raise ValueError("Class boundaries must be finite relative areas between zero and one.")
    if not (np.diff(boundaries) > 0).all():
        raise ValueError("Class boundaries must be strictly increasing.")
    return boundaries


def prepare_test_data(project_root, paths, classes, boundaries):
    annotations = read_annotations(project_root, paths["config"])
    annotations = index_images(annotations, "image_filename", ["size_class"], classes, "annotations")
    areas = pd.to_numeric(annotations["relative_area"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(areas).all() or not ((areas >= 0) & (areas <= 1)).all():
        raise ValueError("Annotation relative areas must be finite values between zero and one.")
    derived_classes = np.asarray(classes)[np.searchsorted(boundaries, areas, side="left")]
    if not np.array_equal(derived_classes, annotations["size_class"].to_numpy()):
        raise ValueError("The saved boundaries do not reproduce the annotation class labels.")
    annotations["relative_area"] = areas

    test_split = read_csv(project_root, paths["test_split"], ["image_filename", "size_class"])
    test_split = index_images(test_split, "image_filename", ["size_class"], classes, "test split")

    predictions = read_csv(project_root, paths["predictions"], ["image_path", "true_class", "predicted_class"])
    predictions = index_images(predictions, "image_path", ["true_class", "predicted_class"], classes, "predictions")

    if set(predictions.index) != set(test_split.index):
        raise ValueError("Prediction image identifiers do not match the test split.")

    if not set(test_split.index).issubset(annotations.index):
        raise ValueError("Some test images are missing from the annotation CSV.")

    annotations = annotations.loc[test_split.index]
    predictions = predictions.loc[test_split.index]
    reference_labels = test_split["size_class"].to_numpy()

    if not np.array_equal(annotations["size_class"].to_numpy(), reference_labels):
        raise ValueError("Annotation labels differ from the test split.")

    if not np.array_equal(predictions["true_class"].to_numpy(), reference_labels):
        raise ValueError("Prediction reference labels differ from the test split.")

    data = predictions.loc[:, ["true_class", "predicted_class"]].copy()
    data["relative_area"] = annotations["relative_area"]
    return data


def summarize_boundary_distances(data, boundary, class_pair):
    rows = []
    for true_class, adjacent_class in [class_pair, class_pair[::-1]]:
        selected = data.loc[data["true_class"] == true_class]
        if selected.empty:
            raise ValueError(f"No test images found for {true_class}.")
        distances = (selected["relative_area"] - boundary).abs()
        predicted = selected["predicted_class"]
        adjacent_group = f"{true_class}_to_{adjacent_class}".lower().replace(" ", "_")
        groups = [
            ("correct", predicted == true_class),
            (adjacent_group, predicted == adjacent_class),
            ("other_error", ~predicted.isin([true_class, adjacent_class])),
        ]
        for group_name, mask in groups:
            values = distances.loc[mask]
            rows.append({
                "class_pair": f"{class_pair[0]} - {class_pair[1]}",
                "true_class": true_class,
                "prediction_group": group_name,
                "image_count": int(values.size),
                "median_boundary_distance": float(values.median()) if not values.empty else np.nan,
                "boundary_relative_area": float(boundary),
            })
    return pd.DataFrame(rows)


def save_summary(summary, project_root, paths):
    output_root = project_path(project_root, paths["output_directory"])

    if output_root == project_root or not output_root.is_relative_to(project_root):
        raise ValueError("The output directory must be inside the project.")

    filename = paths["summary_file"]

    if not filename or filename in [".", ".."] or any(character in filename for character in "/\\:"):
        raise ValueError("The summary filename must not contain a directory.")

    output_directory = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_directory.mkdir(parents=True, exist_ok=False)
    output_file = output_directory / filename
    summary.to_csv(output_file, index=False, mode="x", float_format="%.12f", na_rep="")
    return output_file


def main(project_root, paths, classes):
    project_root = Path(project_root).resolve()
    if len(classes) != 4 or len(set(classes)) != 4:
        raise ValueError("Provide the four distinct canopy-size classes in ascending order.")

    boundaries = read_boundaries(project_root, paths, classes)
    data = prepare_test_data(project_root, paths, classes, boundaries)
    summaries = []

    for index, boundary in enumerate(boundaries):
        class_pair = classes[index:index + 2]
        summaries.append(summarize_boundary_distances(data, float(boundary), class_pair))
    summary = pd.concat(summaries, ignore_index=True)
    output_file = save_summary(summary, project_root, paths)

    for name in ["config", "test_split", "predictions", "boundaries"]:
        print(f"{name}: {paths[name]}")

    print("annotations: dataset.root + dataset.annotation_csv from the configuration")
    print(f"Matched test images: {len(data)}")
    print(f"Adjacent class pairs: {len(summaries)}")
    print("Distance = absolute difference between relative canopy area and the fixed class boundary.")
    print("Distances are relative-area fractions, not percentages or physical distances.")
    print("Each pair is summarized separately; some images occur in two pairs.")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.10f}", na_rep=""))
    print(f"Saved: {output_file.relative_to(project_root).as_posix()}")
    print("Completed using saved predictions of C5.")

    return output_file


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    PATHS = {
        "config": "configs/config.yaml",
        "test_split": "data/splits/test.csv",
        "predictions": "experiments/run_20260915_135736/classifiers/C5/test_predictions.csv",
        "boundaries": "experiments/polygon_annotation_sensitivity/fixed_jenks_boundaries.csv",
        "output_directory": "experiments/boundary_distance",
        "summary_file": "boundary_distance_summary.csv",
    }

    CLASSES = ["Very Small", "Small", "Medium", "Large"]

    print("Summarize C5 boundary distances for all adjacent classes on the stratified-random test split.")
    main(PROJECT_ROOT, PATHS, CLASSES)

