import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import yaml


def project_path(project_root, relative_path):
    text = str(relative_path).replace("\\", "/")
    path = Path(text)
    if not text.strip() or path.is_absolute() or PureWindowsPath(text).drive or ".." in path.parts:
        raise ValueError("Paths must be non-empty and relative to the project root.")
    root = Path(project_root).resolve()
    resolved = (root / path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("The path must remain inside the project.")
    return resolved


def relative_text(project_root, path):
    return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()


def load_uncertainty_settings(settings_path):
    with Path(settings_path).open("r", encoding="utf-8-sig") as stream:
        settings = yaml.safe_load(stream)
    validate_settings(settings)
    return settings


def validate_settings(settings):
    classes = settings["classes"]
    if len(classes) != 4 or len(set(classes)) != 4 or any(not isinstance(x, str) or not x for x in classes):
        raise ValueError("Exactly four distinct class names are required.")
    bootstrap = settings["bootstrap"]
    fixed = {
        "sampling_unit": "image",
        "replacement": True,
        "stratify_by": "true_class",
        "sample_size": "observed_class_counts",
        "paired": True,
        "interval_method": "percentile",
        "interval_sides": "two_sided",
        "quantile_method": "linear",
        "zero_division": 0,
    }
    for key, expected in fixed.items():
        if type(bootstrap[key]) is not type(expected) or bootstrap[key] != expected:
            raise ValueError(f"Unsupported bootstrap setting: {key}. Required value: {expected}")
    if type(bootstrap["resamples"]) is not int or bootstrap["resamples"] < 2:
        raise ValueError("resamples must be an integer of at least 2.")
    if not 0 < float(bootstrap["confidence_level"]) < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if settings["metrics"] != ["accuracy", "balanced_accuracy", "macro_f1"]:
        raise ValueError("Use accuracy, balanced_accuracy and macro_f1 in that order.")
    if not 0 < float(settings["metric_check_tolerance"]) <= 0.000001:
        raise ValueError("metric_check_tolerance must be positive and no larger than 0.000001.")
    for split in settings["splits"].values():
        if type(split["seed"]) is not int or split["seed"] < 0:
            raise ValueError("Each split seed must be a non-negative integer.")
    for analysis in settings["analyses"].values():
        models = analysis["models"]
        splits = analysis["splits"]
        pairs = analysis["comparisons"]
        if not models or len(models) != len(set(models)) or not set(models) <= set(settings["models"]):
            raise ValueError("Analysis models must be distinct configured model names.")
        if not splits or len(splits) != len(set(splits)) or not set(splits) <= set(settings["splits"]):
            raise ValueError("Analysis splits must be distinct configured split names.")
        if not pairs or any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(models) for pair in pairs):
            raise ValueError("Each comparison must contain two different selected models.")
        if len({tuple(pair) for pair in pairs}) != len(pairs):
            raise ValueError("Duplicate comparisons are not allowed.")
        for name in models:
            if not set(splits) <= set(settings["models"][name]["directories"]):
                raise ValueError(f"Missing input directories for {name}.")
    output = settings["output"]
    if type(output["save_resamples"]) is not bool:
        raise ValueError("save_resamples must be true or false.")
    names = list(output["files"].values())
    if len(names) != len(set(names)):
        raise ValueError("Output filenames must be distinct.")
    for name in names:
        if not isinstance(name, str) or not name.strip() or name in [".", ".."] or any(c in name for c in "/\\:"):
            raise ValueError("Output filenames must be simple filenames without directories.")


def normalized_image_id(value):
    text = str(value).strip().replace("\\", "/")
    if not text or Path(text).is_absolute() or PureWindowsPath(text).drive or ".." in Path(text).parts:
        raise ValueError("Image identifiers must be non-empty relative paths.")
    return Path(text).as_posix()


def read_checked_csv(path, columns, image_column, label_columns, classes):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    frame = frame.loc[:, columns].copy()
    if frame.empty:
        raise ValueError("An input CSV is empty.")
    frame[image_column] = frame[image_column].map(normalized_image_id)
    if frame[image_column].duplicated().any():
        raise ValueError("Duplicate image identifiers are present.")
    for name in label_columns:
        if not frame[name].isin(classes).all():
            raise ValueError(f"Unknown or blank class labels in {name}.")
    return frame.set_index(image_column).sort_index()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calculate_metrics(true_labels, predicted_labels, class_count):
    cells = class_count * true_labels + predicted_labels
    matrix = np.bincount(cells, minlength=class_count * class_count).reshape(class_count, class_count)
    support = matrix.sum(axis=1)
    if (support == 0).any():
        raise ValueError("Every true class must be represented.")
    correct = np.diag(matrix).astype(np.float64)
    f1_denominator = support + matrix.sum(axis=0)
    f1 = np.divide(2.0 * correct, f1_denominator, out=np.zeros(class_count), where=f1_denominator > 0)
    return np.array([correct.sum() / matrix.sum(), np.mean(correct / support), np.mean(f1)])


def load_split_inputs(settings, project_root, analysis_name, split_name):
    split_columns = settings["columns"]["test_split"]
    prediction_columns = settings["columns"]["predictions"]
    classes = settings["classes"]
    encoding = {name: index for index, name in enumerate(classes)}
    manifest_path = project_path(project_root, settings["splits"][split_name]["test_manifest"])
    manifest = read_checked_csv(
        manifest_path,
        [split_columns["image"], split_columns["reference_class"]],
        split_columns["image"], [split_columns["reference_class"]], classes,
    )
    true_labels = manifest[split_columns["reference_class"]].map(encoding).to_numpy(dtype=np.int64)
    if set(true_labels) != set(range(len(classes))):
        raise ValueError(f"Missing true classes in {split_name}.")
    predictions = {}
    observed = {}
    checks = []
    for model_name in settings["analyses"][analysis_name]["models"]:
        model = settings["models"][model_name]
        directory = project_path(project_root, model["directories"][split_name])
        prediction_path = project_path(project_root, relative_text(project_root, directory / model["prediction_file"]))
        metrics_path = project_path(project_root, relative_text(project_root, directory / model["metrics_file"]))
        required = [model["image_column"], prediction_columns["reference_class"], prediction_columns["predicted_class"]]
        frame = read_checked_csv(
            prediction_path, required, model["image_column"],
            [prediction_columns["reference_class"], prediction_columns["predicted_class"]], classes,
        )
        if set(frame.index) != set(manifest.index):
            raise ValueError(f"Prediction identifiers do not match the test manifest: {split_name}, {model_name}")
        frame = frame.loc[manifest.index]
        if not np.array_equal(frame[prediction_columns["reference_class"]].to_numpy(), manifest[split_columns["reference_class"]].to_numpy()):
            raise ValueError(f"True labels differ from the test manifest: {split_name}, {model_name}")
        predicted = frame[prediction_columns["predicted_class"]].map(encoding).to_numpy(dtype=np.int64)
        values = calculate_metrics(true_labels, predicted, len(classes))
        with metrics_path.open("r", encoding="utf-8-sig") as stream:
            saved = json.load(stream)["metrics"]["test"]
        reference = np.array([saved[name] for name in settings["metrics"]], dtype=np.float64)
        if not np.isfinite(reference).all() or ((reference < 0) | (reference > 1)).any():
            raise ValueError(f"Invalid saved metrics: {split_name}, {model_name}")
        if int(saved["samples"]) != len(manifest):
            raise ValueError(f"Saved test sample count differs: {split_name}, {model_name}")
        if not np.allclose(values, reference, atol=settings["metric_check_tolerance"], rtol=0):
            raise ValueError(f"Recomputed metrics do not match saved test metrics: {split_name}, {model_name}")
        predictions[model_name] = predicted
        observed[model_name] = values
        checks.append({
            "split": split_name, "model": model_name, "test_images": len(manifest),
            "test_manifest": relative_text(project_root, manifest_path),
            "prediction_file": relative_text(project_root, prediction_path),
            "metrics_file": relative_text(project_root, metrics_path),
            "manifest_sha256": file_hash(manifest_path),
            "predictions_sha256": file_hash(prediction_path),
            "metrics_sha256": file_hash(metrics_path),
            "maximum_metric_difference": float(np.max(np.abs(values - reference))),
            "identifiers_match": True, "labels_match": True, "metrics_match": True,
        })
    return {"true_labels": true_labels, "predictions": predictions, "observed": observed, "checks": checks}


def draw_stratified_sample(generator, class_indices):
    return np.concatenate([generator.choice(indices, size=len(indices), replace=True) for indices in class_indices])


def bootstrap_split(data, settings, seed):
    true_labels = data["true_labels"]
    class_count = len(settings["classes"])
    indices = [np.flatnonzero(true_labels == index) for index in range(class_count)]
    generator = np.random.Generator(np.random.PCG64(seed))
    count = settings["bootstrap"]["resamples"]
    samples = {name: np.empty((count, len(settings["metrics"]))) for name in data["predictions"]}
    for repetition in range(count):
        sampled = draw_stratified_sample(generator, indices)
        for model_name, predicted in data["predictions"].items():
            samples[model_name][repetition] = calculate_metrics(true_labels[sampled], predicted[sampled], class_count)
    return samples


def percentile_interval(values, bootstrap):
    tail = (1.0 - float(bootstrap["confidence_level"])) / 2.0
    return np.quantile(values, [tail, 1.0 - tail], method=bootstrap["quantile_method"])


def summarize_split(data, samples, settings, analysis_name, split_name):
    model_rows = []
    pair_rows = []
    bootstrap = settings["bootstrap"]
    common = {
        "split": split_name, "test_images": len(data["true_labels"]),
        "resamples": bootstrap["resamples"], "seed": settings["splits"][split_name]["seed"],
        "nominal_confidence_level": bootstrap["confidence_level"],
        "interval_method": bootstrap["interval_method"],
        "interpretation": "exploratory_image_level_resampling",
    }
    for model_name in samples:
        for index, metric in enumerate(settings["metrics"]):
            lower, upper = percentile_interval(samples[model_name][:, index], bootstrap)
            model_rows.append({
                **common, "model": model_name, "metric": metric,
                "estimate_percent": 100.0 * data["observed"][model_name][index],
                "lower_percent": 100.0 * lower, "upper_percent": 100.0 * upper,
                "zero_width_interval": bool(lower == upper),
            })
    for first, second in settings["analyses"][analysis_name]["comparisons"]:
        differences = samples[first] - samples[second]
        observed = data["observed"][first] - data["observed"][second]
        for index, metric in enumerate(settings["metrics"]):
            lower, upper = percentile_interval(differences[:, index], bootstrap)
            pair_rows.append({
                **common, "first_model": first, "second_model": second, "metric": metric,
                "difference_percentage_points": 100.0 * observed[index],
                "lower_percentage_points": 100.0 * lower, "upper_percentage_points": 100.0 * upper,
                "zero_width_interval": bool(lower == upper),
            })
    return model_rows, pair_rows


def resample_frame(samples, split_name, metrics):
    frames = []
    for model_name, values in samples.items():
        frame = pd.DataFrame(100.0 * values, columns=[name + "_percent" for name in metrics])
        frame.insert(0, "resample", np.arange(1, len(frame) + 1))
        frame.insert(0, "model", model_name)
        frame.insert(0, "split", split_name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def message(text, log):
    print(text)
    log.write(text + "\n")
    log.flush()


def run_uncertainty(settings, project_root, analysis_name):
    validate_settings(settings)
    analysis = settings["analyses"][analysis_name]
    prepared = {}
    for split_name in analysis["splits"]:
        print(f"Checking saved test predictions: {split_name}")
        prepared[split_name] = load_split_inputs(settings, project_root, analysis_name, split_name)
    base = project_path(project_root, settings["output"]["directory"])
    if not analysis_name or any(c in analysis_name for c in "/\\:") or analysis_name in [".", ".."]:
        raise ValueError("The analysis name must be a simple directory name.")
    output = project_path(project_root, relative_text(project_root, base / analysis_name / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")))
    output.mkdir(parents=True, exist_ok=False)
    files = {key: output / name for key, name in settings["output"]["files"].items()}
    with files["settings"].open("x", encoding="utf-8") as stream:
        yaml.safe_dump(settings, stream, sort_keys=False, allow_unicode=True)
    checks = [row for data in prepared.values() for row in data["checks"]]
    pd.DataFrame(checks).to_csv(files["input_checks"], index=False, mode="x")
    analysis_scope = ("Image-level paired bootstrap analysis of saved test predictions, stratified by reference class.")

    model_rows = []
    pair_rows = []
    supports = {}
    with files["log"].open("x", encoding="utf-8") as log:
        message(analysis_scope, log)
        message(f"Analysis: {analysis_name}; resamples per split: {settings['bootstrap']['resamples']}", log)
        message("Paired differences are first model minus second model, in percentage points.", log)
        message(f"Output: {relative_text(project_root, output)}", log)
        for split_name, data in prepared.items():
            seed = settings["splits"][split_name]["seed"]
            support = np.bincount(data["true_labels"], minlength=len(settings["classes"]))
            supports[split_name] = dict(zip(settings["classes"], support.tolist()))
            message(f"{split_name}: images={len(data['true_labels'])}; seed={seed}; class counts={support.tolist()}", log)
            samples = bootstrap_split(data, settings, seed)
            model_part, pair_part = summarize_split(data, samples, settings, analysis_name, split_name)
            model_rows.extend(model_part)
            pair_rows.extend(pair_part)
            if settings["output"]["save_resamples"]:
                append = files["resamples"].exists()
                frame = resample_frame(samples, split_name, settings["metrics"])
                frame.to_csv(files["resamples"], index=False, mode="a" if append else "x", header=not append)
            message(f"Completed resampling: {split_name}", log)
        pd.DataFrame(model_rows).to_csv(files["model_intervals"], index=False, mode="x")
        pd.DataFrame(pair_rows).to_csv(files["paired_differences"], index=False, mode="x")
        summary = {
            "status": "completed", "analysis": analysis_name, "class_order": settings["classes"],
            "class_counts": supports, "bootstrap": settings["bootstrap"],
            "seeds": {name: settings["splits"][name]["seed"] for name in analysis["splits"]},
            "image_order": "sorted normalized test-manifest image identifiers",
            "difference_direction": "first_model_minus_second_model",
            "score_units": "percent", "difference_units": "percentage_points",
            "model_interval_rows": len(model_rows), "paired_interval_rows": len(pair_rows),
            "scope_warning": analysis_scope, "inputs": checks,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyyaml": yaml.__version__},
            "module_sha256": file_hash(__file__),
            "files": {key: relative_text(project_root, path) for key, path in files.items() if path.exists() or key == "summary"},
        }
        with files["summary"].open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        message(f"Saved {len(model_rows)} model intervals and {len(pair_rows)} paired-difference intervals.", log)
        message("Completed without training models or modifying input files.", log)
    return output
