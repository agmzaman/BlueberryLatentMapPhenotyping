import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import normalize_dataset_relative_path
from .image_descriptors import (
    configuration_feature_columns,
    image_descriptor_columns,
    load_image_descriptor_frame,
    load_image_descriptor_schema,
)
from .image_descriptor_settings import validate_image_descriptor_settings
from .paths import (
    dataset_image_root,
    project_relative_text,
    session_fold_paths,
    split_paths,
)


def descriptor_split_paths(config, project_root, run_name):
    if run_name == "stratified_random":
        configured_paths = split_paths(config, project_root)
    else:
        configured_paths = session_fold_paths(
            config,
            project_root,
            run_name,
        )
    return {
        split_name: configured_paths[split_name]
        for split_name in ["train", "validation", "test"]
    }


def validate_descriptor_split_files(config, project_root):
    validate_image_descriptor_settings(config["image_descriptors"])
    for run_name in config["image_descriptors"]["runs"]:
        paths = descriptor_split_paths(config, project_root, run_name)
        for split_name, path in paths.items():
            if not path.is_file():
                relative_path = project_relative_text(project_root, path)
                raise FileNotFoundError(
                    f"Missing {run_name} {split_name} split: {relative_path}"
                )


def load_descriptor_run_frames(
    config,
    project_root,
    run_name,
    descriptor_frame,
    feature_columns,
):
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]
    image_root = dataset_image_root(config)
    paths = descriptor_split_paths(config, project_root, run_name)
    frames = {}

    for split_name, path in paths.items():
        manifest = pd.read_csv(
            path,
            usecols=[image_column, label_column],
        )
        manifest[image_column] = manifest[image_column].map(
            lambda value: normalize_dataset_relative_path(value, image_root)
        )
        if manifest[image_column].duplicated().any():
            raise ValueError(
                f"Duplicate image identifiers in {run_name} {split_name}."
            )
        frame = manifest.merge(
            descriptor_frame,
            on=image_column,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = frame.loc[frame["_merge"] != "both", image_column].tolist()
        if missing:
            raise ValueError(
                f"Missing descriptors in {run_name} {split_name}: {missing[:5]}"
            )
        frame = frame.drop(columns="_merge")
        values = frame[feature_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite descriptors in {run_name} {split_name}."
            )
        frames[split_name] = frame
    return frames


def encode_descriptor_labels(config, values):
    class_names = config["dataset"]["classes"]
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


def prepare_descriptor_arrays(
    config,
    frames,
    feature_columns,
    run_dir,
):
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]
    training_values = frames["train"][feature_columns].to_numpy(
        dtype=np.float64
    )
    constant_columns = [
        feature_name
        for index, feature_name in enumerate(feature_columns)
        if np.ptp(training_values[:, index]) == 0.0
    ]
    if constant_columns:
        raise ValueError(
            "Zero-variance training descriptors were found: "
            f"{constant_columns}"
        )
    scaler = StandardScaler()
    scaler.fit(training_values)
    joblib.dump(scaler, Path(run_dir) / "feature_scaler.joblib")

    arrays = {}
    for split_name, frame in frames.items():
        features = scaler.transform(
            frame[feature_columns].to_numpy(dtype=np.float64)
        )
        if not np.isfinite(features).all():
            raise ValueError(
                f"Scaled descriptors contain NaN or infinity in {split_name}."
            )
        arrays[split_name] = {
            "features": features,
            "targets": encode_descriptor_labels(
                config,
                frame[label_column].astype(str).tolist(),
            ),
            "paths": frame[image_column].astype(str).tolist(),
        }
    return arrays


def descriptor_metrics(targets, predictions):
    macro = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "samples": int(len(targets)),
    }


def svm_candidates(config):
    settings = config["image_descriptors"]["rbf_svm"]
    return [
        {
            "C": float(c_value),
            "gamma": gamma_value,
        }
        for c_value, gamma_value in product(
            settings["c_values"],
            settings["gamma_values"],
        )
    ]


def random_forest_candidates(config):
    settings = config["image_descriptors"]["random_forest"]
    return [
        {
            "n_estimators": int(settings["n_estimators"]),
            "max_depth": (
                None if max_depth is None else int(max_depth)
            ),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
        }
        for max_depth, min_samples_leaf, max_features in product(
            settings["max_depth"],
            settings["min_samples_leaf"],
            settings["max_features"],
        )
    ]


def build_descriptor_estimator(config, algorithm, parameters):
    settings = config["image_descriptors"]
    if algorithm == "rbf_svm":
        return SVC(
            C=parameters["C"],
            gamma=parameters["gamma"],
            kernel="rbf",
            class_weight=settings["rbf_svm"]["class_weight"],
            probability=False,
            decision_function_shape="ovr",
        )
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["max_features"],
            class_weight=settings["random_forest"]["class_weight"],
            random_state=int(settings["seed"]),
            n_jobs=max(1, int(config["runtime"]["train_workers"])),
        )
    raise ValueError(f"Unsupported image-descriptor classifier: {algorithm}")


def descriptor_candidates(config, algorithm):
    if algorithm == "rbf_svm":
        return svm_candidates(config)
    if algorithm == "random_forest":
        return random_forest_candidates(config)
    raise ValueError(f"Unsupported image-descriptor classifier: {algorithm}")


def selection_score(metrics):
    return (
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        metrics["accuracy"],
    )


def select_descriptor_estimator(config, algorithm, arrays, output_dir):
    candidates = descriptor_candidates(config, algorithm)
    rows = []
    best_score = None
    best_index = None
    best_estimator = None
    best_parameters = None

    for candidate_index, parameters in enumerate(candidates, start=1):
        estimator = build_descriptor_estimator(
            config,
            algorithm,
            parameters,
        )
        estimator.fit(
            arrays["train"]["features"],
            arrays["train"]["targets"],
        )
        predictions = estimator.predict(arrays["validation"]["features"])
        metrics = descriptor_metrics(
            arrays["validation"]["targets"],
            predictions,
        )
        score = selection_score(metrics)
        row = {
            "candidate": candidate_index,
            **parameters,
            "validation_accuracy": metrics["accuracy"],
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
        }
        rows.append(row)
        print(
            f"{algorithm} candidate {candidate_index:02d}/{len(candidates):02d} | "
            f"validation_macro_f1={metrics['macro_f1']:.4f}"
        )
        if best_score is None or score > best_score:
            best_score = score
            best_index = candidate_index
            best_estimator = estimator
            best_parameters = parameters

    for row in rows:
        row["selected"] = row["candidate"] == best_index
    search_path = Path(output_dir) / "hyperparameter_search.csv"
    pd.DataFrame(rows).to_csv(search_path, index=False)
    print(f"Selected {algorithm} parameters: {best_parameters}")
    return best_estimator, best_parameters


def normalized_class_name(class_name):
    return str(class_name).strip().lower().replace(" ", "_")


def ordered_estimator_scores(config, estimator, features, algorithm):
    class_count = len(config["dataset"]["classes"])
    expected_classes = np.arange(class_count, dtype=np.int64)
    observed_classes = np.asarray(estimator.classes_, dtype=np.int64)
    if not np.array_equal(observed_classes, expected_classes):
        raise ValueError(
            "The fitted descriptor classifier does not contain all configured classes."
        )
    if algorithm == "rbf_svm":
        scores = np.asarray(estimator.decision_function(features))
        return scores, "decision"
    probabilities = np.asarray(estimator.predict_proba(features))
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Random Forest probabilities do not sum to one.")
    return probabilities, "probability"


def save_descriptor_predictions(
    config,
    algorithm,
    split_name,
    estimator,
    split_arrays,
    output_dir,
):
    class_names = config["dataset"]["classes"]
    predictions = estimator.predict(split_arrays["features"])
    scores, score_prefix = ordered_estimator_scores(
        config,
        estimator,
        split_arrays["features"],
        algorithm,
    )
    score_columns = [
        f"{score_prefix}_{normalized_class_name(class_name)}"
        for class_name in class_names
    ]
    frame = pd.DataFrame(scores, columns=score_columns)
    frame.insert(
        0,
        "predicted_class",
        [class_names[index] for index in predictions],
    )
    frame.insert(
        0,
        "true_class",
        [class_names[index] for index in split_arrays["targets"]],
    )
    frame.insert(
        0,
        config["dataset"]["image_column"],
        split_arrays["paths"],
    )
    frame.to_csv(
        Path(output_dir) / f"{split_name}_predictions.csv",
        index=False,
    )
    return predictions


def save_descriptor_test_details(config, targets, predictions, output_dir):
    class_names = config["dataset"]["classes"]
    labels = list(range(len(class_names)))
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        Path(output_dir) / "test_classification_report.csv"
    )
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )
    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(
        Path(output_dir) / "test_confusion_matrix.csv",
        index_label="true_class",
    )


def evaluate_descriptor_estimator(
    config,
    algorithm,
    estimator,
    arrays,
    output_dir,
):
    metrics = {}
    for split_name in ["train", "validation", "test"]:
        predictions = save_descriptor_predictions(
            config,
            algorithm,
            split_name,
            estimator,
            arrays[split_name],
            output_dir,
        )
        metrics[split_name] = descriptor_metrics(
            arrays[split_name]["targets"],
            predictions,
        )
        if split_name == "test":
            save_descriptor_test_details(
                config,
                arrays[split_name]["targets"],
                predictions,
                output_dir,
            )
    return metrics


def held_out_session(config, run_name):
    return config["session_held_out"]["folds"].get(run_name)


def train_descriptor_estimator(
    config,
    project_root,
    experiment_dir,
    run_name,
    configuration_name,
    algorithm,
    arrays,
    feature_columns,
):
    output_dir = (
        Path(experiment_dir)
        / run_name
        / algorithm
        / configuration_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator, selected_parameters = select_descriptor_estimator(
        config,
        algorithm,
        arrays,
        output_dir,
    )
    joblib.dump(estimator, output_dir / "model.joblib")
    metrics = evaluate_descriptor_estimator(
        config,
        algorithm,
        estimator,
        arrays,
        output_dir,
    )
    summary = {
        "run": run_name,
        "held_out_session": held_out_session(config, run_name),
        "configuration": configuration_name,
        "feature_groups": config["image_descriptors"]["configurations"][
            configuration_name
        ],
        "algorithm": algorithm,
        "seed": int(config["image_descriptors"]["seed"]),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "standardized_from_training_only": True,
        "selection_metric": config["image_descriptors"]["selection_metric"],
        "selected_parameters": selected_parameters,
        "metrics": metrics,
    }
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    relative_summary = project_relative_text(project_root, summary_path)
    print(
        f"Completed {run_name} {configuration_name} {algorithm}: "
        f"test_accuracy={metrics['test']['accuracy']:.4f}, "
        f"test_macro_f1={metrics['test']['macro_f1']:.4f}"
    )
    print(f"Metrics: {relative_summary}")
    return summary


def descriptor_summary_row(summary):
    test_metrics = summary["metrics"]["test"]
    return {
        "run": summary["run"],
        "held_out_session": summary["held_out_session"],
        "configuration": summary["configuration"],
        "feature_groups": "+".join(summary["feature_groups"]),
        "algorithm": summary["algorithm"],
        "seed": summary["seed"],
        "feature_count": summary["feature_count"],
        "selected_parameters": json.dumps(summary["selected_parameters"]),
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_samples": test_metrics["samples"],
    }


def validate_descriptor_schema(config, schema):
    validate_image_descriptor_settings(config["image_descriptors"])
    expected = int(config["image_descriptors"]["expected_feature_count"])
    feature_columns = schema["feature_columns"]
    if len(feature_columns) != expected:
        raise ValueError(
            f"Expected {expected} descriptor columns but found {len(feature_columns)}."
        )
    forbidden = {
        config["dataset"]["image_column"],
        config["dataset"]["label_column"],
        config["session_held_out"]["session_column"],
        "polygon_points",
        "polygon_area",
        "relative_area",
    }
    overlap = sorted(set(feature_columns) & forbidden)
    if overlap:
        raise ValueError(f"Forbidden model features were found: {overlap}")
    expected_columns = image_descriptor_columns(config)
    if feature_columns != expected_columns:
        raise ValueError(
            "The saved descriptor order does not match the current definition."
        )
    expected_configurations = {
        name: configuration_feature_columns(config, name)
        for name in config["image_descriptors"]["configurations"]
    }
    if schema.get("configurations") != expected_configurations:
        raise ValueError(
            "The saved descriptor configurations do not match the current definition."
        )
    return feature_columns


def train_image_descriptor_classifiers(
    config,
    project_root,
    experiment_dir,
):
    descriptor_frame = load_image_descriptor_frame(experiment_dir)
    schema = load_image_descriptor_schema(experiment_dir)
    feature_columns = validate_descriptor_schema(config, schema)
    rows = []
    summary_path = Path(experiment_dir) / "summaries" / "classifier_results.csv"

    for run_name in config["image_descriptors"]["runs"]:
        print(f"\nPreparing independent descriptor run: {run_name}")
        frames = load_descriptor_run_frames(
            config,
            project_root,
            run_name,
            descriptor_frame,
            feature_columns,
        )
        for configuration_name in config["image_descriptors"]["configurations"]:
            selected_columns = configuration_feature_columns(
                config,
                configuration_name,
            )
            print(
                f"Configuration {configuration_name}: "
                f"{len(selected_columns)} descriptors."
            )
            scaler_dir = (
                Path(experiment_dir)
                / run_name
                / "scalers"
                / configuration_name
            )
            scaler_dir.mkdir(parents=True, exist_ok=True)
            arrays = prepare_descriptor_arrays(
                config,
                frames,
                selected_columns,
                scaler_dir,
            )
            for algorithm in ["rbf_svm", "random_forest"]:
                print(f"Selecting and training {algorithm}.")
                summary = train_descriptor_estimator(
                    config,
                    project_root,
                    experiment_dir,
                    run_name,
                    configuration_name,
                    algorithm,
                    arrays,
                    selected_columns,
                )
                rows.append(descriptor_summary_row(summary))
                pd.DataFrame(rows).to_csv(summary_path, index=False)

    relative_summary = project_relative_text(project_root, summary_path)
    print(f"Saved classifier results: {relative_summary}")
    return rows


def load_descriptor_result_summaries(config, experiment_dir):
    summaries = []
    for run_name in config["image_descriptors"]["runs"]:
        for algorithm in ["rbf_svm", "random_forest"]:
            for configuration_name in config["image_descriptors"][
                "configurations"
            ]:
                path = (
                    Path(experiment_dir)
                    / run_name
                    / algorithm
                    / configuration_name
                    / "metrics_summary.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Descriptor metrics summary does not exist: {path}"
                    )
                summaries.append(
                    json.loads(path.read_text(encoding="utf-8"))
                )
    return summaries


def article_result_table(frame, algorithm, configuration_name):
    columns = [
        "configuration",
        "feature_count",
        "run",
        "held_out_session",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_samples",
    ]
    result = frame.loc[
        (frame["algorithm"] == algorithm)
        & (frame["configuration"] == configuration_name),
        columns,
    ].copy()
    held_out = result.loc[result["run"].isin(["S1", "S2", "S3"])]
    if len(held_out) != 3:
        raise ValueError(
            f"Expected three held-out rows for "
            f"{configuration_name} {algorithm}."
        )
    feature_count = int(result["feature_count"].iloc[0])
    mean_row = {
        "configuration": configuration_name,
        "feature_count": feature_count,
        "run": "session_held_out_mean",
        "held_out_session": "",
        "test_accuracy": held_out["test_accuracy"].mean(),
        "test_balanced_accuracy": held_out["test_balanced_accuracy"].mean(),
        "test_macro_f1": held_out["test_macro_f1"].mean(),
        "test_samples": held_out["test_samples"].sum(),
    }
    standard_deviation_row = {
        "configuration": configuration_name,
        "feature_count": feature_count,
        "run": "session_held_out_sample_sd",
        "held_out_session": "",
        "test_accuracy": held_out["test_accuracy"].std(ddof=1),
        "test_balanced_accuracy": held_out["test_balanced_accuracy"].std(ddof=1),
        "test_macro_f1": held_out["test_macro_f1"].std(ddof=1),
        "test_samples": np.nan,
    }
    return pd.concat(
        [
            result,
            pd.DataFrame([mean_row, standard_deviation_row]),
        ],
        ignore_index=True,
    )


def combined_article_result_table(frame, config, algorithm):
    tables = [
        article_result_table(frame, algorithm, configuration_name)
        for configuration_name in config["image_descriptors"][
            "configurations"
        ]
    ]
    return pd.concat(tables, ignore_index=True)


def mean_sd_rows(frame, config):
    rows = []
    metrics = {
        "accuracy": "test_accuracy",
        "balanced_accuracy": "test_balanced_accuracy",
        "macro_f1": "test_macro_f1",
    }
    for configuration_name in config["image_descriptors"]["configurations"]:
        for algorithm in ["rbf_svm", "random_forest"]:
            held_out = frame.loc[
                (frame["configuration"] == configuration_name)
                & (frame["algorithm"] == algorithm)
                & frame["run"].isin(["S1", "S2", "S3"])
            ]
            if len(held_out) != 3:
                raise ValueError(
                    f"Expected three held-out rows for "
                    f"{configuration_name} {algorithm}."
                )
            for metric_name, column_name in metrics.items():
                mean = float(held_out[column_name].mean())
                sample_sd = float(held_out[column_name].std(ddof=1))
                rows.append(
                    {
                        "configuration": configuration_name,
                        "algorithm": algorithm,
                        "metric": metric_name,
                        "mean": mean,
                        "sample_sd": sample_sd,
                        "mean_sd": f"{mean:.4f} +/- {sample_sd:.4f}",
                    }
                )
    return rows


def summarize_image_descriptor_results(config, project_root, experiment_dir):
    validate_image_descriptor_settings(config["image_descriptors"])
    summaries = load_descriptor_result_summaries(config, experiment_dir)
    rows = [descriptor_summary_row(summary) for summary in summaries]
    frame = pd.DataFrame(rows)
    summary_dir = Path(experiment_dir) / "summaries"

    all_results_path = summary_dir / "classifier_results.csv"
    rbf_path = summary_dir / "rbf_svm_main_results.csv"
    forest_path = summary_dir / "random_forest_results.csv"
    parameter_path = summary_dir / "selected_hyperparameters.csv"
    mean_sd_path = summary_dir / "session_held_out_mean_sd.csv"

    frame.to_csv(all_results_path, index=False)
    combined_article_result_table(frame, config, "rbf_svm").to_csv(
        rbf_path,
        index=False,
    )
    combined_article_result_table(frame, config, "random_forest").to_csv(
        forest_path,
        index=False,
    )
    frame[
        [
            "run",
            "held_out_session",
            "configuration",
            "feature_count",
            "algorithm",
            "selected_parameters",
        ]
    ].to_csv(parameter_path, index=False)
    pd.DataFrame(mean_sd_rows(frame, config)).to_csv(
        mean_sd_path,
        index=False,
    )

    for path in [rbf_path, forest_path, parameter_path, mean_sd_path]:
        print(f"Saved summary: {project_relative_text(project_root, path)}")
    return rows


