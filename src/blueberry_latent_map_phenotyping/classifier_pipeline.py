import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import ArrayDataset
from .features import load_feature_schema
from .models import PlantSizeClassifier
from .runtime import data_loader_generator, seed_worker, set_reproducibility


def load_feature_frames(config, run_dir):
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]
    frames = {}
    for split_name in ["train", "validation", "test"]:
        path = Path(run_dir) / "features" / f"{split_name}_features.csv"
        if not path.exists():
            raise FileNotFoundError(f"Feature CSV does not exist: {path}")
        frame = pd.read_csv(path)
        required = {image_column, label_column}
        if not required.issubset(frame.columns):
            raise KeyError(f"Missing columns in {path}: {sorted(required - set(frame.columns))}")
        frames[split_name] = frame
    return frames


def encode_labels(label_values, class_names):
    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }
    unknown = sorted(set(label_values) - set(class_to_index))
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}")
    return np.asarray(
        [class_to_index[value] for value in label_values],
        dtype=np.int64,
    )


def prepare_arrays(config, frames, feature_columns, labels_by_split, output_dir):
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]
    class_names = config["dataset"]["classes"]

    arrays = {}
    for split_name, frame in frames.items():
        missing_features = [
            column
            for column in feature_columns
            if column not in frame.columns
        ]
        if missing_features:
            raise KeyError(
                f"{split_name} is missing feature columns: {missing_features[:5]}"
            )

        features = frame[feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"{split_name} features contain NaN or infinity.")

        label_values = (
            frame[label_column].astype(str).to_numpy()
            if labels_by_split is None
            else np.asarray(labels_by_split[split_name]).astype(str)
        )
        if len(label_values) != len(frame):
            raise ValueError(f"{split_name} label override has the wrong length.")

        arrays[split_name] = {
            "features": features,
            "labels": encode_labels(label_values, class_names),
            "paths": frame[image_column].astype(str).tolist(),
        }

    if bool(config["classifier"]["standardize_features"]):
        scaler = StandardScaler()
        arrays["train"]["features"] = scaler.fit_transform(
            arrays["train"]["features"]
        ).astype(np.float32)
        for split_name in ["validation", "test"]:
            arrays[split_name]["features"] = scaler.transform(
                arrays[split_name]["features"]
            ).astype(np.float32)
        joblib.dump(scaler, Path(output_dir) / "feature_scaler.joblib")
    else:
        metadata = {"standardized": False}
        path = Path(output_dir) / "feature_scaler.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split_name in arrays:
        if not np.isfinite(arrays[split_name]["features"]).all():
            raise ValueError(f"Scaled {split_name} features contain NaN or infinity.")
    return arrays


def build_classifier_loaders(config, arrays, seed):
    loaders = {}
    for split_name in ["train", "validation", "test"]:
        dataset = ArrayDataset(
            arrays[split_name]["features"],
            arrays[split_name]["labels"],
            arrays[split_name]["paths"],
        )
        workers = (
            int(config["runtime"]["train_workers"])
            if split_name == "train"
            else int(config["runtime"]["evaluation_workers"])
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=int(config["classifier"]["batch_size"]),
            shuffle=split_name == "train",
            num_workers=workers,
            pin_memory=bool(config["runtime"]["pin_memory"]),
            worker_init_fn=seed_worker,
            generator=data_loader_generator(seed),
        )
    return loaders


def build_classifier_optimizer(config, model):
    optimizer_config = config["classifier"]["optimizer"]
    optimizer_types = {
        "Adam": Adam,
        "AdamW": AdamW,
    }
    optimizer_type = optimizer_types[optimizer_config["type"]]
    return optimizer_type(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def build_classifier_scheduler(config, optimizer):
    scheduler_config = config["classifier"]["scheduler"]
    return ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config["mode"],
        factor=float(scheduler_config["lr_update_factor"]),
        patience=int(scheduler_config["lr_update_patience"]),
        threshold=float(scheduler_config["threshold"]),
        min_lr=float(scheduler_config["lr_min"]),
    )


def classifier_scheduler_value(monitor, validation_loss):
    values = {
        "validation_loss": validation_loss,
    }
    return values[monitor]


def classifier_loss_epoch(model, loader, criterion, optimizer, device, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for features, labels, _ in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            predicted = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct_count += int((predicted == labels).sum().item())
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("An empty classifier loader was supplied.")
    return total_loss / sample_count, correct_count / sample_count


def predict_classifier(model, loader, criterion, device):
    model.eval()
    predictions = []
    targets = []
    probabilities = []
    image_paths = []
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for features, labels, paths in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            probability = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probability, dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
            probabilities.extend(probability.cpu().numpy().tolist())
            image_paths.extend(paths)

    if sample_count == 0:
        raise ValueError("An empty classifier loader was supplied.")
    return {
        "loss": total_loss / sample_count,
        "predictions": np.asarray(predictions, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.int64),
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "image_paths": image_paths,
    }


def scalar_metrics(targets, predictions):
    macro = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        targets,
        predictions,
        average="weighted",
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
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "matthews_correlation": float(
            matthews_corrcoef(targets, predictions)
        ),
        "cohen_kappa": float(cohen_kappa_score(targets, predictions)),
    }


def save_split_evaluation(
    split_name,
    prediction_output,
    class_names,
    output_dir,
):
    targets = prediction_output["targets"]
    predictions = prediction_output["predictions"]
    labels = list(range(len(class_names)))
    metrics = scalar_metrics(targets, predictions)
    metrics["loss"] = float(prediction_output["loss"])
    metrics["samples"] = int(len(targets))

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in class_names],
        columns=[f"predicted_{name}" for name in class_names],
    )
    matrix_frame.to_csv(
        Path(output_dir) / f"{split_name}_confusion_matrix.csv"
    )

    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        Path(output_dir) / f"{split_name}_classification_report.csv"
    )

    probability_columns = [
        f"probability_{class_name}"
        for class_name in class_names
    ]
    prediction_frame = pd.DataFrame(
        prediction_output["probabilities"],
        columns=probability_columns,
    )
    prediction_frame.insert(
        0,
        "predicted_class",
        [class_names[index] for index in predictions],
    )
    prediction_frame.insert(
        0,
        "true_class",
        [class_names[index] for index in targets],
    )
    prediction_frame.insert(
        0,
        "image_path",
        prediction_output["image_paths"],
    )
    prediction_frame.to_csv(
        Path(output_dir) / f"{split_name}_predictions.csv",
        index=False,
    )
    return metrics


def train_classifier_run(
    config,
    frames,
    feature_columns,
    output_dir,
    seed,
    labels_by_split,
    keep_model,
    device,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducibility(seed)

    metadata = {
        "seed": int(seed),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "standardize_features": bool(
            config["classifier"]["standardize_features"]
        ),
    }
    (output_dir / "feature_selection.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    arrays = prepare_arrays(
        config,
        frames,
        feature_columns,
        labels_by_split,
        output_dir,
    )
    loaders = build_classifier_loaders(config, arrays, seed)

    model = PlantSizeClassifier(
        input_dimension=len(feature_columns),
        hidden_dimensions=config["classifier"]["hidden_dimensions"],
        class_count=len(config["dataset"]["classes"]),
        dropout=float(config["classifier"]["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_classifier_optimizer(config, model)
    scheduler = build_classifier_scheduler(config, optimizer)
    scheduler_monitor = config["classifier"]["scheduler"]["monitor"]

    checkpoint_path = output_dir / "best_classifier.pt"
    history_path = output_dir / "training_history.csv"
    epoch_count = int(config["classifier"]["epochs"])
    patience = int(config["classifier"]["early_stopping_patience"])
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epoch_count + 1):
        train_loss, train_accuracy = classifier_loss_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            True,
        )
        validation_loss, validation_accuracy = classifier_loss_epoch(
            model,
            loaders["validation"],
            criterion,
            optimizer,
            device,
            False,
        )
        scheduler_value = classifier_scheduler_value(
            scheduler_monitor,
            validation_loss,
        )
        scheduler.step(scheduler_value)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": learning_rate,
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "model_state": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{epoch_count} | "
            f"train={train_loss:.6f}/{train_accuracy:.4f} | "
            f"validation={validation_loss:.6f}/{validation_accuracy:.4f} | "
            f"lr={learning_rate:.8f}"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])

    split_metrics = {}
    for split_name in ["train", "validation", "test"]:
        prediction_output = predict_classifier(
            model,
            loaders[split_name],
            criterion,
            device,
        )
        split_metrics[split_name] = save_split_evaluation(
            split_name,
            prediction_output,
            config["dataset"]["classes"],
            output_dir,
        )

    summary = {
        "seed": int(seed),
        "feature_count": len(feature_columns),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["validation_loss"]),
        "metrics": split_metrics,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if not keep_model:
        checkpoint_path.unlink(missing_ok=True)

    return summary


def train_classifier_ablations(config, run_dir, device):
    frames = load_feature_frames(config, run_dir)
    schema = load_feature_schema(run_dir)
    seed = int(config["classifier"]["seed"])
    summaries = {}

    for run_name in config["classifier"]["runs"]:
        print(f"\nTraining {run_name}")
        feature_columns = schema["run_columns"][run_name]
        output_dir = Path(run_dir) / "classifiers" / run_name
        summaries[run_name] = train_classifier_run(
            config,
            frames,
            feature_columns,
            output_dir,
            seed,
            None,
            True,
            device,
        )

    summary_path = Path(run_dir) / "classifiers" / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    return summaries
