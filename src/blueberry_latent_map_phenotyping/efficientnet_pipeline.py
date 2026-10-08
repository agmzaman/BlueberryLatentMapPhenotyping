import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .data import ImageDataset, random_right_angle
from .paths import (
    dataset_image_root,
    project_relative_text,
    session_fold_paths,
    split_paths,
)
from .runtime import data_loader_generator, seed_worker, set_reproducibility


def efficientnet_weights(config):
    available_weights = {
        "IMAGENET1K_V1": EfficientNet_B0_Weights.IMAGENET1K_V1,
    }
    return available_weights[config["efficientnet"]["pretrained_weights"]]


def efficientnet_transforms(config):
    image_size = int(config["efficientnet"]["input_shape"][1])
    mean = (0.485, 0.456, 0.406)
    standard_deviation = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomApply(
                [transforms.Lambda(random_right_angle)],
                p=0.25,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomResizedCrop(
                        size=(image_size, image_size),
                        scale=(0.9, 1.0),
                        ratio=(1.0, 1.0),
                    )
                ],
                p=0.25,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.15)],
                p=0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    return train_transform, evaluation_transform


def efficientnet_split_paths(config, project_root, run_name):
    if run_name == "stratified_random":
        paths = split_paths(config, project_root)
    else:
        paths = session_fold_paths(config, project_root, run_name)
    return {
        "train": paths["train"],
        "validation": paths["validation"],
        "test": paths["test"],
    }


def validate_efficientnet_split_files(config, project_root):
    for run_name in config["efficientnet"]["runs"]:
        paths = efficientnet_split_paths(config, project_root, run_name)
        for split_name, path in paths.items():
            if not path.is_file():
                relative_path = project_relative_text(project_root, path)
                raise FileNotFoundError(
                    f"Missing {run_name} {split_name} split: {relative_path}"
                )


def build_image_loader(config, csv_path, transform, seed, workers, shuffle):
    dataset = ImageDataset(
        csv_path,
        dataset_image_root(config),
        config["dataset"]["image_column"],
        config["dataset"]["label_column"],
        config["dataset"]["classes"],
        transform,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["efficientnet"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(workers),
        pin_memory=bool(config["runtime"]["pin_memory"]),
        worker_init_fn=seed_worker,
        generator=data_loader_generator(seed),
    )


def build_efficientnet_loaders(config, project_root, run_name, seed):
    paths = efficientnet_split_paths(config, project_root, run_name)
    for split_name, path in paths.items():
        if not path.is_file():
            relative_path = project_relative_text(project_root, path)
            raise FileNotFoundError(f"Missing {split_name} split: {relative_path}")

    train_transform, evaluation_transform = efficientnet_transforms(config)
    train_loader = build_image_loader(
        config,
        paths["train"],
        train_transform,
        seed,
        config["runtime"]["train_workers"],
        True,
    )
    validation_loader = build_image_loader(
        config,
        paths["validation"],
        evaluation_transform,
        seed,
        config["runtime"]["evaluation_workers"],
        False,
    )
    evaluation_loaders = {}
    for split_name, path in paths.items():
        evaluation_loaders[split_name] = build_image_loader(
            config,
            path,
            evaluation_transform,
            seed,
            config["runtime"]["evaluation_workers"],
            False,
        )
    return train_loader, validation_loader, evaluation_loaders


def configure_efficientnet_model(config, model):
    class_count = len(config["dataset"]["classes"])
    input_dimension = int(model.classifier[1].in_features)
    model.classifier = nn.Sequential(
        nn.Dropout(
            p=float(config["efficientnet"]["classifier_dropout"]),
            inplace=True,
        ),
        nn.Linear(input_dimension, class_count),
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    unfreeze_count = int(config["efficientnet"]["unfreeze_last_stages"])
    feature_stages = list(model.features.children())
    if unfreeze_count > len(feature_stages):
        raise ValueError("The configured unfreeze count exceeds the model stages.")
    for stage in feature_stages[-unfreeze_count:]:
        for parameter in stage.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model


def build_efficientnet_model(config):
    model = efficientnet_b0(weights=efficientnet_weights(config))
    return configure_efficientnet_model(config, model)


def set_frozen_stages_to_evaluation(config, model):
    unfreeze_count = int(config["efficientnet"]["unfreeze_last_stages"])
    feature_stages = list(model.features.children())
    for stage in feature_stages[:-unfreeze_count]:
        stage.eval()


def build_efficientnet_optimizer(config, model):
    unfreeze_count = int(config["efficientnet"]["unfreeze_last_stages"])
    feature_stages = list(model.features.children())
    backbone_parameters = [
        parameter
        for stage in feature_stages[-unfreeze_count:]
        for parameter in stage.parameters()
        if parameter.requires_grad
    ]
    classifier_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]
    settings = config["efficientnet"]["optimizer"]
    return AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(settings["backbone_lr"]),
            },
            {
                "params": classifier_parameters,
                "lr": float(settings["classifier_lr"]),
            },
        ],
        weight_decay=float(settings["weight_decay"]),
    )


def build_efficientnet_scheduler(config, optimizer):
    settings = config["efficientnet"]["scheduler"]
    return ReduceLROnPlateau(
        optimizer,
        mode=settings["mode"],
        factor=float(settings["lr_update_factor"]),
        patience=int(settings["lr_update_patience"]),
        threshold=float(settings["threshold"]),
        threshold_mode=settings["threshold_mode"],
        min_lr=float(settings["lr_min"]),
    )


def efficientnet_epoch(config, model, loader, criterion, optimizer, device, training):
    if training:
        model.train()
        set_frozen_stages_to_evaluation(config, model)
    else:
        model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct_count += int((predictions == labels).sum().item())
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("An empty EfficientNet loader was supplied.")
    return total_loss / sample_count, correct_count / sample_count


def save_efficientnet_checkpoint(config, model, epoch, validation_loss, path):
    torch.save(
        {
            "architecture": config["efficientnet"]["architecture"],
            "pretrained_weights": config["efficientnet"]["pretrained_weights"],
            "classes": list(config["dataset"]["classes"]),
            "epoch": int(epoch),
            "validation_loss": float(validation_loss),
            "model_state": model.state_dict(),
        },
        path,
    )


def train_efficientnet_model(
    config,
    project_root,
    run_dir,
    model,
    train_loader,
    validation_loader,
    device,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = build_efficientnet_optimizer(config, model)
    scheduler = build_efficientnet_scheduler(config, optimizer)
    checkpoint_path = Path(run_dir) / "checkpoints" / "best_model.pt"
    history_path = Path(run_dir) / "training_history.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_count = int(config["efficientnet"]["epochs"])
    patience = int(config["efficientnet"]["early_stopping_patience"])
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epoch_count + 1):
        train_loss, train_accuracy = efficientnet_epoch(
            config,
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
        )
        validation_loss, validation_accuracy = efficientnet_epoch(
            config,
            model,
            validation_loader,
            criterion,
            None,
            device,
            False,
        )

        model_saved = validation_loss < best_validation_loss
        if model_saved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_efficientnet_checkpoint(
                config,
                model,
                epoch,
                validation_loss,
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        scheduler.step(validation_loss)
        backbone_learning_rate = float(optimizer.param_groups[0]["lr"])
        classifier_learning_rate = float(optimizer.param_groups[1]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "backbone_learning_rate": backbone_learning_rate,
                "classifier_learning_rate": classifier_learning_rate,
                "model_saved": model_saved,
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch:03d}/{epoch_count} | "
            f"train={train_loss:.6f}/{train_accuracy:.4f} | "
            f"validation={validation_loss:.6f}/{validation_accuracy:.4f} | "
            f"backbone_lr={backbone_learning_rate:.8f} | "
            f"classifier_lr={classifier_learning_rate:.8f} | "
            f"saved={model_saved}"
        )
        if model_saved:
            relative_checkpoint = project_relative_text(project_root, checkpoint_path)
            print(f"Saved best checkpoint: {relative_checkpoint}")
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, criterion, checkpoint, history


def predict_efficientnet(model, loader, criterion, device):
    model.eval()
    predictions = []
    targets = []
    probabilities = []
    image_paths = []
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
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
        raise ValueError("An empty EfficientNet loader was supplied.")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if not np.allclose(probability_array.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("EfficientNet class probabilities do not sum to one.")
    return {
        "loss": total_loss / sample_count,
        "predictions": np.asarray(predictions, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.int64),
        "probabilities": probability_array,
        "image_paths": image_paths,
    }


def efficientnet_metrics(prediction_output):
    targets = prediction_output["targets"]
    predictions = prediction_output["predictions"]
    macro = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "loss": float(prediction_output["loss"]),
        "samples": int(len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(macro[2]),
    }


def probability_column(class_name):
    normalized = str(class_name).strip().lower().replace(" ", "_")
    return f"probability_{normalized}"


def save_predictions(config, split_name, prediction_output, run_dir):
    class_names = config["dataset"]["classes"]
    probability_columns = [
        probability_column(class_name)
        for class_name in class_names
    ]
    frame = pd.DataFrame(
        prediction_output["probabilities"],
        columns=probability_columns,
    )
    frame.insert(
        0,
        "predicted_class",
        [class_names[index] for index in prediction_output["predictions"]],
    )
    frame.insert(
        0,
        "true_class",
        [class_names[index] for index in prediction_output["targets"]],
    )
    frame.insert(
        0,
        config["dataset"]["image_column"],
        prediction_output["image_paths"],
    )

    settings = config["efficientnet"]["predictions"]
    output_path = Path(run_dir) / settings["directory"] / settings["files"][split_name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def save_test_details(config, prediction_output, run_dir):
    class_names = config["dataset"]["classes"]
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        prediction_output["targets"],
        prediction_output["predictions"],
        labels=labels,
        zero_division=0,
    )
    class_frame = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    class_frame.to_csv(Path(run_dir) / "test_class_metrics.csv", index=False)

    matrix = confusion_matrix(
        prediction_output["targets"],
        prediction_output["predictions"],
        labels=labels,
    )
    matrix_frame = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )
    matrix_frame.to_csv(
        Path(run_dir) / "test_confusion_matrix.csv",
        index_label="true_class",
    )


def evaluate_efficientnet(config, model, loaders, criterion, run_dir, device):
    metrics = {}
    for split_name in ["train", "validation", "test"]:
        prediction_output = predict_efficientnet(
            model,
            loaders[split_name],
            criterion,
            device,
        )
        save_predictions(
            config,
            split_name,
            prediction_output,
            run_dir,
        )
        metrics[split_name] = efficientnet_metrics(prediction_output)
        if split_name == "test":
            save_test_details(config, prediction_output, run_dir)
    return metrics


def release_training_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_efficientnet_run(config, project_root, run_name, run_dir, device):
    seed = int(config["efficientnet"]["seed"])
    set_reproducibility(seed)
    train_loader, validation_loader, evaluation_loaders = (
        build_efficientnet_loaders(
            config,
            project_root,
            run_name,
            seed,
        )
    )

    model = build_efficientnet_model(config).to(device)
    print("Initialized a fresh model, classifier, optimizer, and scheduler.")
    print(
        f"Rows: train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)}, "
        f"test={len(evaluation_loaders['test'].dataset)}"
    )
    model, criterion, checkpoint, history = train_efficientnet_model(
        config,
        project_root,
        run_dir,
        model,
        train_loader,
        validation_loader,
        device,
    )
    metrics = evaluate_efficientnet(
        config,
        model,
        evaluation_loaders,
        criterion,
        run_dir,
        device,
    )

    held_out_session = config["session_held_out"]["folds"].get(run_name)
    summary = {
        "run": run_name,
        "held_out_session": held_out_session,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "epochs_trained": len(history),
        "best_validation_loss": float(checkpoint["validation_loss"]),
        "metrics": metrics,
    }
    summary_path = Path(run_dir) / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    del model
    del criterion
    del train_loader
    del validation_loader
    del evaluation_loaders
    release_training_memory()
    return summary


def efficientnet_summary_row(summary):
    test_metrics = summary["metrics"]["test"]
    return {
        "run": summary["run"],
        "held_out_session": summary["held_out_session"],
        "seed": summary["seed"],
        "best_epoch": summary["best_epoch"],
        "epochs_trained": summary["epochs_trained"],
        "best_validation_loss": summary["best_validation_loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_samples": test_metrics["samples"],
    }


def train_efficientnet_experiments(config, project_root, experiment_dir, device):
    summaries = {}
    rows = []
    summary_path = (
        Path(experiment_dir)
        / config["efficientnet"]["experiment_summary_file"]
    )

    for run_name in config["efficientnet"]["runs"]:
        release_training_memory()
        run_dir = Path(experiment_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        print(f"\nStarting independent run: {run_name}")
        try:
            summary = train_efficientnet_run(
                config,
                project_root,
                run_name,
                run_dir,
                device,
            )
        finally:
            release_training_memory()

        summaries[run_name] = summary
        rows.append(efficientnet_summary_row(summary))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(
            f"Completed {run_name}: "
            f"test_accuracy={summary['metrics']['test']['accuracy']:.4f}, "
            f"test_macro_f1={summary['metrics']['test']['macro_f1']:.4f}"
        )

    relative_summary = project_relative_text(project_root, summary_path)
    print(f"Saved experiment summary: {relative_summary}")
    return summaries

