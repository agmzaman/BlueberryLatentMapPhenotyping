import random
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

from .paths import (
    dataset_image_root,
    dataset_path,
    project_relative_text,
    session_split_check_path,
    session_fold_paths,
    split_paths,
)
from .runtime import data_loader_generator, seed_worker


def normalize_dataset_relative_path(raw_value, configured_root):
    raw_text = str(raw_value).strip().replace("\\", "/")
    if not raw_text:
        raise ValueError("An empty image path was found.")

    root = Path(configured_root).resolve()
    raw_path = Path(raw_text)

    if raw_path.is_absolute():
        try:
            relative = raw_path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"Image path is outside dataset.root: {raw_value}") from error
    else:
        parts = list(PurePosixPath(raw_text).parts)
        if parts and parts[0].lower() == root.parent.name.lower():
            parts = parts[1:]
        if parts and parts[0].lower() == root.name.lower():
            parts = parts[1:]
        relative = Path(*parts)

    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid dataset-relative image path: {raw_value}")
    return relative.as_posix()


def load_annotation_rows(config):
    annotation_path = dataset_path(config, config["dataset"]["annotation_csv"])
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]

    source = pd.read_csv(annotation_path, usecols=[image_column, label_column])
    if len(source) != int(config["dataset"]["expected_total"]):
        raise ValueError(
            f"Expected {config['dataset']['expected_total']} annotation rows "
            f"but found {len(source)}."
        )
    if source[image_column].duplicated().any():
        duplicates = source.loc[source[image_column].duplicated(), image_column].tolist()
        raise ValueError(f"Duplicate image paths were found: {duplicates[:5]}")

    root = dataset_image_root(config)
    source[image_column] = source[image_column].map(
        lambda value: normalize_dataset_relative_path(value, root)
    )

    configured_classes = set(config["dataset"]["classes"])
    observed_classes = set(source[label_column].dropna().unique())
    if observed_classes != configured_classes:
        raise ValueError(
            f"Observed classes {sorted(observed_classes)} do not match configured classes {sorted(configured_classes)}"
        )

    missing_images = [
        relative_path
        for relative_path in source[image_column]
        if not (root / Path(relative_path)).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} image files are missing. First entries: {missing_images[:5]}"
        )
    return source


def split_summary_frame(config, split_frames):
    rows = []
    label_column = config["dataset"]["label_column"]
    for split_name in ["train", "validation", "test"]:
        frame = split_frames[split_name]
        counts = frame[label_column].value_counts()
        for class_name in config["dataset"]["classes"]:
            count = int(counts.get(class_name, 0))
            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "count": count,
                    "percentage": 100.0 * count / len(frame),
                }
            )
    return pd.DataFrame(rows)


def validate_split_frames(config, source_frame, split_frames):
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]

    source_paths = set(source_frame[image_column])
    split_sets = {name: set(frame[image_column]) for name, frame in split_frames.items()}
    if split_sets["train"] & split_sets["validation"]:
        raise ValueError("Train and validation splits overlap.")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Train and test splits overlap.")
    if split_sets["validation"] & split_sets["test"]:
        raise ValueError("Validation and test splits overlap.")

    combined_paths = split_sets["train"] | split_sets["validation"] | split_sets["test"]
    if combined_paths != source_paths:
        raise ValueError("The split union does not match the source annotation rows.")

    for frame in split_frames.values():
        if frame[label_column].isna().any():
            raise ValueError("A split contains a missing label.")

    return split_summary_frame(config, split_frames)


def prepare_splits(config, project_root):
    source = load_annotation_rows(config)
    label_column = config["dataset"]["label_column"]

    train_fraction = float(config["split"]["train_fraction"])
    validation_fraction = float(config["split"]["validation_fraction"])
    test_fraction = float(config["split"]["test_fraction"])
    seed = int(config["split"]["seed"])
    stratified = bool(config["split"]["stratified"])
    source_stratification = source[label_column] if stratified else None

    train_frame, temporary_frame = train_test_split(
        source,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=source_stratification,
    )
    validation_share = validation_fraction / (validation_fraction + test_fraction)
    temporary_stratification = (
        temporary_frame[label_column]
        if stratified
        else None
    )
    validation_frame, test_frame = train_test_split(
        temporary_frame,
        train_size=validation_share,
        random_state=seed,
        shuffle=True,
        stratify=temporary_stratification,
    )

    split_frames = {
        "train": train_frame.reset_index(drop=True),
        "validation": validation_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
    }
    summary = validate_split_frames(config, source, split_frames)
    paths = split_paths(config, project_root)

    for split_name, frame in split_frames.items():
        paths[split_name].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths[split_name], index=False)
        relative_output = project_relative_text(project_root, paths[split_name])
        print(f"Saved {split_name}: {len(frame)} rows: {relative_output}")

    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(paths["summary"], index=False)
    relative_summary = project_relative_text(project_root, paths["summary"])
    print(f"Saved split summary: {relative_summary}")
    return split_frames, summary


def session_from_image_path(image_path):
    parts = PurePosixPath(str(image_path).replace("\\", "/")).parts
    if len(parts) < 2:
        raise ValueError(
            f"Image path does not contain a session directory: {image_path}"
        )
    return parts[0]


def session_split_check_frame(config, source):
    session_column = config["session_held_out"]["session_column"]
    label_column = config["dataset"]["label_column"]
    sessions = sorted(source[session_column].unique())
    index = pd.MultiIndex.from_product(
        [sessions, config["dataset"]["classes"]],
        names=[session_column, label_column],
    )
    counts = (
        source.groupby([session_column, label_column])
        .size()
        .reindex(index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    totals = source[session_column].value_counts()
    counts["session_total"] = counts[session_column].map(totals).astype(int)
    counts["percentage"] = 100.0 * counts["count"] / counts["session_total"]
    return counts


def fallback_session_split(config, frame):
    label_column = config["dataset"]["label_column"]
    validation_fraction = float(
        config["session_held_out"]["validation_fraction"]
    )
    seed = int(config["session_held_out"]["seed"])
    training_parts = []
    validation_parts = []

    for class_index, class_name in enumerate(config["dataset"]["classes"]):
        class_frame = frame.loc[frame[label_column] == class_name]
        if class_frame.empty:
            continue

        shuffled = class_frame.sample(
            frac=1.0,
            random_state=seed + class_index,
        )
        validation_count = int(round(len(shuffled) * validation_fraction))
        if len(shuffled) > 1:
            validation_count = max(1, validation_count)
            validation_count = min(validation_count, len(shuffled) - 1)
        else:
            validation_count = 0

        validation_parts.append(shuffled.iloc[:validation_count])
        training_parts.append(shuffled.iloc[validation_count:])

    training = pd.concat(training_parts, ignore_index=True)
    if validation_parts:
        validation = pd.concat(validation_parts, ignore_index=True)
    else:
        validation = frame.iloc[0:0].copy()

    training = training.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if not validation.empty:
        validation = validation.sample(
            frac=1.0,
            random_state=seed,
        ).reset_index(drop=True)
    return training, validation, "classwise_training_retention"


def split_held_in_session(config, frame):
    label_column = config["dataset"]["label_column"]
    train_fraction = float(config["session_held_out"]["train_fraction"])
    validation_fraction = float(
        config["session_held_out"]["validation_fraction"]
    )
    seed = int(config["session_held_out"]["seed"])

    try:
        training, validation = train_test_split(
            frame,
            train_size=train_fraction,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=frame[label_column],
        )
    except ValueError:
        return fallback_session_split(config, frame)

    return (
        training.reset_index(drop=True),
        validation.reset_index(drop=True),
        "stratified",
    )


def ordered_session_manifest(config, frame):
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]
    session_column = config["session_held_out"]["session_column"]
    columns = [image_column, label_column, session_column]
    return (
        frame[columns]
        .sort_values([session_column, image_column], kind="stable")
        .reset_index(drop=True)
    )


def validate_session_held_out_fold(
    config,
    source,
    split_frames,
    held_out_session,
):
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]
    session_column = config["session_held_out"]["session_column"]
    source_paths = set(source[image_column])
    split_sets = {
        name: set(frame[image_column])
        for name, frame in split_frames.items()
    }

    if split_sets["train"] & split_sets["validation"]:
        raise ValueError("Session-held-out train and validation splits overlap.")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Session-held-out train and test splits overlap.")
    if split_sets["validation"] & split_sets["test"]:
        raise ValueError("Session-held-out validation and test splits overlap.")

    combined_paths = (
        split_sets["train"]
        | split_sets["validation"]
        | split_sets["test"]
    )
    if combined_paths != source_paths:
        raise ValueError(
            "The session-held-out split union does not match the annotation rows."
        )

    expected_test_paths = set(
        source.loc[
            source[session_column] == held_out_session,
            image_column,
        ]
    )
    if split_sets["test"] != expected_test_paths:
        raise ValueError("The test split does not match the held-out session.")

    held_in_sessions = set(source[session_column]) - {held_out_session}
    if set(split_frames["train"][session_column]) - held_in_sessions:
        raise ValueError("The training split contains the held-out session.")
    if set(split_frames["validation"][session_column]) - held_in_sessions:
        raise ValueError("The validation split contains the held-out session.")
    if set(split_frames["test"][session_column]) != {held_out_session}:
        raise ValueError("The test split contains a held-in session.")

    for frame in split_frames.values():
        if frame[[image_column, label_column, session_column]].isna().any().any():
            raise ValueError("A session-held-out split contains a missing value.")

    for session in held_in_sessions:
        session_source = source.loc[source[session_column] == session]
        session_training = split_frames["train"].loc[
            split_frames["train"][session_column] == session
        ]
        observed_classes = set(session_source[label_column])
        training_classes = set(session_training[label_column])
        if not observed_classes.issubset(training_classes):
            raise ValueError(
                f"Training does not retain every observed class for session {session}."
            )


def session_held_out_summary_frame(
    config,
    source,
    fold_name,
    held_out_session,
    split_frames,
    allocation_rules,
):
    label_column = config["dataset"]["label_column"]
    session_column = config["session_held_out"]["session_column"]
    sessions = sorted(source[session_column].unique())
    rows = []

    for split_name in ["train", "validation", "test"]:
        frame = split_frames[split_name]
        split_total = len(frame)
        counts = frame.groupby([session_column, label_column]).size()
        for session in sessions:
            if session == held_out_session:
                allocation_rule = "outer_test"
            else:
                allocation_rule = allocation_rules[session]
            for class_name in config["dataset"]["classes"]:
                count = int(counts.get((session, class_name), 0))
                rows.append(
                    {
                        "fold": fold_name,
                        "held_out_session": held_out_session,
                        "split": split_name,
                        session_column: session,
                        label_column: class_name,
                        "count": count,
                        "split_total": split_total,
                        "percentage": 100.0 * count / split_total,
                        "allocation_rule": allocation_rule,
                    }
                )
    return pd.DataFrame(rows)


def prepare_session_held_out_splits(config, project_root):
    source = load_annotation_rows(config)
    image_column = config["dataset"]["image_column"]
    session_column = config["session_held_out"]["session_column"]
    settings = config["session_held_out"]
    source[session_column] = source[image_column].map(session_from_image_path)

    observed_sessions = set(source[session_column])
    configured_sessions = set(settings["folds"].values())
    missing_sessions = configured_sessions - observed_sessions
    if missing_sessions:
        raise ValueError(
            f"Configured held-out sessions were not found: {sorted(missing_sessions)}"
        )

    session_check = session_split_check_frame(config, source)
    check_output = session_split_check_path(config, project_root)
    check_output.parent.mkdir(parents=True, exist_ok=True)
    session_check.to_csv(check_output, index=False)
    relative_check = project_relative_text(project_root, check_output)
    print(f"Saved session split check -> {relative_check}")

    summaries = {}
    for fold_name, held_out_session in settings["folds"].items():
        test = source.loc[source[session_column] == held_out_session].copy()
        held_in = source.loc[source[session_column] != held_out_session]
        training_parts = []
        validation_parts = []
        allocation_rules = {}

        for session in sorted(held_in[session_column].unique()):
            session_frame = held_in.loc[held_in[session_column] == session]
            training, validation, rule = split_held_in_session(
                config,
                session_frame,
            )
            training_parts.append(training)
            validation_parts.append(validation)
            allocation_rules[session] = rule

        split_frames = {
            "train": ordered_session_manifest(
                config,
                pd.concat(training_parts, ignore_index=True),
            ),
            "validation": ordered_session_manifest(
                config,
                pd.concat(validation_parts, ignore_index=True),
            ),
            "test": ordered_session_manifest(config, test),
        }
        validate_session_held_out_fold(
            config,
            source,
            split_frames,
            held_out_session,
        )
        summary = session_held_out_summary_frame(
            config,
            source,
            fold_name,
            held_out_session,
            split_frames,
            allocation_rules,
        )
        paths = session_fold_paths(config, project_root, fold_name)

        for split_name, frame in split_frames.items():
            output_path = paths[split_name]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, index=False)
            relative_output = project_relative_text(project_root, output_path)
            print(
                f"Saved {fold_name} {split_name}: "
                f"{len(frame)} rows -> {relative_output}"
            )

        summary.to_csv(paths["summary"], index=False)
        relative_summary = project_relative_text(project_root, paths["summary"])
        print(f"Saved {fold_name} summary -> {relative_summary}")
        summaries[fold_name] = summary

    return session_check, summaries


def random_right_angle(image):
    angle = random.choice([0, 90, 180, 270])
    return transform_functional.rotate(image, angle)


def image_transforms(config):
    image_size = int(config["vae"]["input_shape"][1])
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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return train_transform, evaluation_transform


class ImageDataset(Dataset):
    def __init__(self, csv_path, root, image_column, label_column, classes, transform):
        self.frame = pd.read_csv(csv_path)
        self.root = Path(root)
        self.image_column = image_column
        self.label_column = label_column
        self.class_to_index = {label: index for index, label in enumerate(classes)}
        self.transform = transform

        required = {image_column, label_column}
        if not required.issubset(self.frame.columns):
            raise KeyError(f"Missing split columns: {sorted(required - set(self.frame.columns))}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        relative_path = normalize_dataset_relative_path(row[self.image_column], self.root)
        image_path = self.root / Path(relative_path)
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        tensor = self.transform(image)
        label = self.class_to_index[row[self.label_column]]
        return tensor, label, relative_path


class ArrayDataset(Dataset):
    def __init__(self, features, labels, image_paths):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.image_paths = list(image_paths)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.image_paths[index]


def loader_options(config, seed, workers, shuffle):
    return {
        "batch_size": int(config["vae"]["batch_size"]),
        "shuffle": shuffle,
        "num_workers": int(workers),
        "pin_memory": bool(config["runtime"]["pin_memory"]),
        "worker_init_fn": seed_worker,
        "generator": data_loader_generator(seed),
    }

