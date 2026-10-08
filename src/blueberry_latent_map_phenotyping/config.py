from copy import deepcopy
from pathlib import Path

import yaml


def load_config(config_path, validate_dataset):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_config(config, config_path.parents[1], validate_dataset)
    return config


def merge_saved_config(current_config, saved_config):
    merged = deepcopy(current_config)
    for key, value in saved_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_saved_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_run_config(
    config_path,
    current_config,
    project_root,
    validate_dataset,
):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        saved_config = yaml.safe_load(file)

    config = merge_saved_config(current_config, saved_config)
    validate_config(config, project_root, validate_dataset)
    return config


def save_config(config, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def require_keys(mapping, keys, section_name):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing keys in {section_name}: {missing}")


def validate_relative_path(value, field_name):
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"{field_name} must be project-relative: {value}")
    if ".." in path.parts:
        raise ValueError(f"{field_name} cannot leave its configured root: {value}")


def validate_optimizer(optimizer, section_name):
    require_keys(
        optimizer,
        ["type", "lr", "weight_decay"],
        section_name,
    )
    if optimizer["type"] not in {"Adam", "AdamW"}:
        raise ValueError(f"Unsupported optimizer in {section_name}: {optimizer['type']}")
    if float(optimizer["lr"]) <= 0.0:
        raise ValueError(f"{section_name}.lr must be positive.")
    if float(optimizer["weight_decay"]) < 0.0:
        raise ValueError(f"{section_name}.weight_decay cannot be negative.")


def validate_scheduler(scheduler, section_name, allowed_monitors):
    require_keys(
        scheduler,
        [
            "name",
            "monitor",
            "mode",
            "threshold",
            "lr_min",
            "lr_update_factor",
            "lr_update_patience",
        ],
        section_name,
    )
    if scheduler["name"] != "ReduceLROnPlateau":
        raise ValueError(f"Unsupported scheduler in {section_name}: {scheduler['name']}")
    if scheduler["monitor"] not in allowed_monitors:
        raise ValueError(
            f"Unsupported monitor in {section_name}: {scheduler['monitor']}"
        )
    if scheduler["mode"] != "min":
        raise ValueError(f"{section_name}.mode must be min for a loss monitor.")
    if float(scheduler["threshold"]) < 0.0:
        raise ValueError(f"{section_name}.threshold cannot be negative.")
    if float(scheduler["lr_min"]) < 0.0:
        raise ValueError(f"{section_name}.lr_min cannot be negative.")
    factor = float(scheduler["lr_update_factor"])
    if not 0.0 < factor < 1.0:
        raise ValueError(f"{section_name}.lr_update_factor must be between 0 and 1.")
    if int(scheduler["lr_update_patience"]) < 0:
        raise ValueError(f"{section_name}.lr_update_patience cannot be negative.")


def validate_efficientnet_optimizer(optimizer):
    require_keys(
        optimizer,
        ["type", "backbone_lr", "classifier_lr", "weight_decay"],
        "efficientnet.optimizer",
    )
    if optimizer["type"] != "AdamW":
        raise ValueError("efficientnet.optimizer.type must be AdamW.")
    if float(optimizer["backbone_lr"]) <= 0.0:
        raise ValueError("efficientnet.optimizer.backbone_lr must be positive.")
    if float(optimizer["classifier_lr"]) <= 0.0:
        raise ValueError("efficientnet.optimizer.classifier_lr must be positive.")
    if float(optimizer["weight_decay"]) < 0.0:
        raise ValueError("efficientnet.optimizer.weight_decay cannot be negative.")


def validate_efficientnet(efficientnet):
    require_keys(
        efficientnet,
        [
            "architecture",
            "pretrained_weights",
            "input_shape",
            "normalization",
            "seed",
            "epochs",
            "batch_size",
            "early_stopping_patience",
            "classifier_dropout",
            "unfreeze_last_stages",
            "checkpoint_monitor",
            "optimizer",
            "scheduler",
            "runs",
            "predictions",
            "experiment_summary_file",
        ],
        "efficientnet",
    )
    require_keys(
        efficientnet["predictions"],
        [
            "directory",
            "files",
        ],
        "efficientnet.predictions",
    )

    if efficientnet["architecture"] != "efficientnet_b0":
        raise ValueError("efficientnet.architecture must be efficientnet_b0.")
    if efficientnet["pretrained_weights"] != "IMAGENET1K_V1":
        raise ValueError("efficientnet.pretrained_weights must be IMAGENET1K_V1.")
    if efficientnet["normalization"] != "imagenet":
        raise ValueError("efficientnet.normalization must be imagenet.")

    input_shape = [int(value) for value in efficientnet["input_shape"]]
    if len(input_shape) != 3 or any(value <= 0 for value in input_shape):
        raise ValueError("efficientnet.input_shape must contain three positive values.")
    if int(efficientnet["seed"]) < 0:
        raise ValueError("efficientnet.seed cannot be negative.")
    training_values = [
        int(efficientnet["epochs"]),
        int(efficientnet["batch_size"]),
        int(efficientnet["early_stopping_patience"]),
    ]
    if any(value <= 0 for value in training_values):
        raise ValueError("EfficientNet epoch, batch, and patience values must be positive.")
    dropout = float(efficientnet["classifier_dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("efficientnet.classifier_dropout must be from 0 up to 1.")

    unfreeze_last_stages = int(efficientnet["unfreeze_last_stages"])
    if not 1 <= unfreeze_last_stages <= 9:
        raise ValueError("efficientnet.unfreeze_last_stages must be from 1 to 9.")

    validate_efficientnet_optimizer(efficientnet["optimizer"])
    validate_scheduler(
        efficientnet["scheduler"],
        "efficientnet.scheduler",
        {"validation_loss"},
    )
    require_keys(
        efficientnet["scheduler"],
        ["threshold_mode"],
        "efficientnet.scheduler",
    )
    if efficientnet["scheduler"]["threshold_mode"] not in {"rel", "abs"}:
        raise ValueError("efficientnet.scheduler.threshold_mode must be rel or abs.")

    if efficientnet["checkpoint_monitor"] != "validation_loss":
        raise ValueError("efficientnet.checkpoint_monitor must be validation_loss.")

    if efficientnet["runs"] != ["stratified_random", "S1", "S2", "S3"]:
        raise ValueError("efficientnet.runs must be stratified_random, S1, S2, and S3.")

    predictions = efficientnet["predictions"]
    validate_relative_path(predictions["directory"], "efficientnet.predictions.directory")
    if set(predictions["files"]) != {"train", "validation", "test"}:
        raise ValueError("EfficientNet prediction files must cover all three splits.")
    for split_name, filename in predictions["files"].items():
        validate_relative_path(filename, f"efficientnet.predictions.files.{split_name}")
    validate_relative_path(
        efficientnet["experiment_summary_file"],
        "efficientnet.experiment_summary_file",
    )


def validate_config(config, project_root, validate_dataset):
    require_keys(
        config,
        [
            "project",
            "dataset",
            "paths",
            "split",
            "session_held_out",
            "runtime",
            "vae",
            "efficientnet",
            "features",
            "classifier",
            "evaluation",
            "figures",
            "controls",
        ],
        "config",
    )
    require_keys(
        config["paths"],
        [
            "train_split",
            "validation_split",
            "test_split",
            "split_summary",
            "session_held_out_splits_dir",
            "experiments_dir",
            "efficientnet_experiments_dir",
            "session_held_out_beta_vae_experiments_dir",
            "current_run_file",
        ],
        "paths",
    )
    require_keys(
        config["dataset"],
        [
            "root",
            "annotation_csv",
            "image_directory",
            "image_column",
            "label_column",
            "classes",
            "expected_total",
        ],
        "dataset",
    )
    require_keys(
        config["split"],
        [
            "seed",
            "stratified",
            "train_fraction",
            "validation_fraction",
            "test_fraction",
        ],
        "split",
    )
    require_keys(
        config["session_held_out"],
        [
            "seed",
            "train_fraction",
            "validation_fraction",
            "session_column",
            "check_file",
            "train_file",
            "validation_file",
            "test_file",
            "summary_file",
            "folds",
        ],
        "session_held_out",
    )
    require_keys(
        config["runtime"],
        [
            "preferred_gpu",
            "train_workers",
            "evaluation_workers",
            "pin_memory",
        ],
        "runtime",
    )
    require_keys(
        config["vae"],
        [
            "input_shape",
            "latent_channels",
            "scaling_factor",
            "log_variance_min",
            "log_variance_max",
            "epochs",
            "batch_size",
            "seed",
            "beta_start",
            "beta_final",
            "warmup_epochs",
            "save_after_epoch",
            "early_stopping_patience",
            "optimizer",
            "scheduler",
        ],
        "vae",
    )
    require_keys(
        config["features"],
        [
            "safe_skew_epsilon",
            "global_statistics",
            "quadrant_statistics",
            "region_grid_size",
            "center_fraction",
            "center_edge_epsilon",
            "kl_statistics",
            "kl_quadrant_statistics",
        ],
        "features",
    )
    require_keys(
        config["classifier"],
        [
            "seed",
            "epochs",
            "batch_size",
            "early_stopping_patience",
            "hidden_dimensions",
            "dropout",
            "standardize_features",
            "optimizer",
            "scheduler",
            "runs",
            "expected_feature_counts",
        ],
        "classifier",
    )
    require_keys(
        config["evaluation"],
        [
            "vae_splits",
            "reconstruction_split",
            "maximum_saved_reconstructions",
            "reference_classifier",
        ],
        "evaluation",
    )
    require_keys(
        config["figures"],
        ["formats", "png_dpi", "font_family"],
        "figures",
    )
    require_keys(
        config["controls"],
        ["seed_robustness", "shuffled_labels"],
        "controls",
    )
    validate_efficientnet(config["efficientnet"])
    for control_name, control in config["controls"].items():
        require_keys(
            control,
            ["run", "first_seed", "repetitions", "keep_models"],
            f"controls.{control_name}",
        )

    for field_name, value in config["paths"].items():
        validate_relative_path(value, f"paths.{field_name}")
    validate_relative_path(
        config["dataset"]["annotation_csv"],
        "dataset.annotation_csv",
    )
    validate_relative_path(
        config["dataset"]["image_directory"],
        "dataset.image_directory",
    )
    for field_name in [
        "check_file",
        "train_file",
        "validation_file",
        "test_file",
        "summary_file",
    ]:
        validate_relative_path(
            config["session_held_out"][field_name],
            f"session_held_out.{field_name}",
        )

    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.is_absolute():
        raise ValueError("dataset.root must be the only absolute data location.")

    fractions = [
        float(config["split"]["train_fraction"]),
        float(config["split"]["validation_fraction"]),
        float(config["split"]["test_fraction"]),
    ]
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")
    if not isinstance(config["split"]["stratified"], bool):
        raise TypeError("split.stratified must be true or false.")

    session_fractions = [
        float(config["session_held_out"]["train_fraction"]),
        float(config["session_held_out"]["validation_fraction"]),
    ]
    if abs(sum(session_fractions) - 1.0) > 1e-9:
        raise ValueError(
            "Session-holdout train and validation fractions must sum to 1.0."
        )
    if any(value <= 0.0 for value in session_fractions):
        raise ValueError("Session-holdout fractions must be positive.")

    session_column = str(config["session_held_out"]["session_column"]).strip()
    if not session_column:
        raise ValueError("session_held_out.session_column cannot be empty.")
    if session_column in {
        config["dataset"]["image_column"],
        config["dataset"]["label_column"],
    }:
        raise ValueError(
            "session_held_out.session_column must be a separate column name."
        )

    folds = config["session_held_out"]["folds"]
    if len(folds) != 3:
        raise ValueError("Exactly three session-held-out folds must be configured.")
    fold_sessions = [str(session).strip() for session in folds.values()]
    if any(not session for session in fold_sessions):
        raise ValueError("Session-holdout fold identifiers cannot be empty.")
    if len(fold_sessions) != len(set(fold_sessions)):
        raise ValueError("Session-holdout folds must use different sessions.")
    for fold_name in folds:
        validate_relative_path(fold_name, f"session_held_out.folds.{fold_name}")
        if len(Path(fold_name).parts) != 1:
            raise ValueError("Session-holdout fold names must be directory names.")

    classes = config["dataset"]["classes"]
    if len(classes) != len(set(classes)):
        raise ValueError("dataset.classes contains duplicate labels.")

    positive_values = [
        int(config["dataset"]["expected_total"]),
        int(config["vae"]["epochs"]),
        int(config["vae"]["batch_size"]),
        int(config["vae"]["latent_channels"]),
        int(config["efficientnet"]["epochs"]),
        int(config["efficientnet"]["batch_size"]),
        int(config["classifier"]["epochs"]),
        int(config["classifier"]["batch_size"]),
        int(config["figures"]["png_dpi"]),
    ]
    if any(value <= 0 for value in positive_values):
        raise ValueError(
            "Configured count, epoch, batch, and DPI values must be positive."
        )

    worker_counts = [
        int(config["runtime"]["train_workers"]),
        int(config["runtime"]["evaluation_workers"]),
    ]
    if any(value < 0 for value in worker_counts):
        raise ValueError("Runtime worker counts cannot be negative.")

    run_names = list(config["classifier"]["runs"])
    expected_names = list(config["classifier"]["expected_feature_counts"])
    if run_names != expected_names:
        raise ValueError(
            "Classifier run order and expected count order must match."
        )

    reference_run = config["evaluation"]["reference_classifier"]
    if reference_run not in run_names:
        raise ValueError(
            "evaluation.reference_classifier must name a classifier run."
        )

    valid_splits = {"train", "validation", "test"}
    evaluation_splits = config["evaluation"]["vae_splits"]
    if set(evaluation_splits) != valid_splits:
        raise ValueError(
            "evaluation.vae_splits must contain train, validation, and test."
        )
    if config["evaluation"]["reconstruction_split"] not in valid_splits:
        raise ValueError(
            "evaluation.reconstruction_split must be a valid split name."
        )
    if int(config["evaluation"]["maximum_saved_reconstructions"]) < 0:
        raise ValueError(
            "evaluation.maximum_saved_reconstructions cannot be negative."
        )

    for control_name, control in config["controls"].items():
        if control["run"] not in run_names:
            raise ValueError(
                f"controls.{control_name}.run must name a classifier run."
            )
        if int(control["repetitions"]) <= 0:
            raise ValueError(
                f"controls.{control_name}.repetitions must be positive."
            )

    validate_optimizer(config["vae"]["optimizer"], "vae.optimizer")
    validate_optimizer(
        config["classifier"]["optimizer"],
        "classifier.optimizer",
    )
    validate_scheduler(
        config["vae"]["scheduler"],
        "vae.scheduler",
        {
            "validation_reconstruction_loss",
            "validation_total_loss",
        },
    )
    validate_scheduler(
        config["classifier"]["scheduler"],
        "classifier.scheduler",
        {"validation_loss"},
    )

    formats = config["figures"]["formats"]
    if not formats or len(formats) != len(set(formats)):
        raise ValueError("figures.formats must be a non-empty list without duplicates.")
    unsupported_formats = set(formats) - {"png", "pdf", "svg"}
    if unsupported_formats:
        raise ValueError(f"Unsupported figure formats: {sorted(unsupported_formats)}")
    if not str(config["figures"]["font_family"]).strip():
        raise ValueError("figures.font_family cannot be empty.")

    if validate_dataset:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
        annotation_path = dataset_root / config["dataset"]["annotation_csv"]
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation CSV does not exist: {annotation_path}")

    project_root = Path(project_root).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")


