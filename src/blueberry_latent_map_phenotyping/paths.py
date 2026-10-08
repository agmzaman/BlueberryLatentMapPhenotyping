from datetime import datetime
from pathlib import Path

from .config import save_config


def project_path(project_root, relative_path):
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Expected a project-relative path: {relative_path}")
    return Path(project_root).resolve() / path


def project_relative_text(project_root, path):
    project_root = Path(project_root).resolve()
    return Path(path).resolve().relative_to(project_root).as_posix()


def dataset_root(config):
    return Path(config["dataset"]["root"]).resolve()


def dataset_path(config, relative_path):
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Expected a dataset-relative path: {relative_path}")
    return dataset_root(config) / path


def dataset_image_root(config):
    return dataset_path(
        config,
        config["dataset"]["image_directory"],
    )


def split_paths(config, project_root):
    return {
        "train": project_path(
            project_root,
            config["paths"]["train_split"],
        ),
        "validation": project_path(
            project_root,
            config["paths"]["validation_split"],
        ),
        "test": project_path(
            project_root,
            config["paths"]["test_split"],
        ),
        "summary": project_path(
            project_root,
            config["paths"]["split_summary"],
        ),
    }


def session_held_out_root(config, project_root):
    return project_path(
        project_root,
        config["paths"]["session_held_out_splits_dir"],
    )


def session_split_check_path(config, project_root):
    return session_held_out_root(config, project_root) / config[
        "session_held_out"
    ]["check_file"]


def session_fold_paths(config, project_root, fold_name):
    fold_directory = session_held_out_root(config, project_root) / fold_name
    settings = config["session_held_out"]
    return {
        "train": fold_directory / settings["train_file"],
        "validation": fold_directory / settings["validation_file"],
        "test": fold_directory / settings["test_file"],
        "summary": fold_directory / settings["summary_file"],
    }


def create_efficientnet_experiment(config, project_root):
    experiment_root = project_path(
        project_root,
        config["paths"]["efficientnet_experiments_dir"],
    )
    experiment_root.mkdir(parents=True, exist_ok=True)

    experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    experiment_dir = experiment_root / experiment_name
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = experiment_root / f"{experiment_name}_{suffix}"
        suffix += 1

    experiment_dir.mkdir(parents=True)
    (experiment_dir / "logs").mkdir()
    save_config(config, experiment_dir / "config_used.yaml")
    return experiment_dir


def session_beta_vae_root(config, project_root):
    return project_path(
        project_root,
        config["paths"]["session_held_out_beta_vae_experiments_dir"],
    )


def create_session_beta_vae_experiment(config, project_root):
    experiment_root = session_beta_vae_root(config, project_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    experiment_dir = experiment_root / experiment_name
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = experiment_root / f"{experiment_name}_{suffix:02d}"
        suffix += 1

    directories = [experiment_dir / "logs"]
    for fold_name in config["session_held_out"]["folds"]:
        fold_dir = experiment_dir / fold_name
        directories.extend(
            [
                fold_dir / "vae" / "checkpoints",
                fold_dir / "vae" / "metrics",
                fold_dir / "vae" / "reconstructions",
                fold_dir / "latent",
                fold_dir / "features",
                fold_dir / "classifiers" / "C5",
            ]
        )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=False)

    save_config(config, experiment_dir / "config_used.yaml")
    current_file = experiment_root / "current_run.txt"
    current_file.write_text(
        project_relative_text(project_root, experiment_dir),
        encoding="utf-8",
    )
    return experiment_dir


def current_session_beta_vae_experiment(config, project_root):
    experiment_root = session_beta_vae_root(config, project_root)
    current_file = experiment_root / "current_run.txt"
    if not current_file.exists():
        raise FileNotFoundError(
            "No current session-held-out beta-VAE experiment. "
            "Run 11_train_session_held_out_beta_vae.py first."
        )

    relative_run = current_file.read_text(encoding="utf-8").strip()
    if not relative_run:
        raise ValueError(f"Current experiment file is empty: {current_file}")

    experiment_dir = project_path(project_root, relative_run)
    try:
        experiment_dir.relative_to(experiment_root.resolve())
    except ValueError as error:
        raise ValueError(
            "The current session-held-out beta-VAE experiment is outside its root."
        ) from error
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Current experiment directory does not exist: {experiment_dir}"
        )
    return experiment_dir


def image_descriptor_root(config, project_root):
    return project_path(
        project_root,
        config["image_descriptors"]["output_directory"],
    )


def create_image_descriptor_experiment(config, project_root):
    experiment_root = image_descriptor_root(config, project_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    experiment_dir = experiment_root / experiment_name
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = experiment_root / f"{experiment_name}_{suffix:02d}"
        suffix += 1

    directories = [
        experiment_dir / "logs",
        experiment_dir / "features",
        experiment_dir / "summaries",
    ]
    for run_name in config["image_descriptors"]["runs"]:
        directories.extend(
            [
                experiment_dir / run_name / "rbf_svm",
                experiment_dir / run_name / "random_forest",
            ]
        )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=False)

    save_config(config, experiment_dir / "config_used.yaml")
    current_file = experiment_root / "current_run.txt"
    current_file.write_text(
        project_relative_text(project_root, experiment_dir),
        encoding="utf-8",
    )
    return experiment_dir


def current_image_descriptor_experiment(config, project_root):
    experiment_root = image_descriptor_root(config, project_root)
    current_file = experiment_root / "current_run.txt"
    if not current_file.exists():
        raise FileNotFoundError(
            "No current image-descriptor experiment. "
            "Run 17_extract_image_descriptors.py first."
        )

    relative_run = current_file.read_text(encoding="utf-8").strip()
    if not relative_run:
        raise ValueError(f"Current experiment file is empty: {current_file}")

    experiment_dir = project_path(project_root, relative_run)
    try:
        experiment_dir.relative_to(experiment_root.resolve())
    except ValueError as error:
        raise ValueError(
            "The current image-descriptor experiment is outside its root."
        ) from error
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Current experiment directory does not exist: {experiment_dir}"
        )
    return experiment_dir


def create_run(config, project_root):
    experiments_dir = project_path(
        project_root,
        config["paths"]["experiments_dir"],
    )
    experiments_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = experiments_dir / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = experiments_dir / f"{run_name}_{suffix:02d}"
        suffix += 1

    directories = [
        run_dir / "logs",
        run_dir / "vae" / "checkpoints",
        run_dir / "vae" / "metrics",
        run_dir / "vae" / "reconstructions",
        run_dir / "latent",
        run_dir / "features",
        run_dir / "classifiers",
        run_dir / "controls",
        run_dir / "reports",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    save_config(config, run_dir / "config_used.yaml")
    current_file = project_path(
        project_root,
        config["paths"]["current_run_file"],
    )
    current_file.parent.mkdir(parents=True, exist_ok=True)
    current_file.write_text(
        str(run_dir.relative_to(project_root)),
        encoding="utf-8",
    )
    return run_dir


def current_run_directory(config, project_root):
    current_file = project_path(
        project_root,
        config["paths"]["current_run_file"],
    )
    if not current_file.exists():
        raise FileNotFoundError(
            "No current experiment. Run 02_train_beta_vae.py first."
        )

    relative_run = current_file.read_text(encoding="utf-8").strip()
    if not relative_run:
        raise ValueError(f"Current experiment file is empty: {current_file}")

    run_dir = project_path(project_root, relative_run)
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Current experiment directory does not exist: {run_dir}"
        )
    return run_dir


def run_config_path(run_dir):
    path = Path(run_dir) / "config_used.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment configuration does not exist: {path}"
        )
    return path


