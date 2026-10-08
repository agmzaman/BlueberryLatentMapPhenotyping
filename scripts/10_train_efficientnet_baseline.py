from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.efficientnet_pipeline import (
    train_efficientnet_experiments,
    validate_efficientnet_split_files,
)
from blueberry_latent_map_phenotyping.paths import (
    create_efficientnet_experiment,
    project_relative_text,
)
from blueberry_latent_map_phenotyping.runtime import capture_console, select_device


SCRIPT_PURPOSE = "Train fresh EfficientNet-B0 models for all configured data splits."


def main():
    config = load_config(CONFIG_PATH, True)
    validate_efficientnet_split_files(config, PROJECT_ROOT)
    experiment_dir = create_efficientnet_experiment(config, PROJECT_ROOT)
    device = select_device(int(config["runtime"]["preferred_gpu"]))
    log_path = experiment_dir / "logs" / "10_train_efficientnet_baseline.txt"

    with capture_console(log_path, "w"):
        relative_experiment = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_experiment}")
        train_efficientnet_experiments(
            config,
            PROJECT_ROOT,
            experiment_dir,
            device,
        )
        print("All EfficientNet runs completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()

