from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.image_descriptor_pipeline import (
    train_image_descriptor_classifiers,
    validate_descriptor_split_files,
)
from blueberry_latent_map_phenotyping.image_descriptor_settings import (
    config_with_image_descriptor_settings,
)
from blueberry_latent_map_phenotyping.paths import (
    current_image_descriptor_experiment,
    project_relative_text,
    run_config_path,
)
from blueberry_latent_map_phenotyping.runtime import capture_console


SCRIPT_PURPOSE = "Train RBF-SVM and Random Forest image-descriptor classifiers."
SETTINGS_PATH = PROJECT_ROOT / "configs" / "image_descriptor.yaml"


def main():
    current_config = config_with_image_descriptor_settings(
        load_config(CONFIG_PATH, False),
        SETTINGS_PATH,
    )
    experiment_dir = current_image_descriptor_experiment(
        current_config,
        PROJECT_ROOT,
    )
    config = load_config(run_config_path(experiment_dir), False)
    validate_descriptor_split_files(config, PROJECT_ROOT)
    log_path = experiment_dir / "logs" / "18_train_descriptor_classifiers.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_run}")
        train_image_descriptor_classifiers(
            config,
            PROJECT_ROOT,
            experiment_dir,
        )
        print("Image-descriptor classifier training completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()


