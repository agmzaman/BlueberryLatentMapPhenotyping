from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.image_descriptor_pipeline import (
    summarize_image_descriptor_results,
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


SCRIPT_PURPOSE = "Summarize primary and supplementary descriptor results."
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
    log_path = experiment_dir / "logs" / "19_summarize_descriptor_results.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_run}")
        summarize_image_descriptor_results(
            config,
            PROJECT_ROOT,
            experiment_dir,
        )
        print("Image-descriptor result summary completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()


