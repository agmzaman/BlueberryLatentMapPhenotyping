from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.image_descriptors import (
    extract_image_descriptors,
    load_descriptor_image_frame,
)
from blueberry_latent_map_phenotyping.image_descriptor_settings import (
    config_with_image_descriptor_settings,
)
from blueberry_latent_map_phenotyping.paths import (
    create_image_descriptor_experiment,
    project_relative_text,
)
from blueberry_latent_map_phenotyping.runtime import capture_console


SCRIPT_PURPOSE = "Extract engineered image descriptors without canopy annotations."
SETTINGS_PATH = PROJECT_ROOT / "configs" / "image_descriptor.yaml"


def main():
    config = config_with_image_descriptor_settings(
        load_config(CONFIG_PATH, False),
        SETTINGS_PATH,
    )
    image_frame = load_descriptor_image_frame(config, PROJECT_ROOT)
    experiment_dir = create_image_descriptor_experiment(config, PROJECT_ROOT)
    log_path = experiment_dir / "logs" / "17_extract_image_descriptors.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_run}")
        extract_image_descriptors(
            config,
            PROJECT_ROOT,
            experiment_dir,
            image_frame,
        )
        print("Image-descriptor extraction completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()
