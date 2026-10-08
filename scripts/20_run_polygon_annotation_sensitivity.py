from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.polygon_sensitivity import (
    load_polygon_sensitivity_settings,
    polygon_sensitivity_paths,
    run_polygon_sensitivity,
)
from blueberry_latent_map_phenotyping.runtime import capture_console


SCRIPT_PURPOSE = "Run polygon-annotation sensitivity analysis."
SETTINGS_PATH = PROJECT_ROOT / "configs" / "polygon_sensitivity.yaml"


def main():
    config = load_config(CONFIG_PATH, True)
    settings = load_polygon_sensitivity_settings(SETTINGS_PATH)
    paths = polygon_sensitivity_paths(settings, PROJECT_ROOT)
    paths["directory"].mkdir(parents=True, exist_ok=True)

    with capture_console(paths["log"], "w"):
        print(f"Repetitions per image: {settings['repetitions']}")
        print(f"Noise percentages: {settings['noise_percentages']}")
        run_polygon_sensitivity(config, settings, PROJECT_ROOT)
        print("Polygon-annotation sensitivity analysis completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()

