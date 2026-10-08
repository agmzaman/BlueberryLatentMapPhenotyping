from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import (
    load_config,
    load_run_config,
)
from blueberry_latent_map_phenotyping.controls import run_shuffled_label_control
from blueberry_latent_map_phenotyping.paths import (
    current_run_directory,
    project_relative_text,
    run_config_path,
)
from blueberry_latent_map_phenotyping.runtime import capture_console, select_device


SCRIPT_PURPOSE = "Run the optional C5 shuffled-label control."


def main():
    current_config = load_config(CONFIG_PATH, True)
    run_dir = current_run_directory(current_config, PROJECT_ROOT)
    config = load_run_config(run_config_path(run_dir), current_config,
    PROJECT_ROOT, True,
    )
    control_config = current_config["controls"]["shuffled_labels"]
    device = select_device(int(config["runtime"]["preferred_gpu"]))
    log_path = run_dir / "logs" / "09_run_shuffled_label_control.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, run_dir)
        print(f"Experiment: {relative_run}")
        print(f"Repetitions: {control_config['repetitions']}")
        run_shuffled_label_control(config, control_config, run_dir, device)
        print("Shuffled-label control completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()

