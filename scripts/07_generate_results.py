from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.paths import (
    current_run_directory,
    run_config_path, project_relative_text,
)
from blueberry_latent_map_phenotyping.reports import generate_results
from blueberry_latent_map_phenotyping.runtime import capture_console



def main():
    current_config = load_config(CONFIG_PATH, True)
    run_dir = current_run_directory(current_config, PROJECT_ROOT)
    config = load_config(run_config_path(run_dir), True)
    log_path = run_dir / "logs" / "07_generate_results.txt"

    with capture_console(log_path, "w"):
        print(f"Experiment: {project_relative_text(PROJECT_ROOT, run_dir)}")
        outputs = generate_results(config, PROJECT_ROOT, run_dir)
        print(f"Results: {outputs}")


if __name__ == "__main__":
    print("Generate results and figures.")
    main()
