from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.paths import (
    current_run_directory,
    project_relative_text,
    run_config_path,
)
from blueberry_latent_map_phenotyping.runtime import capture_console, select_device
from blueberry_latent_map_phenotyping.vae_pipeline import evaluate_beta_vae


def main():
    current_config = load_config(CONFIG_PATH, True)
    run_dir = current_run_directory(current_config, PROJECT_ROOT)
    config = load_config(run_config_path(run_dir), True)
    device = select_device(int(config["runtime"]["preferred_gpu"]))
    log_path = run_dir / "logs" / "03_evaluate_beta_vae.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, run_dir)
        print(f"Experiment: {relative_run}")
        evaluate_beta_vae(
            config,
            PROJECT_ROOT,
            run_dir,
            device,
        )
        print("Reconstruction evaluation completed.")


if __name__ == "__main__":
    print("Evaluate deterministic beta-VAE reconstructions.")
    main()

