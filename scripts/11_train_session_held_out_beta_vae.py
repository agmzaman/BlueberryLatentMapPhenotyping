from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.paths import (
    create_session_beta_vae_experiment,
    project_relative_text,
)
from blueberry_latent_map_phenotyping.runtime import capture_console, select_device
from blueberry_latent_map_phenotyping.session_held_out_pipeline import (
    train_session_beta_vae_experiments,
    validate_session_beta_vae_splits,
)


SCRIPT_PURPOSE = "Train independent beta-VAEs for held-out sessions S1-S3."


def main():
    config = load_config(CONFIG_PATH, True)
    validate_session_beta_vae_splits(config, PROJECT_ROOT)
    device = select_device(int(config["runtime"]["preferred_gpu"]))
    experiment_dir = create_session_beta_vae_experiment(config, PROJECT_ROOT)
    log_path = experiment_dir / "logs" / "11_train_session_held_out_beta_vae.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_run}")
        train_session_beta_vae_experiments(
            config,
            PROJECT_ROOT,
            experiment_dir,
            device,
        )
        print("Session-held-out beta-VAE training completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()

