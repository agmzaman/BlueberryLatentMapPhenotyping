from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.paths import (
    current_session_beta_vae_experiment,
    project_relative_text,
    run_config_path,
)
from blueberry_latent_map_phenotyping.runtime import capture_console
from blueberry_latent_map_phenotyping.session_held_out_pipeline import (
    extract_session_beta_vae_features,
)


SCRIPT_PURPOSE = "Extract latent-map features for held-out sessions S1-S3."


def main():
    current_config = load_config(CONFIG_PATH, True)
    experiment_dir = current_session_beta_vae_experiment(
        current_config,
        PROJECT_ROOT,
    )
    config = load_config(run_config_path(experiment_dir), True)
    log_path = experiment_dir / "logs" / "14_extract_session_held_out_features.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_run}")
        extract_session_beta_vae_features(
            config,
            PROJECT_ROOT,
            experiment_dir,
        )
        print("Session-held-out feature extraction completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()

