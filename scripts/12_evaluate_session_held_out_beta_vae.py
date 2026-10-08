from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.paths import (
    current_session_beta_vae_experiment,
    project_relative_text,
    run_config_path,
)
from blueberry_latent_map_phenotyping.runtime import capture_console, select_device
from blueberry_latent_map_phenotyping.session_held_out_pipeline import (
    evaluate_session_beta_vae_experiments,
    validate_session_beta_vae_splits,
)


SCRIPT_PURPOSE = "Evaluate the three session-held-out beta-VAEs."


def main():
    current_config = load_config(CONFIG_PATH, True)
    experiment_dir = current_session_beta_vae_experiment(
        current_config,
        PROJECT_ROOT,
    )
    config = load_config(run_config_path(experiment_dir), True)
    validate_session_beta_vae_splits(config, PROJECT_ROOT)
    device = select_device(int(config["runtime"]["preferred_gpu"]))
    log_path = experiment_dir / "logs" / "12_evaluate_session_held_out_beta_vae.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_run}")
        evaluate_session_beta_vae_experiments(
            config,
            PROJECT_ROOT,
            experiment_dir,
            device,
        )
        print("Session-held-out reconstruction evaluation completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()

