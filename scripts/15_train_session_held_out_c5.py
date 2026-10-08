from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.paths import (
    current_session_beta_vae_experiment,
    project_relative_text,
    run_config_path,
)
from blueberry_latent_map_phenotyping.runtime import capture_console, select_device
from blueberry_latent_map_phenotyping.session_held_out_pipeline import (
    train_session_beta_vae_c5,
)


SCRIPT_PURPOSE = "Train independent C5 classifiers for held-out sessions S1-S3."


def main():
    current_config = load_config(CONFIG_PATH, True)
    experiment_dir = current_session_beta_vae_experiment(
        current_config,
        PROJECT_ROOT,
    )
    config = load_config(run_config_path(experiment_dir), True)
    device = select_device(int(config["runtime"]["preferred_gpu"]))
    log_path = experiment_dir / "logs" / "15_train_session_held_out_c5.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_run}")
        train_session_beta_vae_c5(
            config,
            PROJECT_ROOT,
            experiment_dir,
            device,
        )
        print("Session-held-out C5 training completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()

