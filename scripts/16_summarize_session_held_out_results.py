from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.paths import (
    current_session_beta_vae_experiment,
    project_relative_text,
    run_config_path,
)
from blueberry_latent_map_phenotyping.runtime import capture_console
from blueberry_latent_map_phenotyping.session_held_out_pipeline import (
    summarize_session_beta_vae_results,
)


SCRIPT_PURPOSE = "Summarize the three session-held-out C5 results."


def main():
    current_config = load_config(CONFIG_PATH, True)
    experiment_dir = current_session_beta_vae_experiment(
        current_config,
        PROJECT_ROOT,
    )
    config = load_config(run_config_path(experiment_dir), True)
    log_path = experiment_dir / "logs" / "16_summarize_session_held_out_results.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, experiment_dir)
        print(f"Experiment: {relative_run}")
        summarize_session_beta_vae_results(
            config,
            PROJECT_ROOT,
            experiment_dir,
        )
        print("Session-held-out result summary completed.")


if __name__ == "__main__":
    print(SCRIPT_PURPOSE)
    main()

