from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.paths import create_run, project_relative_text
from blueberry_latent_map_phenotyping.runtime import capture_console, select_device
from blueberry_latent_map_phenotyping.vae_pipeline import train_beta_vae



def main():
    config = load_config(CONFIG_PATH, True)
    run_dir = create_run(config, PROJECT_ROOT)
    device = select_device(int(config["runtime"]["preferred_gpu"]))
    log_path = run_dir / "logs" / "02_train_beta_vae.txt"

    with capture_console(log_path, "w"):
        relative_run = project_relative_text(PROJECT_ROOT, run_dir)
        print(f"Experiment: {relative_run}")
        summary = train_beta_vae(
            config,
            PROJECT_ROOT,
            run_dir,
            device,
        )
        print(f"Training completed: {summary}")


if __name__ == "__main__":
    print("Spatial beta-VAE experiment.")
    main()

