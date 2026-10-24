from _bootstrap import PROJECT_ROOT
from blueberry_latent_map_phenotyping.prediction_uncertainty import (
    load_uncertainty_settings,
    run_uncertainty,
)


def main():
    settings = load_uncertainty_settings(SETTINGS_PATH)
    run_uncertainty(settings, PROJECT_ROOT, ANALYSIS_NAME)


if __name__ == "__main__":
    SETTINGS_PATH = PROJECT_ROOT / "configs" / "uncertainty.yaml"
    ANALYSIS_NAME = "cross_model"
    print("Compare saved test predictions using paired, class-stratified bootstrap resampling.")
    main()
