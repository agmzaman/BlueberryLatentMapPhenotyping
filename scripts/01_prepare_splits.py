from _bootstrap import CONFIG_PATH, PROJECT_ROOT
from blueberry_latent_map_phenotyping.config import load_config
from blueberry_latent_map_phenotyping.data import (
    prepare_session_held_out_splits,
    prepare_splits,
)

def main():
    config = load_config(CONFIG_PATH, True)
    _, conventional_summary = prepare_splits(config, PROJECT_ROOT)
    session_check, session_summaries = prepare_session_held_out_splits(config, PROJECT_ROOT)

    print("\nStratified random split summary")
    print(conventional_summary.to_string(index=False))

    print("\nSession split check")
    print(session_check.to_string(index=False))

    for fold_name, summary in session_summaries.items():
        split_totals = summary.groupby("split", sort=False)["count"].sum()
        print(f"\n{fold_name} split totals")
        print(split_totals.to_string())

    print("\nAll split preparation completed.")


if __name__ == "__main__":
    print("Prepare the stratified random and session-held-out splits.")
    main()

