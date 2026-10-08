# BlueberryLatentMapPhenotyping

Code for the study **Statistical Descriptors of β-Variational Autoencoder Latent Feature Maps for Image-Based Blueberry Canopy-Size Phenotyping**.

The proposed framework uses a β-VAE to learn spatial posterior maps, from which statistical descriptors are extracted for canopy-size classification. Comparative baselines include EfficientNet-B0 transfer learning and RBF-SVM and Random Forest classifiers, both using the same set of 22 image-derived descriptors.

The four reference classes are **Very Small, Small, Medium, and Large**. They describe image-relative canopy extent, not physically calibrated plant size. Manual canopy polygons define the reference labels; polygon measurements are not used as predictor inputs.

## Repository structure

```text
BlueberryLatentMapPhenotyping/
├── configs/
│   ├── config.example.yaml
│   ├── image_descriptor.yaml
│   └── polygon_sensitivity.yaml
├── data/
│   ├── splits/
│   └── session_held_out_splits/
├── experiments/
├── figures/
├── scripts/
├── src/blueberry_latent_map_phenotyping/
├── CITATION.cff
├── README.md
└── requirements.txt
```

The release includes data splits, settings, results, logs, and PDF figures, but excludes trained models and large intermediate files.

## Setup and dataset

Use a clean Python 3.12.6 environment. From the project root, run:

```bat
python -m pip install -r requirements.txt
python -m pip check
```
The requirements use CUDA 12.4 PyTorch builds; adjust these for other hardware.
Images are not included. Dataset record: [blueberry canopy dataset](https://doi.org/10.5281/zenodo.19597014).

```text
blueberry_canopy_dataset/
├── images/
│   ├── 2025-08-07/
│   ├── 2025-08-22/
│   ├── 2025-09-05/
│   └── 2025-09-29/
└── annotations/
    └── blueberry_annotations_labeled.csv
```
Copy `configs/config.example.yaml` to `configs/config.yaml` and set `dataset.root` to your dataset folder’s absolute path. When reusing a saved experiment, also update `dataset.root` in its `config_used.yaml` file. Output paths are relative to the project root. Keep the remaining experiment settings unchanged to reproduce the reported setup.

## Evaluation splits

- **Stratified random:** 1,225 training, 409 validation, and 409 test images (split seed 131).
- **S1–S3:** one of the first three acquisition sessions is held out for testing. Each remaining session is split 75:25 for training and validation, stratified by class.

All methods use the same splits. The β-VAE is trained separately for each evaluation split. Session summaries report the unweighted mean ± sample SD across S1–S3 only. The session-held-out splits separate acquisition sessions, but not necessarily individual plants.

## Analysis pipeline

Run a stage only after its required inputs exist.

| Scripts | Purpose                                                                                       |
| --- |-----------------------------------------------------------------------------------------------|
| `01_prepare_splits.py` | Generate stratified-random and S1–S3 partitions                                               |
| `02`–`07` | Train/evaluate the β-VAE under the stratified-random split, extract maps/descriptors, train C1–C5, summarize results |
| `08_run_seed_robustness.py` | 50 C5 classifier seeds with the encoder and descriptors fixed                                 |
| `09_run_shuffled_label_control.py` | 50 independent train/validation-label permutations; true test labels retained                 |
| `10_train_efficientnet_baseline.py` | Independent EfficientNet-B0 training for all four evaluation splits                           |
| `11`–`16` | Train/evaluate three session-held-out β-VAEs, extract descriptors and train C5                |
| `17`–`19` | Extract 22 image descriptors, select classical models by validation results and summarize     |
| `20_run_polygon_annotation_sensitivity.py` | Simulated polygon-boundary perturbation analysis                                              |
| `21`–`24` | β-VAE, EfficientNet, image-descriptor and cross-experiment figures                            |

Use the provided splits when reviewing saved results; rerunning Script 01 overwrites them. New training runs create timestamped experiment folders and can change the run selected by subsequent analysis and figure scripts.

## Descriptor configurations

| Configuration | Descriptor components | Number of features |
| --- | --- | ---: |
| C1 | Global posterior-mean statistics | 45 |
| C2 | C1 plus spatial posterior-mean summaries | 180 |
| C3 | C2 plus KL descriptors | 215 |
| C4 | C2 plus log-variance descriptors | 265 |
| C5 | Posterior mean, log variance and KL descriptors | 300 |

C5 is the reference configuration for S1–S3. Both RBF-SVM and Random Forest use all 22 image descriptors: six morphology, six colour, and ten GLCM texture features. Each classifier is trained and evaluated separately using the stratified-random split and the three session-held-out splits (S1–S3). Validation selects one model per classifier per split, giving eight final models (2 classifiers × 4 evaluation splits).

## Figures and reproducibility

Scripts 21–24 provide adjustable figure settings. Figure generation requires Calibri, which is not bundled. Most figures use saved results without retraining. Reconstruction examples require the original and reconstructed images; disable that plotting call if these are unavailable.

Numerical results may vary across software versions and hardware.

## Citation and license

Please cite the software using `CITATION.cff` and cite the dataset separately. The software and dataset are subject to their respective licensing terms, as specified in the accompanying license file and dataset record.