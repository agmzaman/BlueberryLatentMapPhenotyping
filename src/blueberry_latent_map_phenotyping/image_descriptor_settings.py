from pathlib import Path

import yaml


def require_keys(mapping, keys, section_name):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing keys in {section_name}: {missing}")


def validate_image_descriptor_settings(settings):
    require_keys(
        settings,
        [
            "image_size",
            "seed",
            "expected_feature_count",
            "standardize_features",
            "selection_metric",
            "runs",
            "feature_groups",
            "configurations",
            "mask",
            "color",
            "glcm",
            "rbf_svm",
            "random_forest",
            "output_directory",
        ],
        "image_descriptor",
    )
    if [int(value) for value in settings["image_size"]] != [512, 512]:
        raise ValueError("image_descriptor.image_size must be [512, 512].")
    if int(settings["seed"]) < 0:
        raise ValueError("image_descriptor.seed cannot be negative.")
    if int(settings["expected_feature_count"]) != 22:
        raise ValueError("image_descriptor.expected_feature_count must be 22.")
    if settings["standardize_features"] is not True:
        raise ValueError("image_descriptor.standardize_features must be true.")
    if settings["selection_metric"] != "macro_f1":
        raise ValueError("image_descriptor.selection_metric must be macro_f1.")
    if settings["runs"] != ["stratified_random", "S1", "S2", "S3"]:
        raise ValueError(
            "image_descriptor.runs must be stratified_random, S1, S2, and S3."
        )

    group_counts = {
        name: int(value)
        for name, value in settings["feature_groups"].items()
    }
    if group_counts != {"morphology": 6, "color": 6, "glcm": 10}:
        raise ValueError(
            "image_descriptor.feature_groups must be morphology=6, "
            "color=6, and glcm=10."
        )
    expected_configurations = {
        "morphology_color_texture": ["morphology", "color", "glcm"],
    }
    if settings["configurations"] != expected_configurations:
        raise ValueError(
            "image_descriptor.configurations must define the single "
            "22-feature morphology_color_texture configuration."
        )

    require_keys(
        settings["mask"],
        [
            "method",
            "opening_radius",
            "closing_radius",
            "minimum_component_fraction",
        ],
        "image_descriptor.mask",
    )
    mask = settings["mask"]
    if mask["method"] != "exg_otsu":
        raise ValueError("image_descriptor.mask.method must be exg_otsu.")
    if int(mask["opening_radius"]) < 0 or int(mask["closing_radius"]) < 0:
        raise ValueError("Mask morphology radii cannot be negative.")
    component_fraction = float(mask["minimum_component_fraction"])
    if not 0.0 < component_fraction < 1.0:
        raise ValueError(
            "image_descriptor.mask.minimum_component_fraction "
            "must be between 0 and 1."
        )

    require_keys(
        settings["color"],
        ["channels", "statistics"],
        "image_descriptor.color",
    )
    if settings["color"]["channels"] != [
        "exg",
        "hsv_saturation",
        "hsv_value",
    ]:
        raise ValueError(
            "Color channels must be ExG, HSV saturation, and HSV value."
        )
    if settings["color"]["statistics"] != ["mean", "std"]:
        raise ValueError("Color statistics must be mean and population SD.")

    require_keys(
        settings["glcm"],
        [
            "levels",
            "grid_steps",
            "angles_degrees",
            "grayscale_weights",
            "properties",
            "summaries",
        ],
        "image_descriptor.glcm",
    )
    glcm = settings["glcm"]
    if int(glcm["levels"]) != 32:
        raise ValueError("image_descriptor.glcm.levels must be 32.")
    if glcm["grid_steps"] != [1, 2]:
        raise ValueError("GLCM grid steps must be 1 and 2.")
    if glcm["angles_degrees"] != [0, 45, 90, 135]:
        raise ValueError("GLCM angles must be 0, 45, 90, and 135 degrees.")
    weights = [float(value) for value in glcm["grayscale_weights"]]
    if len(weights) != 3 or any(value < 0.0 for value in weights):
        raise ValueError("GLCM grayscale weights must be nonnegative.")
    if abs(sum(weights) - 1.0) > 0.000001:
        raise ValueError("GLCM grayscale weights must sum to one.")
    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    ]
    if glcm["properties"] != properties:
        raise ValueError("GLCM properties do not match the 10-feature schema.")
    if glcm["summaries"] != ["mean", "std"]:
        raise ValueError("GLCM summaries must be mean and population SD.")

    require_keys(
        settings["rbf_svm"],
        ["class_weight", "c_values", "gamma_values"],
        "image_descriptor.rbf_svm",
    )
    svm = settings["rbf_svm"]
    if svm["class_weight"] is not None:
        raise ValueError("image_descriptor.rbf_svm.class_weight must be null.")
    if not svm["c_values"] or any(
        float(value) <= 0.0 for value in svm["c_values"]
    ):
        raise ValueError("RBF-SVM C values must be positive.")
    for value in svm["gamma_values"]:
        if isinstance(value, str):
            if value != "scale":
                raise ValueError("The only supported text gamma is scale.")
        elif float(value) <= 0.0:
            raise ValueError("Numeric RBF-SVM gamma values must be positive.")

    require_keys(
        settings["random_forest"],
        [
            "class_weight",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
        ],
        "image_descriptor.random_forest",
    )
    forest = settings["random_forest"]
    if forest["class_weight"] is not None:
        raise ValueError(
            "image_descriptor.random_forest.class_weight must be null."
        )
    if int(forest["n_estimators"]) <= 0:
        raise ValueError("Random Forest n_estimators must be positive.")
    for value in forest["max_depth"]:
        if value is not None and int(value) <= 0:
            raise ValueError("Random Forest max_depth values must be positive.")
    if any(int(value) <= 0 for value in forest["min_samples_leaf"]):
        raise ValueError("Random Forest min_samples_leaf values must be positive.")
    for value in forest["max_features"]:
        if isinstance(value, str):
            if value != "sqrt":
                raise ValueError("The only supported text max_features is sqrt.")
        elif not 0.0 < float(value) <= 1.0:
            raise ValueError("Numeric max_features values must be in (0, 1].")

    output_path = Path(str(settings["output_directory"]))
    if output_path.is_absolute() or ".." in output_path.parts:
        raise ValueError(
            "image_descriptor.output_directory must be project-relative."
        )


def load_image_descriptor_settings(settings_path):
    settings_path = Path(settings_path).resolve()
    with settings_path.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    validate_image_descriptor_settings(settings)
    return settings


def config_with_image_descriptor_settings(config, settings_path):
    combined = dict(config)
    combined["image_descriptors"] = load_image_descriptor_settings(
        settings_path
    )
    return combined
