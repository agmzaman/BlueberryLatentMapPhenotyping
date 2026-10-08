import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from pytorch_msssim import ms_ssim, ssim
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import ImageDataset, image_transforms
from .models import SpatialBetaVAE
from .paths import dataset_image_root, project_relative_text, split_paths
from .runtime import data_loader_generator, seed_worker, set_reproducibility


def build_vae(config, device):
    vae_config = config["vae"]
    channels = int(vae_config["input_shape"][0])
    model = SpatialBetaVAE(
        latent_channels=int(vae_config["latent_channels"]),
        scaling_factor=float(vae_config["scaling_factor"]),
        log_variance_bounds=(
            float(vae_config["log_variance_min"]),
            float(vae_config["log_variance_max"]),
        ),
        input_channels=channels,
        output_channels=channels,
    )
    return model.to(device)


def build_image_loaders_from_paths(config, configured_paths, training_mode):
    train_transform, evaluation_transform = image_transforms(config)
    classes = config["dataset"]["classes"]
    image_column = config["dataset"]["image_column"]
    label_column = config["dataset"]["label_column"]
    root = dataset_image_root(config)
    batch_size = int(config["vae"]["batch_size"])
    seed = int(config["vae"]["seed"])

    loaders = {}
    for split_name in ["train", "validation", "test"]:
        transform = (
            train_transform
            if training_mode and split_name == "train"
            else evaluation_transform
        )
        dataset = ImageDataset(
            configured_paths[split_name],
            root,
            image_column,
            label_column,
            classes,
            transform,
        )
        workers = (
            int(config["runtime"]["train_workers"])
            if training_mode and split_name == "train"
            else int(config["runtime"]["evaluation_workers"])
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training_mode and split_name == "train",
            num_workers=workers,
            pin_memory=bool(config["runtime"]["pin_memory"]),
            worker_init_fn=seed_worker,
            generator=data_loader_generator(seed),
        )
    return loaders


def build_image_loaders(config, project_root, training_mode):
    configured_paths = split_paths(config, project_root)
    return build_image_loaders_from_paths(
        config,
        configured_paths,
        training_mode,
    )


def beta_for_epoch(config, epoch):
    start = float(config["vae"]["beta_start"])
    final = float(config["vae"]["beta_final"])
    warmup = int(config["vae"]["warmup_epochs"])

    if warmup <= 1 or epoch >= warmup:
        return final
    fraction = (epoch - 1) / (warmup - 1)
    return start + fraction * (final - start)


def vae_loss(reconstruction, image, mean, log_variance, beta):
    reconstruction_loss = functional.l1_loss(
        reconstruction,
        image,
        reduction="sum",
    )
    kl_loss = -0.5 * torch.sum(
        1 + log_variance - mean.pow(2) - log_variance.exp()
    )
    weighted_kl_loss = beta * kl_loss
    total_loss = reconstruction_loss + weighted_kl_loss
    return total_loss, reconstruction_loss, kl_loss, weighted_kl_loss


def run_loss_epoch(model, loader, optimizer, beta, device, training):
    if training:
        model.train()
    else:
        model.eval()

    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "kl_loss": 0.0,
        "weighted_kl_loss": 0.0,
        "samples": 0,
    }

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstruction, mean, log_variance = model(images)
            losses = vae_loss(
                reconstruction,
                images,
                mean,
                log_variance,
                beta,
            )

            if training:
                losses[0].backward()
                optimizer.step()

            batch_size = images.size(0)
            totals["samples"] += batch_size
            totals["total_loss"] += float(losses[0].item())
            totals["reconstruction_loss"] += float(losses[1].item())
            totals["kl_loss"] += float(losses[2].item())
            totals["weighted_kl_loss"] += float(losses[3].item())

    sample_count = totals.pop("samples")
    if sample_count == 0:
        raise ValueError("An empty image loader was supplied.")
    return {name: value / sample_count for name, value in totals.items()}


def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_reconstruction_loss,
    config,
    path,
):
    checkpoint = {
        "epoch": int(epoch),
        "validation_reconstruction_loss": float(
            validation_reconstruction_loss
        ),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vae_config": config["vae"],
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def build_vae_optimizer(config, model):
    optimizer_config = config["vae"]["optimizer"]
    optimizer_classes = {
        "Adam": Adam,
        "AdamW": AdamW,
    }
    optimizer_class = optimizer_classes[optimizer_config["type"]]
    return optimizer_class(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def build_vae_scheduler(config, optimizer):
    scheduler_config = config["vae"]["scheduler"]
    return ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config["mode"],
        factor=float(scheduler_config["lr_update_factor"]),
        patience=int(scheduler_config["lr_update_patience"]),
        threshold=float(scheduler_config["threshold"]),
        min_lr=float(scheduler_config["lr_min"]),
    )


def vae_scheduler_value(monitor, validation_metrics):
    monitor_values = {
        "validation_reconstruction_loss": validation_metrics[
            "reconstruction_loss"
        ],
        "validation_total_loss": validation_metrics["total_loss"],
    }
    return monitor_values[monitor]


def train_beta_vae_from_paths(
    config,
    project_root,
    run_dir,
    device,
    configured_paths,
):
    seed = int(config["vae"]["seed"])
    set_reproducibility(seed)
    loaders = build_image_loaders_from_paths(
        config,
        configured_paths,
        True,
    )
    model = build_vae(config, device)

    optimizer = build_vae_optimizer(config, model)
    scheduler = build_vae_scheduler(config, optimizer)
    scheduler_monitor = config["vae"]["scheduler"]["monitor"]

    epoch_count = int(config["vae"]["epochs"])
    save_after = int(config["vae"]["save_after_epoch"])
    patience = int(config["vae"]["early_stopping_patience"])
    checkpoint_path = Path(run_dir) / "vae" / "checkpoints" / "best_beta_vae.pt"
    history_path = Path(run_dir) / "vae" / "metrics" / "training_history.csv"

    best_validation_reconstruction_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epoch_count + 1):
        beta = beta_for_epoch(config, epoch)
        train_metrics = run_loss_epoch(
            model,
            loaders["train"],
            optimizer,
            beta,
            device,
            True,
        )
        validation_metrics = run_loss_epoch(
            model,
            loaders["validation"],
            optimizer,
            beta,
            device,
            False,
        )
        scheduler.step(
            vae_scheduler_value(
                scheduler_monitor,
                validation_metrics,
            )
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])

        model_saved = False
        if epoch >= save_after:
            validation_reconstruction_loss = validation_metrics[
                "reconstruction_loss"
            ]
            if (
                validation_reconstruction_loss
                < best_validation_reconstruction_loss
            ):
                best_validation_reconstruction_loss = (
                    validation_reconstruction_loss
                )
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    validation_reconstruction_loss,
                    config,
                    checkpoint_path,
                )
                model_saved = True
            else:
                epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "beta": beta,
            "learning_rate": learning_rate,
            "model_saved": model_saved,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            }
        )
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch:03d}/{epoch_count} | "
            f"beta={beta:.6f} | "
            f"train={train_metrics['total_loss']:.4f} | "
            f"validation={validation_metrics['total_loss']:.4f} | "
            f"lr={learning_rate:.8f} | "
            f"saved={model_saved}"
        )
        if model_saved:
            relative_checkpoint = project_relative_text(
                project_root,
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {relative_checkpoint}")
        if epoch >= save_after and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if not checkpoint_path.exists():
        final_row = history[-1]
        best_validation_reconstruction_loss = float(
            final_row["validation_reconstruction_loss"]
        )
        best_epoch = int(final_row["epoch"])
        save_checkpoint(
            model,
            optimizer,
            int(final_row["epoch"]),
            best_validation_reconstruction_loss,
            config,
            checkpoint_path,
        )
        history[-1]["model_saved"] = True
        pd.DataFrame(history).to_csv(history_path, index=False)

    summary = {
        "completed_epochs": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_reconstruction_loss": float(
            best_validation_reconstruction_loss
        ),
        "checkpoint": project_relative_text(project_root, checkpoint_path),
    }
    summary_path = Path(run_dir) / "vae" / "metrics" / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_beta_vae(config, project_root, run_dir, device):
    configured_paths = split_paths(config, project_root)
    return train_beta_vae_from_paths(
        config,
        project_root,
        run_dir,
        device,
        configured_paths,
    )


def metric_sum(metric_values):
    if metric_values.ndim == 0:
        return float(metric_values.item())
    return float(metric_values.sum().item())


def evaluate_loader(model, loader, device):
    model.eval()
    ssim_total = 0.0
    ms_ssim_total = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            reconstruction, _, _ = model.reconstruct_from_mean(images)
            images_zero_one = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
            reconstruction_zero_one = torch.clamp(
                (reconstruction + 1.0) / 2.0,
                0.0,
                1.0,
            )
            ssim_values = ssim(
                reconstruction_zero_one,
                images_zero_one,
                data_range=1.0,
                size_average=False,
            )
            ms_ssim_values = ms_ssim(
                reconstruction_zero_one,
                images_zero_one,
                data_range=1.0,
                size_average=False,
            )
            ssim_total += metric_sum(ssim_values)
            ms_ssim_total += metric_sum(ms_ssim_values)
            sample_count += images.size(0)

    if sample_count == 0:
        raise ValueError("An empty image loader was supplied.")
    return {
        "samples": sample_count,
        "ssim": ssim_total / sample_count,
        "ms_ssim": ms_ssim_total / sample_count,
    }


def reconstruction_name(relative_path):
    path = Path(relative_path)
    return f"{path.stem}_recon{path.suffix}"

def save_reconstructions(config, model, loader, run_dir, device):
    maximum = int(config["evaluation"]["maximum_saved_reconstructions"])
    if maximum <= 0:
        return 0

    output_dir = Path(run_dir) / "vae" / "reconstructions"
    saved = 0
    model.eval()
    with torch.no_grad():
        for images, _, relative_paths in loader:
            images = images.to(device, non_blocking=True)
            reconstruction, _, _ = model.reconstruct_from_mean(images)
            reconstruction = torch.clamp((reconstruction + 1.0) / 2.0,0.0,1.0)

            for item_index, relative_path in enumerate(relative_paths):
                filename = reconstruction_name(relative_path)
                save_image(reconstruction[item_index], output_dir / filename)
                saved += 1
                if saved >= maximum:
                    return saved
    return saved


def evaluate_beta_vae_from_paths(
    config,
    project_root,
    run_dir,
    device,
    configured_paths,
):
    loaders = build_image_loaders_from_paths(
        config,
        configured_paths,
        False,
    )
    model = build_vae(config, device)
    checkpoint_path = Path(run_dir) / "vae" / "checkpoints" / "best_beta_vae.pt"
    checkpoint = load_checkpoint(model, checkpoint_path, device)

    rows = []
    for split_name in config["evaluation"]["vae_splits"]:
        metrics = evaluate_loader(model, loaders[split_name], device)
        row = {"split": split_name, **metrics}
        rows.append(row)
        print(
            f"{split_name}: SSIM={metrics['ssim']:.6f}, "
            f"MS-SSIM={metrics['ms_ssim']:.6f}"
        )

    metrics_path = Path(run_dir) / "vae" / "metrics" / "reconstruction_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    reconstruction_split = config["evaluation"]["reconstruction_split"]

    saved_count = save_reconstructions(
        config,
        model,
        loaders[reconstruction_split],
        run_dir,
        device,
    )
    summary = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_reconstruction_loss": float(
            checkpoint["validation_reconstruction_loss"]
        ),
        "saved_reconstructions": saved_count,
        "metrics_csv": project_relative_text(project_root, metrics_path),
    }
    summary_path = Path(run_dir) / "vae" / "metrics" / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows


def evaluate_beta_vae(config, project_root, run_dir, device):
    configured_paths = split_paths(config, project_root)
    return evaluate_beta_vae_from_paths(
        config,
        project_root,
        run_dir,
        device,
        configured_paths,
    )


def extract_latent_split(model, loader, device):
    means = []
    log_variances = []
    labels = []
    relative_paths = []

    model.eval()
    with torch.no_grad():
        for images, batch_labels, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            mean, log_variance = model.encoder(images)
            means.append(mean.cpu().numpy().astype(np.float32))
            log_variances.append(log_variance.cpu().numpy().astype(np.float32))
            labels.append(batch_labels.numpy().astype(np.int64))
            relative_paths.extend(batch_paths)

    if not means:
        raise ValueError("An empty image loader was supplied.")
    return {
        "mean": np.concatenate(means, axis=0),
        "log_variance": np.concatenate(log_variances, axis=0),
        "label_index": np.concatenate(labels, axis=0),
        "image_path": np.asarray(relative_paths),
    }


def extract_latent_maps_from_paths(
    config,
    project_root,
    run_dir,
    device,
    configured_paths,
):
    loaders = build_image_loaders_from_paths(
        config,
        configured_paths,
        False,
    )
    model = build_vae(config, device)
    checkpoint_path = Path(run_dir) / "vae" / "checkpoints" / "best_beta_vae.pt"
    load_checkpoint(model, checkpoint_path, device)

    summaries = {}
    for split_name in ["train", "validation", "test"]:
        arrays = extract_latent_split(model, loaders[split_name], device)
        output_path = Path(run_dir) / "latent" / f"{split_name}_latent_maps.npz"
        np.savez_compressed(output_path, **arrays)
        summaries[split_name] = {
            "samples": int(arrays["mean"].shape[0]),
            "mean_shape": list(arrays["mean"].shape),
            "log_variance_shape": list(arrays["log_variance"].shape),
            "path": project_relative_text(project_root, output_path),
        }
        relative_output = project_relative_text(project_root, output_path)
        print(f"Saved {split_name} latent maps -> {relative_output}")

    summary_path = Path(run_dir) / "latent" / "latent_map_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def extract_latent_maps(config, project_root, run_dir, device):
    configured_paths = split_paths(config, project_root)
    return extract_latent_maps_from_paths(
        config,
        project_root,
        run_dir,
        device,
        configured_paths,
    )

