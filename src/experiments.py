import argparse
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cnn import CustomResNet, EnhancedLeNet, LeNet
from config import BATCH_SIZE, CSV_DIR_TRAIN, IMG_DIR_TRAIN, param_config
from dataset import DigitDataset
from transforms import transform


DEFAULT_FULL_SEEDS = [42, 52, 62]
MODEL_ORDER = ["lenet", "enhancedlenet", "resnet"]
OPTIMIZER_ORDER = ["adam", "adamw", "sgd"]
TRANSFORM_ORDER = ["no_aug_no_norm", "no_aug_norm", "aug_norm"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(batch_size, normalize=True, augment=True):
    df = pd.read_csv(CSV_DIR_TRAIN)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Category"]
    )

    train_dataset = DigitDataset(
        train_df, IMG_DIR_TRAIN, transform(train=True, normalize=normalize, augment=augment)
    )
    val_dataset = DigitDataset(
        val_df, IMG_DIR_TRAIN, transform(train=False, normalize=normalize, augment=False)
    )

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_val


def build_model(model_name):
    model_name = model_name.lower()
    if model_name == "lenet":
        return LeNet()
    if model_name == "enhancedlenet":
        return EnhancedLeNet()
    if model_name == "resnet":
        return CustomResNet()
    raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, run_config, steps_per_epoch):
    scheduler_name = run_config.get("scheduler", "").lower()
    if scheduler_name != "onecycle":
        return None
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=run_config["lr"],
        epochs=run_config["epochs"],
        steps_per_epoch=steps_per_epoch,
        pct_start=run_config.get("pct_start", 0.3),
        div_factor=run_config.get("div_factor", 25.0),
        final_div_factor=run_config.get("final_div_factor", 1e4),
    )


def train_single_run(run_config, seed, batch_size, study_name, factor, setting_name):
    set_seed(seed)
    device = get_device()
    normalize = run_config.get("normalize", True)
    augment = run_config.get("augment", True)
    dataloader_train, dataloader_val = build_dataloaders(
        batch_size=batch_size, normalize=normalize, augment=augment
    )

    model = build_model(run_config["model"]).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=run_config.get("label_smoothing", 0.0),
    )
    optimizer = build_optimizer(
        run_config["optimizer"],
        model.parameters(),
        run_config["lr"],
        run_config.get("weight_decay", 0.0),
    )
    scheduler = build_scheduler(optimizer, run_config, len(dataloader_train))

    history_rows = []
    best_val_acc = -1.0
    best_epoch = -1
    best_val_loss = float("inf")

    epoch_iterator = tqdm(
        range(1, run_config["epochs"] + 1),
        desc=f"{study_name}:{factor}:{setting_name}:seed{seed}",
        leave=False,
    )
    for epoch in epoch_iterator:
        model.train()
        running_train_loss = 0.0
        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_train_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = running_train_loss / len(dataloader_train)
        val_loss = running_val_loss / len(dataloader_val)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_loss = val_loss

        history_rows.append(
            {
                "study": study_name,
                "factor": factor,
                "setting": setting_name,
                "model": run_config["model"].lower(),
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        epoch_iterator.set_postfix(
            {
                "epoch": epoch,
                "val_acc": f"{val_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
            }
        )

    run_row = {
        "study": study_name,
        "factor": factor,
        "setting": setting_name,
        "model": run_config["model"].lower(),
        "seed": seed,
        "optimizer": run_config["optimizer"].lower(),
        "lr": run_config["lr"],
        "weight_decay": run_config.get("weight_decay", 0.0),
        "label_smoothing": run_config.get("label_smoothing", 0.0),
        "scheduler": run_config.get("scheduler", ""),
        "epochs": run_config["epochs"],
        "normalize": normalize,
        "augment": augment,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_val_acc": history_rows[-1]["val_acc"],
        "final_val_loss": history_rows[-1]["val_loss"],
    }

    return run_row, history_rows


def summarize_results(df, group_cols):
    return (
        df.groupby(group_cols)["best_val_acc"]
        .agg(best_val_acc_mean="mean", best_val_acc_std="std", runs="count")
        .reset_index()
    )


def save_dataframe(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def configure_publication_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def get_zoom_ylim(values, margin=0.0006, min_span=0.0020):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.9900, 1.0000

    ymin = float(arr.min() - margin)
    ymax = float(arr.max() + margin)
    if ymax - ymin < min_span:
        center = (ymin + ymax) / 2
        ymin = center - min_span / 2
        ymax = center + min_span / 2

    ymin = max(0.0, ymin)
    ymax = min(1.0, ymax)
    if ymax - ymin < 1e-6:
        ymax = min(1.0, ymin + min_span)
        ymin = max(0.0, ymax - min_span)
    return ymin, ymax


def style_accuracy_axis(ax, values):
    ymin, ymax = get_zoom_ylim(values)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def order_factor_settings(factor_df, factor):
    if factor == "optimizer":
        factor_df["setting"] = pd.Categorical(
            factor_df["setting"], categories=OPTIMIZER_ORDER, ordered=True
        )
        return factor_df.sort_values("setting")
    if factor == "transform":
        factor_df["setting"] = pd.Categorical(
            factor_df["setting"], categories=TRANSFORM_ORDER, ordered=True
        )
        return factor_df.sort_values("setting")
    if factor in {"weight_decay", "label_smoothing"}:
        factor_df["sort_value"] = factor_df["setting"].astype(float)
        factor_df = factor_df.sort_values("sort_value").drop(columns=["sort_value"])
        return factor_df
    return factor_df.sort_values("setting")


def plot_model_comparison(summary_df, output_dir):
    plot_df = summary_df.copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=MODEL_ORDER, ordered=True)
    plot_df = plot_df.sort_values("model")

    x = np.arange(len(plot_df))
    means = plot_df["best_val_acc_mean"].to_numpy()
    errs = plot_df["best_val_acc_std"].fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, means, yerr=errs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"].tolist())
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("CNN Backbone Comparison (mean ± std across seeds)")
    style_accuracy_axis(ax, means)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_best_val_acc.png", dpi=300)
    plt.close(fig)


def plot_ablation_summary(summary_df, output_dir):
    factors = ["optimizer", "weight_decay", "label_smoothing", "transform"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, factor in enumerate(factors):
        ax = axes[idx]
        factor_df = summary_df[summary_df["factor"] == factor].copy()
        if factor_df.empty:
            ax.set_axis_off()
            continue

        factor_df = order_factor_settings(factor_df, factor)
        x = np.arange(len(factor_df))
        means = factor_df["best_val_acc_mean"].to_numpy()
        errs = factor_df["best_val_acc_std"].fillna(0).to_numpy()

        ax.bar(x, means, yerr=errs, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(factor_df["setting"].tolist(), rotation=20, ha="right")
        ax.set_ylabel("Best Validation Accuracy")
        ax.set_title(f"ResNet ablation: {factor}")
        style_accuracy_axis(ax, means)

    fig.tight_layout()
    fig.savefig(output_dir / "resnet_ablation_best_val_acc.png", dpi=300)
    plt.close(fig)


def plot_model_learning_curves(history_df, output_dir):
    curve_df = (
        history_df.groupby(["model", "epoch"], as_index=False)["val_acc"]
        .agg(val_acc_mean="mean", val_acc_std="std")
    )
    curve_df["model"] = pd.Categorical(curve_df["model"], categories=MODEL_ORDER, ordered=True)
    curve_df = curve_df.sort_values(["model", "epoch"])

    fig, ax = plt.subplots(figsize=(9, 5))
    all_means = []
    for model_name in MODEL_ORDER:
        model_df = curve_df[curve_df["model"] == model_name]
        if model_df.empty:
            continue
        x = model_df["epoch"].to_numpy()
        y = model_df["val_acc_mean"].to_numpy()
        s = model_df["val_acc_std"].fillna(0).to_numpy()
        all_means.extend(y.tolist())
        ax.plot(x, y, marker="o", linewidth=2, label=model_name)
        ax.fill_between(x, np.maximum(0, y - s), np.minimum(1, y + s), alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("CNN Validation Accuracy Curves (mean ± std across seeds)")
    style_accuracy_axis(ax, all_means)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_val_acc_lines.png", dpi=300)
    plt.close(fig)


def plot_ablation_line_summary(summary_df, output_dir):
    factors = ["optimizer", "weight_decay", "label_smoothing", "transform"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, factor in enumerate(factors):
        ax = axes[idx]
        factor_df = summary_df[summary_df["factor"] == factor].copy()
        if factor_df.empty:
            ax.set_axis_off()
            continue

        factor_df = order_factor_settings(factor_df, factor)
        x = np.arange(len(factor_df))
        means = factor_df["best_val_acc_mean"].to_numpy()
        errs = factor_df["best_val_acc_std"].fillna(0).to_numpy()

        ax.plot(x, means, marker="o", linewidth=2)
        ax.errorbar(x, means, yerr=errs, fmt="none", ecolor="black", capsize=4, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(factor_df["setting"].tolist(), rotation=20, ha="right")
        ax.set_ylabel("Best Validation Accuracy")
        ax.set_title(f"ResNet ablation line plot: {factor}")
        style_accuracy_axis(ax, means)

    fig.tight_layout()
    fig.savefig(output_dir / "resnet_ablation_best_val_acc_lines.png", dpi=300)
    plt.close(fig)


def run_cnn_comparison(base_config, seeds, batch_size, output_dir):
    run_rows = []
    history_rows = []
    run_plan = [(model_name, seed) for model_name in MODEL_ORDER for seed in seeds]
    run_iterator = tqdm(run_plan, desc="cnn_comparison runs")
    for model_name, seed in run_iterator:
            run_config = deepcopy(base_config)
            run_config["model"] = model_name
            run_config["normalize"] = True
            run_config["augment"] = True
            run_row, history = train_single_run(
                run_config=run_config,
                seed=seed,
                batch_size=batch_size,
                study_name="cnn_comparison",
                factor="model",
                setting_name=model_name,
            )
            run_rows.append(run_row)
            history_rows.extend(history)
            run_iterator.set_postfix(
                {
                    "model": model_name,
                    "seed": seed,
                    "best_acc": f"{run_row['best_val_acc']:.4f}",
                }
            )

    runs_df = pd.DataFrame(run_rows)
    history_df = pd.DataFrame(history_rows)
    summary_df = summarize_results(runs_df, ["model"])

    save_dataframe(runs_df, output_dir / "cnn_comparison_runs.csv")
    save_dataframe(history_df, output_dir / "cnn_comparison_history.csv")
    save_dataframe(summary_df, output_dir / "cnn_comparison_summary.csv")
    plot_model_comparison(summary_df, output_dir)
    plot_model_learning_curves(history_df, output_dir)
    return runs_df, history_df, summary_df


def run_resnet_ablations(base_config, seeds, batch_size, output_dir):
    ablation_grid = {
        "optimizer": [
            ("adam", {"optimizer": "Adam"}),
            ("adamw", {"optimizer": "AdamW"}),
            ("sgd", {"optimizer": "SGD"}), # missing implementation in training
        ],
        "weight_decay": [
            ("0", {"weight_decay": 0.0}),
            ("1e-5", {"weight_decay": 1e-5}),
            ("1e-4", {"weight_decay": 1e-4}),
            ("1e-3", {"weight_decay": 1e-3}),
        ],
        "label_smoothing": [
            ("0.0", {"label_smoothing": 0.0}),
            ("0.05", {"label_smoothing": 0.05}),
            ("0.1", {"label_smoothing": 0.1}),
        ],
        "transform": [
            (
                "no_aug_no_norm",
                {"augment": False, "normalize": False},
            ),
            (
                "no_aug_norm",
                {"augment": False, "normalize": True},
            ),
            (
                "aug_norm",
                {"augment": True, "normalize": True},
            ),
        ],
    }

    run_rows = []
    history_rows = []
    run_plan = [
        (factor, setting_name, overrides, seed)
        for factor, settings in ablation_grid.items()
        for setting_name, overrides in settings
        for seed in seeds
    ]
    run_iterator = tqdm(run_plan, desc="resnet_ablation runs")
    for factor, setting_name, overrides, seed in run_iterator:
                run_config = deepcopy(base_config)
                run_config["model"] = "resnet"
                run_config["normalize"] = True
                run_config["augment"] = True
                run_config.update(overrides)

                run_row, history = train_single_run(
                    run_config=run_config,
                    seed=seed,
                    batch_size=batch_size,
                    study_name="resnet_ablation",
                    factor=factor,
                    setting_name=setting_name,
                )
                run_rows.append(run_row)
                history_rows.extend(history)
                run_iterator.set_postfix(
                    {
                        "factor": factor,
                        "setting": setting_name,
                        "seed": seed,
                        "best_acc": f"{run_row['best_val_acc']:.4f}",
                    }
                )

    runs_df = pd.DataFrame(run_rows)
    history_df = pd.DataFrame(history_rows)
    summary_df = summarize_results(runs_df, ["factor", "setting"])

    save_dataframe(runs_df, output_dir / "resnet_ablation_runs.csv")
    save_dataframe(history_df, output_dir / "resnet_ablation_history.csv")
    save_dataframe(summary_df, output_dir / "resnet_ablation_summary.csv")
    plot_ablation_summary(summary_df, output_dir)
    plot_ablation_line_summary(summary_df, output_dir)
    return runs_df, history_df, summary_df


def parse_seed_list(seed_text):
    if not seed_text:
        return None
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment runner for CNN comparison and ResNet ablations.")
    parser.add_argument(
        "--mode",
        choices=["compare_models", "ablate_resnet", "all"],
        default="all",
        help="Which experiment block to run.",
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="full",
        help="Quick profile runs fewer epochs/seeds for faster iteration.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments",
        help="Directory for CSV and plot artifacts.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for dataloaders.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list, e.g. 42,52,62.",
    )
    return parser.parse_args()


def resolve_runtime(base_config, args):
    run_config = deepcopy(base_config)
    if args.epochs is not None:
        run_config["epochs"] = args.epochs
    elif args.profile == "quick":
        run_config["epochs"] = min(5, run_config["epochs"])

    seeds = parse_seed_list(args.seeds)
    if seeds is None:
        seeds = [42] if args.profile == "quick" else DEFAULT_FULL_SEEDS
    return run_config, seeds


def main():
    configure_publication_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config, seeds = resolve_runtime(param_config, args)
    print(f"Running mode={args.mode} profile={args.profile} seeds={seeds}")
    print(f"Base config: {base_config}")

    if args.mode in {"compare_models", "all"}:
        run_cnn_comparison(
            base_config=base_config,
            seeds=seeds,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )

    if args.mode in {"ablate_resnet", "all"}:
        run_resnet_ablations(
            base_config=base_config,
            seeds=seeds,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )

    print(f"Saved experiment artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()