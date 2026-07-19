# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=4.0",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "torch>=2.5",
#   "torchvision>=0.20",
# ]
# ///
"""Scaled CIFAR-10 repeated-run evidence for ML-Agent Claim 3.

This is deliberately labeled a model/task proxy: it measures repeated CIFAR-10
accuracy from fresh training, but it is not the unreleased ML-Agent checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class CifarConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class HubCifar10(Dataset):
    """Torch adapter for the Hub-hosted uoft-cs/cifar10 parquet dataset."""

    def __init__(self, split: str, transform, cache_dir: Path, source_dir: Path | None) -> None:
        source = str(source_dir) if source_dir is not None else "uoft-cs/cifar10"
        print(f"DATASET_LOAD_START split={split} source={source}", flush=True)
        if source_dir is None:
            self.dataset = load_dataset("uoft-cs/cifar10", split=split, cache_dir=str(cache_dir))
        else:
            mounted_path = source_dir / "plain_text" / f"{split}-00000-of-00001.parquet"
            local_source_dir = cache_dir / "staged-source"
            local_source_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_source_dir / mounted_path.name
            if not local_path.exists() or local_path.stat().st_size != mounted_path.stat().st_size:
                print(f"DATASET_LOCAL_COPY_START split={split} bytes={mounted_path.stat().st_size}", flush=True)
                shutil.copyfile(mounted_path, local_path)
                print(f"DATASET_LOCAL_COPY_DONE split={split} sha256={checksum(local_path)}", flush=True)
            self.dataset = load_dataset(
                "parquet",
                data_files={split: str(local_path)},
                split=split,
                cache_dir=str(cache_dir),
            )
        self.transform = transform
        print(f"DATASET_LOAD_DONE split={split} samples={len(self.dataset)}", flush=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        return self.transform(row["img"].convert("RGB")), int(row["label"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_loaders(data_dir: Path, source_dir: Path | None, batch_size: int, workers: int, train_limit: int, synthetic: bool, seed: int):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if synthetic:
        train = datasets.FakeData(size=512, image_size=(3, 32, 32), num_classes=10, transform=train_transform, random_offset=seed)
        test = datasets.FakeData(size=256, image_size=(3, 32, 32), num_classes=10, transform=test_transform, random_offset=seed + 1)
    else:
        train = HubCifar10("train", train_transform, data_dir, source_dir)
        test = HubCifar10("test", test_transform, data_dir, source_dir)
        if train_limit < len(train):
            rng = np.random.default_rng(seed)
            train = Subset(train, rng.choice(len(train), size=train_limit, replace=False).tolist())
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available(), generator=generator)
    test_loader = DataLoader(test, batch_size=batch_size * 2, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        correct += int((model(inputs).argmax(1) == targets).sum())
        total += int(targets.numel())
    return correct / total


def train_one(seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, float | int]:
    print(f"SEED_START={seed}", flush=True)
    seed_everything(seed)
    train_loader, test_loader = make_loaders(args.data_dir, args.dataset_source_dir, args.batch_size, args.workers, args.train_limit, args.synthetic, seed)
    model = CifarConvNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    best = 0.0
    final = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            if batch_index % 50 == 0 or batch_index == len(train_loader):
                print(
                    f"SEED={seed} EPOCH={epoch}/{args.epochs} BATCH={batch_index}/{len(train_loader)} "
                    f"MEAN_LOSS={running_loss / batch_index:.6f}",
                    flush=True,
                )
        scheduler.step()
        final = evaluate(model, test_loader, device)
        best = max(best, final)
        print(f"SEED={seed} EPOCH={epoch}/{args.epochs} TEST_ACCURACY={final:.6f}", flush=True)
    return {
        "seed": seed,
        "final_accuracy": final,
        "best_epoch_accuracy": best,
        "test_samples": len(test_loader.dataset),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cifar_gpu"))
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/cifar10"))
    parser.add_argument("--dataset-source-dir", type=Path, help="read staged CIFAR parquet from this directory")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-limit", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17897, 17898, 17899, 17900, 17901])
    parser.add_argument("--synthetic", action="store_true", help="smoke only; verifier rejects this as evidence")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE={device} GPU={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)
    runs = [train_one(seed, args, device) for seed in args.seeds]
    values = np.asarray([float(run["final_accuracy"]) for run in runs])
    summary = {
        "protocol": {
            "label": "scaled_cifar10_proxy_not_ml_agent",
            "dataset": "FakeData" if args.synthetic else "CIFAR-10",
            "synthetic": args.synthetic,
            "model": "CifarConvNet (fresh training; not ML-Agent/Qwen2.5-7B)",
            "epochs": args.epochs,
            "train_limit": args.train_limit,
            "dataset_source_dir": str(args.dataset_source_dir) if args.dataset_source_dir else None,
            "seeds": args.seeds,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "measurement": {
            "mean_accuracy": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "best_run_accuracy": float(values.max()),
            "worst_run_accuracy": float(values.min()),
            "ci95_half_width": float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None,
            "runs": runs,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar([str(seed) for seed in args.seeds], values * 100, color="#247BA0")
    axis.axhline(values.mean() * 100, color="#E76F51", linestyle="--", label=f"measured mean {values.mean() * 100:.2f}%")
    axis.set(xlabel="Seed", ylabel="Final CIFAR-10 accuracy (%)", title="Fresh repeated CIFAR-10 runs (scaled proxy)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "cifar_repeats.png", dpi=180)
    plt.close(figure)
    (args.output_dir / "CHECKSUMS.json").write_text(json.dumps({"results.json": checksum(result_path)}, indent=2) + "\n")
    print(f"RESULTS_JSON={result_path}")
    print(f"MEAN_ACCURACY={values.mean():.6f}")
    print(f"BEST_RUN_ACCURACY={values.max():.6f}")
    print(f"RUNS={len(values)}")


if __name__ == "__main__":
    main()
