#!/usr/bin/env python3
"""Train a DINOv3-based binary ROI segmenter for huajiao RGB images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from oil_content_detection.segmentation.data import (
    HuajiaoROIDataset,
    load_oil_split_samples,
)
from oil_content_detection.segmentation.dinov3_model import (
    DinoV3BackboneConfig,
    DinoV3ROISegmenter,
)
from oil_content_detection.segmentation.metrics import (
    binary_dice_loss,
    binary_segmentation_metrics,
)


DEFAULT_REPO_DIR = "/home/yr/yr/code/cv/large_models/dinov3_all/dinov3"
DEFAULT_WEIGHTS = "/home/yr/yr/code/cv/large_models/dinov3_all/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_PROCESSED = "data/processed/huajiao_refined"
DEFAULT_XML = "/home/yr/yr/data/huajiao_all/annotations_20251231_197.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", default=DEFAULT_REPO_DIR)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED)
    parser.add_argument("--annotation-xml", default=DEFAULT_XML)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-train", type=int, default=None)
    parser.add_argument("--subset-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Use CUDA autocast mixed precision."
    )
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
        help="BCE weight; dice weight is 1 - bce_weight.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def maybe_subset(dataset, n: int | None, seed: int):
    if n is None or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, indices)


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [
        "dice",
        "iou",
        "precision",
        "recall",
        "accuracy",
        "pred_coverage",
        "target_coverage",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate_model(
    model: DinoV3ROISegmenter,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="val", leave=False, ncols=100):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            pred = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy()
            target = masks.detach().cpu().numpy() > 0.5
            for i in range(pred.shape[0]):
                rows.append(binary_segmentation_metrics(pred[i, 0], target[i, 0]))
    return average_metric_rows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"device={device} amp={use_amp}")
    if device.type == "cuda":
        print(
            f"gpu={torch.cuda.get_device_name(0)} mem_gb={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}"
        )

    train_samples = load_oil_split_samples(
        args.processed_dir, "train", args.annotation_xml
    )
    val_samples = load_oil_split_samples(args.processed_dir, "val", args.annotation_xml)
    print(f"samples train={len(train_samples)} val={len(val_samples)}")

    train_set = HuajiaoROIDataset(
        args.processed_dir,
        "train",
        args.annotation_xml,
        image_size=args.image_size,
        augment=True,
        samples=train_samples,
    )
    val_set = HuajiaoROIDataset(
        args.processed_dir,
        "val",
        args.annotation_xml,
        image_size=args.image_size,
        augment=False,
        samples=val_samples,
    )
    train_set = maybe_subset(train_set, args.subset_train, args.seed)
    val_set = maybe_subset(val_set, args.subset_val, args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    backbone_cfg = DinoV3BackboneConfig(
        repo_dir=args.dinov3_repo,
        weights_path=args.weights,
        model_name=args.model_name,
        image_size=args.image_size,
        patch_size=16,
        freeze_backbone=args.freeze_backbone,
    )
    model = DinoV3ROISegmenter(
        backbone_config=backbone_cfg,
        head_channels=args.head_channels,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    bce_loss = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "artifact_type": "huajiao-dinov3-roi-segmentation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": args.image_size,
        "threshold": args.threshold,
        "backbone": backbone_cfg.to_dict(),
        "head": {
            "feature_dim": model.feature_dim,
            "head_channels": args.head_channels,
            "dropout": args.dropout,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "freeze_backbone": args.freeze_backbone,
            "amp": use_amp,
            "bce_weight": args.bce_weight,
        },
        "data": {
            "processed_dir": str(Path(args.processed_dir).resolve()),
            "annotation_xml": str(Path(args.annotation_xml).resolve()),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "label_names": ["peper", "pepper"],
        },
        "weights_sha256": sha256_file(args.weights),
    }
    write_json(out_dir / "config.json", config)

    history: list[dict[str, float]] = []
    best_score = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", ncols=100)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss_bce = bce_loss(logits, masks)
                loss_dice = binary_dice_loss(logits, masks)
                loss = args.bce_weight * loss_bce + (1.0 - args.bce_weight) * loss_dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        val_metrics = evaluate_model(
            model, val_loader, device, threshold=args.threshold
        )
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False, indent=2))

        checkpoint = {
            "epoch": epoch + 1,
            "head_state_dict": model.head.state_dict(),
            "config": config,
            "metrics": epoch_row,
        }
        torch.save(checkpoint, out_dir / "latest_model.pt")
        score = float(val_metrics.get("dice", 0.0))
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            torch.save(checkpoint, out_dir / "best_model.pt")
            print(f"saved best_model.pt epoch={best_epoch} val_dice={best_score:.4f}")
        write_json(
            out_dir / "metrics.json",
            {"history": history, "best_epoch": best_epoch, "best_val_dice": best_score},
        )

    manifest = {
        "artifact_type": "huajiao-dinov3-roi-segmentation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": best_epoch,
        "best_val_dice": best_score,
        "model_name": args.model_name,
        "backbone_weights": str(Path(args.weights).resolve()),
        "backbone_weights_sha256": config["weights_sha256"],
        "best_model_sha256": (
            sha256_file(out_dir / "best_model.pt")
            if (out_dir / "best_model.pt").exists()
            else None
        ),
        "config": "config.json",
        "metrics": "metrics.json",
    }
    write_json(out_dir / "manifest.json", manifest)
    model_card = f"""# 花椒 ROI DINOv3 分割模型\n\n- 模型: {args.model_name} + lightweight binary segmentation head\n- Backbone 权重: {args.weights}\n- 训练 split: 当前含油率 train，样本数 {len(train_samples)}\n- 测试/验证 split: 当前含油率 val，样本数 {len(val_samples)}\n- 输入: RGB 图片，训练时 resize 到 {args.image_size}x{args.image_size}\n- 输出: 二值花椒 ROI mask，默认阈值 {args.threshold}\n- 最佳 epoch: {best_epoch}\n- 最佳 val Dice: {best_score:.6f}\n\n注意：artifact 不包含原始训练数据和 DINOv3 backbone 权重；部署环境需要能访问 config.json 中记录的本地权重路径，或由发布流程同步该权重。\n"""
    (out_dir / "model_card.md").write_text(model_card, encoding="utf-8")
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
