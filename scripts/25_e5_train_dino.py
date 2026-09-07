"""E5a/c：在 REAL-Colon 影格上自訓 DINO（ViT-S/16），從隨機初始化開始
（不是微調 ImageNet 預訓練權重）——這樣才能乾淨地測「SSL 目標函數本身有沒有
在這批資料上主動放大採集指紋」，跟 E1b 的 dinov2_random（完全沒訓練過）、
dinov2_pretrained（在 ImageNet 上訓練過）兩個對照組放在同一條軸線上比較。

用法（預設參數是給這台 Mac 本機做 smoke test 用的小規模設定；正式訓練在遠端
GPU server 上執行時，務必調大 --epochs / --batch-size / --n-local-crops，並拿掉
--max-frames-per-video 用完整的訓練語料）：

    # 本機 smoke test（每支影片只抽 3 張、共 180 張、1 個 epoch、CPU/MPS）：
    python3 25_e5_train_dino.py --max-frames-per-video 3 --epochs 1 --batch-size 4 \\
        --n-local-crops 2 --output-dir ../results/e5_smoke_test

    # 遠端 GPU 正式訓練（標準 augmentation，對應 E5a/b）：
    python3 25_e5_train_dino.py --epochs 100 --batch-size 64 --n-local-crops 8 \\
        --output-dir ../results/e5_dino_standard

    # 遠端 GPU 正式訓練（E4 驗證過的雜訊白化 augmentation，對應 E5c）：
    python3 25_e5_train_dino.py --epochs 100 --batch-size 64 --n-local-crops 8 \\
        --add-noise-whitening --output-dir ../results/e5_dino_whitened

輸出：`{output-dir}/checkpoint.pt`（student/teacher backbone + head 權重、
optimizer state、目前 epoch，可斷點續訓）、`{output-dir}/train_log.csv`
（每個 log 間隔的 loss，供畫學習曲線）。
"""

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from e5_vision_transformer import vit_small
from e5_dino_model import DINOHead, MultiCropWrapper, DINOLoss
from e5_dino_augmentations import DINOMultiCropAugmentation
from e5_dino_dataset import build_frame_pool, DINOFrameDataset, multi_crop_collate

EMBED_DIM = 384


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_schedule(base_value: float, final_value: float, epochs: int, steps_per_epoch: int, warmup_epochs: int = 0) -> list:
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    schedule = []
    for step in range(total_steps):
        if step < warmup_steps:
            schedule.append(base_value * step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            schedule.append(final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress)))
    return schedule


def build_model(n_local_crops: int, out_dim: int, device: torch.device):
    student_backbone = vit_small(patch_size=16)
    teacher_backbone = vit_small(patch_size=16)
    student_head = DINOHead(EMBED_DIM, out_dim)
    teacher_head = DINOHead(EMBED_DIM, out_dim)

    student = MultiCropWrapper(student_backbone, student_head).to(device)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head).to(device)

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-labels-csv", type=str, default=None,
                        help="預設用 results/pilot_frame_labels.csv（全部 60 支影片、6000 張）")
    parser.add_argument("--max-frames-per-video", type=int, default=None,
                        help="從既有抽樣結果再抽小子集，只給 smoke test 用；不指定就用全部")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-local-crops", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=8192,
                        help="DINO head 投影維度；官方 ImageNet 規模用 65536，REAL-Colon"
                             "規模小很多，預設縮小到 8192")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--weight-decay-end", type=float, default=0.4)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--add-noise-whitening", action="store_true",
                        help="E5c：加上 E4 驗證過的雜訊白化 augmentation")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="強制用 CPU（除錯用）")
    args = parser.parse_args()

    device = get_device(force_cpu=args.cpu)
    print(f"Using device: {device}")

    frame_paths = build_frame_pool(
        frame_labels_csv=Path(args.frame_labels_csv) if args.frame_labels_csv else None,
        max_frames_per_video=args.max_frames_per_video,
    )
    print(f"訓練語料：{len(frame_paths)} 張影格")
    if len(frame_paths) < args.batch_size:
        raise SystemExit(
            f"影格數（{len(frame_paths)}）小於 batch size（{args.batch_size}），"
            "請減少 --batch-size 或增加 --max-frames-per-video。"
        )

    transform = DINOMultiCropAugmentation(
        n_local_crops=args.n_local_crops, add_noise_whitening=args.add_noise_whitening
    )
    dataset = DINOFrameDataset(frame_paths, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=multi_crop_collate,
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise SystemExit("一個 epoch 都跑不完一個 batch（drop_last=True 且資料量太小），請調整參數。")

    student, teacher = build_model(args.n_local_crops, args.out_dim, device)
    dino_loss = DINOLoss(args.out_dim, n_global_crops=2).to(device)

    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    lr_schedule = cosine_schedule(args.lr, args.min_lr, args.epochs, steps_per_epoch, args.warmup_epochs)
    wd_schedule = cosine_schedule(args.weight_decay, args.weight_decay_end, args.epochs, steps_per_epoch)
    momentum_schedule = cosine_schedule(args.teacher_momentum, 1.0, args.epochs, steps_per_epoch)

    n_crops = 2 + args.n_local_crops
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []

    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_schedule[global_step]
                g["weight_decay"] = wd_schedule[global_step]

            crops = [c.to(device, non_blocking=True) for c in batch]

            student_out = student(crops)
            with torch.no_grad():
                teacher_out = teacher(crops[:2])  # teacher 只看兩個 global crop

            loss = dino_loss(student_out, teacher_out, teacher_temp=args.teacher_temp, n_crops=n_crops)

            if not torch.isfinite(loss):
                raise SystemExit(f"loss 變成 {loss.item()}（非有限值），訓練發散，檢查學習率/資料。")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_teacher(student, teacher, momentum_schedule[global_step])

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"epoch={epoch} step={global_step}/{args.epochs * steps_per_epoch} "
                      f"loss={loss.item():.4f} lr={lr_schedule[global_step]:.2e} elapsed={elapsed:.1f}s", flush=True)
                log_rows.append({"epoch": epoch, "step": global_step, "loss": loss.item(),
                                  "lr": lr_schedule[global_step]})

            global_step += 1

        torch.save({
            "epoch": epoch,
            "student_backbone": student.backbone.state_dict(),
            "teacher_backbone": teacher.backbone.state_dict(),
            "student_head": student.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, out_dir / "checkpoint.pt")
        print(f"epoch {epoch} 結束，checkpoint 已存到 {out_dir / 'checkpoint.pt'}")

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    print(f"訓練結束，共 {global_step} 步，log 存到 {out_dir / 'train_log.csv'}")


if __name__ == "__main__":
    main()
