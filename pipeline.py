"""
pipeline.py
===========
End-to-end pipeline: load teachers → train 3 students → ensemble → evaluate.

Usage
-----
    from pipeline import run_full_pipeline

    student_models, results = run_full_pipeline(
        teacher_paths = {...},
        datasets      = {...},
        ...
    )
"""

import os
import torch

from models import ResNet50WithFeatures
from trainers import (
    train_student_logits,
    train_student_at,
    train_student_rkd,
    train_student_mt_logits,
    train_student_mt_at,
    train_student_mt_rkd,
)
from evaluation import evaluate, evaluate_ensemble


def load_teacher_model(path: str, device: str = "cuda") -> ResNet50WithFeatures:
    """Load a pretrained ResNet-50 teacher from a checkpoint file."""
    teacher = ResNet50WithFeatures(num_classes=1, pretrained=False)
    state_dict = torch.load(path, map_location=device)

    # handle DataParallel "module." prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # partial-load safety
    model_keys = set(teacher.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    teacher.load_state_dict(filtered, strict=False)
    teacher = teacher.to(device)
    teacher.eval()
    print(f"  ✓ teacher loaded: {path}  ({len(filtered)}/{len(model_keys)} keys)")
    return teacher


def run_full_pipeline(
    teacher_paths: dict,
    datasets: dict,
    num_epochs: int   = 50,
    lr: float         = 0.01,
    alpha: float      = 1.0,
    beta: float       = 0.5,
    rkd_dist_w: float = 1.0,
    rkd_angle_w: float = 2.0,
    device: str       = "cuda",
    save_dir: str     = "./kd_checkpoints",
    multi_teacher: bool = True,
    train_dataset: str  = "combined",
):
    """
    Full pipeline
    -------------
    1. Load teacher models from checkpoints
    2. Prepare dataloaders
    3. Train student-1  →  Logits KD   (Response-based,  Eq. 3)
    4. Train student-2  →  AT KD       (Feature-based,   Eq. 5)
    5. Train student-3  →  RKD KD      (Relation-based,  Eq. 8)
    6. Evaluate all students + ensemble (Soft Voting, Eq. 9)
    7. Save models & print comparison table

    Parameters
    ----------
    teacher_paths : dict
        {'200k': '/path/to/teacher_200k.pth', '140k': ..., '190k': ...}
    datasets : dict
        {'200k': Dataset_selector, '140k': ..., '190k': ...}
        Each must have .loader_train / .loader_val / .loader_test
    multi_teacher : bool
        If True, distillation loss is averaged over ALL loaded teachers.
    train_dataset : str
        '200k' | '140k' | '190k' | 'combined'
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Load teachers ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1 · Loading Teacher Models")
    print("=" * 70)

    teacher_models = []
    for name in ("200k", "140k", "190k"):
        p = teacher_paths.get(name)
        if p and os.path.exists(p):
            teacher_models.append(load_teacher_model(p, device))
        else:
            print(f"  ⚠ skipping {name} teacher  (path missing)")
    if not teacher_models:
        raise RuntimeError("No teacher models loaded — check paths!")
    print(f"  Teachers ready: {len(teacher_models)}")

    # ── 2. Prepare dataloaders ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2 · Preparing DataLoaders")
    print("=" * 70)

    active_keys = [k for k in ("200k", "140k", "190k") if k in datasets]
    if train_dataset == "combined":
        keys = active_keys
    elif train_dataset in datasets:
        keys = [train_dataset]
    else:
        keys = active_keys

    train_loader = datasets[keys[0]].loader_train
    val_loader   = datasets[keys[0]].loader_val
    test_loader  = datasets[keys[0]].loader_test
    print(f"  Loaders from: {keys[0]}")

    # ── 3-5. Train three students ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEPS 3-5 · Training Three Student Models")
    print("=" * 70)

    mt = multi_teacher and len(teacher_models) > 1

    s_logits = (
        train_student_mt_logits(teacher_models, train_loader, val_loader,
                                num_epochs, lr, alpha, beta, device,
                                os.path.join(save_dir, "student_logits_best.pth"))
        if mt else
        train_student_logits(teacher_models[0], train_loader, val_loader,
                             num_epochs, lr, alpha, beta, device,
                             os.path.join(save_dir, "student_logits_best.pth"))
    )

    s_at = (
        train_student_mt_at(teacher_models, train_loader, val_loader,
                            num_epochs, lr, device,
                            os.path.join(save_dir, "student_at_best.pth"))
        if mt else
        train_student_at(teacher_models[0], train_loader, val_loader,
                         num_epochs, lr, device,
                         os.path.join(save_dir, "student_at_best.pth"))
    )

    s_rkd = (
        train_student_mt_rkd(teacher_models, train_loader, val_loader,
                             num_epochs, lr, rkd_dist_w, rkd_angle_w, device,
                             os.path.join(save_dir, "student_rkd_best.pth"))
        if mt else
        train_student_rkd(teacher_models[0], train_loader, val_loader,
                          num_epochs, lr, rkd_dist_w, rkd_angle_w, device,
                          os.path.join(save_dir, "student_rkd_best.pth"))
    )

    student_models = [s_logits, s_at, s_rkd]
    student_names  = ["Logits KD", "AT KD", "RKD KD"]

    # ── 6. Evaluate ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 6 · Evaluation on Test Set")
    print("=" * 70)

    results = {}
    for name, model in zip(student_names, student_models):
        m = evaluate(model, test_loader, device)
        results[name] = m
        print(f"\n  {name}:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    # Ensemble (Soft Voting — Eq. 9)
    ens_m = evaluate_ensemble(student_models, test_loader, device)
    results["Ensemble (Soft Voting)"] = ens_m
    print(f"\n  Ensemble (Soft Voting — Eq. 9):")
    for k, v in ens_m.items():
        print(f"    {k}: {v:.4f}")

    # Teachers for comparison
    for i, t in enumerate(teacher_models):
        name = f"Teacher {i+1}"
        m = evaluate(t, test_loader, device)
        results[name] = m
        print(f"\n  {name}:  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # ── 7. Save final models ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 7 · Saving Final Models")
    print("=" * 70)

    for tag, model in zip(
        ["student_logits", "student_at", "student_rkd"],
        student_models,
    ):
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"{tag}_final.pth"))
    print(f"  All models saved to {save_dir}")

    # ── Comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    hdr = f"{'Method':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")

    return student_models, results
