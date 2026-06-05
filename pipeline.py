"""
Main pipeline — orchestrates the full KD ensemble workflow.

  1. Load teacher models
  2. Train 3 student models (Logits, AT, RKD)
  3. Evaluate individual students + Soft Voting ensemble
  4. Save models & print comparison table
"""

import os
import torch

from models import ResNet50WithFeatures
from utils import load_model
from losses import StudentLoss, LogitsDistillationLoss, AttentionTransferLoss, RKDLoss
from train import train_logits, train_at, train_rkd
from evaluate import evaluate, evaluate_ensemble, print_results_table


def run_pipeline(
    teacher_paths,
    datasets,
    num_epochs=50,
    lr=0.01,
    alpha=1.0,
    beta=0.5,
    rkd_dist_w=1.0,
    rkd_angle_w=2.0,
    device='cuda',
    save_dir='./kd_checkpoints',
    multi_teacher=True,
    train_dataset='combined',
):
    """
    ┌──────────────────────────────────────────────────────┐
    │  Full Knowledge Distillation Ensemble Pipeline       │
    │                                                      │
    │  1. Load 3 pretrained ResNet-50 teacher models       │
    │  2. Train student-Logits  (Response-based KD)        │
    │  3. Train student-AT     (Feature-based KD)          │
    │  4. Train student-RKD    (Relation-based KD)         │
    │  5. Evaluate all + Ensemble (Soft Voting)            │
    │  6. Save & report                                    │
    └──────────────────────────────────────────────────────┘

    Args:
        teacher_paths:  dict {'200k': path, '140k': path, '190k': path}
        datasets:       dict {'200k': Dataset_selector, ...}
        num_epochs:     training epochs per student
        lr:             learning rate
        alpha:          student loss weight  (Logits KD, Eq. 3)
        beta:           distillation weight  (Logits KD, Eq. 3)
        rkd_dist_w:     distance-wise weight (RKD, Eq. 6)
        rkd_angle_w:    angle-wise weight    (RKD, Eq. 6)
        device:         'cuda' or 'cpu'
        save_dir:       checkpoint directory
        multi_teacher:  average distillation over all teachers
        train_dataset:  '200k'|'140k'|'190k'|'combined'
    """
    os.makedirs(save_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────
    # STEP 1: Load Teachers
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1 — Loading Teacher Models")
    print("=" * 70)

    teacher_models = []
    for name in ['200k', '140k', '190k']:
        p = teacher_paths.get(name)
        if p and os.path.exists(p):
            teacher_models.append(load_model(p, device=device))
        else:
            print(f"  Skipping {name} teacher (path missing)")

    if not teacher_models:
        raise RuntimeError("No teacher models loaded — check paths!")
    print(f"  Teachers ready: {len(teacher_models)}")

    # ────────────────────────────────────────────────────────────
    # STEP 2: Prepare Dataloaders
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2 — Preparing DataLoaders")
    print("=" * 70)

    available = [k for k in ['200k', '140k', '190k'] if k in datasets]
    if train_dataset in datasets:
        keys = [train_dataset]
    else:
        keys = available

    train_loader = datasets[keys[0]].loader_train
    val_loader   = datasets[keys[0]].loader_val
    test_loader  = datasets[keys[0]].loader_test
    print(f"  Using dataset: {keys[0]}")

    # ────────────────────────────────────────────────────────────
    # STEP 3-5: Train Three Students
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3-5 — Training Student Models")
    print("=" * 70)

    teachers = teacher_models if multi_teacher and len(teacher_models) > 1 \
               else teacher_models[0]

    s_logits = train_logits(
        teachers, train_loader, val_loader,
        num_epochs=num_epochs, lr=lr, alpha=alpha, beta=beta,
        device=device,
        save_path=os.path.join(save_dir, 'student_logits_best.pth'),
    )

    s_at = train_at(
        teachers, train_loader, val_loader,
        num_epochs=num_epochs, lr=lr,
        device=device,
        save_path=os.path.join(save_dir, 'student_at_best.pth'),
    )

    s_rkd = train_rkd(
        teachers, train_loader, val_loader,
        num_epochs=num_epochs, lr=lr,
        distance_weight=rkd_dist_w, angle_weight=rkd_angle_w,
        device=device,
        save_path=os.path.join(save_dir, 'student_rkd_best.pth'),
    )

    student_models = [s_logits, s_at, s_rkd]
    student_names  = ['Logits KD', 'AT KD', 'RKD KD']

    # ────────────────────────────────────────────────────────────
    # STEP 6: Evaluate
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 6 — Evaluation on Test Set")
    print("=" * 70)

    results = {}
    for name, model in zip(student_names, student_models):
        m = evaluate(model, test_loader, device)
        results[name] = m
        print(f"\n  {name}:  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # Ensemble (Soft Voting — Eq. 9)
    ens = evaluate_ensemble(student_models, test_loader, device)
    results['Ensemble (Soft Voting)'] = ens
    print(f"\n  Ensemble (Soft Voting):  acc={ens['accuracy']:.4f}  "
          f"f1={ens['f1']:.4f}  auc={ens['auc']:.4f}")

    # Teachers for comparison
    for i, t in enumerate(teacher_models):
        m = evaluate(t, test_loader, device)
        results[f'Teacher {i+1}'] = m
        print(f"  Teacher {i+1}:  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    # ────────────────────────────────────────────────────────────
    # STEP 7: Save Final Models
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 7 — Saving Final Models")
    print("=" * 70)

    for name, model in zip(
        ['student_logits', 'student_at', 'student_rkd'], student_models
    ):
        path = os.path.join(save_dir, f'{name}_final.pth')
        torch.save(model.state_dict(), path)
    print(f"  All models saved to {save_dir}")

    # ────────────────────────────────────────────────────────────
    # Summary Table
    # ────────────────────────────────────────────────────────────
    print_results_table(results)

    return student_models, results
