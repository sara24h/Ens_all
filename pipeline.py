"""
Main pipeline — orchestrates the full KD ensemble workflow.

Design (matching the paper + 3-teacher setup):
  ┌───────────────────────────────────────────────────────────────┐
  │  Student-Logits  ←  Teacher-200k   on  Dataset-200k          │
  │  Student-AT      ←  Teacher-140k   on  Dataset-140k          │
  │  Student-RKD     ←  Teacher-190k   on  Dataset-190k          │
  │                                                               │
  │  Then ensemble all 3 students via Soft Voting (Eq. 9)        │
  └───────────────────────────────────────────────────────────────┘

Each student learns complementary knowledge from a different
teacher-dataset pair, making the ensemble more diverse and powerful.
"""

import os
import sys
import torch

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import load_model
from train import train_logits, train_at, train_rkd
from evaluate import evaluate, evaluate_ensemble, print_results_table


# Mapping: which KD method uses which teacher & dataset
KD_ASSIGNMENTS = {
    'logits': {'teacher_key': '200k', 'dataset_key': '200k'},
    'at':     {'teacher_key': '140k', 'dataset_key': '140k'},
    'rkd':    {'teacher_key': '190k', 'dataset_key': '190k'},
}


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
    test_dataset='200k',      # which dataset to use for final test
):
    """
    ┌──────────────────────────────────────────────────────────┐
    │  Full Knowledge Distillation Ensemble Pipeline           │
    │                                                          │
    │  Student-Logits  ←  Teacher-200k  on  Dataset-200k      │
    │  Student-AT      ←  Teacher-140k  on  Dataset-140k      │
    │  Student-RKD     ←  Teacher-190k  on  Dataset-190k      │
    │                                                          │
    │  Final: Soft Voting Ensemble (Eq. 9)                    │
    └──────────────────────────────────────────────────────────┘

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
        test_dataset:   which dataset's test split to use for final eval
    """
    os.makedirs(save_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────
    # STEP 1: Load Teachers
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1 — Loading Teacher Models")
    print("=" * 70)

    teacher_models = {}
    for name in ['200k', '140k', '190k']:
        p = teacher_paths.get(name)
        if p and os.path.exists(p):
            teacher_models[name] = load_model(p, device=device)
        else:
            print(f"  WARNING: {name} teacher path missing or not found: {p}")

    if len(teacher_models) == 0:
        raise RuntimeError("No teacher models loaded — check paths!")

    # Check that all required teacher-dataset pairs exist
    for kd_name, assignment in KD_ASSIGNMENTS.items():
        t_key = assignment['teacher_key']
        d_key = assignment['dataset_key']
        if t_key not in teacher_models:
            print(f"  WARNING: Teacher for {kd_name} ({t_key}) not loaded!")
        if d_key not in datasets:
            print(f"  WARNING: Dataset for {kd_name} ({d_key}) not available!")

    print(f"  Teachers loaded: {list(teacher_models.keys())}")

    # Auto-detect num_classes from first teacher's FC layer
    first_teacher = next(iter(teacher_models.values()))
    num_classes = first_teacher.fc.out_features
    print(f"  Detected num_classes={num_classes} from teacher FC layer")

    # ────────────────────────────────────────────────────────────
    # STEP 2: Verify DataLoaders
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2 — Verifying DataLoaders")
    print("=" * 70)

    for kd_name, assignment in KD_ASSIGNMENTS.items():
        d_key = assignment['dataset_key']
        t_key = assignment['teacher_key']
        ds = datasets.get(d_key)
        tc = teacher_models.get(t_key)
        if ds and tc:
            print(f"  {kd_name:8s} → Teacher-{t_key} + Dataset-{d_key}  "
                  f"(train={len(ds.loader_train.dataset)}, "
                  f"val={len(ds.loader_val.dataset)}, "
                  f"test={len(ds.loader_test.dataset)})")
        else:
            print(f"  {kd_name:8s} → MISSING teacher or dataset!")

    # ────────────────────────────────────────────────────────────
    # STEP 3: Train Student-Logits  ←  Teacher-200k  on  Dataset-200k
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3 — Training Student-Logits")
    print("  Teacher: 200k  |  Dataset: 200k  |  Method: Response-based (Logits)")
    print("=" * 70)

    s_logits = None
    assignment = KD_ASSIGNMENTS['logits']
    t_key, d_key = assignment['teacher_key'], assignment['dataset_key']

    if t_key in teacher_models and d_key in datasets:
        ds = datasets[d_key]
        s_logits = train_logits(
            teacher_models=teacher_models[t_key],
            train_loader=ds.loader_train,
            val_loader=ds.loader_val,
            num_epochs=num_epochs, lr=lr, alpha=alpha, beta=beta,
            num_classes=num_classes, device=device,
            save_path=os.path.join(save_dir, 'student_logits_best.pth'),
        )
    else:
        print("  SKIPPED — missing teacher or dataset")

    # ────────────────────────────────────────────────────────────
    # STEP 4: Train Student-AT  ←  Teacher-140k  on  Dataset-140k
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 4 — Training Student-AT")
    print("  Teacher: 140k  |  Dataset: 140k  |  Method: Feature-based (AT)")
    print("=" * 70)

    s_at = None
    assignment = KD_ASSIGNMENTS['at']
    t_key, d_key = assignment['teacher_key'], assignment['dataset_key']

    if t_key in teacher_models and d_key in datasets:
        ds = datasets[d_key]
        s_at = train_at(
            teacher_models=teacher_models[t_key],
            train_loader=ds.loader_train,
            val_loader=ds.loader_val,
            num_epochs=num_epochs, lr=lr,
            num_classes=num_classes, device=device,
            save_path=os.path.join(save_dir, 'student_at_best.pth'),
        )
    else:
        print("  SKIPPED — missing teacher or dataset")

    # ────────────────────────────────────────────────────────────
    # STEP 5: Train Student-RKD  ←  Teacher-190k  on  Dataset-190k
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 5 — Training Student-RKD")
    print("  Teacher: 190k  |  Dataset: 190k  |  Method: Relation-based (RKD)")
    print("=" * 70)

    s_rkd = None
    assignment = KD_ASSIGNMENTS['rkd']
    t_key, d_key = assignment['teacher_key'], assignment['dataset_key']

    if t_key in teacher_models and d_key in datasets:
        ds = datasets[d_key]
        s_rkd = train_rkd(
            teacher_models=teacher_models[t_key],
            train_loader=ds.loader_train,
            val_loader=ds.loader_val,
            num_epochs=num_epochs, lr=lr,
            distance_weight=rkd_dist_w, angle_weight=rkd_angle_w,
            num_classes=num_classes, device=device,
            save_path=os.path.join(save_dir, 'student_rkd_best.pth'),
        )
    else:
        print("  SKIPPED — missing teacher or dataset")

    # ────────────────────────────────────────────────────────────
    # STEP 6: Evaluate on Test Set
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 6 — Evaluation on Test Set")
    print("=" * 70)

    # Use the specified test dataset
    if test_dataset not in datasets:
        test_dataset = list(datasets.keys())[0]
        print(f"  test_dataset not found, using: {test_dataset}")

    test_loader = datasets[test_dataset].loader_test
    print(f"  Test dataset: {test_dataset}")

    results = {}
    student_models = []
    student_names  = ['Logits KD', 'AT KD', 'RKD KD']

    for name, model in zip(student_names, [s_logits, s_at, s_rkd]):
        if model is not None:
            m = evaluate(model, test_loader, device)
            results[name] = m
            student_models.append(model)
            print(f"\n  {name}:  acc={m['accuracy']:.4f}  "
                  f"f1={m['f1']:.4f}  auc={m['auc']:.4f}")
        else:
            print(f"\n  {name}:  SKIPPED (not trained)")

    # Ensemble (Soft Voting — Eq. 9)
    if len(student_models) >= 2:
        ens = evaluate_ensemble(student_models, test_loader, device)
        results['Ensemble (Soft Voting)'] = ens
        print(f"\n  Ensemble (Soft Voting — Eq. 9):  "
              f"acc={ens['accuracy']:.4f}  f1={ens['f1']:.4f}  auc={ens['auc']:.4f}")
    else:
        print("\n  Ensemble SKIPPED (need at least 2 trained students)")

    # Teachers (for comparison)
    for name, t in teacher_models.items():
        m = evaluate(t, test_loader, device)
        results[f'Teacher-{name}'] = m
        print(f"  Teacher-{name}:  acc={m['accuracy']:.4f}  "
              f"f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # ────────────────────────────────────────────────────────────
    # STEP 7: Save Final Models
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 7 — Saving Final Models")
    print("=" * 70)

    for name, model in zip(
        ['student_logits', 'student_at', 'student_rkd'],
        [s_logits, s_at, s_rkd],
    ):
        if model is not None:
            path = os.path.join(save_dir, f'{name}_final.pth')
            torch.save(model.state_dict(), path)
            print(f"  Saved: {path}")

    print(f"\n  All models saved to {save_dir}")

    # ────────────────────────────────────────────────────────────
    # Summary Table
    # ────────────────────────────────────────────────────────────
    print_results_table(results)

    return student_models, results
