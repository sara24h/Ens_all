"""
Training loops for the three Knowledge Distillation methods.

  1. train_logits()  — Response-based KD  (Eq. 3)
  2. train_at()      — Feature-based KD   (Eq. 5)
  3. train_rkd()     — Relation-based KD  (Eq. 8)

Each function supports:
  - Single-teacher mode
  - Multi-teacher mode (distillation loss averaged over all teachers)
"""

import os
import copy
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models import ResNet50WithFeatures
from losses import StudentLoss, LogitsDistillationLoss, AttentionTransferLoss, RKDLoss


# ═══════════════════════════════════════════════════════════════════
# 1. Response-based KD  (Logits) — Eq. 3
# ═══════════════════════════════════════════════════════════════════

def train_logits(teacher_models, train_loader, val_loader,
                 num_epochs=50, lr=0.01, alpha=1.0, beta=0.5,
                 device='cuda', save_path='student_logits_best.pth'):
    """
    Train student with Response-based KD (Logits MSE).
    Eq. 3: L = α · L_S + β · L_logits

    Args:
        teacher_models: list of teacher models (or single model)
        train_loader:   training DataLoader
        val_loader:     validation DataLoader
        num_epochs:     number of training epochs
        lr:             learning rate
        alpha:          student loss weight
        beta:           distillation loss weight
        device:         'cuda' or 'cpu'
        save_path:      path to save best model checkpoint
    """
    print("\n" + "=" * 70)
    print("  Response-based KD (Logits) — Eq. 3")
    print("=" * 70)

    # ── Setup ──
    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    if not isinstance(teacher_models, list):
        teacher_models = [teacher_models]
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    logits_loss_fn  = LogitsDistillationLoss()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * m) for m in [0.5, 0.75]],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    # ── Training loop ──
    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Logits Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Teacher forward
            with torch.no_grad():
                t_logits_list = [t(images) for t in teacher_models]

            # Student forward
            s_logits = student(images)

            # Loss
            l_s  = student_loss_fn(s_logits, labels)
            l_kd = torch.stack(
                [logits_loss_fn(tl, s_logits) for tl in t_logits_list]
            ).mean()
            loss = alpha * l_s + beta * l_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend(
                (torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy()
            )
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = _quick_eval(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m:.4f}")

        if val_m > best_acc:
            best_acc = val_m
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    -> Saved best (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


# ═══════════════════════════════════════════════════════════════════
# 2. Feature-based KD (Attention Transfer) — Eq. 5
# ═══════════════════════════════════════════════════════════════════

def train_at(teacher_models, train_loader, val_loader,
             num_epochs=50, lr=0.01, device='cuda',
             save_path='student_at_best.pth'):
    """
    Train student with Feature-based KD (Attention Transfer).
    Eq. 5: L = L_S + L_AT
    """
    print("\n" + "=" * 70)
    print("  Feature-based KD (Attention Transfer) — Eq. 5")
    print("=" * 70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    if not isinstance(teacher_models, list):
        teacher_models = [teacher_models]
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    at_loss_fn      = AttentionTransferLoss()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * m) for m in [0.5, 0.75]],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"AT Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                t_attn_list = [
                    t(images, return_attention=True)[1] for t in teacher_models
                ]

            s_logits, s_attn = student(images, return_attention=True)

            l_s  = student_loss_fn(s_logits, labels)
            l_at = torch.stack(
                [at_loss_fn(ta, s_attn) for ta in t_attn_list]
            ).mean()
            loss = l_s + l_at

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend(
                (torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy()
            )
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = _quick_eval(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m:.4f}")

        if val_m > best_acc:
            best_acc = val_m
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    -> Saved best (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


# ═══════════════════════════════════════════════════════════════════
# 3. Relation-based KD (RKD) — Eq. 8
# ═══════════════════════════════════════════════════════════════════

def train_rkd(teacher_models, train_loader, val_loader,
              num_epochs=50, lr=0.01,
              distance_weight=1.0, angle_weight=2.0,
              device='cuda', save_path='student_rkd_best.pth'):
    """
    Train student with Relation-based KD (RKD).
    Eq. 8: L = L_S + L_RKD
    """
    print("\n" + "=" * 70)
    print("  Relation-based KD (RKD) — Eq. 8")
    print("=" * 70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    if not isinstance(teacher_models, list):
        teacher_models = [teacher_models]
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    rkd_loss_fn     = RKDLoss(distance_weight, angle_weight)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * m) for m in [0.5, 0.75]],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"RKD Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                t_feat_list = [
                    t(images, return_features=True)[1] for t in teacher_models
                ]

            s_logits, s_feats = student(images, return_features=True)

            l_s   = student_loss_fn(s_logits, labels)
            l_rkd = torch.stack(
                [rkd_loss_fn(tf, s_feats) for tf in t_feat_list]
            ).mean()
            loss = l_s + l_rkd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend(
                (torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy()
            )
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = _quick_eval(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m:.4f}")

        if val_m > best_acc:
            best_acc = val_m
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    -> Saved best (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


# ═══════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _quick_eval(model, dataloader, device):
    """Return validation accuracy only (for checkpoint selection)."""
    model.eval()
    correct, total = 0, 0
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits).squeeze() > 0.5).long()
        correct += (preds.cpu() == labels.long()).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)
