"""
trainers.py
===========
Training loops for each KD method — both single-teacher and multi-teacher.

Single-teacher functions
------------------------
- train_student_logits()   →  Response-based KD (Eq. 3)
- train_student_at()       →  Feature-based KD   (Eq. 5)
- train_student_rkd()      →  Relation-based KD  (Eq. 8)

Multi-teacher functions
-----------------------
- train_student_mt_logits()
- train_student_mt_at()
- train_student_mt_rkd()

All trainers:
  • accept a *new* student model (pretrained on ImageNet) and return the
    best student (loaded from checkpoint).
  • use SGD + Nesterov + MultiStepLR  (0.1× at 50 % and 75 % of epochs).
  • save the best checkpoint (by validation accuracy) and reload it at the end.
"""

import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models import ResNet50WithFeatures
from losses import (
    StudentLoss,
    LogitsDistillationLoss,
    LogitsKDLoss,
    AttentionTransferLoss,
    ATKDLoss,
    RKDLoss,
    RKDKDLoss,
)
from evaluation import evaluate


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE-TEACHER TRAINERS
# ═══════════════════════════════════════════════════════════════════════════

def train_student_logits(teacher_model, train_loader, val_loader,
                         num_epochs=50, lr=0.01, alpha=1.0, beta=0.5,
                         device='cuda',
                         save_path='student_logits_best.pth'):
    """Response-based KD (Logits)  — Eq. 3: L = α·L_S + β·L_logits"""
    print("\n" + "=" * 70)
    print("  [1/3] Response-based KD  (Logits)  —  Eq. 3")
    print("=" * 70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    criterion = LogitsKDLoss(alpha=alpha, beta=beta)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.5), int(num_epochs * 0.75)],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Logits  epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                t_logits = teacher_model(images)

            s_logits = student(images)
            loss = criterion(s_logits, t_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend(
                (torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy()
            )
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}:  loss={running_loss / len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}  "
              f"val_f1={val_m['f1']:.4f}")

        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    ✓ saved best  (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


def train_student_at(teacher_model, train_loader, val_loader,
                     num_epochs=50, lr=0.01, device='cuda',
                     save_path='student_at_best.pth'):
    """Feature-based KD (Attention Transfer)  — Eq. 5: L = L_S + L_AT"""
    print("\n" + "=" * 70)
    print("  [2/3] Feature-based KD  (AT)  —  Eq. 5")
    print("=" * 70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    criterion = ATKDLoss()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.5), int(num_epochs * 0.75)],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"AT  epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                _, t_attn = teacher_model(images, return_attention=True)

            s_logits, s_attn = student(images, return_attention=True)
            loss = criterion(s_logits, labels, t_attn, s_attn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend(
                (torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy()
            )
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}:  loss={running_loss / len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}  "
              f"val_f1={val_m['f1']:.4f}")

        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    ✓ saved best  (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


def train_student_rkd(teacher_model, train_loader, val_loader,
                      num_epochs=50, lr=0.01,
                      distance_weight=1.0, angle_weight=2.0,
                      device='cuda',
                      save_path='student_rkd_best.pth'):
    """Relation-based KD (RKD)  — Eq. 8: L = L_S + L_RKD"""
    print("\n" + "=" * 70)
    print("  [3/3] Relation-based KD  (RKD)  —  Eq. 8")
    print("=" * 70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    criterion = RKDKDLoss(distance_weight, angle_weight)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.5), int(num_epochs * 0.75)],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"RKD  epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                _, t_feats = teacher_model(images, return_features=True)

            s_logits, s_feats = student(images, return_features=True)
            loss = criterion(s_logits, labels, t_feats, s_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend(
                (torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy()
            )
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}:  loss={running_loss / len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}  "
              f"val_f1={val_m['f1']:.4f}")

        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    ✓ saved best  (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-TEACHER TRAINERS
# ═══════════════════════════════════════════════════════════════════════════

def train_student_mt_logits(teacher_models, train_loader, val_loader,
                            num_epochs=50, lr=0.01, alpha=1.0, beta=0.5,
                            device='cuda',
                            save_path='student_logits_best.pth'):
    """Multi-teacher Response-based KD — distillation loss averaged over teachers."""
    print("\n" + "=" * 70)
    print("  [1/3] Multi-Teacher Response-based KD  (Logits)")
    print("=" * 70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    logits_loss_fn  = LogitsDistillationLoss()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.5), int(num_epochs * 0.75)],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Logits-MT  epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                t_logits_list = [t(images) for t in teacher_models]

            s_logits = student(images)

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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}:  loss={running_loss / len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}")

        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    ✓ saved best  (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


def train_student_mt_at(teacher_models, train_loader, val_loader,
                        num_epochs=50, lr=0.01, device='cuda',
                        save_path='student_at_best.pth'):
    """Multi-teacher Feature-based KD (AT)."""
    print("\n" + "=" * 70)
    print("  [2/3] Multi-Teacher Feature-based KD  (AT)")
    print("=" * 70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    at_loss_fn      = AttentionTransferLoss()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.5), int(num_epochs * 0.75)],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"AT-MT  epoch {epoch+1}/{num_epochs}")
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}:  loss={running_loss / len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}")

        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    ✓ saved best  (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


def train_student_mt_rkd(teacher_models, train_loader, val_loader,
                         num_epochs=50, lr=0.01,
                         distance_weight=1.0, angle_weight=2.0,
                         device='cuda',
                         save_path='student_rkd_best.pth'):
    """Multi-teacher Relation-based KD (RKD)."""
    print("\n" + "=" * 70)
    print("  [3/3] Multi-Teacher Relation-based KD  (RKD)")
    print("=" * 70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    rkd_loss_fn     = RKDLoss(distance_weight, angle_weight)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.5), int(num_epochs * 0.75)],
        gamma=0.1,
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"RKD-MT  epoch {epoch+1}/{num_epochs}")
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}:  loss={running_loss / len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}")

        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = copy.deepcopy(student.state_dict())
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    ✓ saved best  (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student
