"""
=============================================================================
Ensemble Learning of Lightweight Deep Learning Models Using Knowledge
Distillation for Deepfake Image Classification
=============================================================================

Based on the paper:
  "Ensemble Learning of Lightweight Deep Learning Models Using Knowledge
   Distillation for Image Classification" — Kang & Gwak, Mathematics 2020

Three Knowledge Distillation Methods:
  1. Response-based KD  (Logits MSE)          — Eq. 2, 3
  2. Feature-based KD   (Attention Transfer)   — Eq. 4, 5
  3. Relation-based KD  (RKD)                  — Eq. 6, 7, 8

Ensemble Method:
  Soft Voting (averaging logits from 3 student models)  — Eq. 9

Architecture: ResNet-50 for both Teacher and Student
Task: Binary classification — Real vs Fake (Deepfake detection)
Datasets: 200k, 140k, 190k deepfake datasets

Usage:
  python knowledge_distillation_ensemble.py
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: ResNet-50 with Intermediate Feature Extraction
# ============================================================================

class ResNet50WithFeatures(nn.Module):
    """
    ResNet-50 wrapper that can return:
      - logits (final output)
      - attention maps from layer1, layer2, layer3, layer4
      - penultimate features (output of avgpool, before FC)

    Needed for both teacher and student to extract intermediate
    representations for Feature-based (AT) and Relation-based (RKD) distillation.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(ResNet50WithFeatures, self).__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )

        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1   # 256 channels
        self.layer2  = backbone.layer2   # 512 channels
        self.layer3  = backbone.layer3   # 1024 channels
        self.layer4  = backbone.layer4   # 2048 channels
        self.avgpool = backbone.avgpool
        self.fc      = nn.Linear(backbone.fc.in_features, num_classes)

        self.feature_channels = [256, 512, 1024, 2048]

    def forward(self, x, return_features=False, return_attention=False):
        """
        Forward pass.

        Returns:
            If no extra flags: logits [B, num_classes]
            If return_features: (logits, features)  — features [B, 2048]
            If return_attention: (logits, attention_maps)
            If both: (logits, features, attention_maps)
                attention_maps: list of 4 tensors [B, C_i, H_i, W_i]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        attention_maps = []
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            if return_attention:
                attention_maps.append(x)

        features = self.avgpool(x)
        features = torch.flatten(features, 1)
        logits = self.fc(features)

        if not return_features and not return_attention:
            return logits

        outputs = [logits]
        if return_features:
            outputs.append(features)
        if return_attention:
            outputs.append(attention_maps)
        return tuple(outputs)


# ============================================================================
# SECTION 2: Attention Map Computation (AT)
# ============================================================================

def compute_attention_map(feature_map):
    """
    Compute attention map from a feature map (AT [29]).

    A(F) = sum_c |F_c|  →  vectorize  →  L2-normalize

    Args:
        feature_map: [B, C, H, W]
    Returns:
        attention: [B, H*W]  L2-normalized
    """
    # Sum absolute values across channel dim → [B, H, W]
    attention = torch.sum(torch.abs(feature_map), dim=1)
    # Flatten spatial dims → [B, H*W]
    attention = attention.view(attention.size(0), -1)
    # L2 normalize
    norm = torch.norm(attention, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    attention = attention / norm
    return attention


# ============================================================================
# SECTION 3: Knowledge Distillation Loss Functions
# ============================================================================

class StudentLoss(nn.Module):
    """
    Eq. 1 — Cross-entropy between student soft-target and ground truth.
    Binary classification → BCEWithLogitsLoss.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, labels):
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return self.criterion(student_logits, labels)


class LogitsDistillationLoss(nn.Module):
    """
    Eq. 2 — Response-based KD using logits (MSE).
    L_logits(z_t, z_s) = sum_i (z_t_i - z_s_i)^2
    """
    def __init__(self):
        super().__init__()

    def forward(self, teacher_logits, student_logits):
        return F.mse_loss(teacher_logits, student_logits)


class AttentionTransferLoss(nn.Module):
    """
    Eq. 4 — Feature-based KD (AT).
    L_AT = sum_j  ||  Q_S_j / ||Q_S_j||_2  -  Q_T_j / ||Q_T_j||_2  ||^2
    """
    def __init__(self):
        super().__init__()

    def forward(self, teacher_attention_maps, student_attention_maps):
        loss = 0.0
        for t_map, s_map in zip(teacher_attention_maps, student_attention_maps):
            # Resize student map if spatial dims differ
            if t_map.shape[2:] != s_map.shape[2:]:
                s_map = F.interpolate(
                    s_map, size=t_map.shape[2:],
                    mode='bilinear', align_corners=False
                )
            t_att = compute_attention_map(t_map)
            s_att = compute_attention_map(s_map)
            loss += torch.sum((s_att - t_att) ** 2)
        return loss


class RKDLoss(nn.Module):
    """
    Eq. 6, 7 — Relation-based KD (RKD).
    Combines distance-wise and angle-wise potentials with Huber loss.
    """
    def __init__(self, distance_weight=1.0, angle_weight=2.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    @staticmethod
    def huber_loss(pred, target, delta=1.0):
        """Eq. 7: Smooth L1 / Huber loss."""
        diff = torch.abs(pred - target)
        quadratic = torch.min(diff, torch.tensor(delta, device=diff.device))
        linear = diff - quadratic
        return 0.5 * quadratic ** 2 + linear

    def distance_wise_potential(self, features):
        """Pairwise Euclidean distances (upper triangle)."""
        diff = features.unsqueeze(1) - features.unsqueeze(0)       # [B,B,D]
        dist_matrix = torch.norm(diff, p=2, dim=2)                 # [B,B]
        mask = torch.triu(torch.ones_like(dist_matrix, dtype=torch.bool), diagonal=1)
        return dist_matrix[mask]

    def angle_wise_potential(self, features):
        """Angles at each vertex for all unique triplets."""
        diff = features.unsqueeze(1) - features.unsqueeze(0)       # [B,B,D]
        norm = torch.norm(diff, p=2, dim=2, keepdim=True).clamp(min=1e-8)
        diff_n = diff / norm                                        # [B,B,D]

        cos_sim = torch.bmm(diff_n.transpose(0, 1), diff_n)       # [B,B,B]
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.acos(cos_sim)                                # [B,B,B]

        B = features.size(0)
        idx = []
        for j in range(B):
            for i in range(B):
                if i == j:
                    continue
                for k in range(i + 1, B):
                    if k == j:
                        continue
                    idx.append((i, j, k))
        if not idx:
            return torch.tensor(0.0, device=features.device)

        ii, jj, kk = zip(*idx)
        return angles[torch.tensor(ii, device=features.device),
                       torch.tensor(jj, device=features.device),
                       torch.tensor(kk, device=features.device)]

    def forward(self, teacher_features, student_features):
        # Distance-wise
        t_dist = self.distance_wise_potential(teacher_features)
        s_dist = self.distance_wise_potential(student_features)
        t_dist = t_dist / (t_dist.sum() + 1e-8)
        s_dist = s_dist / (s_dist.sum() + 1e-8)
        dist_loss = self.huber_loss(s_dist, t_dist).mean()

        # Angle-wise
        t_ang = self.angle_wise_potential(teacher_features)
        s_ang = self.angle_wise_potential(student_features)
        if isinstance(t_ang, torch.Tensor) and t_ang.numel() > 0:
            ang_loss = self.huber_loss(s_ang, t_ang).mean()
        else:
            ang_loss = torch.tensor(0.0, device=teacher_features.device)

        return self.distance_weight * dist_loss + self.angle_weight * ang_loss


# ============================================================================
# SECTION 4: Combined KD Loss Functions (per paper equations)
# ============================================================================

class LogitsKDLoss(nn.Module):
    """Eq. 3: L = alpha * L_S + beta * L_logits"""
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.student_loss = StudentLoss()
        self.logits_loss = LogitsDistillationLoss()

    def forward(self, student_logits, teacher_logits, labels):
        return (self.alpha * self.student_loss(student_logits, labels)
                + self.beta * self.logits_loss(teacher_logits, student_logits))


class ATKDLoss(nn.Module):
    """Eq. 5: L = L_S + L_AT"""
    def __init__(self):
        super().__init__()
        self.student_loss = StudentLoss()
        self.at_loss = AttentionTransferLoss()

    def forward(self, student_logits, labels,
                teacher_attention_maps, student_attention_maps):
        return (self.student_loss(student_logits, labels)
                + self.at_loss(teacher_attention_maps, student_attention_maps))


class RKDKDLoss(nn.Module):
    """Eq. 8: L = L_S + L_RKD"""
    def __init__(self, distance_weight=1.0, angle_weight=2.0):
        super().__init__()
        self.student_loss = StudentLoss()
        self.rkd_loss = RKDLoss(distance_weight, angle_weight)

    def forward(self, student_logits, labels,
                teacher_features, student_features):
        return (self.student_loss(student_logits, labels)
                + self.rkd_loss(teacher_features, student_features))


# ============================================================================
# SECTION 5: Evaluation Utilities
# ============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate a single model. Returns dict with acc/prec/rec/f1/auc."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze()
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'auc':       roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
    }


@torch.no_grad()
def evaluate_ensemble(student_models, dataloader, device):
    """
    Eq. 9 — Soft Voting ensemble.
    y_hat = argmax_i  sum_j  z_ij
    For binary classification: average logits → sigmoid → threshold.
    """
    for m in student_models:
        m.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in dataloader:
        images = images.to(device)
        # Sum logits from all students, then average
        ensemble_logits = sum(m(images) for m in student_models) / len(student_models)
        probs = torch.sigmoid(ensemble_logits).squeeze()
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'auc':       roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
    }


# ============================================================================
# SECTION 6: Training Functions — Single Teacher
# ============================================================================

def train_student_logits(teacher_model, train_loader, val_loader,
                         num_epochs=50, lr=0.01, alpha=1.0, beta=0.5,
                         device='cuda', save_path='student_logits_best.pth'):
    """
    Train a student with Response-based KD (Logits).  Eq. 3
    """
    print("\n" + "="*70)
    print("  Response-based KD (Logits) — Eq. 3")
    print("="*70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    criterion = LogitsKDLoss(alpha=alpha, beta=beta)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Logits Epoch {epoch+1}/{num_epochs}")
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
            preds_list.extend((torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}  "
              f"val_f1={val_m['f1']:.4f}")

        if val_m['accuracy'] > best_acc:
            best_acc = val_m['accuracy']
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, save_path)
            print(f"    -> Saved best model (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


def train_student_at(teacher_model, train_loader, val_loader,
                     num_epochs=50, lr=0.01, device='cuda',
                     save_path='student_at_best.pth'):
    """
    Train a student with Feature-based KD (Attention Transfer).  Eq. 5
    """
    print("\n" + "="*70)
    print("  Feature-based KD (Attention Transfer) — Eq. 5")
    print("="*70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    criterion = ATKDLoss()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"AT Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Teacher: logits + attention maps  →  index 0, 1
            with torch.no_grad():
                t_logits, t_attn = teacher_model(images, return_attention=True)

            # Student: logits + attention maps  →  index 0, 1
            s_logits, s_attn = student(images, return_attention=True)

            loss = criterion(s_logits, labels, t_attn, s_attn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend((torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}  "
              f"val_f1={val_m['f1']:.4f}")

        if val_m['accuracy'] > best_acc:
            best_acc = val_m['accuracy']
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, save_path)
            print(f"    -> Saved best model (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


def train_student_rkd(teacher_model, train_loader, val_loader,
                      num_epochs=50, lr=0.01,
                      distance_weight=1.0, angle_weight=2.0,
                      device='cuda', save_path='student_rkd_best.pth'):
    """
    Train a student with Relation-based KD (RKD).  Eq. 8
    """
    print("\n" + "="*70)
    print("  Relation-based KD (RKD) — Eq. 8")
    print("="*70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    criterion = RKDKDLoss(distance_weight, angle_weight)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"RKD Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Teacher: logits + features  →  index 0, 1
            with torch.no_grad():
                t_logits, t_feats = teacher_model(images, return_features=True)

            # Student: logits + features  →  index 0, 1
            s_logits, s_feats = student(images, return_features=True)

            loss = criterion(s_logits, labels, t_feats, s_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend((torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}  "
              f"val_f1={val_m['f1']:.4f}")

        if val_m['accuracy'] > best_acc:
            best_acc = val_m['accuracy']
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, save_path)
            print(f"    -> Saved best model (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


# ============================================================================
# SECTION 7: Training Functions — Multi-Teacher
# ============================================================================

def train_student_mt_logits(teacher_models, train_loader, val_loader,
                            num_epochs=50, lr=0.01, alpha=1.0, beta=0.5,
                            device='cuda', save_path='student_logits_best.pth'):
    """
    Multi-teacher Response-based KD (Logits).
    Distillation loss = average over all teachers.
    """
    print("\n" + "="*70)
    print("  Multi-Teacher Response-based KD (Logits)")
    print("="*70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    logits_loss_fn  = LogitsDistillationLoss()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"Logits-MT Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                t_logits_list = [t(images) for t in teacher_models]

            s_logits = student(images)

            l_s = student_loss_fn(s_logits, labels)
            l_kd = torch.stack([logits_loss_fn(tl, s_logits) for tl in t_logits_list]).mean()
            loss = alpha * l_s + beta * l_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend((torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}")

        if val_m['accuracy'] > best_acc:
            best_acc = val_m['accuracy']
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, save_path)
            print(f"    -> Saved best (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


def train_student_mt_at(teacher_models, train_loader, val_loader,
                        num_epochs=50, lr=0.01, device='cuda',
                        save_path='student_at_best.pth'):
    """
    Multi-teacher Feature-based KD (AT).
    """
    print("\n" + "="*70)
    print("  Multi-Teacher Feature-based KD (AT)")
    print("="*70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    at_loss_fn      = AttentionTransferLoss()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"AT-MT Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                t_attn_list = [t(images, return_attention=True)[1] for t in teacher_models]

            s_logits, s_attn = student(images, return_attention=True)

            l_s = student_loss_fn(s_logits, labels)
            l_at = torch.stack([at_loss_fn(ta, s_attn) for ta in t_attn_list]).mean()
            loss = l_s + l_at

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend((torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}")

        if val_m['accuracy'] > best_acc:
            best_acc = val_m['accuracy']
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, save_path)
            print(f"    -> Saved best (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


def train_student_mt_rkd(teacher_models, train_loader, val_loader,
                         num_epochs=50, lr=0.01,
                         distance_weight=1.0, angle_weight=2.0,
                         device='cuda', save_path='student_rkd_best.pth'):
    """
    Multi-teacher Relation-based KD (RKD).
    """
    print("\n" + "="*70)
    print("  Multi-Teacher Relation-based KD (RKD)")
    print("="*70)

    student = ResNet50WithFeatures(num_classes=1, pretrained=True).to(device)
    for t in teacher_models:
        t.to(device); t.eval()

    student_loss_fn = StudentLoss()
    rkd_loss_fn     = RKDLoss(distance_weight, angle_weight)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1
    )

    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        student.train()
        running_loss, preds_list, labels_list = 0.0, [], []

        pbar = tqdm(train_loader, desc=f"RKD-MT Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                t_feat_list = [t(images, return_features=True)[1] for t in teacher_models]

            s_logits, s_feats = student(images, return_features=True)

            l_s = student_loss_fn(s_logits, labels)
            l_rkd = torch.stack([rkd_loss_fn(tf, s_feats) for tf in t_feat_list]).mean()
            loss = l_s + l_rkd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_list.extend((torch.sigmoid(s_logits) > 0.5).float().squeeze().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = accuracy_score(labels_list, preds_list)
        val_m = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_m['accuracy']:.4f}")

        if val_m['accuracy'] > best_acc:
            best_acc = val_m['accuracy']
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, save_path)
            print(f"    -> Saved best (val_acc={best_acc:.4f})")

    student.load_state_dict(best_state)
    return student


# ============================================================================
# SECTION 8: Complete Pipeline
# ============================================================================

def load_teacher_model(path, device='cuda'):
    """Load a pretrained teacher model from checkpoint."""
    teacher = ResNet50WithFeatures(num_classes=1, pretrained=False)
    state_dict = torch.load(path, map_location=device)

    # Handle DataParallel 'module.' prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Filter only matching keys (for partial loading)
    model_keys = set(teacher.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    teacher.load_state_dict(filtered, strict=False)
    teacher = teacher.to(device)
    teacher.eval()
    print(f"  Loaded teacher: {path}  ({len(filtered)}/{len(model_keys)} keys)")
    return teacher


def run_full_pipeline(
    teacher_paths,          # dict: {'200k': path, '140k': path, '190k': path}
    datasets,               # dict: {'200k': Dataset_selector, ...}
    num_epochs=50,
    lr=0.01,
    alpha=1.0,
    beta=0.5,
    rkd_dist_w=1.0,
    rkd_angle_w=2.0,
    device='cuda',
    save_dir='./kd_checkpoints',
    multi_teacher=True,
    train_dataset='combined',   # '200k'|'140k'|'190k'|'combined'
):
    """
    ┌─────────────────────────────────────────────────────────┐
    │  Full Pipeline                                          │
    │  1. Load teacher models                                 │
    │  2. Train 3 students (Logits, AT, RKD)                 │
    │  3. Ensemble via Soft Voting                            │
    │  4. Evaluate & save                                     │
    └─────────────────────────────────────────────────────────┘
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load Teachers ----
    print("\n" + "="*70)
    print("  STEP 1: Loading Teacher Models")
    print("="*70)
    teacher_models = []
    for name in ['200k', '140k', '190k']:
        p = teacher_paths.get(name)
        if p and os.path.exists(p):
            teacher_models.append(load_teacher_model(p, device))
        else:
            print(f"  Skipping {name} teacher (path missing or not found)")
    if not teacher_models:
        raise RuntimeError("No teacher models loaded — check paths!")
    print(f"  Teachers ready: {len(teacher_models)}")

    # ---- Prepare Dataloaders ----
    print("\n" + "="*70)
    print("  STEP 2: Preparing DataLoaders")
    print("="*70)

    active_keys = [k for k in ['200k', '140k', '190k'] if k in datasets]
    if train_dataset == 'combined':
        keys = active_keys
    elif train_dataset in datasets:
        keys = [train_dataset]
    else:
        keys = active_keys

    train_loader = datasets[keys[0]].loader_train
    val_loader   = datasets[keys[0]].loader_val
    test_loader  = datasets[keys[0]].loader_test
    print(f"  Train/Val/Test loaders from: {keys[0]}")
    if len(keys) > 1:
        print(f"  Note: Only first dataset's loader used. "
              f"For combined, modify to use ConcatDataset.")

    # ---- Train Students ----
    print("\n" + "="*70)
    print("  STEP 3-5: Training Three Student Models")
    print("="*70)

    if multi_teacher and len(teacher_models) > 1:
        s_logits = train_student_mt_logits(
            teacher_models, train_loader, val_loader,
            num_epochs, lr, alpha, beta, device,
            os.path.join(save_dir, 'student_logits_best.pth'))
        s_at = train_student_mt_at(
            teacher_models, train_loader, val_loader,
            num_epochs, lr, device,
            os.path.join(save_dir, 'student_at_best.pth'))
        s_rkd = train_student_mt_rkd(
            teacher_models, train_loader, val_loader,
            num_epochs, lr, rkd_dist_w, rkd_angle_w, device,
            os.path.join(save_dir, 'student_rkd_best.pth'))
    else:
        s_logits = train_student_logits(
            teacher_models[0], train_loader, val_loader,
            num_epochs, lr, alpha, beta, device,
            os.path.join(save_dir, 'student_logits_best.pth'))
        s_at = train_student_at(
            teacher_models[0], train_loader, val_loader,
            num_epochs, lr, device,
            os.path.join(save_dir, 'student_at_best.pth'))
        s_rkd = train_student_rkd(
            teacher_models[0], train_loader, val_loader,
            num_epochs, lr, rkd_dist_w, rkd_angle_w, device,
            os.path.join(save_dir, 'student_rkd_best.pth'))

    student_models = [s_logits, s_at, s_rkd]
    student_names  = ['Logits KD', 'AT KD', 'RKD KD']

    # ---- Evaluate ----
    print("\n" + "="*70)
    print("  STEP 6: Evaluation on Test Set")
    print("="*70)

    results = {}
    for name, model in zip(student_names, student_models):
        m = evaluate(model, test_loader, device)
        results[name] = m
        print(f"\n  {name}:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    # Ensemble
    ens_m = evaluate_ensemble(student_models, test_loader, device)
    results['Ensemble (Soft Voting)'] = ens_m
    print(f"\n  Ensemble (Soft Voting — Eq. 9):")
    for k, v in ens_m.items():
        print(f"    {k}: {v:.4f}")

    # Teachers (for comparison)
    for i, t in enumerate(teacher_models):
        name = f"Teacher {i+1}"
        m = evaluate(t, test_loader, device)
        results[name] = m
        print(f"\n  {name}: acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # ---- Save Final Models ----
    print("\n" + "="*70)
    print("  STEP 7: Saving Models")
    print("="*70)
    for name, model in zip(['student_logits', 'student_at', 'student_rkd'], student_models):
        torch.save(model.state_dict(), os.path.join(save_dir, f'{name}_final.pth'))
    print(f"  All models saved to {save_dir}")

    # ---- Summary Table ----
    print("\n" + "="*70)
    print("  FINAL COMPARISON TABLE")
    print("="*70)
    header = f"{'Method':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")

    return student_models, results


# ============================================================================
# SECTION 9: Inference Helper — Load & Predict
# ============================================================================

class EnsemblePredictor:
    """
    Load trained student models and perform ensemble inference.
    """
    def __init__(self, model_paths, device='cuda'):
        """
        Args:
            model_paths: dict with keys 'logits', 'at', 'rkd'
                         values are paths to .pth checkpoints
        """
        self.device = device
        self.models = {}
        for kd_type, path in model_paths.items():
            model = ResNet50WithFeatures(num_classes=1, pretrained=False)
            state_dict = torch.load(path, map_location=device)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device).eval()
            self.models[kd_type] = model

    @torch.no_grad()
    def predict(self, images, method='ensemble'):
        """
        Args:
            images: tensor [B, 3, H, W] (already preprocessed)
            method: 'logits', 'at', 'rkd', or 'ensemble'

        Returns:
            probs: numpy array [B] — probability of being REAL
            preds: numpy array [B] — binary predictions (0=fake, 1=real)
        """
        images = images.to(self.device)

        if method == 'ensemble':
            # Soft Voting — Eq. 9
            avg_logits = sum(m(images) for m in self.models.values()) / len(self.models)
        elif method in self.models:
            avg_logits = self.models[method](images)
        else:
            raise ValueError(f"Unknown method: {method}")

        probs = torch.sigmoid(avg_logits).squeeze().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        return probs, preds

    @torch.no_grad()
    def predict_individual(self, images):
        """
        Get predictions from each student separately.

        Returns:
            dict: {method_name: (probs, preds)}
        """
        images = images.to(self.device)
        results = {}
        for name, model in self.models.items():
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            results[name] = (probs, preds)
        return results


# ============================================================================
# SECTION 10: Main — Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  USAGE EXAMPLE                                                   ║
    ║  Update the paths below to match your environment.               ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """

    # ---- Configuration ----
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS  = 50
    LR          = 0.01
    BATCH_SIZE  = 32
    SAVE_DIR    = './kd_checkpoints'

    # Paths to pretrained teacher models (UPDATE THESE)
    TEACHER_PATHS = {
        '200k': '/path/to/teacher_200k_best.pth',
        '140k': '/path/to/teacher_140k_best.pth',
        '190k': '/path/to/teacher_190k_best.pth',
    }

    # ---- Create dataset selectors ----
    # Import your Dataset_selector class:
    # from your_dataset_file import Dataset_selector

    dataset_200k = Dataset_selector(
        dataset_mode='200k',
        realfake200k_train_csv='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/train_labels.csv',
        realfake200k_val_csv='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/val_labels.csv',
        realfake200k_test_csv='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/test_labels.csv',
        realfake200k_root_dir='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset',
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )

    dataset_140k = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )

    dataset_190k = Dataset_selector(
        dataset_mode='190k',
        realfake190k_root_dir='/kaggle/input/deepfake-and-real-images/Dataset',
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )

    datasets = {
        '200k': dataset_200k,
        '140k': dataset_140k,
        '190k': dataset_190k,
    }

    # ---- Run Full Pipeline ----
    student_models, results = run_full_pipeline(
        teacher_paths=TEACHER_PATHS,
        datasets=datasets,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        alpha=1.0,         # student loss weight (Logits KD)
        beta=0.5,          # logits distillation weight (Logits KD)
        rkd_dist_w=1.0,    # distance-wise weight (RKD)
        rkd_angle_w=2.0,   # angle-wise weight (RKD)
        device=DEVICE,
        save_dir=SAVE_DIR,
        multi_teacher=True,
        train_dataset='combined',
    )

    # ---- Inference Example ----
    # After training, use EnsemblePredictor for inference:
    #
    # predictor = EnsemblePredictor({
    #     'logits': f'{SAVE_DIR}/student_logits_final.pth',
    #     'at':     f'{SAVE_DIR}/student_at_final.pth',
    #     'rkd':    f'{SAVE_DIR}/student_rkd_final.pth',
    # }, device=DEVICE)
    #
    # # Get a batch of test images
    # test_images, test_labels = next(iter(datasets['200k'].loader_test))
    # probs, preds = predictor.predict(test_images, method='ensemble')
    # print(f"Ensemble probs: {probs}")
    # print(f"Ensemble preds: {preds}")
