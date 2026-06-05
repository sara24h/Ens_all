"""
==========================================================================
  اسکریپت ساخت تمام فایل‌های گم‌شده در کگل
  فقط این فایل رو اجرا کن: python setup_all_missing_files.py
==========================================================================
"""

import os

BASE = '/kaggle/working/Ens_all'

# ============================================================
# 1) evaluate.py  —  مطابق با رابط pipeline.py
# ============================================================
evaluate_py = r'''import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from config import DEVICE


def evaluate(model, dataloader, device=None, num_classes=None):
    """
    Evaluate a single model on a dataloader.
    Returns dict with: accuracy, precision, recall, f1, auc

    Args:
        model: the model to evaluate
        dataloader: test/val dataloader
        device: 'cuda' or 'cpu' (default: from config)
        num_classes: if None, auto-detect from model
    """
    if device is None:
        device = DEVICE

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    # Auto-detect num_classes from model
    if num_classes is None:
        num_classes = model.fc.out_features

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Handle models that return tuples (e.g., ResNet50WithFeatures)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            if num_classes == 1:
                # Binary classification
                probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            else:
                # Multi-class
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else list(preds))
            all_labels.extend(labels.cpu().numpy().tolist())
            if num_classes == 1:
                all_probs.extend(probs.tolist() if hasattr(probs, 'tolist') else list(probs))
            else:
                all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    if num_classes == 1 or len(np.unique(all_labels)) <= 2:
        average = 'binary'
    else:
        average = 'macro'

    prec = precision_score(all_labels, all_preds, average=average, zero_division=0)
    rec = recall_score(all_labels, all_preds, average=average, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)

    # AUC
    try:
        if num_classes == 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception:
        auc = 0.0

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
    }


def evaluate_ensemble(models, dataloader, device=None, num_classes=None):
    """
    Evaluate ensemble of models using Soft Voting (Eq.9).
    Sum the logits from all models, then take argmax/sigmoid.
    Returns dict with: accuracy, precision, recall, f1, auc
    """
    if device is None:
        device = DEVICE
    for m in models:
        m.eval()

    # Auto-detect num_classes from first model
    if num_classes is None and len(models) > 0:
        num_classes = models[0].fc.out_features

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ensemble Evaluating", leave=False):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]

            images = images.to(device)
            labels = labels.to(device)

            ensemble_logits = None

            for model in models:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                if ensemble_logits is None:
                    ensemble_logits = logits.clone()
                else:
                    ensemble_logits += logits

            # Soft Voting: argmax of summed logits (Eq.9)
            if num_classes == 1:
                probs = torch.sigmoid(ensemble_logits.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(ensemble_logits, dim=1).cpu().numpy()
                preds = ensemble_logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else list(preds))
            all_labels.extend(labels.cpu().numpy().tolist())
            if num_classes == 1:
                all_probs.extend(probs.tolist() if hasattr(probs, 'tolist') else list(probs))
            else:
                all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    if num_classes == 1 or len(np.unique(all_labels)) <= 2:
        average = 'binary'
    else:
        average = 'macro'

    prec = precision_score(all_labels, all_preds, average=average, zero_division=0)
    rec = recall_score(all_labels, all_preds, average=average, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)

    try:
        if num_classes == 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception:
        auc = 0.0

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
    }


def print_results_table(results):
    """Print a formatted results table matching pipeline.py expectations."""
    print("\n" + "=" * 80)
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 80)

    for name, metrics in results.items():
        acc = metrics.get('accuracy', 0)
        f1 = metrics.get('f1', 0)
        auc = metrics.get('auc', 0)
        print(f"  {name:<30} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")

    print("=" * 80)
'''

# ============================================================
# 2) models/resnet50.py
# ============================================================
resnet50_py = r'''import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50WithFeatures(nn.Module):
    """
    ResNet50 wrapper that can also return:
      - attention maps (from layer2, layer3, layer4) for AT loss
      - penultimate features (before fc) for RKD loss

    Compatible with standard ResNet50 checkpoints (auto-loads fc weights).
    """
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Copy all layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

        # Store feature dim for external use
        self.feature_dim = resnet.fc.in_features  # 2048

    def forward(self, x, return_attention=False, return_features=False):
        # Standard forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        attn2 = None
        attn3 = None
        attn4 = None

        if return_attention:
            x = self.layer2(x)
            attn2 = x  # [B, 512, H, W]
            x = self.layer3(x)
            attn3 = x  # [B, 1024, H, W]
            x = self.layer4(x)
            attn4 = x  # [B, 2048, H, W]
        else:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        features = None
        if return_features:
            features = self.avgpool(x)       # [B, 2048, 1, 1]
            features = features.flatten(1)   # [B, 2048]

        x = self.avgpool(x)
        x = x.flatten(1)
        logits = self.fc(x)                 # [B, num_classes]

        # Return based on flags
        if return_attention and return_features:
            return logits, [attn2, attn3, attn4], features
        elif return_attention:
            return logits, [attn2, attn3, attn4]
        elif return_features:
            return logits, features
        else:
            return logits

    def get_attention_maps(self, x):
        """Convenience method to get attention maps."""
        _, attention = self.forward(x, return_attention=True, return_features=False)
        return attention

    def get_features(self, x):
        """Convenience method to get penultimate features."""
        _, features = self.forward(x, return_attention=False, return_features=True)
        return features
'''

# ============================================================
# 3) losses/student_loss.py
# ============================================================
student_loss_py = r'''import torch
import torch.nn as nn


class StudentLoss(nn.Module):
    """
    Student classification loss (Eq.1): Binary Cross-Entropy with Logits
    L_S = BCE(y_student, y_true)
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, labels):
        """
        Args:
            student_logits: [B, 1] or [B] - raw logits from student
            labels: [B] - ground truth labels (0 or 1)
        """
        if student_logits.dim() == 2 and student_logits.size(1) == 1:
            student_logits = student_logits.squeeze(1)
        loss = self.bce(student_logits, labels.float())
        return loss
'''

# ============================================================
# 4) losses/logits_loss.py
# ============================================================
logits_loss_py = r'''import torch
import torch.nn as nn


class LogitsDistillationLoss(nn.Module):
    """
    Logits-based Knowledge Distillation Loss (Eq.2, Eq.3)

    L_logits = MSE(z_student, z_teacher)    (Eq.2)
    L_total  = alpha * L_S + beta * L_logits (Eq.3)

    where z = raw logits (before sigmoid/softmax)
    """
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: [B, 1] or [B]
            teacher_logits: [B, 1] or [B] - from teacher (no grad)
            labels: [B] - ground truth
        """
        if student_logits.dim() == 2 and student_logits.size(1) == 1:
            student_logits = student_logits.squeeze(1)
        if teacher_logits.dim() == 2 and teacher_logits.size(1) == 1:
            teacher_logits = teacher_logits.squeeze(1)

        # L_S: student classification loss (Eq.1)
        l_s = self.bce(student_logits, labels.float())

        # L_logits: MSE between student and teacher logits (Eq.2)
        l_logits = self.mse(student_logits, teacher_logits)

        # Combined (Eq.3)
        total_loss = self.alpha * l_s + self.beta * l_logits

        return total_loss, l_s, l_logits
'''

# ============================================================
# 5) losses/at_loss.py
# ============================================================
at_loss_py = r'''import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer Loss (Eq.4, Eq.5)

    F_s = sum of squared attention map elements (Eq.4)
    L_AT = || (F_s_teacher / ||F_s_teacher||) - (F_s_student / ||F_s_student||) ||^2  (Eq.5)

    L_total = L_S + gamma * L_AT
    """
    def __init__(self, gamma=1000.0):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()

    def attention_map(self, feature_map):
        """
        Compute attention map from feature map (Eq.4).
        Sum over channels, then square + normalize.
        Args:
            feature_map: [B, C, H, W]
        Returns:
            attention: [B, H*W] normalized
        """
        attn = F.normalize(feature_map.pow(2).mean(1).view(feature_map.size(0), -1))
        return attn

    def at_loss(self, student_attns, teacher_attns):
        """
        Compute AT loss between student and teacher attention maps (Eq.5).
        Args:
            student_attns: list of [attn2, attn3, attn4] from student
            teacher_attns: list of [attn2, attn3, attn4] from teacher
        """
        loss = 0.0
        for s_attn, t_attn in zip(student_attns, teacher_attns):
            s_map = self.attention_map(s_attn)
            t_map = self.attention_map(t_attn)
            loss += ((s_map - t_map) ** 2).sum()
        return loss

    def forward(self, student_logits, labels, student_attns, teacher_attns):
        """
        Args:
            student_logits: [B, 1] or [B]
            labels: [B]
            student_attns: list of feature maps from student
            teacher_attns: list of feature maps from teacher (no grad)
        """
        if student_logits.dim() == 2 and student_logits.size(1) == 1:
            student_logits = student_logits.squeeze(1)

        # L_S: student classification loss
        l_s = self.bce(student_logits, labels.float())

        # L_AT: attention transfer loss (Eq.5)
        l_at = self.at_loss(student_attns, teacher_attns)

        # Combined
        total_loss = l_s + self.gamma * l_at

        return total_loss, l_s, l_at
'''

# ============================================================
# 6) losses/rkd_loss.py
# ============================================================
rkd_loss_py = r'''import torch
import torch.nn as nn
import torch.nn.functional as F


class RKDLoss(nn.Module):
    """
    Relational Knowledge Distillation Loss (Eq.6, Eq.7, Eq.8)

    L_RKD = lambda_d * L_distance + lambda_a * L_angle

    Distance potential (Eq.7):
        psi_d(t_i, t_j) = 1/beta_d * ||t_i - t_j||_2
        L_distance = Huber(psi_d(s) / psi_d(s).detach().sum(),
                          psi_d(t) / psi_d(t).detach().sum())

    Angle potential (Eq.8):
        psi_a(t_i, t_j, t_k) = cos(angle(t_i - t_j, t_k - t_j))
        L_angle = Huber(psi_a(s) / psi_a(s).detach().sum(),
                       psi_a(t) / psi_a(t).detach().sum())

    L_total = L_S + L_RKD
    """
    def __init__(self, lambda_d=25.0, lambda_a=50.0, beta_d=4.0, huber_delta=1.0):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_a = lambda_a
        self.beta_d = beta_d
        self.huber = nn.SmoothL1Loss(reduction='mean', beta=huber_delta)
        self.bce = nn.BCEWithLogitsLoss()

    def pairwise_distance(self, features):
        """
        Compute pairwise distance matrix (Eq.7).
        Args:
            features: [B, D]
        Returns:
            distances: [B, B]
        """
        diff = features.unsqueeze(1) - features.unsqueeze(0)  # [B, B, D]
        dist = torch.norm(diff, p=2, dim=2) / self.beta_d     # [B, B]
        return dist

    def angle_potential(self, features):
        """
        Compute angle potential (Eq.8).
        Args:
            features: [B, D]
        Returns:
            angles: [B, B, B] - angle between triplets
        """
        diff = features.unsqueeze(1) - features.unsqueeze(0)  # [B, B, D]

        # Normalize
        norm = torch.norm(diff, p=2, dim=2, keepdim=True).clamp(min=1e-8)
        diff_norm = diff / norm  # [B, B, D]

        # Angle: cos(v_ij, v_kj)
        cos_sim = torch.bmm(diff_norm, diff_norm.transpose(1, 2))  # [B, B, B]

        return cos_sim

    def distance_loss(self, student_features, teacher_features):
        """L_distance (Eq.7)."""
        s_dist = self.pairwise_distance(student_features)
        t_dist = self.pairwise_distance(teacher_features)

        # Normalize
        s_dist_norm = s_dist / (s_dist.detach().sum() + 1e-8)
        t_dist_norm = t_dist / (t_dist.detach().sum() + 1e-8)

        return self.huber(s_dist_norm, t_dist_norm.detach())

    def angle_loss(self, student_features, teacher_features):
        """L_angle (Eq.8)."""
        s_angle = self.angle_potential(student_features)
        t_angle = self.angle_potential(teacher_features)

        # Flatten and normalize
        s_flat = s_angle.view(-1)
        t_flat = t_angle.view(-1)

        s_norm = s_flat / (s_flat.detach().abs().sum() + 1e-8)
        t_norm = t_flat / (t_flat.detach().abs().sum() + 1e-8)

        return self.huber(s_norm, t_norm.detach())

    def forward(self, student_logits, labels, student_features, teacher_features):
        """
        Args:
            student_logits: [B, 1] or [B]
            labels: [B]
            student_features: [B, D] - penultimate features from student
            teacher_features: [B, D] - penultimate features from teacher (no grad)
        """
        if student_logits.dim() == 2 and student_logits.size(1) == 1:
            student_logits = student_logits.squeeze(1)

        # L_S: student classification loss
        l_s = self.bce(student_logits, labels.float())

        # RKD losses
        l_dist = self.distance_loss(student_features, teacher_features)
        l_angle = self.angle_loss(student_features, teacher_features)

        # L_RKD (Eq.6)
        l_rkd = self.lambda_d * l_dist + self.lambda_a * l_angle

        # Total
        total_loss = l_s + l_rkd

        return total_loss, l_s, l_rkd
'''

# ============================================================
# 7) utils/helpers.py  —  load_model فقط مدل برمی‌گردونه (نه تاپل)
# ============================================================
helpers_py = r'''import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet50 import ResNet50WithFeatures


def load_model(checkpoint_path, num_classes='auto', device='cpu'):
    """
    Load a ResNet50WithFeatures model from a checkpoint.
    Returns ONLY the model (not a tuple), matching pipeline.py expectations.

    Args:
        checkpoint_path: path to .pth checkpoint
        num_classes: number of output classes. Use 'auto' to detect from checkpoint.
        device: 'cpu' or 'cuda'

    Returns:
        model: loaded ResNet50WithFeatures on device
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Auto-detect num_classes from fc.weight shape
    if num_classes == 'auto':
        fc_weight_key = None
        for key in state_dict.keys():
            if key == 'fc.weight' or key.endswith('.fc.weight'):
                fc_weight_key = key
                break

        if fc_weight_key is not None:
            num_classes = state_dict[fc_weight_key].shape[0]
            print(f"  [Auto-detect] num_classes = {num_classes} (from {fc_weight_key})")
        else:
            raise ValueError("Cannot auto-detect num_classes: fc.weight not found in checkpoint")

    # Create model
    model = ResNet50WithFeatures(num_classes=num_classes)

    # Remove 'module.' prefix if saved with DataParallel
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = value

    # Load state dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Warning: Missing keys: {missing}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {unexpected}")

    model = model.to(device)
    return model  # فقط مدل برمی‌گردونه


def convert_standard_to_feature_model(standard_state_dict, num_classes=1):
    """
    Convert a standard ResNet50 state_dict to ResNet50WithFeatures format.
    They have the same keys, so this is just a pass-through with verification.
    """
    new_state_dict = {}
    for key, value in standard_state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = value

    return new_state_dict
'''

# ============================================================
# 8) __init__.py files
# ============================================================
models_init = r'''from .resnet50 import ResNet50WithFeatures

__all__ = ['ResNet50WithFeatures']
'''

losses_init = r'''from .student_loss import StudentLoss
from .logits_loss import LogitsDistillationLoss
from .at_loss import AttentionTransferLoss
from .rkd_loss import RKDLoss

__all__ = [
    'StudentLoss',
    'LogitsDistillationLoss',
    'AttentionTransferLoss',
    'RKDLoss'
]
'''

utils_init = r'''from .helpers import load_model, convert_standard_to_feature_model

__all__ = ['load_model', 'convert_standard_to_feature_model']
'''

# ============================================================
# Create all files
# ============================================================
files_to_create = {
    'evaluate.py': evaluate_py,
    'models/__init__.py': models_init,
    'models/resnet50.py': resnet50_py,
    'losses/__init__.py': losses_init,
    'losses/student_loss.py': student_loss_py,
    'losses/logits_loss.py': logits_loss_py,
    'losses/at_loss.py': at_loss_py,
    'losses/rkd_loss.py': rkd_loss_py,
    'utils/__init__.py': utils_init,
    'utils/helpers.py': helpers_py,
}

print("=" * 60)
print("  Creating all missing files...")
print("=" * 60)

for rel_path, content in files_to_create.items():
    full_path = os.path.join(BASE, rel_path)
    dir_path = os.path.dirname(full_path)

    os.makedirs(dir_path, exist_ok=True)

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')

    print(f"  [OK] {rel_path}")

# ============================================================
# Verify
# ============================================================
print("\n" + "=" * 60)
print("  Verifying all files...")
print("=" * 60)

required_files = [
    'evaluate.py',
    'models/__init__.py',
    'models/resnet50.py',
    'losses/__init__.py',
    'losses/student_loss.py',
    'losses/logits_loss.py',
    'losses/at_loss.py',
    'losses/rkd_loss.py',
    'utils/__init__.py',
    'utils/helpers.py',
]

all_ok = True
for rel_path in required_files:
    full_path = os.path.join(BASE, rel_path)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        print(f"  [OK] {rel_path} ({size:,} bytes)")
    else:
        print(f"  [MISSING] {rel_path}")
        all_ok = False

if all_ok:
    print("\n  All files created successfully!")
    print("  Now run: python main.py")
else:
    print("\n  Some files are missing. Check errors above.")
