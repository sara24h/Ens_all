"""
losses.py
=========
All knowledge-distillation loss functions from the paper:

  Eq. 1  — StudentLoss              (cross-entropy with ground truth)
  Eq. 2  — LogitsDistillationLoss   (MSE between teacher / student logits)
  Eq. 3  — LogitsKDLoss             (combined:  α·L_S + β·L_logits)
  Eq. 4  — AttentionTransferLoss    (AT — L2 distance of normalised attention maps)
  Eq. 5  — ATKDLoss                 (combined:  L_S + L_AT)
  Eq. 6  — RKDLoss                  (distance-wise + angle-wise with Huber loss)
  Eq. 7  — huber_loss               (smooth L1 used inside RKD)
  Eq. 8  — RKDKDLoss                (combined:  L_S + L_RKD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helper ────────────────────────────────────────────────────────────────

def compute_attention_map(feature_map: torch.Tensor) -> torch.Tensor:
    """
    AT [29]:  A(F) = sum_c |F_c|  →  vectorise  →  L2-normalise.

    Args:
        feature_map: [B, C, H, W]
    Returns:
        attention:   [B, H·W]  (L2-normalised)
    """
    att = torch.sum(torch.abs(feature_map), dim=1)          # [B, H, W]
    att = att.view(att.size(0), -1)                          # [B, H·W]
    norm = torch.norm(att, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    return att / norm


# ── Eq. 1 ────────────────────────────────────────────────────────────────

class StudentLoss(nn.Module):
    """Cross-entropy between student soft-target and ground-truth label."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, labels):
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return self.criterion(student_logits, labels)


# ── Eq. 2 ────────────────────────────────────────────────────────────────

class LogitsDistillationLoss(nn.Module):
    """L_logits(z_t, z_s) = Σ (z_t − z_s)²   (MSE over logits)."""

    def __init__(self):
        super().__init__()

    def forward(self, teacher_logits, student_logits):
        return F.mse_loss(teacher_logits, student_logits)


# ── Eq. 3 ────────────────────────────────────────────────────────────────

class LogitsKDLoss(nn.Module):
    """L = α · L_S + β · L_logits"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.student_loss = StudentLoss()
        self.logits_loss  = LogitsDistillationLoss()

    def forward(self, student_logits, teacher_logits, labels):
        return (self.alpha * self.student_loss(student_logits, labels)
                + self.beta * self.logits_loss(teacher_logits, student_logits))


# ── Eq. 4 ────────────────────────────────────────────────────────────────

class AttentionTransferLoss(nn.Module):
    """
    L_AT = Σ_j  ‖  Q_S_j / ‖Q_S_j‖₂  −  Q_T_j / ‖Q_T_j‖₂  ‖²

    Sums over all student-teacher activation-layer pairs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, teacher_attention_maps, student_attention_maps):
        loss = 0.0
        for t_map, s_map in zip(teacher_attention_maps, student_attention_maps):
            # resize student map if spatial dims differ (shouldn't for same arch)
            if t_map.shape[2:] != s_map.shape[2:]:
                s_map = F.interpolate(s_map, size=t_map.shape[2:],
                                      mode='bilinear', align_corners=False)
            t_att = compute_attention_map(t_map)
            s_att = compute_attention_map(s_map)
            loss = loss + torch.sum((s_att - t_att) ** 2)
        return loss


# ── Eq. 5 ────────────────────────────────────────────────────────────────

class ATKDLoss(nn.Module):
    """L = L_S + L_AT"""

    def __init__(self):
        super().__init__()
        self.student_loss = StudentLoss()
        self.at_loss      = AttentionTransferLoss()

    def forward(self, student_logits, labels,
                teacher_attention_maps, student_attention_maps):
        return (self.student_loss(student_logits, labels)
                + self.at_loss(teacher_attention_maps, student_attention_maps))


# ── Eq. 6 & 7 ────────────────────────────────────────────────────────────

class RKDLoss(nn.Module):
    """
    Relational Knowledge Distillation (RKD) [30].

    Combines:
      • distance-wise potential  (pairwise Euclidean distances)
      • angle-wise potential     (angles formed by triplets)
    Both penalised with Huber loss (Eq. 7).
    """

    def __init__(self, distance_weight: float = 1.0,
                       angle_weight: float   = 2.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight    = angle_weight

    # ── Eq. 7 ──

    @staticmethod
    def huber_loss(pred, target, delta: float = 1.0):
        diff      = torch.abs(pred - target)
        quadratic = torch.min(diff, torch.tensor(delta, device=diff.device))
        linear    = diff - quadratic
        return 0.5 * quadratic ** 2 + linear

    # ── potentials ──

    def distance_wise_potential(self, features):
        """Upper-triangle pairwise Euclidean distances."""
        diff = features.unsqueeze(1) - features.unsqueeze(0)   # [B,B,D]
        dist = torch.norm(diff, p=2, dim=2)                    # [B,B]
        mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
        return dist[mask]

    def angle_wise_potential(self, features):
        """Angles at each vertex for all unique triplets."""
        diff = features.unsqueeze(1) - features.unsqueeze(0)   # [B,B,D]
        norm = torch.norm(diff, p=2, dim=2, keepdim=True).clamp(min=1e-8)
        diff_n = diff / norm                                    # [B,B,D]

        cos_sim = torch.bmm(diff_n.transpose(0, 1), diff_n)   # [B,B,B]
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        angles  = torch.acos(cos_sim)

        B = features.size(0)
        idx = [(i, j, k)
               for j in range(B)
               for i in range(B) if i != j
               for k in range(i + 1, B) if k != j]
        if not idx:
            return torch.tensor(0.0, device=features.device)

        ii, jj, kk = zip(*idx)
        return angles[torch.tensor(ii, device=features.device),
                       torch.tensor(jj, device=features.device),
                       torch.tensor(kk, device=features.device)]

    # ── forward ──

    def forward(self, teacher_features, student_features):
        # distance-wise
        t_dist = self.distance_wise_potential(teacher_features)
        s_dist = self.distance_wise_potential(student_features)
        t_dist = t_dist / (t_dist.sum() + 1e-8)
        s_dist = s_dist / (s_dist.sum() + 1e-8)
        dist_loss = self.huber_loss(s_dist, t_dist).mean()

        # angle-wise
        t_ang = self.angle_wise_potential(teacher_features)
        s_ang = self.angle_wise_potential(student_features)
        if isinstance(t_ang, torch.Tensor) and t_ang.numel() > 0:
            ang_loss = self.huber_loss(s_ang, t_ang).mean()
        else:
            ang_loss = torch.tensor(0.0, device=teacher_features.device)

        return self.distance_weight * dist_loss + self.angle_weight * ang_loss


# ── Eq. 8 ────────────────────────────────────────────────────────────────

class RKDKDLoss(nn.Module):
    """L = L_S + L_RKD"""

    def __init__(self, distance_weight: float = 1.0,
                       angle_weight: float   = 2.0):
        super().__init__()
        self.student_loss = StudentLoss()
        self.rkd_loss     = RKDLoss(distance_weight, angle_weight)

    def forward(self, student_logits, labels,
                teacher_features, student_features):
        return (self.student_loss(student_logits, labels)
                + self.rkd_loss(teacher_features, student_features))
