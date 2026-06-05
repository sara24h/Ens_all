# ═══════════════════════════════════════════════════════════════════
# این سلول رو کپی کن و تو Kaggle Notebook ران کن
# همه فایل‌های گم‌شده ساخته میشن
# ═══════════════════════════════════════════════════════════════════

import os

base = '/kaggle/working/Ens_all'

# ─── models/resnet50.py ───
os.makedirs(f'{base}/models', exist_ok=True)
with open(f'{base}/models/resnet50.py', 'w') as f:
    f.write(r'''import torch.nn as nn
from torchvision import models

class ResNet50WithFeatures(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.feature_channels = [256, 512, 1024, 2048]

    def forward(self, x, return_features=False, return_attention=False):
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
        features = features.flatten(1)
        logits = self.fc(features)
        if not return_features and not return_attention:
            return logits
        outputs = [logits]
        if return_features:
            outputs.append(features)
        if return_attention:
            outputs.append(attention_maps)
        return tuple(outputs)
''')

with open(f'{base}/models/__init__.py', 'w') as f:
    f.write('from .resnet50 import ResNet50WithFeatures\n__all__ = ["ResNet50WithFeatures"]\n')

# ─── losses/student_loss.py ───
os.makedirs(f'{base}/losses', exist_ok=True)
with open(f'{base}/losses/student_loss.py', 'w') as f:
    f.write(r'''import torch.nn as nn

class StudentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, labels):
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return self.criterion(student_logits, labels)
''')

# ─── losses/logits_loss.py ───
with open(f'{base}/losses/logits_loss.py', 'w') as f:
    f.write(r'''import torch.nn as nn
import torch.nn.functional as F
from .student_loss import StudentLoss

class LogitsDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_logits, student_logits):
        return F.mse_loss(teacher_logits, student_logits)

class LogitsKDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.student_loss = StudentLoss()
        self.logits_loss = LogitsDistillationLoss()

    def forward(self, student_logits, teacher_logits, labels):
        l_s = self.student_loss(student_logits, labels)
        l_kd = self.logits_loss(teacher_logits, student_logits)
        return self.alpha * l_s + self.beta * l_kd
''')

# ─── losses/at_loss.py ───
with open(f'{base}/losses/at_loss.py', 'w') as f:
    f.write(r'''import torch
import torch.nn as nn
import torch.nn.functional as F
from .student_loss import StudentLoss

def compute_attention_map(feature_map):
    attention = torch.sum(torch.abs(feature_map), dim=1)
    attention = attention.view(attention.size(0), -1)
    norm = torch.norm(attention, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    attention = attention / norm
    return attention

class AttentionTransferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_attention_maps, student_attention_maps):
        loss = 0.0
        for t_map, s_map in zip(teacher_attention_maps, student_attention_maps):
            if t_map.shape[2:] != s_map.shape[2:]:
                s_map = F.interpolate(s_map, size=t_map.shape[2:], mode='bilinear', align_corners=False)
            t_att = compute_attention_map(t_map)
            s_att = compute_attention_map(s_map)
            loss += torch.sum((s_att - t_att) ** 2)
        return loss

class ATKDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_loss = StudentLoss()
        self.at_loss = AttentionTransferLoss()

    def forward(self, student_logits, labels, teacher_attention_maps, student_attention_maps):
        return self.student_loss(student_logits, labels) + self.at_loss(teacher_attention_maps, student_attention_maps)
''')

# ─── losses/rkd_loss.py ───
with open(f'{base}/losses/rkd_loss.py', 'w') as f:
    f.write(r'''import torch
import torch.nn as nn
from .student_loss import StudentLoss

class RKDLoss(nn.Module):
    def __init__(self, distance_weight=1.0, angle_weight=2.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    @staticmethod
    def huber_loss(pred, target, delta=1.0):
        diff = torch.abs(pred - target)
        quadratic = torch.min(diff, torch.tensor(delta, device=diff.device))
        linear = diff - quadratic
        return 0.5 * quadratic ** 2 + linear

    def distance_wise_potential(self, features):
        diff = features.unsqueeze(1) - features.unsqueeze(0)
        dist_matrix = torch.norm(diff, p=2, dim=2)
        mask = torch.triu(torch.ones_like(dist_matrix, dtype=torch.bool), diagonal=1)
        return dist_matrix[mask]

    def angle_wise_potential(self, features):
        diff = features.unsqueeze(1) - features.unsqueeze(0)
        norm = torch.norm(diff, p=2, dim=2, keepdim=True).clamp(min=1e-8)
        diff_n = diff / norm
        cos_sim = torch.bmm(diff_n.transpose(0, 1), diff_n)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.acos(cos_sim)
        B = features.size(0)
        ii, jj, kk = [], [], []
        for j in range(B):
            for i in range(B):
                if i == j: continue
                for k in range(i + 1, B):
                    if k == j: continue
                    ii.append(i); jj.append(j); kk.append(k)
        if not ii:
            return torch.tensor(0.0, device=features.device)
        return angles[torch.tensor(ii, device=features.device),
                      torch.tensor(jj, device=features.device),
                      torch.tensor(kk, device=features.device)]

    def forward(self, teacher_features, student_features):
        t_dist = self.distance_wise_potential(teacher_features)
        s_dist = self.distance_wise_potential(student_features)
        t_dist = t_dist / (t_dist.sum() + 1e-8)
        s_dist = s_dist / (s_dist.sum() + 1e-8)
        dist_loss = self.huber_loss(s_dist, t_dist).mean()
        t_ang = self.angle_wise_potential(teacher_features)
        s_ang = self.angle_wise_potential(student_features)
        if isinstance(t_ang, torch.Tensor) and t_ang.numel() > 0:
            ang_loss = self.huber_loss(s_ang, t_ang).mean()
        else:
            ang_loss = torch.tensor(0.0, device=teacher_features.device)
        return self.distance_weight * dist_loss + self.angle_weight * ang_loss

class RKDKDLoss(nn.Module):
    def __init__(self, distance_weight=1.0, angle_weight=2.0):
        super().__init__()
        self.student_loss = StudentLoss()
        self.rkd_loss = RKDLoss(distance_weight, angle_weight)

    def forward(self, student_logits, labels, teacher_features, student_features):
        return self.student_loss(student_logits, labels) + self.rkd_loss(teacher_features, student_features)
''')

# ─── losses/__init__.py ───
with open(f'{base}/losses/__init__.py', 'w') as f:
    f.write('''from .student_loss import StudentLoss
from .logits_loss import LogitsDistillationLoss, LogitsKDLoss
from .at_loss import AttentionTransferLoss, ATKDLoss
from .rkd_loss import RKDLoss, RKDKDLoss
__all__ = ['StudentLoss', 'LogitsDistillationLoss', 'LogitsKDLoss',
           'AttentionTransferLoss', 'ATKDLoss', 'RKDLoss', 'RKDKDLoss']
''')

# ─── utils/helpers.py ───
os.makedirs(f'{base}/utils', exist_ok=True)
with open(f'{base}/utils/helpers.py', 'w') as f:
    f.write(r'''import os, sys, torch
import torch.nn as nn
from torchvision import models

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models import ResNet50WithFeatures

def _detect_num_classes(state_dict):
    if 'fc.weight' in state_dict:
        return state_dict['fc.weight'].shape[0]
    return 1

def load_model(path, model_type='auto', num_classes='auto', device='cuda'):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if num_classes == 'auto':
        num_classes = _detect_num_classes(state_dict)
        print(f"  Auto-detected num_classes={num_classes}")
    model = ResNet50WithFeatures(num_classes=num_classes, pretrained=False)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()
    print(f"  Loaded [{model_type}]: {path}")
    print(f"    Keys: {len(filtered)}/{len(model_keys)} matched")
    if 'fc.weight' in filtered:
        print(f"    FC: weight{list(filtered['fc.weight'].shape)} bias{list(filtered['fc.bias'].shape)}")
    return model
''')

# ─── utils/__init__.py ───
with open(f'{base}/utils/__init__.py', 'w') as f:
    f.write('from .helpers import load_model\n__all__ = ["load_model"]\n')

# ─── بررسی نهایی ───
print("=" * 50)
print("  فایل‌های ساخته شده:")
print("=" * 50)
for root, dirs, files in os.walk(base):
    if '.git' in root or 'data' in root:
        continue
    level = root.replace(base, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    for file in sorted(files):
        if file.endswith('.py'):
            print(f'{indent}  {file}')

# ─── تست ایمپورت ───
print("\n" + "=" * 50)
print("  تست ایمپورت:")
print("=" * 50)
sys.path.insert(0, base)
try:
    from models import ResNet50WithFeatures
    print("  ✅ models.ResNet50WithFeatures")
except Exception as e:
    print(f"  ❌ models: {e}")
try:
    from losses import StudentLoss, LogitsDistillationLoss, AttentionTransferLoss, RKDLoss
    print("  ✅ losses (all 4)")
except Exception as e:
    print(f"  ❌ losses: {e}")
try:
    from utils import load_model
    print("  ✅ utils.load_model")
except Exception as e:
    print(f"  ❌ utils: {e}")

print("\nاگه همه ✅ شد، حالا python main.py رو ران کن!")
