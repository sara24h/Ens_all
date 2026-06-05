import torch
import torch.nn as nn
import torch.nn.functional as F

class ResponseKDLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0):
        super(ResponseKDLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        # تغییر به CrossEntropyLoss که استاندارد برای خروجی‌های [batch, classes] است
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # labels باید [64] باشد (مقادیر 0 یا 1)
        loss_ce = self.ce_loss(student_logits, labels)
        
        # محاسبه با dim=1 برای کلاس‌ها
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        
        loss_kl = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        return (1.0 - self.alpha) * loss_ce + self.alpha * loss_kl

class FeatureKDLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(FeatureKDLoss, self).__init__()
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss() # اصلاح به CrossEntropy
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, student_features, teacher_features, labels):
        loss_ce = self.ce_loss(student_logits, labels)
        loss_feat = self.mse_loss(student_features, teacher_features)
        return loss_ce + self.beta * loss_feat

class RelationKDLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(RelationKDLoss, self).__init__()
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def _compute_similarity_matrix(self, features):
        flattened = features.view(features.size(0), -1)
        # در اینجا dim=1 را برای نرمال‌سازی در نظر بگیرید (برای هر تصویر در بچ)
        norm_flat = F.normalize(flattened, p=2, dim=1) 
        similarity_matrix = torch.mm(norm_flat, norm_flat.t())
        return similarity_matrix

    def forward(self, student_logits, student_features, teacher_features, labels):
        loss_ce = self.ce_loss(student_logits, labels)
        
        s_sim = self._compute_similarity_matrix(student_features)
        t_sim = self._compute_similarity_matrix(teacher_features)
        
        loss_relation = self.mse_loss(s_sim, t_sim)
        return loss_ce + self.gamma * loss_relation
