# models.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes=2, checkpoint_path=None):
    """
    ساخت مدل رزنیت ۵۰ با قابلیت بارگذاری چک‌پوینت معلم
    """
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # در صورت وجود کلید متفرقه هدایت مستقیم به استیت دیکشنری اصلی انجام می‌شود
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    return model

class DistillationWrapper(nn.Module):
    """
    رپری روی رزنیت ۵۰ برای دسترسی آسان به فیچرهای لایه ۴ و لاجیت‌های خروجی مدل
    """
    def __init__(self, model):
        super(DistillationWrapper, self).__init__()
        self.model = model
        self.features = None
        
        # ثبت هوک روی آخرین بلوک کانولوشنی رزنیت ۵۰ (layer4)
        self.model.layer4.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # ذخیره فیچر مپ میانی خروجی لایه ۴
        self.features = output

    def forward(self, x):
        logits = self.model(x)
        return logits, self.features
