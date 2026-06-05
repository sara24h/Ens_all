# train.py
import os
import argparse
import torch
import torch.optim as optim
from dataset import Dataset_selector
import torch.nn as nn
from tqdm import tqdm
from models import get_resnet50, DistillationWrapper
from losses import ResponseKDLoss, FeatureKDLoss, RelationKDLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training for DeepFake")
    parser.add_argument('--dataset', type=str, required=True, choices=['140k', '190k', '200k'])
    parser.add_argument('--kd_method', type=str, required=True, choices=['response', 'feature', 'relation'])
    parser.add_argument('--teacher_ckpt', type=str, required=True, help="Path to teacher checkpoint")
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ۱. بارگذاری دیتابیس بر اساس آرگومان ورودی
    print(f"Initializing {args.dataset} dataset...")
    # متغیرهای مسیرها را بر اساس محیط سیستم یا کگل خود بازتنظیم کنید
    selector = Dataset_selector(
        dataset_mode=args.dataset,
        realfake140k_train_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces',
        realfake200k_train_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/train_labels.csv',
        realfake200k_val_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/val_labels.csv',
        realfake200k_test_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/test_labels.csv',
        realfake200k_root_dir='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset',
        realfake190k_root_dir='/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset',
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size
    )
    
    # ۲. راه‌اندازی مدل معلم و قرار دادن در رپر هوک
    # ۲. راه‌اندازی مدل معلم
    print("Loading Teacher model...")
    teacher_base = get_resnet50(num_classes=2, checkpoint_path=args.teacher_ckpt)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher = DistillationWrapper(teacher_base).to(device)
    teacher.eval()

# ۳. راه‌اندازی مدل دانش‌آموز (موازی‌سازی)
    print("Initializing Student model (ResNet-50)...")
    student_base = get_resnet50(num_classes=2, checkpoint_path=None)
    student = DistillationWrapper(student_base)

# بررسی تعداد GPUها و موازی‌سازی
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        student = nn.DataParallel(student)

# انتقال مدل موازی‌سازی شده به کارت اول (کارت‌های دیگر خودکار اضافه می‌شوند)
    student = student.to(device)
    
    # ۴. تعیین تابع اتلاف براساس روش انتخابی
    if args.kd_method == 'response':
        criterion = ResponseKDLoss(alpha=0.5, temperature=4.0)
    elif args.kd_method == 'feature':
        criterion = FeatureKDLoss(beta=1.0)
    elif args.kd_method == 'relation':
        criterion = RelationKDLoss(gamma=1.0)
        
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    
    # ۵. حلقه اصلی آموزش
    # ۵. حلقه اصلی آموزش
    # ۵. حلقه اصلی آموزش
    for epoch in range(args.epochs):
        student.train()
        
        # نوار پیشرفت را داخل حلقه اپوک قرار دهید
        progress_bar = tqdm(selector.loader_train, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # --- عملیات فوروارد ---
            with torch.no_grad():
                t_logits, t_features = teacher(images)
            s_logits, s_features = student(images)
            
            # --- محاسبه Loss ---
            if args.kd_method == 'response':
                loss = criterion(s_logits, t_logits, labels)
            elif args.kd_method == 'feature':
                loss = criterion(s_logits, s_features, t_features, labels)
            else: # relation
                loss = criterion(s_logits, s_features, t_features, labels) # متناسب با متد خود تنظیم کنید
            
            loss.backward()
            optimizer.step()
            
            # --- به‌روزرسانی نوار پیشرفت (بعد از محاسبه loss) ---
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    # ۶. ذخیره وزن‌ها (خارج از حلقه epoch قرار می‌گیرد)
    save_path = os.path.join(args.save_dir, f"student_{args.dataset}_{args.kd_method}.pth")
    # دقت کنید اگر از DataParallel استفاده کردید، برای ذخیره درست باید از student.module.state_dict() استفاده کنید
    state_dict = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
    torch.save(state_dict, save_path)
    print(f"Saved student model weights to {save_path}\n")
        

if __name__ == '__main__':
    train()
