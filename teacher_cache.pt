import torch
from dataset import Dataset_selector
from models import get_resnet50, DistillationWrapper
import os

# تنظیمات
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_mode = '140k' # یا هر کدام که استفاده می‌کنید
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
        train_batch_size=64, # یا مقداری که در آرگومان‌ها داشتید
        eval_batch_size=64
    )
teacher = get_resnet50(num_classes=2, checkpoint_path='/kaggle/input/models/sara24h/teacher_model_best/pytorch/default/1/teacher_model_best.pth')
teacher = DistillationWrapper(teacher).to(device)
teacher.eval()

# ذخیره خروجی‌ها
cache = []
print("Caching teacher outputs...")
with torch.no_grad():
    for images, labels in selector.loader_train: # مطمئن شوید shuffle=False است
        images = images.to(device)
        t_logits, t_features = teacher(images)
        cache.append((t_logits.cpu(), t_features.cpu(), labels))

torch.save(cache, 'teacher_cache.pt')
print("Done! teacher_cache.pt saved.")
