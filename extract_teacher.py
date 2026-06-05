# extract_teacher.py
import torch
from tqdm import tqdm
import pickle
from dataset import Dataset_selector
from models import get_resnet50, DistillationWrapper

def extract_teacher_cache(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Teacher...")
    teacher_base = get_resnet50(num_classes=2, checkpoint_path=args.teacher_ckpt)
    teacher = DistillationWrapper(teacher_base).to(device)
    teacher.eval()
    
    selector = Dataset_selector(...)  # همان تنظیمات قبلی
    
    cache = []
    with torch.no_grad():
        for images, labels in tqdm(selector.train_loader, desc="Extracting teacher outputs"):
            images = images.to(device)
            logits, features = teacher(images)
            cache.append((
                logits.cpu(), 
                features.cpu() if features is not None else None,
                labels.cpu()
            ))
    
    save_path = f"teacher_cache_{args.dataset}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Teacher cache saved to {save_path}")
