# ensemble.py
import torch
import torch.nn.functional as F
from dataset import Dataset_selector
from models import get_resnet50

def evaluate_ensemble(student_paths, target_dataset='140k'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # بارگذاری دیتابیس تست هدف برای ارزیابی انسمبل
    print(f"Loading test set of {target_dataset} for ensemble verification...")
    selector = Dataset_selector(
        dataset_mode=target_dataset,
        realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
        realfake200k_train_csv='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/train_labels.csv',
        realfake200k_val_csv='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/val_labels.csv',
        realfake200k_test_csv='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/test_labels.csv',
        realfake200k_root_dir='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset',
        realfake190k_root_dir='/kaggle/input/deepfake-and-real-images/Dataset',
        eval_batch_size=64
    )
    test_loader = selector.loader_test

    # بارگذاری و آماده‌سازی هر ۳ مدل دانش‌آموز
    models_list = []
    for path in student_paths:
        model = get_resnet50(num_classes=2, checkpoint_path=path)
        model = model.to(device)
        model.eval()
        models_list.append(model)
        
    correct = 0
    total = 0
    
    print("Evaluating Ensemble Model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # جمع‌آوری احتمالات نرمال‌شده (Softmax) از هر مدل دانش‌آموز
            ensemble_probs = torch.zeros((images.size(0), 2)).to(device)
            
            for model in models_list:
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                ensemble_probs += probs
                
            # میانگین‌گیری از احتمالات طبق روش انسمبل مقاله
            ensemble_probs /= len(models_list)
            
            # استخراج پیش‌بینی نهایی
            _, predicted = ensemble_probs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    ensemble_accuracy = 100. * correct / total
    print(f"\nFinal Ensemble Accuracy on {target_dataset} test set: {ensemble_accuracy:.2f}%")

if __name__ == '__main__':
    # آدرس چک‌پوینت‌های ذخیره شده پس از اتمام آموزش ۳ مدل دانش‌آموز
    paths = [
        './checkpoints/student_140k_response.pth',
        './checkpoints/student_190k_feature.pth',
        './checkpoints/student_200k_relation.pth'
    ]
    evaluate_ensemble(paths, target_dataset='140k')
