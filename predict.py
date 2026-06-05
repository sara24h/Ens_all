"""
EnsemblePredictor — Load trained student models and run inference.

Student model assignments:
  Student-Logits  ←  Teacher-200k  on  Dataset-200k   (Response-based KD)
  Student-AT      ←  Teacher-140k  on  Dataset-140k   (Feature-based KD)
  Student-RKD     ←  Teacher-190k  on  Dataset-190k   (Relation-based KD)

Usage:
    from predict import EnsemblePredictor

    predictor = EnsemblePredictor({
        'logits': './kd_checkpoints/student_logits_final.pth',   # trained on 200k
        'at':     './kd_checkpoints/student_at_final.pth',       # trained on 140k
        'rkd':    './kd_checkpoints/student_rkd_final.pth',      # trained on 190k
    }, device='cuda')

    probs, preds = predictor.predict(images, method='ensemble')
"""

import torch
from models import ResNet50WithFeatures


# Which dataset each student was trained on (for logging / reference)
STUDENT_DATASETS = {
    'logits': '200k',   # Response-based KD  ← Teacher-200k
    'at':     '140k',   # Feature-based KD   ← Teacher-140k
    'rkd':    '190k',   # Relation-based KD  ← Teacher-190k
}


class EnsemblePredictor:
    """
    Load trained student checkpoints and perform inference.

    Supports:
      - Individual student predictions (Logits / AT / RKD)
      - Soft Voting ensemble (Eq. 9)
    """

    def __init__(self, model_paths, device='cuda'):
        """
        Args:
            model_paths: dict {kd_type: path_to_pth}
                         keys: 'logits', 'at', 'rkd'
            device:      'cuda' or 'cpu'
        """
        self.device = device
        self.models = {}

        for kd_type, path in model_paths.items():
            model = ResNet50WithFeatures(num_classes=1, pretrained=False)
            state_dict = torch.load(path, map_location=device)

            # Handle DataParallel prefix
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {
                    k.replace('module.', ''): v for k, v in state_dict.items()
                }

            model.load_state_dict(state_dict, strict=False)
            model = model.to(device).eval()
            self.models[kd_type] = model

            ds = STUDENT_DATASETS.get(kd_type, '?')
            print(f"  Loaded {kd_type} (trained on {ds}): {path}")

    @torch.no_grad()
    def predict(self, images, method='ensemble'):
        """
        Predict using a single student or the Soft Voting ensemble.

        Args:
            images: tensor [B, 3, H, W] (already preprocessed)
            method: 'ensemble' | 'logits' | 'at' | 'rkd'

        Returns:
            probs: numpy [B] — P(real)
            preds: numpy [B] — binary predictions (0=fake, 1=real)
        """
        images = images.to(self.device)

        if method == 'ensemble':
            # Soft Voting — Eq. 9
            avg_logits = (
                sum(m(images) for m in self.models.values())
                / len(self.models)
            )
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
            dict: {method_name: {'probs': array, 'preds': array, 'dataset': str}}
        """
        images = images.to(self.device)
        results = {}

        for name, model in self.models.items():
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            results[name] = {
                'probs': probs,
                'preds': preds,
                'trained_on': STUDENT_DATASETS.get(name, 'unknown'),
            }

        return results
