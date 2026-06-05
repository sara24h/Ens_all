"""
EnsemblePredictor — Load trained student models and run inference.

Supports:
  - Individual student predictions (Logits / AT / RKD)
  - Soft Voting ensemble (Eq. 9)
"""

import torch
from models import ResNet50WithFeatures


class EnsemblePredictor:
    """
    Load trained student checkpoints and perform inference.

    Usage:
        predictor = EnsemblePredictor({
            'logits': './kd_checkpoints/student_logits_final.pth',
            'at':     './kd_checkpoints/student_at_final.pth',
            'rkd':    './kd_checkpoints/student_rkd_final.pth',
        }, device='cuda')

        probs, preds = predictor.predict(images, method='ensemble')
    """

    def __init__(self, model_paths, device='cuda'):
        """
        Args:
            model_paths: dict {kd_type: path_to_pth}
                         keys are 'logits', 'at', 'rkd'
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
            print(f"  Loaded {kd_type}: {path}")

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
            dict: {method_name: (probs, preds)}
        """
        images = images.to(self.device)
        results = {}

        for name, model in self.models.items():
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            results[name] = (probs, preds)

        return results
