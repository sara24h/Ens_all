"""
predictor.py
============
EnsemblePredictor — load trained student models and run inference.

Example
-------
    predictor = EnsemblePredictor({
        'logits': './kd_checkpoints/student_logits_final.pth',
        'at':     './kd_checkpoints/student_at_final.pth',
        'rkd':    './kd_checkpoints/student_rkd_final.pth',
    })

    probs, preds = predictor.predict(images, method='ensemble')
"""

import torch
import numpy as np
from models import ResNet50WithFeatures


class EnsemblePredictor:
    """
    Load the three trained student models and perform inference
    using soft-voting ensemble  (Eq. 9)  or any individual model.
    """

    def __init__(self, model_paths: dict, device: str = "cuda"):
        """
        Parameters
        ----------
        model_paths : dict
            Keys: 'logits', 'at', 'rkd'
            Values: paths to the corresponding .pth checkpoints
        device : str
            'cuda' or 'cpu'
        """
        self.device = device
        self.models = {}

        for kd_type, path in model_paths.items():
            model = ResNet50WithFeatures(num_classes=1, pretrained=False)
            state_dict = torch.load(path, map_location=device)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device).eval()
            self.models[kd_type] = model

    @torch.no_grad()
    def predict(self, images: torch.Tensor, method: str = "ensemble"):
        """
        Parameters
        ----------
        images : Tensor [B, 3, H, W]   (already pre-processed)
        method : 'ensemble' | 'logits' | 'at' | 'rkd'

        Returns
        -------
        probs : ndarray [B]   — probability of being REAL
        preds : ndarray [B]   — binary prediction (0 = fake, 1 = real)
        """
        images = images.to(self.device)

        if method == "ensemble":
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
    def predict_individual(self, images: torch.Tensor) -> dict:
        """
        Get predictions from each student separately.

        Returns
        -------
        dict  {method_name: (probs, preds)}
        """
        images = images.to(self.device)
        results = {}

        for name, model in self.models.items():
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            results[name] = (probs, preds)

        return results
