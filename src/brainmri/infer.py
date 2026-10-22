"""Загрузка обученной модели и предсказание с картой внимания. Нужны torch, timm, grad-cam."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from brainmri.data.preprocess import preprocess_gray, center_eval
from brainmri.serving import nn_distance


class Predictor:
    """weights_dir должна содержать config.json и model_state.pt (их создаёт блокнот обучения, ячейка 4.7).
    Необязательно: prototype_config.json (пороги) и reference_bank.npz (опорный набор для «непохожести»)."""

    def __init__(self, weights_dir, device: str | None = None):
        import torch
        import timm
        self.torch = torch
        wd = Path(weights_dir)
        self.cfg = json.load(open(wd / "config.json", encoding="utf-8"))
        self.classes = list(self.cfg["classes"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(self.cfg["model"], pretrained=False, num_classes=len(self.classes),
                                       in_chans=int(self.cfg.get("in_chans", 1)))
        self.model.load_state_dict(torch.load(wd / "model_state.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        self.mean, self.std = float(self.cfg["mean"]), float(self.cfg["std"])

        pc = wd / "prototype_config.json"
        proto = json.load(open(pc, encoding="utf-8")) if pc.exists() else {}
        self.t_conf = float(proto.get("t_conf", 0.894))
        self.t_dist = proto.get("t_dist")
        rb = wd / "reference_bank.npz"
        if rb.exists():
            z = np.load(rb)
            self.ref_mean, self.ref_bank = z["mean"], z["bank"]
        else:
            self.ref_mean = self.ref_bank = None
        self._cam_algo = None

    def _tensor(self, img01: np.ndarray):
        x = ((img01 - self.mean) / self.std).astype(np.float32)
        return self.torch.from_numpy(x)[None, None].to(self.device)

    def _cam(self, xb) -> np.ndarray:
        from pytorch_grad_cam import GradCAM
        if self._cam_algo is None:
            self._cam_algo = GradCAM(model=self.model, target_layers=[self.model.stages[-1]])
        return self._cam_algo(input_tensor=xb, targets=None)[0].astype(np.float32)

    def predict(self, gray_u8: np.ndarray) -> dict:
        """gray_u8: uint8 HxW. Возвращает снимок после стандартизации, карту внимания, вероятности, класс, уверенность, непохожесть."""
        img = center_eval(preprocess_gray(gray_u8))
        xb = self._tensor(img)
        with self.torch.no_grad():
            f = self.model.forward_features(xb)
            feat = self.model.forward_head(f, pre_logits=True)[0].float().cpu().numpy()
            probs = self.model.forward_head(f).float().softmax(1)[0].cpu().numpy()
        pred = int(probs.argmax())
        dist = float(nn_distance(feat, self.ref_mean, self.ref_bank)[0]) if self.ref_bank is not None else None
        return dict(img=img, cam=self._cam(xb), probs=probs, pred=pred, conf=float(probs[pred]), dist=dist, classes=self.classes)
