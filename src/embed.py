"""
이미지 임베딩 추출기. 백본 3종을 같은 인터페이스로 제공한다.
  - dinov2  : facebook/dinov2-small   (self-supervised, 검색에 강함) — 메인
  - clip    : openai/clip-vit-base-patch32 (image-text 대조학습) — 비교
  - resnet50: ImageNet 지도학습 백본 — 비교 baseline

모든 임베딩은 L2 정규화해서 반환 → 내적 = 코사인.
전처리 옵션(center-crop)으로 "배경 제거가 검색에 도움 되나"를 실험할 수 있게 했다.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _load_rgb(path, center_crop=False):
    img = Image.open(path).convert("RGB")
    if center_crop:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img


class Embedder:
    def __init__(self, backbone="dinov2", device=None):
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._build()

    def _build(self):
        if self.backbone == "dinov2":
            from transformers import AutoImageProcessor, AutoModel
            self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        elif self.backbone == "clip":
            from transformers import CLIPModel, CLIPImageProcessor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif self.backbone == "resnet50":
            import torchvision as tv
            from torchvision import transforms
            weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
            m = tv.models.resnet50(weights=weights)
            m.fc = torch.nn.Identity()          # 분류 헤드 제거 → 2048차 특징
            self.model = m
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"unknown backbone: {self.backbone}")
        self.model.to(self.device).eval()

    def _forward(self, imgs):
        if self.backbone == "dinov2":
            inp = self.proc(images=imgs, return_tensors="pt").to(self.device)
            out = self.model(**inp).last_hidden_state[:, 0]        # CLS 토큰
        elif self.backbone == "clip":
            inp = self.proc(images=imgs, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inp)
        else:  # resnet50
            batch = torch.stack([self.tf(im) for im in imgs]).to(self.device)
            out = self.model(batch)
        return torch.nn.functional.normalize(out, dim=-1)          # 코사인용

    @torch.no_grad()
    def encode(self, paths, batch_size=64, center_crop=False, desc=""):
        vecs = []
        for i in tqdm(range(0, len(paths), batch_size), desc=desc or self.backbone):
            imgs = [_load_rgb(p, center_crop) for p in paths[i:i + batch_size]]
            vecs.append(self._forward(imgs).cpu().numpy())
        return np.concatenate(vecs).astype("float32")
