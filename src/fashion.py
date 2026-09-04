"""
패션 특화(domain) 임베딩 — FashionCLIP.
범용 DINOv2와 비교/경쟁시키기 위한 모듈.

FashionCLIP(patrickjohncyh/fashion-clip)은 대규모 패션 상품 이미지-텍스트로 학습된
CLIP이라, "상품 검색"에 곧바로 맞는 도메인 표현을 준다. (쇼포트 JD 직결)

제공: zero-shot 임베딩(FashionEmbedder) + 백본 fine-tune(finetune_fashion) — DINOv2와 같은
supervised-contrastive 방식으로 비교 공정성을 맞춤.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .finetune import supcon_loss, pk_batches

# OpenAI CLIP 정규화 (FashionCLIP은 CLIP ViT-B/32 기반)
_MEAN = [0.48145466, 0.4578275, 0.40821073]
_STD = [0.26862954, 0.26130258, 0.27577711]
FASHION_CLIP = "patrickjohncyh/fashion-clip"


def _train_tf(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _eval_tf(size=224):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _load(p):
    return Image.open(p).convert("RGB")


def _build(device, name=FASHION_CLIP):
    from transformers import CLIPModel
    return CLIPModel.from_pretrained(name).to(device)


def _image_features(model, pixel_values):
    """CLIP 이미지 임베딩을 명시적으로 계산 (transformers 버전에 안 의존)."""
    pooled = model.vision_model(pixel_values=pixel_values).pooler_output
    return model.visual_projection(pooled)


class FashionEmbedder:
    """FashionCLIP zero-shot 이미지 임베딩 (DINOv2 Embedder와 동일 인터페이스)."""
    def __init__(self, name=FASHION_CLIP, device=None, size=224):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build(self.device, name).eval()
        self.tf = _eval_tf(size)

    @torch.no_grad()
    def encode(self, paths, batch_size=64, desc="fashion"):
        vecs = []
        for i in tqdm(range(0, len(paths), batch_size), desc=desc):
            imgs = torch.stack([self.tf(_load(p)) for p in paths[i:i + batch_size]]).to(self.device)
            out = _image_features(self.model, imgs)   # 투영된 이미지 임베딩
            vecs.append(F.normalize(out, dim=-1).cpu().numpy())
        return np.concatenate(vecs).astype("float32")


def finetune_fashion(train_paths, train_labels, name=FASHION_CLIP, epochs=3, P=8, K=4,
                     steps_per_epoch=200, lr=1e-5, temp=0.1, size=224,
                     unfreeze_layers=4, device=None, seed=0, log_every=50):
    """
    FashionCLIP 비전 인코더를 supervised-contrastive로 fine-tune.
    unfreeze_layers: 비전 트랜스포머 뒤 N개 레이어 + visual_projection 학습.
    lr은 CLIP이 민감해 DINOv2(2e-5)보다 낮게(1e-5) 권장.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _build(device, name)

    for p in model.parameters():
        p.requires_grad = False
    for blk in model.vision_model.encoder.layers[-unfreeze_layers:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.visual_projection.parameters():
        p.requires_grad = True
    for p in model.vision_model.post_layernorm.parameters():
        p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"학습 파라미터: {sum(p.numel() for p in params)/1e6:.2f}M "
          f"(FashionCLIP, unfreeze_layers={unfreeze_layers})")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    tf = _train_tf(size)
    labels = np.asarray(train_labels)

    model.train()
    step = 0
    for ep in range(epochs):
        for bidx in pk_batches(labels, P, K, steps_per_epoch, seed + ep):
            imgs = torch.stack([tf(_load(train_paths[i])) for i in bidx]).to(device)
            out = _image_features(model, imgs)
            z = F.normalize(out, dim=-1)
            loss = supcon_loss(z, labels[bidx], temp)
            opt.zero_grad(); loss.backward(); opt.step()
            if log_every and step % log_every == 0:
                print(f"  epoch {ep} | step {step:4d} | loss {loss.item():.4f}")
            step += 1
    model.eval()
    return model


@torch.no_grad()
def encode_fashion(model, paths, batch_size=64, size=224, device=None, desc="fashion-ft"):
    """fine-tune된 FashionCLIP으로 임베딩 추출 (L2 정규화)."""
    device = device or next(model.parameters()).device
    tf = _eval_tf(size)
    vecs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        imgs = torch.stack([tf(_load(p)) for p in paths[i:i + batch_size]]).to(device)
        out = _image_features(model, imgs)
        vecs.append(F.normalize(out, dim=-1).cpu().numpy())
    return np.concatenate(vecs).astype("float32")
