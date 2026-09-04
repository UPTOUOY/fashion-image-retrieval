"""
제대로 된 fine-tune: DINOv2 백본을 직접 학습해 "같은 상품 = 가깝게" 표현을 재학습.

head-only([[finetune]])와의 차이
- head-only: 얼린 백본 위 작은 head만 → 얼린 특징의 한계에 갇힘.
- 여기: 백본 파라미터를 직접 업데이트 + 데이터 증강 → 포즈/구도 편향을 근본적으로 완화.

핵심 설계 (무료 Colab T4 고려)
- Supervised Contrastive Loss + P상품×K장 배치 (같은 상품의 다른 포즈가 한 배치에 → positive).
- 증강(RandomResizedCrop/flip/jitter)으로 포즈·구도에 덜 민감하게.
- unfreeze_blocks로 뒤쪽 N개 블록만 풀어 속도·안정성 조절(None이면 전체 학습).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .finetune import supcon_loss, pk_batches

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


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
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _load(path):
    return Image.open(path).convert("RGB")


def _build_dinov2(device, model_name="facebook/dinov2-small"):
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_name).to(device)


def finetune_backbone(train_paths, train_labels, epochs=3, P=8, K=4,
                      steps_per_epoch=200, lr=2e-5, temp=0.1, size=224,
                      unfreeze_blocks=4, model_name="facebook/dinov2-small",
                      device=None, seed=0, log_every=50):
    """
    DINOv2 백본을 supervised-contrastive로 fine-tune.
    model_name: 'facebook/dinov2-small'(384d) 또는 'facebook/dinov2-base'(768d, 더 강함).
    unfreeze_blocks: 뒤에서 몇 개 transformer 블록을 학습할지 (None=전체).
    반환: 학습된 model (encode_backbone로 임베딩 추출).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_dinov2(device, model_name)

    if unfreeze_blocks is not None:                       # 뒤쪽 블록만 학습
        for p in model.parameters():
            p.requires_grad = False
        for blk in model.encoder.layer[-unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.layernorm.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"학습 파라미터: {n_trainable/1e6:.2f}M "
          f"(unfreeze_blocks={unfreeze_blocks})")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    tf = _train_tf(size)
    labels = np.asarray(train_labels)

    model.train()
    step = 0
    for ep in range(epochs):
        for bidx in pk_batches(labels, P, K, steps_per_epoch, seed + ep):
            imgs = torch.stack([tf(_load(train_paths[i])) for i in bidx]).to(device)
            out = model(pixel_values=imgs).last_hidden_state[:, 0]
            z = F.normalize(out, dim=-1)
            loss = supcon_loss(z, labels[bidx], temp)
            opt.zero_grad(); loss.backward(); opt.step()
            if log_every and step % log_every == 0:
                print(f"  epoch {ep} | step {step:4d} | loss {loss.item():.4f}")
            step += 1
    model.eval()
    return model


@torch.no_grad()
def encode_backbone(model, paths, batch_size=64, size=224, device=None, desc="encode"):
    """학습된 백본으로 임베딩 추출 (L2 정규화). 평가는 결정적 전처리 사용."""
    device = device or next(model.parameters()).device
    tf = _eval_tf(size)
    vecs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        imgs = torch.stack([tf(_load(p)) for p in paths[i:i + batch_size]]).to(device)
        out = model(pixel_values=imgs).last_hidden_state[:, 0]
        vecs.append(F.normalize(out, dim=-1).cpu().numpy())
    return np.concatenate(vecs).astype("float32")
