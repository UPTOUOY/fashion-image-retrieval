"""
"모델이 이미지의 어디를 보는가" 시각화 — fine-tune 전/후 비교.

방법: DINOv2의 마지막 layer에서 각 patch 토큰과 CLS 토큰의 코사인 유사도를 구하면,
      전역 표현(CLS)에 크게 기여하는 영역이 밝게 나온다. (attention 가중치 대신
      patch-CLS 정렬도를 쓰므로 attn_implementation에 의존하지 않아 안정적)

스토리 증명: fine-tune 전에는 배경·사람에, 후에는 옷에 집중이 옮겨간 것을 히트맵으로 보여준다.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _tf(size=224):
    return transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


@torch.no_grad()
def focus_map(model, pil, size=224, device=None):
    """patch–CLS 코사인 유사도 히트맵 (0~1 정규화된 grid×grid)."""
    device = device or next(model.parameters()).device
    x = _tf(size)(pil.convert("RGB")).unsqueeze(0).to(device)
    tokens = model(pixel_values=x).last_hidden_state[0]     # (1+P, D): [0]=CLS, [1:]=patch
    cls, patches = tokens[0], tokens[1:]
    sims = F.cosine_similarity(patches, cls.unsqueeze(0), dim=-1)   # (P,)
    g = int(round(patches.shape[0] ** 0.5))
    m = sims[:g * g].reshape(g, g).float().cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def _upsample(m, size):
    return np.array(Image.fromarray((m * 255).astype("uint8"))
                    .resize((size, size), Image.BILINEAR)) / 255.0


def compare_figure(model_before, model_after, paths, out_path, size=224,
                   titles=("원본", "fine-tune 전", "fine-tune 후")):
    """
    각 이미지에 대해 [원본 | 학습 전 집중 | 학습 후 집중] 3열 그리드를 저장.
    model_before: 사전학습 DINOv2, model_after: fine-tune된 DINOv2 (같은 아키텍처).
    """
    n = len(paths)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3.1 * n))
    if n == 1:
        axes = axes[None, :]
    for r, p in enumerate(paths):
        pil = Image.open(p).convert("RGB")
        base = pil.resize((size, size))
        hb = _upsample(focus_map(model_before, pil, size), size)
        ha = _upsample(focus_map(model_after, pil, size), size)
        for c, (im, heat) in enumerate([(base, None), (base, hb), (base, ha)]):
            ax = axes[r, c]
            ax.imshow(im)
            if heat is not None:
                ax.imshow(heat, cmap="jet", alpha=0.5)
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def load_dinov2_eager(name="facebook/dinov2-small", state_dict=None, device=None):
    """
    attention 비교용 DINOv2 로더. state_dict를 주면 fine-tune 가중치를 얹는다
    (finetune_backbone에서 학습한 model.state_dict()를 그대로 사용).
    """
    from transformers import AutoModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = AutoModel.from_pretrained(name).to(device).eval()
    if state_dict is not None:
        m.load_state_dict(state_dict, strict=False)
    return m
