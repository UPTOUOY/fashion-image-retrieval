"""
경량 fine-tune: 얼린 백본(DINOv2) 임베딩 위에 projection head를 metric learning으로 학습.

왜 이 방식인가 (무료 Colab 제약에 맞춘 설계 결정)
- 백본 전체를 학습하려면 매 세션 원본 이미지 + 무거운 backprop이 필요 → 세션 끊김 리스크 큼.
- 대신 train 임베딩을 한 번 뽑아두고, 그 위 작은 head만 supervised-contrastive로 학습하면
  몇 분 만에 수렴하고 임베딩만 있으면 재현 가능 → "학습으로 검색 품질이 오르나"를 안전하게 검증.
- 도메인(패션 상품)에 맞게 같은 상품은 가깝게, 다른 상품은 멀게 임베딩 공간을 재배치한다.

핵심: Supervised Contrastive Loss + P개 상품 × K장 배치 샘플링(같은 상품이 배치에 여러 장 있어야 positive가 생김).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjHead(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x) + x, dim=-1)   # residual: 원 임베딩을 크게 안 망가뜨림


def pk_batches(labels, P, K, steps, seed=0):
    """상품(label) P개를 뽑고 각 상품에서 K장씩 → 배치 인덱스 생성기."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    by = {l: np.where(labels == l)[0] for l in np.unique(labels)}
    usable = [l for l, idx in by.items() if len(idx) >= 2]
    for _ in range(steps):
        chosen = rng.choice(usable, size=min(P, len(usable)), replace=False)
        batch = []
        for l in chosen:
            idx = by[l]
            batch.extend(rng.choice(idx, size=min(K, len(idx)),
                                    replace=len(idx) < K).tolist())
        yield np.array(batch)


def supcon_loss(z, labels, temp=0.1):
    """Supervised Contrastive Loss. z: (B, D) 정규화됨, labels: (B,) 문자열/정수 무관."""
    labels = np.asarray(labels)
    same = torch.as_tensor(labels[:, None] == labels[None, :], device=z.device)  # numpy로 비교 후 텐서화
    sim = z @ z.t() / temp
    B = z.size(0)
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, -1e9)                      # 자기 자신 제외
    pos = same & ~self_mask
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # log-softmax
    pos_cnt = pos.sum(1).clamp(min=1)
    loss = -(logp * pos).sum(1) / pos_cnt                  # positive들의 평균 log-prob
    valid = pos.sum(1) > 0                                 # positive 없는 앵커 제외
    return loss[valid].mean()


def train_head(train_emb, train_labels, dim, P=16, K=4, steps=400,
               lr=1e-3, temp=0.1, device=None, seed=0, log_every=100):
    """train 임베딩으로 projection head 학습 후 반환."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head = ProjHead(dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    emb = torch.as_tensor(train_emb, dtype=torch.float32, device=device)
    head.train()
    for step, bidx in enumerate(pk_batches(train_labels, P, K, steps, seed)):
        z = head(emb[bidx])
        loss = supcon_loss(z, np.asarray(train_labels)[bidx], temp)
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f}")
    head.eval()
    return head


@torch.no_grad()
def apply_head(head, emb, device=None):
    """학습된 head를 임베딩에 적용해 새 임베딩 반환 (L2 정규화됨)."""
    device = device or next(head.parameters()).device
    x = torch.as_tensor(emb, dtype=torch.float32, device=device)
    return head(x).cpu().numpy().astype("float32")
