"""
실패 케이스 갤러리 — 이 프로젝트의 심장.

단순 Recall 숫자에서 멈추지 않고, "왜 틀렸는지"를 눈으로 보고 유형화한다.
(치수 추정에서 '높이 오차의 진짜 원인 = 납작한 물건'을 케이스 분석으로 규명한 방식의 재현)

제공 기능
1) find_failures : top-k 안에 정답이 없는 query 인덱스 목록
2) auto_tag      : 각 실패에 자동 힌트 태그(같은 카테고리 오검색 / 다른 카테고리 등)
3) save_grid     : query + top-k 결과를 정답/오답 테두리로 한 장 이미지에 저장
4) build_gallery : 실패 N개를 그리드로 덤프 + 유형 분포 요약
"""
from __future__ import annotations
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .metrics import relevance_matrix


def find_failures(idx, query_ids, gallery_ids, k=5):
    """top-k 안에 정답이 하나도 없는 query 인덱스."""
    rel = relevance_matrix(idx[:, :k], query_ids, gallery_ids)
    return np.where(~rel.any(axis=1))[0]


def auto_tag(qi, idx, query, gallery):
    """
    실패 하나에 대한 자동 힌트 태그(사람 검수의 출발점).
    - same_category_confusion: top-1이 같은 카테고리인데 다른 상품 → 색/형태 과의존 의심
    - cross_category         : top-1이 아예 다른 카테고리 → 배경/전역특징에 끌린 것 의심
    """
    top1 = idx[qi, 0]
    q_cat, g_cat = query.categories[qi], gallery.categories[top1]
    if q_cat == g_cat:
        return "same_category_confusion"
    return "cross_category"


def save_grid(qi, idx, sims, query, gallery, k, out_path):
    """query 1장 + top-k 결과를 한 줄로. 정답=초록, 오답=빨강 테두리."""
    fig, axes = plt.subplots(1, k + 1, figsize=(2.1 * (k + 1), 2.6))
    # query
    axes[0].imshow(Image.open(query.paths[qi]).convert("RGB"))
    axes[0].set_title("query", fontsize=9)
    axes[0].axis("off")
    for j in range(k):
        gi = idx[qi, j]
        correct = gallery.item_ids[gi] == query.item_ids[qi]
        ax = axes[j + 1]
        ax.imshow(Image.open(gallery.paths[gi]).convert("RGB"))
        ax.set_title(f"top{j+1}  {sims[qi, j]:.2f}", fontsize=8)
        ax.axis("off")
        color = "#2ecc71" if correct else "#e74c3c"
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(color); s.set_linewidth(3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_gallery(idx, sims, query, gallery, out_dir, k=5, n=60, seed=0):
    """
    실패 케이스 n개를 그리드 이미지로 저장하고, 자동 태그 분포를 반환.
    반환: dict {tag: count} — 어떤 실패 유형이 많은지 = 다음에 뭘 고칠지의 근거.
    """
    os.makedirs(out_dir, exist_ok=True)
    fails = find_failures(idx, query.item_ids, gallery.item_ids, k=k)
    rng = np.random.default_rng(seed)
    pick = rng.permutation(fails)[:n]

    tags = Counter()
    for rank, qi in enumerate(pick):
        tag = auto_tag(qi, idx, query, gallery)
        tags[tag] += 1
        save_grid(qi, idx, sims, query, gallery, k,
                  os.path.join(out_dir, f"fail_{rank:03d}_{tag}.png"))
    return dict(tags), len(fails)
