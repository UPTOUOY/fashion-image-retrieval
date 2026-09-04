"""
검색 지표: FAISS 코사인 검색 + Recall@k / mAP@k.

핵심 개념
- 임베딩을 L2 정규화한 뒤 내적(IndexFlatIP)으로 검색 = 코사인 유사도 랭킹.
- Recall@k: query 중에서 top-k 안에 "같은 상품(item_id)"이 하나라도 있는 비율.
- mAP@k: 여러 정답이 있을 때 랭킹 품질까지 반영(정답을 위에 올릴수록 높음).
"""
from __future__ import annotations
import numpy as np


def build_index(gallery_vecs: np.ndarray):
    """L2 정규화된 gallery 벡터로 코사인(내적) 인덱스를 만든다."""
    import faiss
    x = np.ascontiguousarray(gallery_vecs.astype("float32"))
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    return index


def search(gallery_vecs, query_vecs, topk=50, exclude_self=False):
    """
    query마다 gallery에서 top-k 이웃을 찾는다.
    exclude_self=True: query와 gallery가 같은 풀일 때(폴백 데이터셋) 자기 자신 제거.
                       (In-Shop은 query/gallery가 분리돼 있어 False면 됨)
    반환: sims (Nq, topk), idx (Nq, topk)  ← idx는 gallery 인덱스
    """
    import faiss
    index = build_index(gallery_vecs)
    q = np.ascontiguousarray(query_vecs.astype("float32"))
    k = topk + (1 if exclude_self else 0)
    sims, idx = index.search(q, k)
    if exclude_self:
        # 첫 열이 자기 자신(유사도 1.0)이면 제거
        sims, idx = sims[:, 1:], idx[:, 1:]
    return sims, idx


def relevance_matrix(idx, query_ids, gallery_ids) -> np.ndarray:
    """idx(Nq, k)의 각 위치가 '정답(같은 item_id)'인지 표시한 bool 행렬."""
    gids = np.asarray(gallery_ids)
    qids = np.asarray(query_ids)
    return gids[idx] == qids[:, None]


def recall_at_k(rel: np.ndarray, k: int) -> float:
    """top-k 안에 정답이 하나라도 있으면 성공. 성공한 query 비율."""
    return float(rel[:, :k].any(axis=1).mean())


def map_at_k(rel: np.ndarray, k: int, n_relevant=None) -> float:
    """
    mean Average Precision @k.
    n_relevant: query별 실제 정답 개수(있으면 정확한 정규화). 없으면 top-k 내 정답 수로 근사.
    """
    r = rel[:, :k].astype(float)
    ranks = np.arange(1, r.shape[1] + 1)
    precision_at_i = np.cumsum(r, axis=1) / ranks       # 위치별 정밀도
    ap = (precision_at_i * r).sum(axis=1)                # 정답 위치에서만 누적
    if n_relevant is None:
        denom = np.maximum(r.sum(axis=1), 1)
    else:
        denom = np.maximum(np.minimum(np.asarray(n_relevant), k), 1)
    return float((ap / denom).mean())


def evaluate(idx, query_ids, gallery_ids, ks=(1, 5, 10), n_relevant=None) -> dict:
    """Recall@k들과 mAP@max(k)를 한 번에 계산해 dict로 반환."""
    rel = relevance_matrix(idx, query_ids, gallery_ids)
    out = {f"recall@{k}": recall_at_k(rel, k) for k in ks}
    out[f"mAP@{max(ks)}"] = map_at_k(rel, max(ks), n_relevant)
    return out


def n_relevant_per_query(query_ids, gallery_ids) -> np.ndarray:
    """query별로 gallery에 존재하는 같은 item_id 이미지 수(=정답 총 개수)."""
    from collections import Counter
    c = Counter(gallery_ids)
    return np.array([c.get(q, 0) for q in query_ids])


# ---------------------------------------------------------------------------
# Re-ranking (검색 후처리, 학습 0) — mAP 개선용
# ---------------------------------------------------------------------------
def query_expansion(gallery_vecs, query_vecs, topk=10, alpha=3.0):
    """
    Average Query Expansion (αQE): 각 query를 top-k gallery 이웃으로 보강 후 재정규화.
    학습 없이 query 표현을 풍부하게 만들어 재검색 → recall·mAP 개선.
    반환: 보강된 query 벡터 (이후 search/evaluate에 그대로 사용).
    """
    sims, idx = search(gallery_vecs, query_vecs, topk=topk)
    g = gallery_vecs.astype("float32")
    new_q = query_vecs.astype("float32").copy()
    for qi in range(len(query_vecs)):
        w = np.maximum(sims[qi], 0.0) ** alpha            # 유사도^alpha 가중
        agg = (w[:, None] * g[idx[qi]]).sum(0)
        v = query_vecs[qi] + agg
        new_q[qi] = v / (np.linalg.norm(v) + 1e-8)
    return new_q.astype("float32")


def idx_from_dist(dist, topk=50):
    """거리행렬(작을수록 유사)에서 상위 topk 인덱스."""
    return np.argsort(dist, axis=1)[:, :topk].astype(np.int64)


def rerank_kreciprocal(query_vecs, gallery_vecs, k1=20, k2=6, lambda_value=0.3):
    """
    k-reciprocal encoding 재순위 (Zhong et al., 2017).
    상호 최근접(k-reciprocal) 이웃 기반 Jaccard 거리 + 원거리를 결합해 재순위.
    반환: (Nq, Ng) 최종 거리행렬 (작을수록 유사) → idx_from_dist로 랭킹.
    """
    feat = np.concatenate([query_vecs, gallery_vecs]).astype(np.float32)
    q_num, all_num = len(query_vecs), len(feat)
    # 정규화 벡터 → 유클리드² = 2-2cos
    original_dist = 2.0 - 2.0 * (feat @ feat.T)
    original_dist = np.transpose(original_dist / (np.max(original_dist, axis=0) + 1e-12))
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    for i in range(all_num):
        forward = initial_rank[i, :k1 + 1]
        backward = initial_rank[forward, :k1 + 1]
        recip = forward[np.where(backward == i)[0]]
        recip_exp = recip
        for j in recip:
            cand = initial_rank[j, :int(round(k1 / 2)) + 1]
            cand_back = initial_rank[cand, :int(round(k1 / 2)) + 1]
            cand_recip = cand[np.where(cand_back == j)[0]]
            if len(np.intersect1d(cand_recip, recip)) > 2.0 / 3 * len(cand_recip):
                recip_exp = np.append(recip_exp, cand_recip)
        recip_exp = np.unique(recip_exp)
        w = np.exp(-original_dist[i, recip_exp])
        V[i, recip_exp] = (w / np.sum(w)).astype(np.float32)

    if k2 != 1:                                            # local query expansion
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe

    orig_q = original_dist[:q_num]
    inv_idx = [np.where(V[:, i] != 0)[0] for i in range(all_num)]  # gallery별 비영 행
    jaccard = np.zeros((q_num, all_num), dtype=np.float32)
    for i in range(q_num):
        tmp_min = np.zeros(all_num, dtype=np.float32)
        nz = np.where(V[i] != 0)[0]
        for j in nz:
            tmp_min[inv_idx[j]] += np.minimum(V[i, j], V[inv_idx[j], j])
        jaccard[i] = 1.0 - tmp_min / (2.0 - tmp_min + 1e-12)

    final = jaccard * (1 - lambda_value) + orig_q * lambda_value
    return final[:, q_num:]                                # query×gallery 부분만
