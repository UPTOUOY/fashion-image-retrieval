"""
DeepFashion In-Shop 데이터 로딩 + 서브샘플.

In-Shop 폴더 구조 (공식/캐글 미러 공통):
  <root>/
    Anno/list_eval_partition.txt   ← image_name  item_id  evaluation_status
    Img/img/....jpg                ← 실제 이미지

list_eval_partition.txt 형식:
  <총개수>
  image_name item_id evaluation_status
  img/MEN/Denim/id_00000080/01_1_front.jpg   id_00000080   train
  ...
evaluation_status ∈ {train, query, gallery}
- query/gallery: 같은 item_id지만 서로 다른 이미지 → "같은 상품 찾기"
- query와 gallery의 item 집합은 train과 겹치지 않음
"""
from __future__ import annotations
import os
import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Split:
    paths: list          # 이미지 절대경로
    item_ids: list       # 같은 상품이면 같은 문자열 (예: 'id_00000080')
    categories: list     # 대략적 카테고리 (예: 'MEN/Denim') — 실패 분석용

    def __len__(self):
        return len(self.paths)


def _category_from_name(image_name: str) -> str:
    # 'img/MEN/Denim/id_00000080/01_1_front.jpg' -> 'MEN/Denim'
    parts = image_name.replace("\\", "/").split("/")
    return "/".join(parts[1:3]) if len(parts) >= 3 else "unknown"


def _find_file(root: str, filename: str) -> str:
    """root 아래 어디에 있든 filename을 찾아 절대경로 반환 (Kaggle 미러 구조 대응)."""
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"{filename} not found under {root}")


def _resolve_img_root(root: str, sample_name: str) -> str:
    """
    image_name(예: 'img/MEN/...')이 실제로 붙을 상위 폴더를 자동 탐색.
    'img' 폴더를 찾아 그 부모를 img_root로 삼고, 실제 파일 존재로 검증.
    """
    for dirpath, dirs, _ in os.walk(root):
        if os.path.basename(dirpath).lower() == "img":
            candidate = os.path.dirname(dirpath)
            if os.path.exists(os.path.join(candidate, sample_name)):
                return candidate
    # 폴백: root 자체에서 바로 붙는지 확인
    if os.path.exists(os.path.join(root, sample_name)):
        return root
    raise FileNotFoundError(
        f"이미지 폴더를 찾지 못함. sample='{sample_name}' 가 붙는 위치를 확인하세요.")


def load_inshop(root: str):
    """
    root: In-Shop 데이터 폴더.
    - 주석파일(list_eval_partition.txt)이 있으면 공식 split 사용
    - 없으면(이미지 폴더만 있는 미러) id 폴더 구조에서 직접 split 구성
    반환: dict(train=Split, query=Split, gallery=Split)
    """
    try:
        return _load_from_partition(root)
    except FileNotFoundError:
        print("주석파일 없음 → 폴더 구조에서 split 자동 구성")
        return build_splits_from_folders(root)


def _load_from_partition(root: str):
    """공식 list_eval_partition.txt 기반 로딩."""
    anno = _find_file(root, "list_eval_partition.txt")
    with open(anno, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 첫 데이터 행의 image_name으로 실제 이미지 폴더 위치를 자동 해석
    first_name = lines[2].split()[0]
    img_root = _resolve_img_root(root, first_name)

    buckets = defaultdict(lambda: ([], [], []))  # split -> (paths, ids, cats)
    for line in lines[2:]:                        # 첫 두 줄(개수/헤더) 건너뜀
        parts = line.split()
        if len(parts) < 3:
            continue
        name, item_id, status = parts[0], parts[1], parts[2]
        p, i, c = buckets[status]
        p.append(os.path.join(img_root, name))
        i.append(item_id)
        c.append(_category_from_name(name))

    return {k: Split(*v) for k, v in buckets.items()}


def subsample_items(split: Split, n_items: int, seed: int = 0) -> Split:
    """상품(item_id) 단위로 n_items개만 남긴다. 상품의 모든 이미지는 유지."""
    if n_items is None:
        return split
    by_item = defaultdict(list)
    for idx, iid in enumerate(split.item_ids):
        by_item[iid].append(idx)
    items = sorted(by_item)
    random.Random(seed).shuffle(items)
    keep_idx = sorted(i for iid in items[:n_items] for i in by_item[iid])
    return Split(
        [split.paths[i] for i in keep_idx],
        [split.item_ids[i] for i in keep_idx],
        [split.categories[i] for i in keep_idx],
    )


def cap_gallery(query: Split, gallery: Split, max_gallery: int, seed: int = 0):
    """
    gallery를 max_gallery장으로 줄이되, 정답이 사라지지 않게
    query에 등장하는 item_id의 gallery 이미지는 반드시 남기고,
    나머지(distractor)만 무작위로 채운다.
    """
    if max_gallery is None or len(gallery) <= max_gallery:
        return gallery
    q_items = set(query.item_ids)
    keep, fill = [], []
    for idx, iid in enumerate(gallery.item_ids):
        (keep if iid in q_items else fill).append(idx)
    random.Random(seed).shuffle(fill)
    take = keep + fill[: max(0, max_gallery - len(keep))]
    take = sorted(take)
    return Split(
        [gallery.paths[i] for i in take],
        [gallery.item_ids[i] for i in take],
        [gallery.categories[i] for i in take],
    )


# ---------------------------------------------------------------------------
# 폴더 구조 기반 (주석파일 없는 미러: img_highres/MEN|WOMEN/<cat>/id_XXXX/*.jpg)
# id 폴더 = 같은 상품 = 정답 그룹
# ---------------------------------------------------------------------------
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _scan_item_folders(root: str):
    """이미지가 들어있는 말단 폴더(=상품)를 모아 {key: [paths]}, {key: category} 반환."""
    items = defaultdict(list)
    cats = {}
    for dp, _, fn in os.walk(root):
        imgs = [f for f in fn if f.lower().endswith(_IMG_EXT)]
        if not imgs:
            continue
        rel = os.path.relpath(dp, root).replace("\\", "/")   # 예: img_highres/MEN/Denim/id_00000080
        key = rel                                            # 폴더경로 = 고유 상품키
        parts = rel.split("/")
        cat = "unknown"
        for i, p in enumerate(parts):                        # MEN/WOMEN + 다음 요소 = 카테고리
            if p in ("MEN", "WOMEN") and i + 1 < len(parts):
                cat = f"{p}/{parts[i + 1]}"
                break
        for f in imgs:
            items[key].append(os.path.join(dp, f))
        cats[key] = cat
    return items, cats


def build_splits_from_folders(root: str, train_ratio: float = 0.5, seed: int = 0):
    """
    id 폴더에서 train/query/gallery를 직접 구성.
    - 이미지 2장 이상인 상품만 사용 (query 1장 + gallery 최소 1장 보장)
    - 상품 단위로 train/eval 분리 → eval 상품은 첫 장=query, 나머지=gallery
    """
    items, cats = _scan_item_folders(root)
    keys = [k for k, v in items.items() if len(v) >= 2]
    random.Random(seed).shuffle(keys)
    n_train = int(len(keys) * train_ratio)
    train_keys, eval_keys = keys[:n_train], keys[n_train:]

    def make(subset, mode):
        paths, ids, cs = [], [], []
        for k in subset:
            imgs = sorted(items[k])
            sel = imgs if mode == "train" else (imgs[:1] if mode == "query" else imgs[1:])
            for p in sel:
                paths.append(p); ids.append(k); cs.append(cats[k])
        return Split(paths, ids, cs)

    return {
        "train":   make(train_keys, "train"),
        "query":   make(eval_keys, "query"),
        "gallery": make(eval_keys, "gallery"),
    }
