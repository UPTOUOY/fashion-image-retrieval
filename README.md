# Fashion Image Retrieval — Zero-shot 임베딩에서 실패를 파고들어 개선하기

이미지 한 장으로 **"같은 상품"을 찾는** 이미지 임베딩 검색 파이프라인.
DeepFashion In-Shop 벤치마크에서 zero-shot 임베딩을 baseline으로 두고, **검색 실패 사례를 유형별로 규명한 뒤 그 원인을 겨냥해 개선**했다.

> 이 프로젝트의 초점은 완성도가 아니라 **"왜 안 맞는지 파고들어 데이터·모델·전처리 중 무엇을 고칠지 판단하는 과정"**이다.

---

## 문제 정의

- **입력**: 상품 이미지 1장 (query)
- **출력**: gallery(후보 이미지 풀)에서 유사도 상위 k개
- **정답**: query와 **같은 상품(item_id)** 이미지를 top-k 안에 올렸는가
- **데이터**: DeepFashion In-Shop Clothes Retrieval (같은 상품을 다른 포즈/각도로 촬영 → 인스턴스 검색)
- **지표**: Recall@1 / @5 / @10, mAP@10

## 파이프라인

```
이미지 → [임베딩 모델] → L2 정규화 → [FAISS 코사인 검색] → top-k
                                              │
                                    [실패 사례 유형화]
                                              │
                              데이터 · 모델 · 전처리 개선 실험
```

## 접근 (2일 실험 로그)

| 단계 | 내용 |
|---|---|
| **Baseline** | DINOv2 / CLIP zero-shot 임베딩으로 Recall@k 측정 |
| **실패 분석** | top-k에 정답 없는 query를 그리드로 덤프 → 유형 태깅 (같은 카테고리 오검색 / 다른 카테고리 등) |
| **개선 실험** | 백본 비교(DINOv2·CLIP·ResNet50), 전처리(center-crop), projection head fine-tune |
| **검증** | 동일 프로토콜로 재평가 → 채택/기각을 수치로 기록 |

## 결과

평가 세트: query 500 상품 / gallery 8,000장 (train·eval 상품 분리 → leakage 없음).

| 방법 | 학습 | Recall@1 | Recall@5 | Recall@10 | mAP@10 | |
|---|---|---|---|---|---|---|
| DINOv2 zero-shot | 없음 | 0.492 | 0.664 | 0.726 | 0.200 | baseline |
| + center-crop | 전처리 | 0.494 | 0.662 | 0.726 | 0.200 | ❌ 기각 |
| + head-only fine-tune | 얼린 백본 + head | 0.696 | 0.886 | 0.930 | 0.408 | ✅ |
| + backbone fine-tune (DINOv2-S) | 백본 직접 학습 | 0.852 | 0.946 | 0.962 | 0.553 | ✅ |
| + backbone fine-tune (DINOv2-B) | 더 큰 백본 | 0.874 | 0.954 | 0.968 | 0.600 | ✅ |
| **+ 강화 (24k장·6ep·6블록)** | | **0.884** | **0.962** | **0.980** | **0.621** | ✅ 최종 |

**Recall@1 0.492 → 0.884 (+80%), mAP@10 0.20 → 0.621 (3.1×), top-5 검색 실패 168 → 19.**

### 핵심 발견 (실패 분석)
top-5에 정답이 없는 168개 실패를 눈으로 유형화한 결과, 모델이 **"옷"이 아니라 "장면 전체(사람·포즈·구도)"를 매칭**하는 것이 주 원인이었다.
- 결정적 증거: "의자에 앉은 사람" query → top-1~5가 전부 *의자에 앉은 사람* (옷은 제각각). 포즈/구도로 뭉침.
- 전신샷이라 옷이 화면 일부에 불과 → 배경·사람이 임베딩을 지배.
- **center-crop(전처리)로는 안 풀림** → 같은 상품의 다른 포즈를 가깝게 배우는 **학습(supervised contrastive)**이 근본 처방임을 사다리로 증명.

## 구조

```
src/
  data.py       In-Shop 파싱 + 상품단위 서브샘플 (폴더구조 자동탐색)
  embed.py      DINOv2 / CLIP / ResNet50 임베딩 (공통 인터페이스)
  metrics.py    FAISS 검색 + Recall@k / mAP@k
  failures.py   실패 케이스 갤러리 생성 + 자동 태깅
  finetune.py           얼린 백본 위 projection head를 supervised-contrastive로 학습 (head-only)
  finetune_backbone.py  DINOv2 백본 직접 fine-tune (supervised-contrastive + 증강, model_name 선택)
retrieval.ipynb  Colab 실행 노트북 (GPU)
```

## 실행

Colab(T4)에서 `retrieval.ipynb`를 열고 위에서부터 실행. 데이터는 노트북이 Kaggle에서 자동 다운로드.
로컬 재현은 `pip install -r requirements.txt` 후 동일 모듈 사용.

## 한계 · 다음

- **남은 실패는 주로 `same_category`** (같은 카테고리·비슷한 색/실루엣의 다른 상품). fine-grained 디테일(로고·패턴) 구분이 약함 → 높은 입력 해상도(224→256/336)나 부위별 crop으로 개선 여지.
- **개선폭은 포화 근처** (0.874 → 0.884). 다음 후보: k-reciprocal re-ranking(학습 0, mAP↑), 더 큰 백본(DINOv2-L), object detection으로 의류 영역만 crop.
- 평가는 In-Shop 서브셋(gallery 8k) 기준. 전체 gallery(52k)로 키우면 절대 수치는 낮아지되 경향은 유지될 것.
