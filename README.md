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

> 실행 후 채워짐. `results/metrics.md`, `results/failures/` 참고.

| 실험 | Recall@1 | Recall@5 | Recall@10 | mAP@10 |
|---|---|---|---|---|
| DINOv2 (zero-shot) | – | – | – | – |
| CLIP (zero-shot) | – | – | – | – |
| + center-crop | – | – | – | – |
| + projection head fine-tune | – | – | – | – |

## 구조

```
src/
  data.py       In-Shop 파싱 + 상품단위 서브샘플 (폴더구조 자동탐색)
  embed.py      DINOv2 / CLIP / ResNet50 임베딩 (공통 인터페이스)
  metrics.py    FAISS 검색 + Recall@k / mAP@k
  failures.py   실패 케이스 갤러리 생성 + 자동 태깅
  finetune.py   얼린 백본 위 projection head를 supervised-contrastive로 학습
retrieval.ipynb  Colab 실행 노트북 (GPU)
```

## 실행

Colab(T4)에서 `retrieval.ipynb`를 열고 위에서부터 실행. 데이터는 노트북이 Kaggle에서 자동 다운로드.
로컬 재현은 `pip install -r requirements.txt` 후 동일 모듈 사용.

## 한계 · 다음

- (실행 후 관찰한 실패 유형과 남은 한계를 여기에 기록)
