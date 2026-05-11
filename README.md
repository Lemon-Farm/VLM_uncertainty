# VLM Uncertainty

BLIP-family vision-language models에서 dataset, metric, uncertainty method를 바꿔가며 신뢰도와 불확실성을 측정하기 위한 연구용 프로젝트입니다.

## Directory Tree

```text
VLM_uncertainty/
├── configs/                    # 실험 설정: dataset/model/metric/uncertainty 조합
│   ├── default.yaml             # 기본 실행 설정
│   ├── data/                    # 데이터셋별 설정
│   ├── model/                   # BLIP/BLIP-2 등 모델별 설정
│   ├── metric/                  # accuracy, ECE, AUROC 등 metric 설정
│   └── uncertainty/             # entropy, margin, MC dropout 등 방법 설정
├── data/                        # 로컬 데이터 위치, git에는 실제 데이터 제외
│   ├── raw/                     # 원본 데이터
│   ├── interim/                 # 전처리 중간 산출물
│   └── processed/               # 학습/평가에 바로 쓰는 데이터
├── notebooks/                   # 빠른 분석, 시각화, 프로토타입
├── outputs/                     # Hydra/실험 실행 결과
├── reports/                     # 표, 그래프, 최종 분석 자료
│   └── figures/                 # 논문/보고서용 이미지
├── scripts/                     # CLI 진입점 래퍼
├── src/
│   └── vlm_uncertainty/
│       ├── cli/                 # train/evaluate/uncertainty 실행 명령
│       ├── data/                # dataset loading, preprocessing, collator
│       ├── evaluation/          # evaluation loop, prediction export
│       ├── metrics/             # task metric과 calibration metric
│       ├── models/              # BLIP 계열 wrapper, checkpoint loading
│       ├── uncertainty/         # uncertainty score 계산 로직
│       └── utils/               # config, seed, logging 등 공통 유틸
└── tests/                       # 단위 테스트와 smoke test
```

## Where To Work

- `configs/`: 실험 조합을 바꾸는 곳입니다. 코드 수정 없이 `data=coco model=blip2 uncertainty=entropy metric=default`처럼 조합해 실행하는 것을 목표로 합니다.
- `src/vlm_uncertainty/data/`: COCO, VQA, Flickr30k 같은 데이터셋 로딩과 BLIP 입력 형식 변환을 둡니다.
- `src/vlm_uncertainty/models/`: Hugging Face BLIP/BLIP-2 모델을 감싸는 wrapper를 둡니다. 모델별 output format 차이는 여기서 흡수합니다.
- `src/vlm_uncertainty/uncertainty/`: predictive entropy, max probability, margin, mutual information, MC dropout, ensemble score 등을 구현합니다.
- `src/vlm_uncertainty/metrics/`: accuracy, BLEU/CIDEr 같은 task metric과 ECE, NLL, Brier score, AUROC 같은 uncertainty/calibration metric을 둡니다.
- `src/vlm_uncertainty/evaluation/`: dataset을 돌며 prediction, uncertainty score, metric을 계산하고 `outputs/`에 저장합니다.
- `notebooks/`: 새 아이디어 검증, 결과 플롯, 에러 케이스 확인용입니다. 재사용 로직은 `src/`로 옮깁니다.
- `reports/`: 실험 결과 표와 figure를 정리합니다.
- `tests/`: 새 dataset/model/metric이 들어올 때 최소한 shape, key, deterministic behavior를 확인합니다.

## Quick Start

```bash
pip install -e ".[dev]"
python scripts/evaluate.py
python scripts/compute_uncertainty.py
```

아직 실제 BLIP 로딩과 데이터셋 연결은 placeholder 상태입니다. 다음 단계에서 사용할 task를 정하면 해당 dataset adapter와 model wrapper부터 채우면 됩니다.
