# METT — R → PyTorch 
R 패키지 [`METT`](https://sites.google.com/view/yeonheepark/research) (Park 2021; Park & Chen 2023,
원본은 Yeonhee Park 교수님 사이트에서 Google Drive로 배포)의 핵심 `METT2E`
(exponential survival) 함수를 PyTorch로 변환합니다.
**Median Event Time Test** 기반 단일군 phase II 임상시험 2단계 설계 (Park 2021,
Park & Chen 2023, PLOS ONE) 의 최적 설계 탐색을 가속합니다.

## Highlights

- **알고리즘 동등성**: R 원본을 ground-truth oracle로 두고 검증
  (KM median 0 차이, MC 비교 2σ 이내, cell-level α̂/β̂ 일치)
- **속도**:
  - i9-14900K CPU: R 대비 약 **36,000× 평균 가속** (Park & Chen Table 2 nsim=1000)
  - RTX 5090 CUDA fp32: R 대비 약 **290,000× 평균 가속**
- **재현성**: 단일 `environment.yml` 한 파일로 conda env 한 번에 구성

## Quickstart

```bash
git clone https://github.com/myshin22/METT2Pytorch.git

# conda env 구성
conda env create -f METT/environment.yml
conda activate mett

# 노트북 실행
conda run -n mett --no-capture-output jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    METT/python/01_validation.ipynb

# 또는 파이썬 스크립트
conda run -n mett --no-capture-output python METT/python/validation.py

# Figures 재생성
conda run -n mett --no-capture-output python METT/python/generate_figures.py
```

## 디렉토리 구조

```
.
├── README.md
├── LICENSE                # GPL-2
└── METT/                  # 기존 R 패키지 구조 그대로 + python/ 추가
    ├── DESCRIPTION
    ├── NAMESPACE
    ├── R/METT_update.R    # R 원본
    ├── man/               # R 도움말
    ├── python/            # PyTorch 포팅 (이번 작업)
    │   ├── 01_validation.ipynb   # 5단계 라이브 검증 노트북
    │   ├── km.py                 # batched Kaplan-Meier median
    │   ├── opres.py              # grid-batched opres.exp (CRN)
    │   ├── mett2e.py             # mett2e + mett2e_cell_by_cell
    │   ├── oracles.py            # R(rpy2) / lifelines KM 래퍼
    │   ├── validation.py         # 검증 헬퍼
    │   └── generate_figures.py   # figure 생성
    ├── figures/           # PNG 산출물
    └── environment.yml    # conda env 정의 (Python + R + 의존성)
```

## 환경

- Python 3.12, R 4.5.3, PyTorch 2.12 (+ CUDA 13), rpy2 3.6.7, lifelines 0.30
- 측정 하드웨어: Intel i9-14900K (32-thread) / 128GB / NVIDIA RTX 5090

## 참조

- Park, Y.* (2021). Optimal Two-Stage Design of Single arm Phase II Clinical Trials based on Median Event Time Test. PLoS One. 16(2): e0246448
- Park, Y.* and Chen, Y.† (2023). Sample Size Determination and Evaluation for Two-stage Adaptive Designs of Single-arm Clinical Trials based on Median Event Time Test. Contemporary Clinical Trials Communications. 

상세 검증 절차·결과·설계 결정은 [`METT/python/01_validation.ipynb`](METT/python/01_validation.ipynb) 의 5단계 라이브 검증 노트북 참조.

## License

GPL-2 (원 R 패키지 `METT` 의 라이선스 계승).
원 R 패키지 저자: Yeonhee Park &lt;ypark56@wisc.edu&gt;.
