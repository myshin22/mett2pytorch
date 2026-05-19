"""Batched opres.exp port (grid-batched).

R 등가 (METT_update.R `opres.exp`):
    opres.exp <- function(t.n, t.a, mu_true, lam, n.interim, rate, FUP, nsim) { ... }

설계: 한 호출에 대해 `(n_grid × lam_grid)` cell을 텐서 한 번에 평가.
    - arrival/event를 한 번만 생성, 모든 (n2, λ) cell이 공유 — Common Random
      Numbers (CRN). R은 cell마다 새 시뮬레이션을 돌리므로 cell 간 RNG가 독립이지만
      여기는 correlated. 각 cell의 (phat, earlystop, …) 추정은 unbiased.
    - Stage 1 KM은 n2 무관 → 한 번 계산 후 lam broadcast로 재사용
    - Stage 2 KM은 (n_count, nsim, n_max) padded 텐서로 batch (padding은
      [n_valid:] 위치에 t_event=+inf, t_ind=0, at_risk clamp(min=1))

R quirk (Issue 7):
    nobs        = n.interim + 1; nobs[end] = nmax       # R 1-indexed [n1+1, n2]
    tobs[1]     = arrival[n1+1]                          # stage 1 cutoff
    tobs[2]     = arrival[n2] + FUP                      # stage 2 cutoff
    → Python 0-indexed: tobs1 = arrival[:, n1], tobs2 = arrival[:, n2-1] + FUP

분기 (Issue 5):
    out1 == 21 : stage 1 early stop  (phihat_1 <= lam)
    out1 == 1  : stage 2 reject H0   (phihat_2 > lam, not early stop)
    out1 == 2  : stage 2 accept H0
    phat       = mean(out1 == 1)
    earlystop  = mean(out1 == 21)

Reviewer 지적 #2 (RNG stream): H0/H1을 같은 generator로 stream advance시켜야
    R의 set.seed-once 패턴과 호환. 호출자 (mett2e)가 generator를 공유해서 전달.
Reviewer 지적 #5 (rand=0): `_exp_sample`에서 clamp_min(tiny)로 -log(0) 방어.
"""
from __future__ import annotations

import numpy as np
import torch

from km import batched_km_median, batched_km_median_padded


def _exp_sample(shape, rate, *, device, dtype, generator):
    """Exp(rate) 샘플. rand=0 방어로 -log(0)=inf 방지."""
    U = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    U = U.clamp_min(torch.finfo(dtype).tiny)
    return -torch.log(U) / rate


def batched_opres_exp(
    t_n: float,
    t_a: float,
    mu_true: float,
    lam_grid,            # iterable[float]
    n1: int,
    n_grid,              # iterable[int], 모두 > n1
    rate: float,
    FUP: float,
    nsim: int,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    generator: torch.Generator | None = None,
) -> dict:
    """한 n1에 대해 `(n_grid × lam_grid)` cell을 한 번에 평가.

    단일 cell 평가는 `lam_grid=[lam]`, `n_grid=[n2]`로 호출하면 됨 (validation.py
    `compare_opres_exp`가 이런 식으로 사용).

    Args:
        t_n, t_a:   signature 호환용 (내부 계산엔 미사용)
        mu_true:    데이터 생성용 median, event time ~ Exp(rate=log(2)/mu_true)
        lam_grid:   λ threshold 후보들
        n1:         stage 1 sample size (모든 cell 공통)
        n_grid:     stage 2 sample size 후보들 (각각 > n1)
        rate:       환자 도착 rate, wait time ~ Exp(rate)
        FUP:        마지막 stage 추가 follow-up 기간
        nsim:       simulation 개수
        device, dtype, generator: 텐서 옵션

    Returns:
        dict (shape `(|n_grid|, |lam_grid|)` numpy arrays):
            phat, earlystop, mpts, mtrial
    """
    n_grid_t = torch.tensor(list(n_grid), dtype=torch.long, device=device)
    lam_grid_t = torch.tensor(list(lam_grid), dtype=dtype, device=device)
    n_count = int(n_grid_t.numel())
    lam_count = int(lam_grid_t.numel())
    n_max = int(n_grid_t.max().item())
    assert int(n_grid_t.min().item()) > n1, "all n_grid > n1 required"

    # ── RNG 한 번 ─────────────────────────────────────────────────────
    wait_t = _exp_sample((nsim, n_max), rate, device=device, dtype=dtype, generator=generator)
    event_t = _exp_sample((nsim, n_max), np.log(2.0) / mu_true,
                          device=device, dtype=dtype, generator=generator)
    arrival = torch.cumsum(wait_t, dim=-1)

    # ── Stage 1: n2 무관 (n1 고정) ───────────────────────────────────
    tobs1 = arrival[:, n1]                              # (nsim,)
    arr1 = arrival[:, :n1]
    evt1 = event_t[:, :n1]
    is_event_1 = (arr1 + evt1) <= tobs1.unsqueeze(-1)
    t_event_1 = torch.where(is_event_1, evt1, tobs1.unsqueeze(-1) - arr1)
    t_ind_1 = is_event_1.to(dtype)
    phihat_1 = batched_km_median(t_event_1, t_ind_1)    # (nsim,)
    early_stop = phihat_1.unsqueeze(-1) <= lam_grid_t.unsqueeze(0)  # (nsim, lam_count)

    # ── Stage 2: (n_count, nsim, n_max) padded ───────────────────────
    tobs2 = arrival.index_select(-1, n_grid_t - 1).T + FUP   # (n_count, nsim)

    pos = torch.arange(n_max, device=device).view(1, 1, n_max)
    valid = pos < n_grid_t.view(n_count, 1, 1)               # (n_count, 1, n_max)

    arr_b = arrival.unsqueeze(0)                              # (1, nsim, n_max)
    evt_b = event_t.unsqueeze(0)
    tobs2_b = tobs2.unsqueeze(-1)                             # (n_count, nsim, 1)

    is_event_2 = ((arr_b + evt_b) <= tobs2_b) & valid
    t_event_2 = torch.where(is_event_2, evt_b.expand(n_count, -1, -1),
                            tobs2_b - arr_b)
    INF = torch.tensor(float("inf"), device=device, dtype=dtype)
    t_event_2 = torch.where(valid, t_event_2, INF)            # 패딩 위치 +inf
    t_ind_2 = is_event_2.to(dtype)                            # 패딩 위치 자동 0

    B = n_count * nsim
    n_valid_flat = n_grid_t.view(n_count, 1).expand(n_count, nsim).reshape(B)
    phihat_2 = batched_km_median_padded(
        t_event_2.reshape(B, n_max),
        t_ind_2.reshape(B, n_max),
        n_valid_flat,
    ).reshape(n_count, nsim)

    # ── 집계 ────────────────────────────────────────────────────────
    reject_2 = phihat_2.unsqueeze(-1) > lam_grid_t.view(1, 1, lam_count)
    early_stop_b = early_stop.unsqueeze(0)                    # (1, nsim, lam_count)
    out_one = (~early_stop_b) & reject_2                      # (n_count, nsim, lam_count)

    phat_arr = out_one.to(dtype).mean(dim=1)                  # (n_count, lam_count)
    earlystop_vec = early_stop.to(dtype).mean(dim=0)          # (lam_count,)
    earlystop_arr = earlystop_vec.unsqueeze(0).expand(n_count, lam_count).contiguous()

    n1_t = torch.tensor(float(n1), device=device, dtype=dtype)
    n2_t = n_grid_t.to(dtype).view(n_count, 1, 1)
    pts = torch.where(early_stop_b, n1_t, n2_t)
    mpts_arr = pts.mean(dim=1)

    tobs1_b = tobs1.view(1, nsim, 1)
    tobs2_b3 = tobs2.unsqueeze(-1)
    ttrial = torch.where(early_stop_b, tobs1_b, tobs2_b3)
    mtrial_arr = ttrial.mean(dim=1)

    return {
        "phat": phat_arr.cpu().numpy(),
        "earlystop": earlystop_arr.cpu().numpy(),
        "mpts": mpts_arr.cpu().numpy(),
        "mtrial": mtrial_arr.cpu().numpy(),
    }
