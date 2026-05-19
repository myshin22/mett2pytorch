"""PyTorch batched Kaplan-Meier median estimator (R `quantile.survfit` 호환).

R 등가 로직 (METT_update.R 참고):
    fit <- survfit(Surv(t.event, t.ind) ~ 1)
    if (min(fit$surv) > 0.5) max(t.event) else summary(fit)$table[7]

Tie 처리 (Issue 1):
    R survfit은 같은 시점에 event와 censoring이 함께 있으면 event를 먼저
    처리한 뒤 censoring으로 at-risk를 줄이는 규약. 같은 row-wise cumprod도
    **tie 시 event 먼저** 정렬되면 R과 동일한 S 시퀀스를 만든다.
    구현: 2-pass stable sort — 먼저 ind 내림차순(event=1이 앞), 이후 t 오름차순(stable).
    연속 RNG에서는 측도 0이라 영향 없지만, 이산/이산성 데이터에서도 R과 일치.

Plateau-midpoint 규약 (Issue 6):
    `summary(fit)$table[7]`은 `quantile.survfit(prob=0.5)`:
        t_low  = 최소 t with S(t) <= 0.5
        if S(t_low) < 0.5 (strict):
            median = t_low
        else:  # S(t_low) == 0.5 (plateau)
            t_high = 다음 t with S(t) < 0.5
            if exists:  median = (t_low + t_high) / 2
            else:       median = t_low
    S(t)가 0.5에 도달 못하면 max(t_event)로 대체.
"""
import torch


def _sort_event_first(t_event: torch.Tensor, t_ind: torch.Tensor):
    """Tie 시 event(ind=1) 먼저 오도록 정렬 (R survfit 호환).

    2-pass stable sort: ind 내림차순 → t 오름차순 (stable).
    Returns: (sorted_t, sorted_ind)
    """
    # 1: ind 내림차순으로 (event=1이 censor=0보다 앞)
    _, idx1 = torch.sort(-t_ind.to(t_event.dtype), dim=-1, stable=True)
    t_mid = t_event.gather(-1, idx1)
    ind_mid = t_ind.gather(-1, idx1)
    # 2: t 오름차순 (stable — 동률 시 event-first 유지)
    sorted_t, idx2 = torch.sort(t_mid, dim=-1, stable=True)
    sorted_ind = ind_mid.gather(-1, idx2)
    return sorted_t, sorted_ind


def batched_km_median(
    t_event: torch.Tensor,
    t_ind: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Args:
        t_event: (B, n) or (n,) float — 관측 시간
        t_ind:   (B, n) or (n,) {0, 1} — 1=event, 0=censored
        eps:     plateau 판정 tolerance (S == 0.5 vs S < 0.5 구분)
    Returns:
        median:  (B,) or scalar — R quantile.survfit 호환 median
    """
    squeeze = t_event.dim() == 1
    if squeeze:
        t_event = t_event.unsqueeze(0)
        t_ind = t_ind.unsqueeze(0)

    B, n = t_event.shape
    sorted_t, sorted_ind = _sort_event_first(t_event, t_ind.to(t_event.dtype))
    at_risk = torch.arange(
        n, 0, -1, device=t_event.device, dtype=t_event.dtype
    ).expand(B, -1)
    step = 1.0 - sorted_ind / at_risk
    S = torch.cumprod(step, dim=-1)

    # has_median: min(S) <= 0.5 (R: min(fit$surv) > 0.5 가 NA 조건)
    has_median = S.min(dim=-1).values <= 0.5 + eps

    # t_low: 처음으로 S <= 0.5 (plateau 포함)
    below = S <= 0.5 + eps
    first_idx = below.to(t_event.dtype).argmax(dim=-1)
    t_low = sorted_t.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)
    S_low = S.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)

    # plateau 판정: S_low ≈ 0.5 (strict 미만이 아님)
    on_plateau = (S_low > 0.5 - eps) & (S_low < 0.5 + eps)

    # t_high: 처음으로 S < 0.5 (strict). plateau 다음 강하 시점.
    below_strict = S < 0.5 - eps
    has_strict = below_strict.any(dim=-1)
    next_idx = below_strict.to(t_event.dtype).argmax(dim=-1)
    t_high = sorted_t.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)

    # plateau & strict drop 존재 → midpoint. 아니면 t_low.
    median = torch.where(
        on_plateau & has_strict,
        0.5 * (t_low + t_high),
        t_low,
    )
    # has_median 아니면 max(t_event)
    median = torch.where(has_median, median, sorted_t[:, -1])

    return median.squeeze(0) if squeeze else median


def batched_km_median_padded(
    t_event: torch.Tensor,
    t_ind: torch.Tensor,
    n_valid: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """KM median for batches with variable lengths (padding 처리).

    Padding 규약:
        - 유효 위치 [0, n_valid[b])에는 실제 (t_event, t_ind) 값
        - 패딩 위치 [n_valid[b], M)에는 t_event = +inf, t_ind = 0
          → 정렬 시 끝으로 가고 step=1로 S 변화 없음
        - at_risk는 유효 위치에서 (n_valid - i), 패딩 위치는 clamp(min=1)

    Args:
        t_event: (B, M) — padded
        t_ind:   (B, M) — padded with 0
        n_valid: (B,)   — 유효 위치 개수
    Returns:
        median: (B,)
    """
    B, M = t_event.shape
    dtype = t_event.dtype

    # tie 시 event 먼저 (R survfit 호환). 패딩 위치는 t=+inf, ind=0이므로
    # event-first 정렬도 자동으로 패딩은 끝으로 보냄.
    sorted_t, sorted_ind = _sort_event_first(t_event, t_ind.to(dtype))

    i_arr = torch.arange(M, device=t_event.device, dtype=dtype).unsqueeze(0).expand(B, M)
    at_risk = (n_valid.unsqueeze(-1).to(dtype) - i_arr).clamp(min=1.0)
    step = 1.0 - sorted_ind / at_risk
    S = torch.cumprod(step, dim=-1)

    has_median = S.min(dim=-1).values <= 0.5 + eps

    below = S <= 0.5 + eps
    first_idx = below.to(dtype).argmax(dim=-1)
    t_low = sorted_t.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)
    S_low = S.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)
    on_plateau = (S_low > 0.5 - eps) & (S_low < 0.5 + eps)

    below_strict = S < 0.5 - eps
    has_strict = below_strict.any(dim=-1)
    next_idx = below_strict.to(dtype).argmax(dim=-1)
    t_high = sorted_t.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)

    median = torch.where(
        on_plateau & has_strict,
        0.5 * (t_low + t_high),
        t_low,
    )

    # has_median 아니면 마지막 유효 t (R: max(t.event))
    last_valid_idx = (n_valid - 1).clamp(min=0).long()
    max_valid_t = sorted_t.gather(-1, last_valid_idx.unsqueeze(-1)).squeeze(-1)
    median = torch.where(has_median, median, max_valid_t)

    return median
