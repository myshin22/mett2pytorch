"""METT2E port (grid-batched).

R 등가 (METT_update.R `METT2E`):
    METT2E <- function(alpha, beta, M, t.n, t.a, rate, FUP, nsim,
                       nincm, lamincm, eps1, eps2, n1init, n1last, seed) { ... }

이 포트는 R의 핵심 grid loop만 구현. 자동 n1init 탐색 (R의 n1init.exp 분기)은
첫 버전에서는 생략 — `n1init`, `n1last`를 명시적으로 받아야 함.

알고리즘:
    for k in 1..len(n1):
        seed = base_seed + k
        pp = seq(n1[k]+1, M, by=nincm)        # n2 후보
        qq = seq(t.n,    t.a, by=lamincm)     # lam 후보
        out_H0 = batched_opres_exp(mu=t.n, lam_grid=qq, n_grid=pp, ...)  # n2×lam 한 번에
        out_H1 = batched_opres_exp(mu=t.a, lam_grid=qq, n_grid=pp, ...)
        ahat = out_H0.phat;  phat = 1 - out_H1.phat;  PET = out_H0.earlystop
        ff = (ahat - alpha)² + (phat - beta)²
        (i1, i2) = which.min(ff)              # R: which(==min, arr.ind=T)
        if |ahat-α|<eps1 & |phat-β|<eps2:  mark valid
    select best n1 minimizing EN among valid

R과의 차이 :
    - **R simulation design**: 각 (n2, λ) cell마다 opres.exp를 새로 호출 → cell마다
      독립 RNG 시뮬레이션 (set.seed 한 번 후 stream advance).
    - **이 구현**: 한 n1에 대해 arrival/event를 한 번 생성하고 (n2 × λ) 모든 cell에
      재사용 — Common Random Numbers (CRN) 기법.
      → 각 cell의 (ahat, phat) 추정은 unbiased (기댓값 동일).
      → cell 간 estimate가 correlated (R 독립과 다름) → 같은 nsim에서 cell 선택이 다를
        수 있음. 통계적으로는 정당한 variance-reduction 기법이지만, R bit-equivalence는
        아니라는 점 명시.
    - which.min: R column-major 동률 처리 (`_which_min_arr_ind`).
"""
from __future__ import annotations

import numpy as np
import torch

from opres import batched_opres_exp


def _which_min_arr_ind(ff: np.ndarray) -> tuple[int, int]:
    """R의 `which(ff == min(ff), arr.ind=TRUE)[1, ]`와 동일.

    R은 column-major (Fortran) 순서로 flatten하여 첫 번째 최솟값 위치를 반환.
    동률이 여러 개면 column-major 첫 번째.
    """
    flat = ff.flatten(order="F")
    k = int(np.argmin(flat))
    nrow = ff.shape[0]
    return k % nrow, k // nrow


def mett2e(
    alpha: float,
    beta: float,
    M: int,
    t_n: float,
    t_a: float,
    rate: float,
    FUP: float,
    nsim: int,
    nincm: int,
    lamincm: float,
    eps1: float,
    eps2: float,
    n1init: int,
    n1last: int | None = None,
    seed: int = 840130,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> dict:
    """R METT2E의 PyTorch 포트.

    Args (R 동일): alpha, beta, M, t_n, t_a, rate, FUP, nsim, nincm, lamincm,
                   eps1, eps2, n1init, n1last, seed

    Returns:
        dict (R `res$names`와 동일): n1, n, lambda, EN, PET0, alphahat, betahat
        valid cell이 없으면 모두 None (R: NA).
    """
    if n1last is None:
        n1last = M - 1

    n1_arr = np.arange(n1init, n1last + 1, dtype=int)
    n1l = len(n1_arr)

    n_out = np.full(n1l, np.nan)
    lam_out = np.full(n1l, np.nan)
    PETn = np.full(n1l, np.nan)
    EN = np.full(n1l, np.nan)
    alphahat = np.full(n1l, np.nan)
    betahat = np.full(n1l, np.nan)
    inderr = np.zeros(n1l, dtype=int)

    for k in range(n1l):
        n1 = int(n1_arr[k])
        s = seed + (k + 1)
        gen = torch.Generator(device=device).manual_seed(int(s))

        pp = list(range(n1 + 1, M + 1, nincm))
        qq = list(np.arange(t_n, t_a + 1e-12, lamincm))
        llp, ll = len(pp), len(qq)

        if verbose:
            print(f"  k={k+1}/{n1l}: n1={n1}, |pp|={llp}, |qq|={ll}")

        # H0 (mu_true=t_n) — (n_grid × lam_grid) 한 번에 (CRN: 모든 cell이 같은
        # arrival/event 공유). H1과 별도로 새 arrival/event를 생성.
        out_h0 = batched_opres_exp(
            t_n, t_a, t_n, qq, n1, pp,
            rate, FUP, nsim,
            device=device, dtype=dtype, generator=gen,
        )
        # H1 (mu_true=t_a) — 같은 generator stream 이어감
        out_h1 = batched_opres_exp(
            t_n, t_a, t_a, qq, n1, pp,
            rate, FUP, nsim,
            device=device, dtype=dtype, generator=gen,
        )

        ahat = out_h0["phat"]
        phat = 1.0 - out_h1["phat"]
        PET = out_h0["earlystop"]

        ff = (ahat - alpha) ** 2 + (phat - beta) ** 2
        i1, i2 = _which_min_arr_ind(ff)

        n_out[k] = pp[i1]
        lam_out[k] = qq[i2]
        PETn[k] = PET[i1, i2]
        EN[k] = n1 + (1.0 - PETn[k]) * (n_out[k] - n1)
        alphahat[k] = ahat[i1, i2]
        betahat[k] = phat[i1, i2]

        if abs(ahat[i1, i2] - alpha) < eps1 and abs(phat[i1, i2] - beta) < eps2:
            inderr[k] = 1
            if verbose:
                print(
                    f"    n1={n1}, n={int(n_out[k])}, lam={lam_out[k]:.3f}, "
                    f"alphahat={alphahat[k]:.4f}, betahat={betahat[k]:.4f}, "
                    f"EN={EN[k]:.3f}"
                )

    valid = np.where(inderr == 1)[0]
    if len(valid) == 0:
        if verbose:
            print("---------- Change values of n1init and n1last ----------")
        return {"n1": None, "n": None, "lambda": None, "EN": None,
                "PET0": None, "alphahat": None, "betahat": None}

    best = int(valid[int(np.argmin(EN[valid]))])
    return {
        "n1": int(n1_arr[best]),
        "n": int(n_out[best]),
        "lambda": float(lam_out[best]),
        "EN": float(EN[best]),
        "PET0": float(PETn[best]),
        "alphahat": float(alphahat[best]),
        "betahat": float(betahat[best]),
    }


def mett2e_cell_by_cell(
    alpha: float,
    beta: float,
    M: int,
    t_n: float,
    t_a: float,
    rate: float,
    FUP: float,
    nsim: int,
    nincm: int,
    lamincm: float,
    eps1: float,
    eps2: float,
    n1init: int,
    n1last: int | None = None,
    seed: int = 840130,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> dict:
    """R-equivalent validation oracle — cell마다 fresh RNG simulation (no CRN).

    `mett2e`의 CRN 설계가 R independent-simulation 설계와 통계적으로 등가인지
    검증하는 용도. 각 `(n2, λ)` cell마다 `batched_opres_exp([lam], n1, [n2], ...)`
    를 새로 호출 → R `opres.exp` 호출 패턴과 동일 (set.seed 한 번 후 stream advance).

    느림 — 외부 ll × llp × 2 호출. `mett2e`보다 50-80× 느림. validation 용도로만.
    """
    if n1last is None:
        n1last = M - 1

    n1_arr = np.arange(n1init, n1last + 1, dtype=int)
    n1l = len(n1_arr)

    n_out = np.full(n1l, np.nan)
    lam_out = np.full(n1l, np.nan)
    PETn = np.full(n1l, np.nan)
    EN = np.full(n1l, np.nan)
    alphahat = np.full(n1l, np.nan)
    betahat = np.full(n1l, np.nan)
    inderr = np.zeros(n1l, dtype=int)

    for k in range(n1l):
        n1 = int(n1_arr[k])
        s = seed + (k + 1)
        gen = torch.Generator(device=device).manual_seed(int(s))

        pp = list(range(n1 + 1, M + 1, nincm))
        qq = list(np.arange(t_n, t_a + 1e-12, lamincm))
        llp, ll = len(pp), len(qq)

        if verbose:
            print(f"  k={k+1}/{n1l}: n1={n1}, |pp|={llp}, |qq|={ll}")

        ahat = np.zeros((llp, ll))
        phat = np.zeros((llp, ll))
        PET = np.zeros((llp, ll))

        for i, nval in enumerate(pp):
            for t, lamphi in enumerate(qq):
                # cell마다 새 RNG 소비 (R 호환 패턴)
                out_h0 = batched_opres_exp(
                    t_n, t_a, t_n, [float(lamphi)], n1, [int(nval)],
                    rate, FUP, nsim,
                    device=device, dtype=dtype, generator=gen,
                )
                out_h1 = batched_opres_exp(
                    t_n, t_a, t_a, [float(lamphi)], n1, [int(nval)],
                    rate, FUP, nsim,
                    device=device, dtype=dtype, generator=gen,
                )
                ahat[i, t] = float(out_h0["phat"][0, 0])
                phat[i, t] = 1.0 - float(out_h1["phat"][0, 0])
                PET[i, t] = float(out_h0["earlystop"][0, 0])

        ff = (ahat - alpha) ** 2 + (phat - beta) ** 2
        i1, i2 = _which_min_arr_ind(ff)

        n_out[k] = pp[i1]
        lam_out[k] = qq[i2]
        PETn[k] = PET[i1, i2]
        EN[k] = n1 + (1.0 - PETn[k]) * (n_out[k] - n1)
        alphahat[k] = ahat[i1, i2]
        betahat[k] = phat[i1, i2]

        if abs(ahat[i1, i2] - alpha) < eps1 and abs(phat[i1, i2] - beta) < eps2:
            inderr[k] = 1

    valid = np.where(inderr == 1)[0]
    if len(valid) == 0:
        return {"n1": None, "n": None, "lambda": None, "EN": None,
                "PET0": None, "alphahat": None, "betahat": None}

    best = int(valid[int(np.argmin(EN[valid]))])
    return {
        "n1": int(n1_arr[best]),
        "n": int(n_out[best]),
        "lambda": float(lam_out[best]),
        "EN": float(EN[best]),
        "PET0": float(PETn[best]),
        "alphahat": float(alphahat[best]),
        "betahat": float(betahat[best]),
    }
