"""KM median validation utilities.

Primary: R survfit vs PyTorch batched — 동일한 규약 (R quantile.survfit, plateau midpoint).
         tolerance 1e-4 내 일치해야 함 (assert).

Secondary: lifelines — 다른 규약 (`S < 0.5` strict 또는 `S ≤ 0.5`).
           plateau 케이스에서 R/PyTorch와 legitimately disagree. 참고용으로만 표시.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from km import batched_km_median
from opres import batched_opres_exp
from mett2e import mett2e as torch_mett2e
from oracles import r_km_median, lifelines_km_median, _init_r, load_mett_r


ATOL = 1e-6
RTOL = 1e-4


@dataclass
class TripleResult:
    r: float
    lifelines: float
    torch: float

    def primary_diff(self) -> float:
        """R vs PyTorch — primary 검증 차이."""
        return abs(self.r - self.torch)

    def lifelines_diff(self) -> float:
        """R vs lifelines — secondary 정보."""
        return abs(self.r - self.lifelines)

    def primary_ok(self, atol: float = ATOL, rtol: float = RTOL) -> bool:
        try:
            np.testing.assert_allclose(self.torch, self.r, atol=atol, rtol=rtol)
            return True
        except AssertionError:
            return False


def km_median_triple(t_event, t_ind) -> TripleResult:
    """동일 입력으로 세 oracle 호출. Primary 비교는 R vs PyTorch."""
    te = np.asarray(t_event, dtype=float)
    ti = np.asarray(t_ind, dtype=int)
    return TripleResult(
        r=r_km_median(te, ti),
        lifelines=lifelines_km_median(te, ti),
        torch=batched_km_median(
            torch.tensor(te, dtype=torch.float64),
            torch.tensor(ti, dtype=torch.float64),
        ).item(),
    )


def generate_case_in_r(seed: int, n: int = 20, mu: float = 12.0, p_event: float = 0.7):
    """R에서 결정론적으로 데이터 생성, Python으로 가져옴.

    R 난수를 직접 쓰는 이유: 다른 oracle도 결국 R survfit이 reference라,
    R RNG로 만든 데이터를 공유하는 것이 가장 직접적인 결정론적 비교.
    """
    ro = _init_r()
    ro.r(
        f"""
        set.seed({seed})
        t.event <- rexp({n}, rate = log(2)/{mu})
        t.ind   <- rbinom({n}, 1, {p_event})
        """
    )
    return np.array(ro.r("t.event")), np.array(ro.r("t.ind"))


def run_seed_sweep(
    seeds: list[int] | range,
    n: int = 20,
    mu: float = 12.0,
    p_event: float = 0.7,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> dict:
    """여러 seed에 대해 R vs PyTorch (primary) 비교, lifelines는 정보로 수집."""
    results = []
    primary_failures = []
    lifelines_diffs = []
    max_primary = 0.0

    for s in seeds:
        te, ti = generate_case_in_r(s, n=n, mu=mu, p_event=p_event)
        res = km_median_triple(te, ti)
        results.append((s, res))
        max_primary = max(max_primary, res.primary_diff())
        lifelines_diffs.append(res.lifelines_diff())
        if not res.primary_ok(atol=atol, rtol=rtol):
            primary_failures.append(
                {
                    "seed": s,
                    "t_event": te.tolist(),
                    "t_ind": ti.tolist(),
                    "r": res.r,
                    "torch": res.torch,
                    "diff": res.primary_diff(),
                }
            )

    return {
        "n_total": len(results),
        "primary_pass": len(results) - len(primary_failures),
        "primary_fail": len(primary_failures),
        "max_primary_diff": max_primary,
        "lifelines_diff_mean": float(np.mean(lifelines_diffs)),
        "lifelines_diff_max": float(np.max(lifelines_diffs)),
        "lifelines_disagreements": int(np.sum(np.array(lifelines_diffs) > atol)),
        "primary_failures": primary_failures,
        "atol": atol,
        "rtol": rtol,
    }


def robustness_sweep_opres(cases: list[dict], nsim: int = 5000) -> list[dict]:
    """여러 파라미터 조합에서 R vs PyTorch opres.exp 비교.

    각 케이스가 phat/earlystop이 MC 2σ 이내인지 검사.
    """
    results = []
    for c in cases:
        full = {**c, "nsim": nsim}
        cmp = compare_opres_exp(**full)
        ok_phat = abs(cmp["diff"]["phat"]) <= cmp["se2"]["phat"] + 1e-9
        ok_es = abs(cmp["diff"]["earlystop"]) <= cmp["se2"]["earlystop"] + 1e-9
        results.append(
            {
                "params": c,
                "r": cmp["r"],
                "torch": cmp["torch"],
                "diff": cmp["diff"],
                "se2": cmp["se2"],
                "ok_phat": ok_phat,
                "ok_earlystop": ok_es,
                "ok": ok_phat and ok_es,
            }
        )
    return results


def compare_mett2e(
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
    n1last: int,
    seed: int = 840130,
    verbose: bool = False,
) -> dict:
    """R METT2E vs PyTorch mett2e 결과 비교 (grid 선택 + 수치 일치)."""
    ro = _init_r()
    load_mett_r()
    import time

    if verbose:
        print("Running R METT2E...")
    t0 = time.time()
    r_res = ro.r["METT2E"](
        float(alpha), float(beta), int(M), float(t_n), float(t_a),
        float(rate), float(FUP), int(nsim),
        int(nincm), float(lamincm), float(eps1), float(eps2),
        int(n1init), int(n1last), int(seed),
    )
    t_r = time.time() - t0

    def _maybe(name):
        v = r_res.rx2(name)
        if v is None or len(v) == 0:
            return None
        # R's NA detection via rpy2 (handles NA_integer_=INT_MIN, NA_real_=NaN, etc.)
        try:
            if bool(ro.r("is.na")(v)[0]):
                return None
        except Exception:
            pass
        try:
            f = float(v[0])
            return None if np.isnan(f) else f
        except Exception:
            return None

    r_dict = {k: _maybe(k) for k in ("n1", "n", "lambda", "EN", "PET0", "alphahat", "betahat")}

    if verbose:
        print(f"Running PyTorch mett2e... (R took {t_r:.1f}s)")
    t0 = time.time()
    t_dict = torch_mett2e(
        alpha, beta, M, t_n, t_a, rate, FUP, nsim,
        nincm, lamincm, eps1, eps2, n1init, n1last, seed,
        verbose=verbose,
    )
    t_torch = time.time() - t0

    return {"r": r_dict, "torch": t_dict, "time_r_s": t_r, "time_torch_s": t_torch}


def compare_opres_exp(
    t_n: float,
    t_a: float,
    mu_true: float,
    lam: float,
    n_interim,
    rate: float,
    FUP: float,
    nsim: int,
    r_seed: int = 840130,
    torch_seed: int = 840130,
) -> dict:
    """R opres.exp vs PyTorch batched_opres_exp 통계적 일치 비교.

    RNG는 다르니 bit-exact 불가. 큰 nsim에서 MC 오차 (~1/√nsim) 내 일치 확인.
    phat/earlystop은 proportion이라 2σ 경계 계산 가능; mpts/mtrial은 empirical diff만.
    """
    ro = _init_r()
    load_mett_r()
    ro.r(f"set.seed({r_seed})")
    r_res = ro.r["opres.exp"](
        float(t_n), float(t_a), float(mu_true), float(lam),
        ro.IntVector(list(n_interim)),
        float(rate), float(FUP), int(nsim),
    )
    r_out = {k: float(r_res.rx2(k)[0]) for k in ["phat", "earlystop", "mpts", "mtrial"]}

    # `batched_opres_exp`는 grid 버전. 단일 cell은 [lam], [n2]로 호출.
    gen = torch.Generator().manual_seed(torch_seed)
    n1, n2 = int(n_interim[0]), int(n_interim[1])
    out_grid = batched_opres_exp(
        t_n, t_a, mu_true, [lam], n1, [n2], rate, FUP, nsim, generator=gen
    )
    t_out = {k: float(out_grid[k][0, 0]) for k in ["phat", "earlystop", "mpts", "mtrial"]}

    def se_prop(p: float, n: int) -> float:
        return float(np.sqrt(max(p * (1 - p), 0.0) / n))

    return {
        "r": r_out,
        "torch": t_out,
        "diff": {k: t_out[k] - r_out[k] for k in r_out},
        "se2": {
            "phat": 2 * se_prop(r_out["phat"], nsim),
            "earlystop": 2 * se_prop(r_out["earlystop"], nsim),
        },
    }


if __name__ == "__main__":
    print("=== Step 1: single seed (42, n=20) ===")
    te, ti = generate_case_in_r(42)
    r = km_median_triple(te, ti)
    print(f"  R         : {r.r:.8f}")
    print(f"  PyTorch   : {r.torch:.8f}  (primary diff: {r.primary_diff():.2e})")
    print(f"  lifelines : {r.lifelines:.8f}  (vs R: {r.lifelines_diff():.2e})")
    print(f"  primary ok: {r.primary_ok()}")

    print("\n=== Step 1 extended: 100 seeds (n=20) ===")
    summary = run_seed_sweep(range(100))
    print(f"  primary (R vs PyTorch): {summary['primary_pass']}/{summary['n_total']} pass")
    print(f"     max diff: {summary['max_primary_diff']:.2e}")
    print(f"  lifelines (R vs lifelines, 다른 규약으로 정보용):")
    print(f"     disagreements > atol: {summary['lifelines_disagreements']}/{summary['n_total']}")
    print(f"     mean abs diff: {summary['lifelines_diff_mean']:.4f}")
    print(f"     max abs diff:  {summary['lifelines_diff_max']:.4f}")
    if summary["primary_failures"]:
        print(f"\n  PRIMARY FAILURES ({len(summary['primary_failures'])}):")
        for f in summary["primary_failures"][:5]:
            print(f"    seed={f['seed']}  diff={f['diff']:.2e}  "
                  f"R={f['r']:.6f} torch={f['torch']:.6f}")
    else:
        print(f"\n  ✓ R and PyTorch agree on all {summary['n_total']} seeds within "
              f"atol={ATOL}, rtol={RTOL}")

    print("\n=== Step 2: opres.exp R vs PyTorch (MC) ===")
    # METT 스타일 파라미터: H0 시뮬 (mu_true = t_n) → phat = α 추정
    params = dict(
        t_n=12.0, t_a=18.0, mu_true=12.0, lam=15.0,
        n_interim=[10, 30], rate=2.0 / 3.0, FUP=12.0, nsim=10000,
    )
    print(f"  params: {params}")
    cmp = compare_opres_exp(**params)
    print(f"\n  {'metric':<10} {'R':>12} {'PyTorch':>12} {'diff':>12} {'2σ':>10}")
    for k in ["phat", "earlystop", "mpts", "mtrial"]:
        se = cmp["se2"].get(k, "")
        se_s = f"{se:.4f}" if isinstance(se, float) else ""
        print(f"  {k:<10} {cmp['r'][k]:>12.4f} {cmp['torch'][k]:>12.4f} "
              f"{cmp['diff'][k]:>+12.4f} {se_s:>10}")
