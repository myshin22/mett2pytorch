"""KM median oracles: R (via rpy2) and lifelines.

실행 환경:
    - conda env `mett` (R 4.5.3 + rpy2 conda-forge 빌드).
    - 실행 시 `conda run -n mett python ...` 또는 env 활성화 필수.
    - 시스템 R(4.4/4.6)은 rpy2 wheel과 ABI 불일치라 사용 불가.
"""
from __future__ import annotations

import os
import numpy as np

_ro = None


def _init_r():
    """Lazy rpy2 init + survival package load."""
    global _ro
    if _ro is not None:
        return _ro
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    importr("survival")
    _ro = ro
    return _ro


def r_km_median(t_event, t_ind) -> float:
    """R survfit을 직접 호출. METT_update.R의 fallback 로직 동일."""
    ro = _init_r()
    from rpy2.robjects import FloatVector, IntVector

    t_event = np.asarray(t_event, dtype=float)
    t_ind = np.asarray(t_ind, dtype=int)
    ro.globalenv["t_event"] = FloatVector(t_event)
    ro.globalenv["t_ind"] = IntVector(t_ind)
    val = ro.r(
        """
        fit <- survfit(Surv(t_event, t_ind) ~ 1)
        if (min(fit$surv) > 0.5) max(t_event) else unname(summary(fit)$table["median"])
        """
    )
    return float(val[0])


def lifelines_km_median(t_event, t_ind) -> float:
    """lifelines.KaplanMeierFitter. None/Inf 시 max(t_event) fallback (R과 일치)."""
    from lifelines import KaplanMeierFitter

    t_event = np.asarray(t_event, dtype=float)
    t_ind = np.asarray(t_ind, dtype=int)
    kmf = KaplanMeierFitter().fit(t_event, t_ind)
    m = kmf.median_survival_time_
    if m is None or not np.isfinite(m):
        m = float(t_event.max())
    return float(m)


def load_mett_r(r_path: str | None = None):
    """METT_update.R을 R 세션에 source. 이후 ro.r['opres.exp'] 등으로 호출 가능."""
    ro = _init_r()
    if r_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        r_path = os.path.normpath(os.path.join(here, "..", "R", "METT_update.R"))
    ro.r(f'source("{r_path}")')
    return ro
