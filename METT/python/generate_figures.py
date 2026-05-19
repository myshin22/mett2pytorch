"""Generate all 4 figures for REPORT.md.

Saves to figures/{speedup, mc_noise, km_curves, nsim_convergence}.png.
Run: `KMP_DUPLICATE_LIB_OK=TRUE conda run -n mett python scripts/generate_figures.py`
"""
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트
for f in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"]:
    fonts = [x.name for x in font_manager.fontManager.ttflist if f.lower() in x.name.lower()]
    if fonts:
        plt.rcParams["font.family"] = fonts[0]
        break
plt.rcParams["axes.unicode_minus"] = False


# === Viz 1: speedup bar ===
print("[1/4] speedup bar...", flush=True)
scenarios = ["3→5 (M=54)", "10→17 (M=46)"]
methods = ["R (paper)", "PyTorch CPU fp64 (i9)", "PyTorch CUDA fp32 (5090)"]
times_s = np.array([
    [13.50*3600, 1.68, 0.25],
    [11.92*3600, 0.85, 0.06],
])
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(scenarios)); w = 0.25
colors = ["#aa3300", "#3366aa", "#77aa33"]
for i, (m, c) in enumerate(zip(methods, colors)):
    ax.bar(x + (i-1)*w, times_s[:, i], w, label=m, color=c)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(scenarios)
ax.set_ylabel("Wall-clock (sec, log)")
ax.set_title("Park & Chen Table 2 — nsim=1000")
ax.legend(); ax.grid(True, axis="y", which="major", alpha=0.3)
for i in range(len(scenarios)):
    for j in range(len(methods)):
        v = times_s[i, j]
        lab = f"{v/3600:.1f}h" if v > 3600 else f"{v:.1f}s"
        ax.text(x[i]+(j-1)*w, v*1.6, lab, ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "speedup.png", dpi=80); plt.close()



# === Viz 2: MC noise histogram ===
print("[2/4] MC noise histogram (~10s)...", flush=True)
def mc_alphahats(nsim_h_, n_trials=200):
    # 수렴 cell (n₁=12, n=30, λ=7.0) — Section 4.6 nsim≥2K에서 수렴하는 cell
    out = []
    for seed in range(n_trials):
        gen = torch.Generator().manual_seed(seed)
        r = batched_opres_exp(6, 12, 6, [7.0], 12, [30], 1.0, 12, nsim_h_, generator=gen)
        out.append(float(r["phat"][0, 0]))
    return np.array(out)

a_1k = mc_alphahats(1000)
a_10k = mc_alphahats(10000)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 0.3, 40)
ax.hist(a_1k,  bins=bins, alpha=0.55, label=f"nsim=1000  (σ={a_1k.std():.4f})",  color="#3366aa")
ax.hist(a_10k, bins=bins, alpha=0.55, label=f"nsim=10000 (σ={a_10k.std():.4f})", color="#77aa33")
ax.axvspan(0.05, 0.15, alpha=0.12, color="gray", label="eps1 window (target α=0.10)")
ax.axvline(a_1k.mean(),  color="#3366aa", linestyle="--", linewidth=1)
ax.axvline(a_10k.mean(), color="#77aa33", linestyle="--", linewidth=1)
ax.set_xlabel("α̂ estimate"); ax.set_ylabel("frequency")
ax.set_title("MC noise distribution at converged cell (n₁=12, n=30, λ=7.0), 200 trials")
ax.legend(); plt.tight_layout()
plt.savefig(FIG / "mc_noise.png", dpi=80); plt.close()


# === Viz 3: KM survival curves overlay ===
print("[3/4] KM survival curves...", flush=True)
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from lifelines import KaplanMeierFitter
from km import _sort_event_first
importr("survival")

ro.r("set.seed(42); te <- rexp(20, rate=log(2)/12); ti <- rbinom(20, 1, 0.7)")
te = np.array(ro.r("te"))
ti = np.array(ro.r("ti")).astype(int)

r_times = [0.0] + list(np.array(ro.r("survfit(Surv(te, ti)~1)$time")))
r_surv  = [1.0] + list(np.array(ro.r("survfit(Surv(te, ti)~1)$surv")))

kmf = KaplanMeierFitter().fit(te, ti)
ll_times = list(kmf.survival_function_.index.values)
ll_surv  = list(kmf.survival_function_.values.flatten())

te_t = torch.tensor(te, dtype=torch.float64).unsqueeze(0)
ti_t = torch.tensor(ti, dtype=torch.float64).unsqueeze(0)
sorted_t, sorted_ind = _sort_event_first(te_t, ti_t)
n = sorted_t.shape[-1]
at_risk = torch.arange(n, 0, -1, dtype=torch.float64)
step = 1.0 - sorted_ind / at_risk
S = torch.cumprod(step, dim=-1).squeeze(0).tolist()
py_times = [0.0] + sorted_t.squeeze(0).tolist()
py_surv  = [1.0] + S

fig, ax = plt.subplots(figsize=(9, 5))
ax.step(r_times,  r_surv,  where="post", label="R survfit",  linewidth=4, alpha=0.6, color="#aa3300")
ax.step(ll_times, ll_surv, where="post", label="lifelines",  linewidth=2, linestyle="--", color="#3366aa")
ax.step(py_times, py_surv, where="post", label="PyTorch",    linewidth=1.5, linestyle=":", color="black")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, linewidth=0.8)
ax.set_xlabel("time"); ax.set_ylabel("S(t)")
ax.set_title("KM survival curves — three oracles overlay (seed=42, n=20)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "km_curves.png", dpi=80); plt.close()


# === Viz 4: nsim convergence ===
print("[4/4] nsim convergence (~15s)...", flush=True)
from mett2e import mett2e
import time

base = dict(alpha=0.10, beta=0.20, M=30, t_n=6.0, t_a=12.0,
            rate=1.0, FUP=12.0,
            nincm=1, lamincm=0.5, eps1=0.05, eps2=0.10,
            n1init=8, n1last=15, seed=840130,
            device="cuda", dtype=torch.float32)

# warmup GPU
_ = torch.zeros(1, device="cuda") + 1; torch.cuda.synchronize()

nsim_list = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
results = []
for ns in nsim_list:
    out = mett2e(nsim=ns, **base)
    results.append({"nsim": ns, "out": out})

nsims = [r["nsim"] for r in results]
ens = [r["out"]["EN"] for r in results]
n1s = [r["out"]["n1"] for r in results]
ns_  = [r["out"]["n"] for r in results]
alphas = [r["out"]["alphahat"] for r in results]
betas = [r["out"]["betahat"] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
ax.plot(nsims, ens, "o-", linewidth=2, markersize=9, color="#aa3300")
ax.set_xscale("log"); ax.set_xlabel("nsim (log)"); ax.set_ylabel("EN")
ax.set_title("Expected Sample Size (EN) vs nsim")
# 수렴 EN 표시 (nsim>=2,000 평균)
en_conv = float(np.mean([e for n, e in zip(nsims, ens) if n >= 2000]))
ax.axhline(en_conv, color="gray", linestyle="--", alpha=0.5,
           label=f"converged EN ≈ {en_conv:.2f} (nsim≥2K mean)")
ax.grid(True, alpha=0.3); ax.legend()
for x, y, n1, n in zip(nsims, ens, n1s, ns_):
    ax.annotate(f"({n1},{n})", (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8)

ax = axes[1]
ax.plot(nsims, alphas, "o-", label=r"$\hat\alpha$", color="#3366aa", linewidth=2, markersize=8)
ax.plot(nsims, betas,  "s-", label=r"$\hat\beta$",  color="#77aa33", linewidth=2, markersize=8)
ax.axhline(0.10, color="#3366aa", linestyle="--", alpha=0.6, label=r"target $\alpha$=0.10")
ax.axhline(0.20, color="#77aa33", linestyle="--", alpha=0.6, label=r"target $\beta$=0.20")
ax.set_xscale("log"); ax.set_xlabel("nsim (log)"); ax.set_ylabel(r"$\hat\alpha,\hat\beta$")
ax.set_title("Type I / II error estimates")
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

ax = axes[2]
from collections import defaultdict

coord_groups = defaultdict(list)
for idx, coord in enumerate(zip(n1s, ns_)):
    coord_groups[coord].append(idx)

plot_n1 = np.array(n1s, dtype=float)
plot_n = np.array(ns_, dtype=float)
for (n1, n), idxs in coord_groups.items():
    if len(idxs) == 1:
        continue
    spread = np.linspace(-0.22, 0.22, len(idxs))
    for off, idx in zip(spread, idxs):
        plot_n1[idx] += off
        plot_n[idx] += 0.10 * np.sign(off) if off != 0 else 0.0

ax.plot(n1s, ns_, "-", linewidth=1.5, color="#7733aa", alpha=0.35, zorder=1)
for i in range(len(n1s) - 1):
    if n1s[i] == n1s[i + 1] and ns_[i] == ns_[i + 1]:
        continue
    ax.annotate("", xy=(n1s[i + 1], ns_[i + 1]), xytext=(n1s[i], ns_[i]),
                arrowprops=dict(arrowstyle="->", color="#7733aa", alpha=0.35, linewidth=1.2),
                zorder=1)

colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(nsims)))
ax.scatter(plot_n1, plot_n, s=95, c=colors, edgecolor="white", linewidth=1.0, zorder=3)
for idx, (ns, x, y) in enumerate(zip(nsims, plot_n1, plot_n)):
    dy = 8 if idx % 2 == 0 else -12
    label = f"{ns//1000}k" if ns < 100000 else "100k"
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, dy), fontsize=8)

ax.text(0.03, 0.04, "Repeated cells are jittered slightly for visibility",
        transform=ax.transAxes, fontsize=8, color="#555555")
ax.set_xlabel(r"$n_1$"); ax.set_ylabel("n")
ax.set_title(r"Selected cell $(n_1, n)$ trajectory")
ax.grid(True, alpha=0.3)
ax.set_xlim(min(plot_n1)-1, max(plot_n1)+2); ax.set_ylim(min(plot_n)-2, max(plot_n)+3)

plt.tight_layout()
plt.savefig(FIG / "nsim_convergence.png", dpi=80); plt.close()

# Save data
(FIG / "nsim_sweep.json").write_text(json.dumps(results, indent=2, default=float))

print(f"\nAll 5 figures saved to {FIG}")
