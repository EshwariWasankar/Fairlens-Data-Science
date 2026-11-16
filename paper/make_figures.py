"""Generate the paper's figures from data/verified_benchmark_results.json and paper/paper_stats.json."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fairness.frontier import pareto_frontier

J = json.load(open(ROOT / "data/verified_benchmark_results.json"))
S = json.load(open(ROOT / "paper/paper_stats.json"))
OUT = ROOT / "paper/figures"; OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.1, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"dp": "#1f4e9c", "eo": "#8e44ad", "base": "#e08e0b", "rw": "#1b8a5a", "to": "#c0392b", "front": "#555555"}
NAMES = {"german_credit": "German Credit", "compas": "COMPAS", "adult_income": "Adult Income"}

# ---------- Fig. 2: fairness-accuracy frontier (3 panels) ----------
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15))
for ax, dk in zip(axes, ["german_credit", "compas", "adult_income"]):
    sw = J["eg_sweep_results"][dk]; ex = J["experiments"][dk]; sel = J["selected_epsilons"][dk]
    pts = []
    for cons, col, mk, lab in [("demographic_parity", C["dp"], "o", "EG (DP constraint)"), ("equalized_odds", C["eo"], "s", "EG (EO constraint)")]:
        d = pd.DataFrame(sw[cons])
        ax.scatter(d["dp_diff_raw"], d["accuracy_raw"], s=16, c=col, marker=mk, label=lab, zorder=3, alpha=0.85, edgecolors="white", linewidths=0.4)
        pts += list(zip(d["dp_diff_raw"], d["accuracy_raw"]))
    others = [("Baseline", ex["baseline_fairness"], ex["baseline_pred_metrics"], C["base"], "D"),
              ("Reweighing", ex["mitigations"]["Reweighing"]["metrics"], ex["mitigations"]["Reweighing"]["pred_metrics"], C["rw"], "^"),
              ("Threshold Opt.", ex["mitigations"]["Threshold Optimizer"]["metrics"], ex["mitigations"]["Threshold Optimizer"]["pred_metrics"], C["to"], "v")]
    for lab, fm, pm, col, mk in others:
        x, y = fm["demographic_parity_diff"]["val"], pm["accuracy"]
        ax.scatter([x], [y], s=34, c=col, marker=mk, label=lab, zorder=4, edgecolors="black", linewidths=0.4)
        pts.append((x, y))
    fr = pareto_frontier(pd.DataFrame(pts, columns=["dp_diff", "accuracy"]))
    ax.step(fr["dp_diff"], fr["accuracy"], where="post", color=C["front"], ls=":", lw=1.0, label="Pareto frontier", zorder=2)
    s = next(r for r in sw["demographic_parity"] if r["eps"] == sel)
    ax.scatter([s["dp_diff_raw"]], [s["accuracy_raw"]], s=110, facecolors="none", edgecolors="black", linewidths=1.0, zorder=5,
               label=r"Selected $\varepsilon$ (validation)")
    ax.set_title(f"{NAMES[dk]} ($\\varepsilon^*={sel}$)")
    ax.set_xlabel("Demographic parity difference")
    xmax = max(p[0] for p in pts); ax.set_xlim(-0.05 * xmax, 1.1 * xmax)
axes[0].set_ylabel("Test accuracy")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.08), handletextpad=0.3, columnspacing=0.9)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "frontier.pdf"); plt.close(fig)

# ---------- Fig. 3: calibration by group (COMPAS, Adult) ----------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
pal = ["#1f4e9c", "#e08e0b", "#1b8a5a", "#c0392b"]
for ax, dk, ttl in [(axes[0], "compas", "COMPAS (race)"), (axes[1], "adult_income", "Adult (sex)")]:
    cur = pd.DataFrame(S[dk]["calibration"]["curves"]); summ = pd.DataFrame(S[dk]["calibration"]["summary"])
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=0.7)
    for i, g in enumerate(summ["group"][:3]):
        c = cur[cur["group"] == g]
        e = summ.loc[summ["group"] == g, "ece"].item()
        ax.plot(c["mean_predicted"], c["observed_rate"], marker="o", ms=2.2, c=pal[i], label=f"{g} (ECE {e:.3f})")
    ax.set_title(ttl); ax.set_xlabel("Mean predicted P(y=1)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=5.4, handlelength=1.1, borderpad=0.2)
axes[0].set_ylabel("Observed rate")
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT / "calibration.pdf"); plt.close(fig)

# ---------- Fig. 4: intersectional selection rates (COMPAS race x sex, Adult sex x age) ----------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.85), gridspec_kw={"width_ratios": [1.45, 1]})
for ax, dk, ttl in [(axes[0], "compas", "COMPAS: race × sex"), (axes[1], "adult_income", "Adult: sex × age")]:
    t = pd.DataFrame(S[dk]["intersectional"]["table"], columns=["group", "count", "sel", "tpr", "fpr", "incl"])
    t = t[t["incl"]].copy()
    t[["a", "b"]] = t["group"].str.split(" × ", n=1, expand=True)
    A = [a for a in t.groupby("a")["count"].sum().sort_values(ascending=False).index]
    B = sorted(t["b"].unique())
    w = 0.8 / len(B)
    for j, b in enumerate(B):
        vals = [t[(t["a"] == a) & (t["b"] == b)]["sel"].sum() if ((t["a"] == a) & (t["b"] == b)).any() else np.nan for a in A]
        ax.bar(np.arange(len(A)) + (j - (len(B) - 1) / 2) * w, vals, w, label=b.split(" (")[0], color=pal[j], edgecolor="black", linewidth=0.3)
    short = {"African-American": "Afr.-Am.", "Caucasian": "Cauc."}
    ax.set_xticks(range(len(A))); ax.set_xticklabels([short.get(a, a) for a in A], fontsize=6.5)
    ax.set_title(ttl); ax.legend(frameon=False, fontsize=5.8, handlelength=0.9, loc="upper right")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Selection rate P($\\hat{y}$=1)")
fig.tight_layout(w_pad=0.5)
fig.savefig(OUT / "intersectional.pdf"); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.iterdir()))
