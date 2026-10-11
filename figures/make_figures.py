"""Figures for the quantization-recovery paper. Palette validated (dataviz).

CLEAN/ET/IT/EP counts are computed here from phase3_multi_results.json using
the same same-prompt clean-solvable matching as same_prompt_matched_analysis.py,
instead of being hardcoded -- so this figure always reflects whichever run
produced that JSON (fixes a reviewer-flagged staleness risk: a rerun used to
leave this figure showing the old numbers)."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY, INK, MUT = "#9a9994", "#0b0b0b", "#52514e"

def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)

ARMS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]   # excludes Q2_K, as in the paper's Fig. 3
bits   = [8, 6, 5, 4, 3]
labels = ARMS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
with (DATA_DIR / "phase3_multi_results.json").open() as f:
    R = json.load(f)

# same-prompt clean-solvable matching, identical to same_prompt_matched_analysis.py
CLEAN, ET, IT, EP = [], [], [], []
for arm in ARMS:
    episodes = R[arm]["episodes"]
    clean_eps = [e for e in episodes if e["cond"] == "CLEAN"]
    clean_solved = {e["task_id"] for e in clean_eps if e.get("success")}
    CLEAN.append((len(clean_solved), len(clean_eps)))
    for cond, bucket in (("ET", ET), ("IT", IT), ("EP", EP)):
        cell = [e for e in episodes if e["cond"] == cond and e["task_id"] in clean_solved]
        bucket.append((sum(bool(e.get("success")) for e in cell), len(cell)))

plt.rcParams.update({"font.size":11,"axes.edgecolor":"#d9d8d3",
    "axes.linewidth":0.8,"xtick.color":MUT,"ytick.color":MUT,
    "axes.labelcolor":INK,"text.color":INK})

# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(figsize=(6.9, 4.6))
series = [("Clean (matched set)", CLEAN, GREY, "--", 1.6, "o"),
          ("Transient fault (ET)", ET, BLUE, "-", 2.6, "o"),
          ("Corrupted data (IT)", IT, AQUA, "-", 1.8, "s"),
          ("Dead tool: abstain (EP)", EP, ORANGE, "-", 1.8, "^")]
for name, data, col, ls, lw, mk in series:
    ys  = [k/n for k,n in data]
    los = [wilson(k,n)[0] for k,n in data]
    his = [wilson(k,n)[1] for k,n in data]
    ax.fill_between(bits, los, his, color=col, alpha=0.12, linewidth=0)
    ax.plot(bits, ys, ls, color=col, linewidth=lw, marker=mk, markersize=7,
            markerfacecolor=col, markeredgecolor="white", markeredgewidth=1.2,
            zorder=3)
handles = [plt.Line2D([],[], color=c, ls=l, lw=w, marker=m, markersize=7,
           markerfacecolor=c, markeredgecolor="white", label=n)
           for n,_,c,l,w,m in series]
ax.legend(handles=handles, frameon=False, ncol=2, fontsize=9.5,
          loc="lower left", bbox_to_anchor=(0.0, 1.01), columnspacing=1.6,
          handlelength=2.2)
ax.annotate("recovery cliff", xy=(4.55, 0.63), xytext=(4.15, 0.24),
            fontsize=9.5, color=MUT, ha="center",
            arrowprops=dict(arrowstyle="->", color=MUT, lw=1))
ax.set_xlabel("Nominal bits per weight (GGUF level)")
ax.set_ylabel("Success rate")
ax.set_xticks(bits); ax.set_xticklabels([f"{b}\n{l}" for b,l in zip(bits,labels)],
                                        fontsize=8.5)
ax.set_ylim(0, 1.04); ax.set_xlim(8.45, 2.55)
ax.grid(axis="y", color="#ececea", linewidth=0.7, zorder=0)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig1.pdf"); fig.savefig("fig1.png", dpi=200)

# ---------------------------------------------------------------- Figure 2
fig, ax = plt.subplots(figsize=(6.6, 3.6))
x = range(len(ARMS)); w = 0.38
cy = [k/n for k,n in CLEAN]; ey = [k/n for k,n in ET]
b1 = ax.bar([i-w/2 for i in x], cy, w*0.94, color=GREY, label="Clean", zorder=3)
b2 = ax.bar([i+w/2 for i in x], ey, w*0.94, color=BLUE, label="Transient fault (ET)", zorder=3)
for i,(c,e) in enumerate(zip(cy,ey)):
    gap = round((c-e)*100)
    ax.annotate(f"−{gap} pts", (i, max(c,e)+0.03), ha="center",
                fontsize=9.5, color=ORANGE if gap>25 else MUT,
                fontweight="bold" if gap>25 else "normal")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.08)
ax.grid(axis="y", color="#ececea", linewidth=0.7, zorder=0)
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
          fontsize=9.5)
fig.tight_layout()
fig.savefig("fig2.pdf"); fig.savefig("fig2.png", dpi=200)
print("figures written from data/phase3_multi_results.json")
for name, data in [("CLEAN", CLEAN), ("ET", ET), ("IT", IT), ("EP", EP)]:
    print(f"  {name}: " + "  ".join(f"{a}:{k}/{n}" for a,(k,n) in zip(ARMS, data)))
