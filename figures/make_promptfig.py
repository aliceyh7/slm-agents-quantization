"""Figure: the deficit appears under one prompt and moves around under four others.

Reads the sweep's own aggregate counts directly from promptvar2_results.json
instead of hardcoding numbers, so this figure always reflects whichever run
produced that file (fixes a reviewer-flagged staleness risk: a rerun used to
leave this figure showing the old numbers)."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9994"
INK, MUT = "#0b0b0b", "#52514e"

def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h, c+h

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
with (DATA_DIR / "promptvar2_results.json").open() as f:
    R = json.load(f)

ARMS = ["Q8_0", "Q4_K_M", "Q3_K_M"]
PROMPT_LABELS = {
    "P0": "P0\noriginal", "P1": "P1\nparaphrase", "P2": "P2\nparaphrase",
    "P3": "P3\nnumbered rules", "P4": "P4\nterse",
}

# own matched-reliable-set counts, straight from the sweep's own aggregate
# fields (et_success/et_n) for each level x prompt cell -- no hardcoding.
data = {}
for p, label in PROMPT_LABELS.items():
    cells = []
    for lvl in ARMS:
        cell = R[f"{lvl}:{p}"]
        cells.append((cell["et_success"], cell["et_n"]))
    data[label] = cells

LEVELS = [("Q8_0 (8-bit)", GREY), ("Q4_K_M (4-bit)", BLUE), ("Q3_K_M (3-bit)", ORANGE)]

plt.rcParams.update({"font.size":11,"axes.edgecolor":"#d9d8d3",
    "axes.linewidth":0.8,"xtick.color":MUT,"ytick.color":MUT,
    "axes.labelcolor":INK,"text.color":INK})
fig, ax = plt.subplots(figsize=(7.0, 3.6))
w = 0.26
for i,(pname, cells) in enumerate(data.items()):
    for j,((k,n),(lab,col)) in enumerate(zip(cells, LEVELS)):
        x = i + (j-1)*w
        y = k/n; lo,hi = wilson(k,n)
        ax.bar(x, y, w*0.92, color=col, zorder=3, label=lab if i==0 else None)
        ax.errorbar(x, y, yerr=[[max(0,y-lo)],[max(0,hi-y)]], fmt="none",
                    ecolor=INK, elinewidth=1.0, capsize=2.4, zorder=4)
        ax.annotate(f"{y:.0%}", (x, hi+0.03), ha="center", fontsize=7.6,
                    color=MUT, rotation=0)
ax.set_xticks(range(len(data))); ax.set_xticklabels(list(data.keys()), fontsize=8.8)
ax.set_ylabel("Transient-fault recovery")
ax.set_ylim(0, 1.16)
ax.grid(axis="y", color="#ececea", linewidth=0.7, zorder=0)
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower left",
          bbox_to_anchor=(0.0, 1.02), fontsize=9.5)
fig.tight_layout()
fig.savefig("figP.pdf"); fig.savefig("figP.png", dpi=200)
print("ok - figP.pdf regenerated from data/promptvar2_results.json")
for p, cells in data.items():
    print(" ", p.replace(chr(10)," "), [(k,n,f"{k/n:.0%}") for k,n in cells])
