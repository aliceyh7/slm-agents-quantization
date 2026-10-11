"""Generate the paper's figures from the released confirmatory JSON files.

Figure 1 summarizes the experiment and compares separate-screen,
prompt-matched, and full-pipeline estimates for two models. Figure 2
reports the exploratory Llama-only P0/P1 analysis at three precisions.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9994"
INK, MUT = "#0b0b0b", "#52514e"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parent
LLAMA = json.loads((DATA_DIR / "confirmatory_sweep_results.json").read_text())
QWEN = json.loads((DATA_DIR / "qwen_confirmatory_sweep_results.json").read_text())
PROMPTS = ["P0", "P1", "P2", "P3", "P4"]


def cells(result):
    return result["cells"]


def cell(result, arm, prompt):
    return cells(result)[f"{arm}:{prompt}:R0"]


def stable_sets(result, arms):
    return {
        (arm, prompt): set(cell(result, arm, prompt)["stable_clean_tasks"])
        for arm in arms for prompt in PROMPTS
    }


def task_rates(result, arm, prompt, task_ids):
    by_task = defaultdict(list)
    for episode in cell(result, arm, prompt)["episodes"]:
        if episode["task_id"] in task_ids:
            by_task[episode["task_id"]].append(int(bool(episode.get("success"))))
    return {task_id: sum(values) / len(values) for task_id, values in by_task.items() if values}


def boot_ci(differences, seed=0, samples=20_000):
    rng = random.Random(seed)
    values = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return values[int(.025 * samples)], values[int(.975 * samples)]


def boot_unpaired_ci(left, right, seed=0, samples=20_000):
    rng = random.Random(seed)
    values = sorted(
        sum(rng.choice(left) for _ in left) / len(left)
        - sum(rng.choice(right) for _ in right) / len(right)
        for _ in range(samples)
    )
    return values[int(.025 * samples)], values[int(.975 * samples)]


def three_targets(result):
    arms = ["Q8_0", "Q4_K_M"]
    stable = stable_sets(result, arms)
    all_tasks = set(result["config"]["task_ids"])
    targets = []
    matched_sizes = []

    own_means, own_lo, own_hi = [], [], []
    common_means, common_lo, common_hi = [], [], []
    full_means, full_lo, full_hi = [], [], []
    for prompt in PROMPTS:
        q8 = list(task_rates(result, "Q8_0", prompt, stable[("Q8_0", prompt)]).values())
        q4 = list(task_rates(result, "Q4_K_M", prompt, stable[("Q4_K_M", prompt)]).values())
        mean = sum(q8) / len(q8) - sum(q4) / len(q4)
        lo, hi = boot_unpaired_ci(q8, q4)
        own_means.append(mean); own_lo.append(mean - lo); own_hi.append(hi - mean)

        matched = sorted(stable[("Q8_0", prompt)] & stable[("Q4_K_M", prompt)])
        matched_sizes.append(len(matched))
        q8 = task_rates(result, "Q8_0", prompt, matched)
        q4 = task_rates(result, "Q4_K_M", prompt, matched)
        differences = [q8[t] - q4[t] for t in matched]
        mean = sum(differences) / len(differences)
        lo, hi = boot_ci(differences)
        common_means.append(mean); common_lo.append(mean - lo); common_hi.append(hi - mean)

        q8 = task_rates(result, "Q8_0", prompt, all_tasks)
        q4 = task_rates(result, "Q4_K_M", prompt, all_tasks)
        differences = [q8.get(t, 0.0) - q4.get(t, 0.0) for t in sorted(all_tasks)]
        mean = sum(differences) / len(differences)
        lo, hi = boot_ci(differences)
        full_means.append(mean); full_lo.append(mean - lo); full_hi.append(hi - mean)

    targets.append((own_means, [own_lo, own_hi]))
    targets.append((common_means, [common_lo, common_hi]))
    targets.append((full_means, [full_lo, full_hi]))
    return matched_sizes, targets


plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#d9d8d3",
    "axes.linewidth": 0.8,
    "xtick.color": MUT,
    "ytick.color": MUT,
    "axes.labelcolor": INK,
    "text.color": INK,
})


# ---------------------------------------------------------------- Figure 1
llama_matched_sizes, llama_targets = three_targets(LLAMA)
qwen_matched_sizes, qwen_targets = three_targets(QWEN)

fig = plt.figure(figsize=(7.05, 4.45))
flow = fig.add_axes([0.04, 0.775, 0.92, 0.205])
flow.set_xlim(0, 1); flow.set_ylim(0, 1); flow.axis("off")


def box(x, y, width, height, title, body, fc="#f6f6f3", ec="#c8c7c2"):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=0.9,
    )
    flow.add_patch(patch)
    flow.text(x + 0.015, y + height - 0.10, title, ha="left", va="top",
              fontsize=8.1, fontweight="bold")
    flow.text(x + 0.015, y + height - 0.25, body, ha="left", va="top",
              fontsize=7.2, linespacing=1.18, color=MUT)


def arrow(x1, y1, x2, y2):
    flow.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
        linewidth=0.9, color="#777671",
    ))


flow.text(0.0, 0.99, "Experiment and evaluation targets", fontsize=8.4, fontweight="bold",
          ha="left", va="top")
box(0.01, 0.10, 0.16, 0.65, "Task suite", "20 deterministic\ntool-use tasks")
box(0.22, 0.10, 0.20, 0.65, "Configurations", r"2 models $\times$ 5 prompts" "\n" r"$\times$ 2 precisions")
box(0.47, 0.10, 0.23, 0.65, "Run protocol",
    "Clean screen, then inject\none temporary tool failure")
box(0.76, 0.04, 0.23, 0.77, "Three estimates",
    "Separate task sets\nPrompt-matched tasks\nFull 20-task pipeline",
    fc="#f7f3ee", ec="#d9bcae")
arrow(0.18, 0.43, 0.21, 0.43)
arrow(0.43, 0.43, 0.46, 0.43)
arrow(0.71, 0.43, 0.75, 0.43)

grid = fig.add_gridspec(
    2, 3, left=0.085, right=0.985, bottom=0.07, top=0.715,
    wspace=0.28, hspace=0.85,
)
axes = [[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)]
rows = [
    ("Llama-3.1-8B", llama_matched_sizes, llama_targets),
    ("Qwen2.5-7B", qwen_matched_sizes, qwen_targets),
]
column_titles = ["A. Separate clean-passing sets", "B. Prompt-matched tasks", "C. Full pipeline (20 tasks)"]

for row_index, (model_name, matched_sizes, targets) in enumerate(rows):
    for column_index, (means, errors) in enumerate(targets):
        ax = axes[row_index][column_index]
        xs = list(range(len(PROMPTS)))
        colors = [BLUE if value > 1e-9 else (ORANGE if value < -1e-9 else GREY)
                  for value in means]
        ax.bar(xs, means, 0.60, color=colors, zorder=3)
        ax.errorbar(xs, means, yerr=errors, fmt="none", ecolor=INK,
                    elinewidth=0.9, capsize=2.2, zorder=4)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(xs)
        labels = [f"$P_{i}$" for i in range(5)]
        if column_index == 1:
            labels = [f"$P_{i}$\n$n{{=}}{n}$" for i, n in enumerate(matched_sizes)]
        ax.set_xticklabels(labels, fontsize=6.8 if column_index == 1 else 8)
        title = column_titles[column_index]
        ax.set_title(title, fontsize=8.2, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#ececea", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.88, 0.80)
        ax.yaxis.set_major_formatter(lambda value, position: f"{100 * value:.0f}")
        for x, value in enumerate(means):
            if value >= 0:
                offset, va = errors[1][x] + 0.03, "bottom"
            else:
                offset, va = -(errors[0][x] + 0.03), "top"
            ax.annotate(
                f"{100 * value:+.1f}" if abs(value) > 1e-9 else "0",
                (x, value + offset), ha="center", va=va, fontsize=7.0, color=MUT,
                annotation_clip=False,
            )
        if column_index == 0:
            ax.set_ylabel(f"{model_name}\n8-bit $-$ 4-bit (points)", fontsize=8.2)

fig.savefig(OUT / "figC.pdf", bbox_inches="tight")
fig.savefig(OUT / "figC.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- Figure 2
llama_arms = ["Q8_0", "Q4_K_M", "Q3_K_M"]
llama_stable = stable_sets(LLAMA, llama_arms)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                         gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
labels, means, errlo, errhi, sample_sizes = [], [], [], [], []
for arm, label in zip(llama_arms, ["Q8_0\n(8-bit)", "Q4_K_M\n(4-bit)", "Q3_K_M\n(3-bit)"]):
    common = sorted(llama_stable[(arm, "P0")] & llama_stable[(arm, "P1")])
    p0 = task_rates(LLAMA, arm, "P0", common)
    p1 = task_rates(LLAMA, arm, "P1", common)
    differences = [p0[t] - p1[t] for t in common]
    mean = sum(differences) / len(differences)
    lo, hi = boot_ci(differences)
    labels.append(label); means.append(mean)
    errlo.append(mean - lo); errhi.append(hi - mean); sample_sizes.append(len(common))
xs = range(len(llama_arms))
ax.bar(xs, means, 0.55, color=[BLUE if value > 0 else ORANGE for value in means], zorder=3)
ax.errorbar(xs, means, yerr=[errlo, errhi], fmt="none", ecolor=INK,
            elinewidth=1.0, capsize=2.8, zorder=4)
ax.axhline(0, color=INK, lw=0.8)
for x, (value, sample_size) in enumerate(zip(means, sample_sizes)):
    offset = errhi[x] + 0.04 if value >= 0 else -(errlo[x] + 0.09)
    ax.annotate(f"{100 * value:+.0f}", (x, value + offset), ha="center", fontsize=8.4, color=MUT)
    ax.annotate(f"n={sample_size}", (x, -0.52), ha="center", fontsize=7.4, color=MUT)
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("$P_0 - P_1$ recovery")
ax.set_ylim(-0.58, 0.92)
ax.set_title("A. Llama $P_0-P_1$ difference by precision", fontsize=9, loc="left")
ax.grid(axis="y", color="#ececea", linewidth=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
episodes = cell(LLAMA, "Q3_K_M", "P1")["episodes"]
failed = [episode for episode in episodes if not episode.get("success")]
report_failure = sum(
    "report_failure" in (episode.get("last_content") or "").lower()
    for episode in failed
)
categories = ["report_failure\non a transient fault", "other"]
values = [report_failure, len(failed) - report_failure]
ax.barh(range(len(categories)), values, 0.5, color=[ORANGE, GREY], zorder=3)
for y, value in enumerate(values):
    ax.annotate(f"{value}/{len(failed)}", (value + 0.5, y), va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(categories))); ax.set_yticklabels(categories, fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0, 36)
ax.set_xlabel("Failed Q3_K_M/$P_1$ episodes")
ax.set_title("B. Terminal action of every failure", fontsize=9, loc="left")
ax.grid(axis="x", color="#ececea", linewidth=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "figD.pdf")
fig.savefig(OUT / "figD.png", dpi=200)
plt.close(fig)

print("Figure 1 prompt-matched set sizes:",
      f"Llama n={llama_matched_sizes}", f"Qwen n={qwen_matched_sizes}")
for name, _, targets in rows:
    print(name)
    for label, (means, _) in zip(["separate", "matched", "full"], targets):
        print(" ", label, [f"{100 * value:+.1f}" for value in means])
print(f"Figure 2 failure mode: {report_failure}/{len(failed)}")
