"""Figure 1: what a task is, and what each injected fault does to it."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUT, LINE = "#0b0b0b", "#52514e", "#c9c8c2"

fig, ax = plt.subplots(figsize=(6.9, 3.2))
ax.set_xlim(-1.5, 101.5); ax.set_ylim(-1, 50); ax.axis("off")

def step(x, y, label, w=22.0, h=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#f6f6f4", edgecolor=LINE, linewidth=1.1))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center",
            fontsize=7.0, color=INK, family="monospace")

def arr(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
        mutation_scale=10, color=MUT, linewidth=1.2))

# title line
ax.text(0, 47.0, "A task: for order O241, apply the customer's tier discount "
        "to the subtotal", fontsize=8.2, color=INK, fontweight="bold", va="center")
ax.text(0, 42.4, "one correct answer, known in advance", fontsize=7.0,
        color=MUT, va="center", style="italic")

# chain
ys = 32.5
xs = [0, 26, 52, 78]
for x, l in zip(xs, ["get_order", "get_customer", "get_discount", "submit_answer"]):
    step(x, ys, l)
for x in xs[:-1]:
    arr(x+22.3, x+25.7, ys+3.3)

# fault marker under step 2
ax.add_patch(FancyArrowPatch((37.0, 26.6), (37.0, 31.9), arrowstyle="-|>",
    mutation_scale=11, color=INK, linewidth=1.5))
ax.text(39.0, 27.8, "we break one call", fontsize=7.4, color=INK,
        ha="left", va="center", fontweight="bold")

# three fault cards
y2 = 11.5
for i, (c, tag, what, right) in enumerate([
        (BLUE,  "ET", "errors once, then works",  "correct: call it again"),
        (AQUA,  "IT", "returns a bad id,\nnext call fails", "correct: re-query upstream"),
        (ORANGE,"EP", "errors every single time", "correct: report failure")]):
    x = i*34.0
    w = 32.0
    ax.add_patch(FancyBboxPatch((x, y2), w, 12.2,
        boxstyle="round,pad=0.35,rounding_size=1.1",
        facecolor="white", edgecolor=c, linewidth=1.3))
    ax.text(x+w/2, y2+9.6, tag, fontsize=8.8, color=c, fontweight="bold",
            ha="center", va="center")
    ax.text(x+w/2, y2+6.0, what, fontsize=6.9, color=INK, ha="center",
            va="center", linespacing=1.35)
    ax.text(x+w/2, y2+2.2, right, fontsize=6.9, color=c, ha="center",
            va="center", fontweight="bold")

# control strip
ax.add_patch(FancyBboxPatch((0, 0.5), 100, 7.4,
    boxstyle="round,pad=0.35,rounding_size=1.1",
    facecolor="#eef3fb", edgecolor=BLUE, linewidth=1.2))
ax.text(50, 4.2, "Each level is tested only on tasks it already solves every "
        "time with no fault present,\nso a failure here reflects the fault "
        "rather than the difficulty of the task.",
        ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.4)
fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.015)
fig.savefig("fig0.pdf"); fig.savefig("fig0.png", dpi=200)
print("ok")
