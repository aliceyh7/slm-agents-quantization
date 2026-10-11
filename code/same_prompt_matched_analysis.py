"""Reproduce the paper's same-prompt-matched ladder table and ET contrasts."""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
with (DATA_DIR / "phase3_multi_results.json").open() as f:
    R = json.load(f)
ARMS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
HIGH = ARMS[:3]
LOW = ARMS[3:]


def fisher_two_sided(a, b, c, d):
    n1, n2, successes = a + b, c + d, a + c
    lo, hi = max(0, successes - n2), min(n1, successes)

    def prob(x):
        return math.comb(n1, x) * math.comb(n2, successes - x) / math.comb(n1 + n2, successes)

    observed = prob(a)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-12))


clean = {}
filtered = {}
print("SAME-PROMPT CLEAN-SOLVABLE SETS AND SITE-LEVEL COUNTS")
print(f"{'level':<10}{'clean':>10}{'ET':>12}{'IT':>12}{'EP':>12}{'invented':>11}")
for arm in ARMS:
    episodes = R[arm]["episodes"]
    clean[arm] = {
        e["task_id"] for e in episodes if e["cond"] == "CLEAN" and e.get("success")
    }
    filtered[arm] = {}
    cells = []
    for cond in ("ET", "IT", "EP"):
        cell = [e for e in episodes if e["cond"] == cond and e["task_id"] in clean[arm]]
        filtered[arm][cond] = cell
        cells.append((sum(bool(e.get("success")) for e in cell), len(cell)))
    invented = sum(
        1
        for e in filtered[arm]["EP"]
        if e.get("submitted") and not e.get("success") and e.get("parsed_json")
    )
    values = "".join(f"{k}/{n}: {k / n:.0%}".rjust(12) for k, n in cells)
    print(f"{arm:<10}{str(len(clean[arm])) + '/' + str(len(clean[arm])):>10}{values}{invented:>11}")


def pooled_counts(arms):
    success = sum(sum(bool(e.get("success")) for e in filtered[a]["ET"]) for a in arms)
    total = sum(len(filtered[a]["ET"]) for a in arms)
    return success, total


hi_k, hi_n = pooled_counts(HIGH)
lo_k, lo_n = pooled_counts(LOW)
print("\nPOOLED SAME-PROMPT ET CONTRAST")
print(f"high precision: {hi_k}/{hi_n} = {hi_k / hi_n:.1%}")
print(f"low precision:  {lo_k}/{lo_n} = {lo_k / lo_n:.1%}")
print(f"Fisher exact p = {fisher_two_sided(hi_k, hi_n-hi_k, lo_k, lo_n-lo_k):.6g}")

q8_k, q8_n = pooled_counts(["Q8_0"])
q4_k, q4_n = pooled_counts(["Q4_K_M"])
print("\nQ8_0 VS Q4_K_M")
print(f"Q8_0: {q8_k}/{q8_n} = {q8_k / q8_n:.1%}")
print(f"Q4_K_M: {q4_k}/{q4_n} = {q4_k / q4_n:.1%}")
print(f"Fisher exact p = {fisher_two_sided(q8_k, q8_n-q8_k, q4_k, q4_n-q4_k):.6g}")


rate = {}
for arm in ARMS:
    per_task = defaultdict(lambda: [0, 0])
    for e in filtered[arm]["ET"]:
        per_task[e["task_id"]][0] += bool(e.get("success"))
        per_task[e["task_id"]][1] += 1
    rate[arm] = {task: k / n for task, (k, n) in per_task.items()}

shared = sorted(set.intersection(*(clean[a] for a in ARMS)))
diffs = []
for task in shared:
    high = sum(rate[a][task] for a in HIGH) / len(HIGH)
    low = sum(rate[a][task] for a in LOW) / len(LOW)
    diffs.append(high - low)
positive = sum(d > 0 for d in diffs)
negative = sum(d < 0 for d in diffs)
ties = sum(d == 0 for d in diffs)
n_eff = positive + negative
p_sign = min(
    1.0,
    2 * sum(math.comb(n_eff, i) for i in range(min(positive, negative) + 1)) / 2**n_eff,
)
random.seed(0)
boot = []
for _ in range(20000):
    sample = [random.choice(diffs) for _ in diffs]
    boot.append(sum(sample) / len(sample))
boot.sort()
print("\nTASK-LEVEL ROBUSTNESS CHECK")
print(f"common clean tasks: {len(shared)}")
print(f"favor high: {positive}; favor low: {negative}; ties: {ties}; sign-test p = {p_sign:.4g}")
print(f"cluster-bootstrap 95% CI: [{boot[500]:.2f}, {boot[19499]:.2f}]")
