#!/usr/bin/env python3
"""Task-controlled confirmatory analysis for the repeated prompt sweep."""

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


DATA = Path("../data/confirmatory_sweep_results.json")
OUT = Path("../data/confirmatory_analysis.json")
R = json.loads(DATA.read_text())
CELLS = R["cells"]
ARMS = R["config"]["arms"]
PROMPTS = R["config"]["prompts"]
RUNS = R["config"]["runs"]


def cell(arm, prompt, run):
    return CELLS[f"{arm}:{prompt}:R{run}"]


def stable_intersection(arms, prompts):
    sets = []
    for arm in arms:
        for prompt in prompts:
            for run in range(RUNS):
                sets.append(set(cell(arm, prompt, run)["stable_clean_tasks"]))
    return sorted(set.intersection(*sets)) if sets else []


def task_rates(arm, prompt, task_ids):
    outcomes = defaultdict(list)
    for run in range(RUNS):
        for episode in cell(arm, prompt, run)["episodes"]:
            if episode["task_id"] in task_ids:
                outcomes[episode["task_id"]].append(bool(episode.get("success")))
    return {task: sum(values) / len(values) for task, values in outcomes.items() if values}


def bootstrap_ci(differences, seed=0, samples=20000):
    if not differences:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = [rng.choice(differences) for _ in differences]
        values.append(sum(draw) / len(draw))
    values.sort()
    return [values[int(0.025 * samples)], values[int(0.975 * samples)]]


def sign_test(differences):
    positive = sum(d > 0 for d in differences)
    negative = sum(d < 0 for d in differences)
    ties = sum(d == 0 for d in differences)
    n = positive + negative
    if n == 0:
        p = 1.0
    else:
        k = min(positive, negative)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"positive": positive, "negative": negative, "ties": ties, "p": p}


def comparison(arm_a, arm_b, prompt_a, prompt_b, task_ids):
    rates_a = task_rates(arm_a, prompt_a, task_ids)
    rates_b = task_rates(arm_b, prompt_b, task_ids)
    shared = sorted(set(rates_a) & set(rates_b))
    differences = [rates_a[t] - rates_b[t] for t in shared]
    return {
        "tasks": shared,
        "n_tasks": len(shared),
        "mean_a": sum(rates_a[t] for t in shared) / max(1, len(shared)),
        "mean_b": sum(rates_b[t] for t in shared) / max(1, len(shared)),
        "mean_difference": sum(differences) / max(1, len(differences)),
        "bootstrap_95_ci": bootstrap_ci(differences),
        "sign_test": sign_test(differences),
    }


common_by_arm = {arm: stable_intersection([arm], PROMPTS) for arm in ARMS}
common_q84 = stable_intersection(["Q8_0", "Q4_K_M"], PROMPTS)
common_q3 = stable_intersection(["Q3_K_M"], PROMPTS)

aggregate = {}
for arm in ARMS:
    aggregate[arm] = {}
    for prompt in PROMPTS:
        episodes = [
            episode
            for run in range(RUNS)
            for episode in cell(arm, prompt, run)["episodes"]
        ]
        k = sum(bool(e.get("success")) for e in episodes)
        aggregate[arm][prompt] = {"success": k, "n": len(episodes), "rate": k / max(1, len(episodes))}

q84 = {
    prompt: comparison("Q8_0", "Q4_K_M", prompt, prompt, common_q84)
    for prompt in PROMPTS
}
q3_p0_p1 = comparison("Q3_K_M", "Q3_K_M", "P0", "P1", common_q3)

q3_p1_episodes = [
    episode
    for run in range(RUNS)
    for episode in cell("Q3_K_M", "P1", run)["episodes"]
]
failed = [episode for episode in q3_p1_episodes if not episode.get("success")]
wrong_report_failure = sum(
    "report_failure" in (episode.get("last_content") or "").lower()
    for episode in failed
)

analysis = {
    "config": R["config"],
    "common_clean_tasks_by_arm": common_by_arm,
    "common_q8_q4_tasks": common_q84,
    "common_q3_tasks": common_q3,
    "aggregate_own_set_rates": aggregate,
    "q8_minus_q4_common_task": q84,
    "q3_p0_minus_p1_common_task": q3_p0_p1,
    "q3_p1_failure_mode": {
        "failed_episodes": len(failed),
        "wrong_report_failure": wrong_report_failure,
        "fraction": wrong_report_failure / max(1, len(failed)),
    },
}
OUT.write_text(json.dumps(analysis, indent=2))

print("COMMON CLEAN TASK SETS")
for arm, tasks in common_by_arm.items():
    print(f"{arm}: {len(tasks)}")
print(f"Q8/Q4 across all prompts and runs: {len(common_q84)}")
print(f"Q3 across all prompts and runs: {len(common_q3)}")

print("\nQ8 - Q4 ON THE SAME TASKS")
for prompt in PROMPTS:
    result = q84[prompt]
    lo, hi = result["bootstrap_95_ci"]
    print(
        f"{prompt}: {result['mean_difference']:+.1%} "
        f"[{lo:+.1%}, {hi:+.1%}], n_tasks={result['n_tasks']}"
    )

result = q3_p0_p1
lo, hi = result["bootstrap_95_ci"]
print("\nQ3 P0 - P1 ON THE SAME TASKS")
print(
    f"difference {result['mean_difference']:+.1%} [{lo:+.1%}, {hi:+.1%}], "
    f"n_tasks={result['n_tasks']}, sign-test p={result['sign_test']['p']:.4g}"
)
print(
    f"Q3/P1 failed episodes calling report_failure: "
    f"{wrong_report_failure}/{len(failed)}"
)
print("\nWrote", OUT)
