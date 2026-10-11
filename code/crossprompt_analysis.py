"""Cross-prompt analysis of the five-prompt sweep (promptvar2_results.json).

Generalized from the original version, which only tested Q3_K_M P0 vs P1.
With complete per-episode logs for all 15 prompt x precision cells (as a
fresh run of promptvar2.py produces), this now:

  1. Reports the common clean-solvable set at every level, and which cells
     have complete episode logs (log count == aggregate count) vs. only
     aggregate counts -- so it never silently draws a task-clustered
     inference from an incomplete episode list.
  2. Runs the task-clustered sign test + cluster bootstrap for every
     (level, P0-vs-Pi) pair with complete logs, not just Q3_K_M/P1.
  3. Tests whether the Q8_0-vs-Q4_K_M ordering reversal across prompts
     (Table 1 in the paper) survives on a single task set that is
     clean-solvable under every prompt AND at both levels -- the strongest
     version of the task-composition-confound check, and the one the
     05_run_notebook_review flagged as the difference between a 6/10 and a
     7/10 (word for word: "If the reversal survives on a common task set...").
  4. Runs the failure-mode categorization for every cell whose ET recovery
     is at or below 20%, not just the one cell that happened to be 0%.
"""
import json, re, math, random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
with (DATA_DIR / "promptvar2_results.json").open() as f:
    R = json.load(f)
ARMS = ["Q8_0", "Q4_K_M", "Q3_K_M"]
PROMPTS = ["P0", "P1", "P2", "P3", "P4"]


def has_complete_log(key):
    cell = R[key]
    return len(cell.get("episodes", [])) == cell.get("et_n", -1) and cell.get("et_n", 0) > 0


print("="*70)
print("1) COMMON CLEAN-SOLVABLE SET (intersection across all 5 prompts)")
print("="*70)
common = {}
for lvl in ARMS:
    sets = [set(R[f"{lvl}:{p}"]["solvable"]) for p in PROMPTS]
    inter = set.intersection(*sets)
    common[lvl] = inter
    print(f"{lvl}: per-prompt solvable sizes = {[len(s) for s in sets]}  ->  common = {len(inter)} tasks")

print()
print("EPISODE-LOG COVERAGE (log count vs aggregate et_n, per cell)")
complete_cells = []
for lvl in ARMS:
    cells = []
    for p in PROMPTS:
        key = f"{lvl}:{p}"
        n_log = len(R[key].get("episodes", []))
        n_agg = R[key].get("et_n", 0)
        ok = has_complete_log(key)
        if ok:
            complete_cells.append(key)
        cells.append(f"{p}:{n_log}/{n_agg}{'OK' if ok else '  '}")
    print(f"{lvl}: " + "  ".join(cells))
print(f"\n{len(complete_cells)}/15 cells have complete episode logs; task-clustered "
      "inference below runs only on those.")


def per_task_rate(key, task_ids):
    eps = [e for e in R[key]["episodes"] if e.get("task_id") in task_ids]
    by_task = defaultdict(list)
    for e in eps:
        by_task[e["task_id"]].append(1 if e.get("success") else 0)
    return {t: sum(v)/len(v) for t, v in by_task.items() if v}


def sign_test_and_bootstrap(rateA, rateB, seed=0, n_boot=10000):
    shared_tasks = sorted(set(rateA) & set(rateB))
    diffs = [rateA[t] - rateB[t] for t in shared_tasks]
    favA = sum(1 for d in diffs if d > 0)
    favB = sum(1 for d in diffs if d < 0)
    tie  = sum(1 for d in diffs if d == 0)
    n_eff = favA + favB
    if n_eff > 0:
        k = min(favA, favB)
        p_sign = min(sum(math.comb(n_eff, i) for i in range(0, k + 1)) * 2 / (2 ** n_eff), 1.0)
    else:
        p_sign = float("nan")
    random.seed(seed)
    boots = []
    for _ in range(n_boot):
        samp = [random.choice(diffs) for _ in diffs] if diffs else [0]
        boots.append(sum(samp) / len(samp))
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else float("nan")
    hi = boots[int(0.975 * len(boots))] if boots else float("nan")
    return dict(shared=len(shared_tasks), favA=favA, favB=favB, tie=tie,
                p_sign=p_sign, ci=(lo, hi),
                meanA=sum(rateA[t] for t in shared_tasks)/len(shared_tasks) if shared_tasks else float("nan"),
                meanB=sum(rateB[t] for t in shared_tasks)/len(shared_tasks) if shared_tasks else float("nan"))


print()
print("="*70)
print("2) TASK-CLUSTERED P0-vs-Pi INFERENCE, EVERY LEVEL, EVERY COMPLETE PAIR")
print("="*70)
for lvl in ARMS:
    tset = common[lvl]
    keyA = f"{lvl}:P0"
    if not has_complete_log(keyA):
        print(f"{lvl}: skipped (P0 lacks complete episode log)")
        continue
    rateA = per_task_rate(keyA, tset)
    for p in PROMPTS[1:]:
        keyB = f"{lvl}:{p}"
        if not has_complete_log(keyB):
            print(f"{lvl} P0 vs {p}: skipped (incomplete episode log for {p})")
            continue
        rateB = per_task_rate(keyB, tset)
        r = sign_test_and_bootstrap(rateA, rateB)
        print(f"{lvl} P0 vs {p}: shared={r['shared']} favor-P0={r['favA']} "
              f"favor-{p}={r['favB']} tie={r['tie']}  sign-p={r['p_sign']:.4g}  "
              f"mean P0={r['meanA']:.2f} {p}={r['meanB']:.2f}  "
              f"boot 95% CI(P0-{p})=[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]")

print()
print("="*70)
print("3) DOES THE Q8_0-vs-Q4_K_M ORDERING REVERSAL SURVIVE ON A TASK SET FIXED")
print("   ACROSS BOTH LEVELS AND ALL FIVE PROMPTS?")
print("="*70)
cross_level_common = common["Q8_0"] & common["Q4_K_M"]
print(f"tasks clean-solvable under every prompt at BOTH Q8_0 and Q4_K_M: "
      f"{len(cross_level_common)}  {sorted(cross_level_common)}")
if not cross_level_common:
    print("empty set -- cannot test the reversal on a fully common task set with this data.")
else:
    for p in PROMPTS:
        k8, k4 = f"Q8_0:{p}", f"Q4_K_M:{p}"
        if not (has_complete_log(k8) and has_complete_log(k4)):
            print(f"{p}: skipped (incomplete episode log at Q8_0 or Q4_K_M)")
            continue
        r8 = per_task_rate(k8, cross_level_common)
        r4 = per_task_rate(k4, cross_level_common)
        shared = sorted(set(r8) & set(r4))
        if not shared:
            print(f"{p}: no tasks with episodes at both levels after filtering")
            continue
        m8 = sum(r8[t] for t in shared) / len(shared)
        m4 = sum(r4[t] for t in shared) / len(shared)
        print(f"{p}: n={len(shared)} tasks  Q8_0={m8:.2f}  Q4_K_M={m4:.2f}  "
              f"signed Q8-Q4 diff = {m8-m4:+.2f}")
    print("\nCompare the sign pattern above with Table 1's per-cell signed "
          "differences (+18,+1,0,+6,-21 on the own-reliable-set numbers). If "
          "the signs agree here on one task set shared by every prompt and "
          "both levels, the ordering reversal is not an artifact of which "
          "tasks entered each prompt's own reliable set.")

print()
print("="*70)
print("4) FAILURE-MODE CHARACTERIZATION FOR EVERY LOW-RECOVERY CELL (<=20% ET)")
print("="*70)

def classify(e):
    txt = (e.get("last_content") or "").lower()
    if e.get("success"):
        return "success"
    if e.get("success_lenient") and not e.get("success"):
        return "near-miss (lenient-credit, strict-fail)"
    if e.get("prose_abstain"):
        return "prose abstain (gave up in free text, no tool call)"
    if "report_failure" in txt:
        return "called report_failure on a transient fault (wrong strategy)"
    if re.search(r"\bsorry\b|\bcannot\b|\bunable\b|\bi apologize\b", txt):
        return "apologetic refusal / gave up"
    if txt.strip() == "":
        return "empty/truncated final content"
    if re.search(r"\{.*\}", txt, re.S) is None and len(txt) > 0:
        return "no parseable JSON tool call emitted"
    return "other / unclassified"

any_low = False
for lvl in ARMS:
    for p in PROMPTS:
        key = f"{lvl}:{p}"
        cell = R[key]
        if cell.get("et_n", 0) == 0:
            continue
        rate = cell["et_success"] / cell["et_n"]
        if rate > 0.20:
            continue
        any_low = True
        print(f"\n--- {key}  ({cell['et_success']}/{cell['et_n']} = {rate:.0%}) ---")
        if not has_complete_log(key):
            print("  incomplete episode log -- rerun promptvar2.py from a blank "
                  "results file to get per-episode detail for this cell.")
            continue
        eps = cell["episodes"]
        buckets = defaultdict(int)
        examples = {}
        for e in eps:
            c = classify(e)
            buckets[c] += 1
            examples.setdefault(c, (e.get("task_id"), e.get("site"), (e.get("last_content") or "")[:160]))
        for c, n in sorted(buckets.items(), key=lambda x: -x[1]):
            tid, site, snip = examples[c]
            print(f"  [{n}/{len(eps)}] {c}")
            print(f"    example: task_id={tid} site={site}  last_content={snip!r}")
if not any_low:
    print("no cell at or below 20% ET recovery in this run.")
