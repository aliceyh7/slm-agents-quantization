"""Reproduce the paper's cross-prompt estimates and appendix tables
from data/confirmatory_sweep_results.json. Single source of truth for the
confirmatory statistics; run with no arguments, no GPU needed."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "confirmatory_sweep_results.json"
R = json.loads(DATA.read_text())
C = R["cells"]
ARMS = ["Q8_0", "Q4_K_M", "Q3_K_M"]
PROMPTS = ["P0", "P1", "P2", "P3", "P4"]

def cell(a, p):
    return C[f"{a}:{p}:R0"]

def task_rates(a, p, tids):
    by = defaultdict(list)
    for e in cell(a, p)["episodes"]:
        if e["task_id"] in tids:
            by[e["task_id"]].append(1 if e.get("success") else 0)
    return {t: sum(v) / len(v) for t, v in by.items() if v}

def sign_test(diffs):
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    return pos, neg, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)

def boot_ci(diffs, seed=0, B=20000):
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        s = [rng.choice(diffs) for _ in diffs]
        vals.append(sum(s) / len(s))
    vals.sort()
    return vals[int(.025 * B)], vals[int(.975 * B)]

def boot_unpaired_ci(left, right, seed=0, B=20000):
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        lmean = sum(rng.choice(left) for _ in left) / len(left)
        rmean = sum(rng.choice(right) for _ in right) / len(right)
        vals.append(lmean - rmean)
    vals.sort()
    return vals[int(.025 * B)], vals[int(.975 * B)]

stable = {(a, p): set(cell(a, p)["stable_clean_tasks"]) for a in ARMS for p in PROMPTS}

print("== Separate-screen sweep (Table 1) ==")
for a in ARMS:
    print(f"  {a}: " + "  ".join(
        f"{p}:{cell(a,p)['et_success']}/{cell(a,p)['et_n']}" for p in PROMPTS))
print("  signed Q8-Q4 episode-pooled own-set diffs (pts): " + ", ".join(
    f"{p}:{100*(cell('Q8_0',p)['et_success']/cell('Q8_0',p)['et_n'] - cell('Q4_K_M',p)['et_success']/cell('Q4_K_M',p)['et_n']):+.1f}"
    for p in PROMPTS))
print("  task-mean own-set diffs with two-sample task-bootstrap CIs:")
for p in PROMPTS:
    r8 = list(task_rates("Q8_0", p, stable[("Q8_0", p)]).values())
    r4 = list(task_rates("Q4_K_M", p, stable[("Q4_K_M", p)]).values())
    m = sum(r8) / len(r8) - sum(r4) / len(r4)
    lo, hi = boot_unpaired_ci(r8, r4)
    print(f"  {p}: Q8={100*sum(r8)/len(r8):.1f}% Q4={100*sum(r4)/len(r4):.1f}% "
          f"diff={100*m:+.1f} CI[{100*lo:+.1f},{100*hi:+.1f}]")

print("\n== All-prompt common Q8+Q4 task set ==")
common_q84 = sorted(set.intersection(*[stable[(a, p)] for a in ["Q8_0", "Q4_K_M"] for p in PROMPTS]))
print(f"  set ({len(common_q84)} tasks): {common_q84}")
for p in PROMPTS:
    r8 = task_rates("Q8_0", p, common_q84)
    r4 = task_rates("Q4_K_M", p, common_q84)
    sh = sorted(set(r8) & set(r4))
    d = [r8[t] - r4[t] for t in sh]
    lo, hi = boot_ci(d)
    print(f"  {p}: Q8={sum(r8[t] for t in sh)/len(sh):.3f} Q4={sum(r4[t] for t in sh)/len(sh):.3f} "
          f"diff={100*sum(d)/len(d):+.1f}pts CI[{100*lo:+.1f},{100*hi:+.1f}]")

print("\n== Exploratory Q3 P0 vs P1 on the common Q3 set ==")
common_q3 = sorted(set.intersection(*[stable[("Q3_K_M", p)] for p in PROMPTS]))
print(f"  set ({len(common_q3)} tasks): {common_q3}")
r0 = task_rates("Q3_K_M", "P0", common_q3)
r1 = task_rates("Q3_K_M", "P1", common_q3)
sh = sorted(set(r0) & set(r1))
d = [r0[t] - r1[t] for t in sh]
pos, neg, p_sign = sign_test(d)
lo, hi = boot_ci(d)
print(f"  P0={sum(r0[t] for t in sh)/len(sh):.3f} P1={sum(r1[t] for t in sh)/len(sh):.3f} "
      f"diff={100*sum(d)/len(d):+.1f}pts CI[{100*lo:+.1f},{100*hi:+.1f}] "
      f"favor-P0 {pos}/{len(sh)} sign-p={p_sign:.4g}")
for p in ["P2", "P3", "P4"]:
    rp = task_rates("Q3_K_M", p, common_q3)
    print(f"  {p} on same set: {sum(rp[t] for t in sh)/len(sh):.3f}")

print("\n== Additional sensitivity check: one fixed set across all levels ==")
common_all = sorted(set.intersection(*stable.values()))
print(f"  set ({len(common_all)} tasks): {common_all}")
for a in ARMS:
    rA = task_rates(a, "P0", common_all)
    rB = task_rates(a, "P1", common_all)
    shl = sorted(set(rA) & set(rB))
    dd = [rA[t] - rB[t] for t in shl]
    pos, neg, sp = sign_test(dd)
    lo, hi = boot_ci(dd)
    print(f"  {a}: P0={sum(rA[t] for t in shl)/len(shl):.3f} P1={sum(rB[t] for t in shl)/len(shl):.3f} "
          f"diff={100*sum(dd)/len(dd):+.1f}pts CI[{100*lo:+.1f},{100*hi:+.1f}] p={sp:.4g}")

print("\n== Exploratory Q3/P1 failure modes ==")
eps = cell("Q3_K_M", "P1")["episodes"]
failed = [e for e in eps if not e.get("success")]
rf = [e for e in failed if "report_failure" in (e.get("last_content") or "").lower()]
print(f"  failed={len(failed)}  report_failure={len(rf)}")
for e in failed:
    if e not in rf:
        print(f"  exception: task={e['task_id']} site={e.get('site')} last={e.get('last_content')!r}")

print("\n== Full-suite pipeline yield ==")
for a in ARMS:
    row = []
    for p in PROMPTS:
        tr = task_rates(a, p, set(range(1, 21)))
        row.append(f"{p}:{sum(tr.get(t, 0.0) for t in range(1, 21))/20:.2f}")
    print(f"  {a}: " + "  ".join(row))

print("\n== Provenance ==")
print("  config:", json.dumps(R["config"]))

print("\n== Fig. 2A: pairwise P0-vs-P1 common clean sets per level ==")
for a in ARMS:
    inter = sorted(stable[(a, "P0")] & stable[(a, "P1")])
    r0 = task_rates(a, "P0", inter)
    r1 = task_rates(a, "P1", inter)
    sh = sorted(set(r0) & set(r1))
    dd = [r0[t] - r1[t] for t in sh]
    pos, neg, sp = sign_test(dd)
    lo, hi = boot_ci(dd)
    print(f"  {a}: n={len(sh)} P0={100*sum(r0[t] for t in sh)/len(sh):.1f}% "
          f"P1={100*sum(r1[t] for t in sh)/len(sh):.1f}% diff={100*sum(dd)/len(dd):+.1f} "
          f"CI[{100*lo:+.1f},{100*hi:+.1f}] +{pos}/-{neg} sign-p={sp:.3g}")

print("\n== Fig. 1C: full-suite pipeline yield Q8-Q4 per prompt ==")
ALL20 = set(range(1, 21))
for p in PROMPTS:
    tr8 = task_rates("Q8_0", p, ALL20)
    tr4 = task_rates("Q4_K_M", p, ALL20)
    dd = [tr8.get(t, 0.0) - tr4.get(t, 0.0) for t in range(1, 21)]
    lo, hi = boot_ci(dd)
    print(f"  {p}: diff={100*sum(dd)/len(dd):+.1f} CI[{100*lo:+.1f},{100*hi:+.1f}]")

print("\n== Exploratory-vs-confirmatory consistency check ==")
EXP = Path(__file__).resolve().parents[1] / "data" / "promptvar2_results.json"
E = json.loads(EXP.read_text())
mismatches = []
for a in ARMS:
    for p in PROMPTS:
        conf = cell(a, p)
        expl = E[f"{a}:{p}"]
        checks = [
            ("et_success", conf["et_success"], expl["et_success"]),
            ("et_n", conf["et_n"], expl["et_n"]),
            ("clean set", sorted(conf["stable_clean_tasks"]), sorted(expl["solvable"])),
        ]
        for name, cv, ev in checks:
            if cv != ev:
                mismatches.append(f"{a}:{p} {name}: confirmatory={cv} exploratory={ev}")
if mismatches:
    for m in mismatches:
        print("  MISMATCH", m)
    raise SystemExit("exploratory and confirmatory runs disagree; "
                     "the paper's identical-counts claim would be false")
print("  all 15 cells: clean-task sets, numerators, and denominators identical")


print("\n== Qwen2.5-7B confirmatory sweep: validation and Figure 1 ==")
QWEN_DATA = Path(__file__).resolve().parents[1] / "data" / "qwen_confirmatory_sweep_results.json"
Q = json.loads(QWEN_DATA.read_text())
QC = Q["cells"]
QARMS = ["Q8_0", "Q4_K_M"]


def qcell(a, p):
    return QC[f"{a}:{p}:R0"]


def qtask_rates(a, p, tids):
    by = defaultdict(list)
    for e in qcell(a, p)["episodes"]:
        if e["task_id"] in tids:
            by[e["task_id"]].append(1 if e.get("success") else 0)
    return {t: sum(v) / len(v) for t, v in by.items() if v}


expected_cells = {f"{a}:{p}:R0" for a in QARMS for p in PROMPTS}
if set(QC) != expected_cells:
    raise SystemExit(f"Qwen cells differ from expected 10-cell grid: {sorted(QC)}")
qstable = {}
all_task_ids = set(Q["config"]["task_ids"])
for a in QARMS:
    for p in PROMPTS:
        c = qcell(a, p)
        clean = c["clean_episodes"]
        stable_clean = {e["task_id"] for e in clean if e.get("success")}
        if len(clean) != len(all_task_ids) or stable_clean != set(c["stable_clean_tasks"]):
            raise SystemExit(f"Qwen clean-screen mismatch in {a}:{p}")
        if any(e["task_id"] not in stable_clean for e in c["episodes"]):
            raise SystemExit(f"Qwen fault episode outside clean-passing set in {a}:{p}")
        if c["et_n"] != len(c["episodes"]) or c["et_success"] != sum(
                bool(e.get("success")) for e in c["episodes"]):
            raise SystemExit(f"Qwen aggregate-count mismatch in {a}:{p}")
        qstable[(a, p)] = stable_clean
print("  all 10 cells complete; clean sets and aggregate fault counts match episode logs")

qcommon = sorted(set.intersection(*qstable.values()))
print(f"  fixed Q8/Q4 x all-prompt set ({len(qcommon)} tasks): {qcommon}")
for p in PROMPTS:
    q8_own = list(qtask_rates("Q8_0", p, qstable[("Q8_0", p)]).values())
    q4_own = list(qtask_rates("Q4_K_M", p, qstable[("Q4_K_M", p)]).values())
    own = sum(q8_own) / len(q8_own) - sum(q4_own) / len(q4_own)
    own_lo, own_hi = boot_unpaired_ci(q8_own, q4_own)

    q8_common = qtask_rates("Q8_0", p, qcommon)
    q4_common = qtask_rates("Q4_K_M", p, qcommon)
    common_diffs = [q8_common[t] - q4_common[t] for t in qcommon]
    common = sum(common_diffs) / len(common_diffs)
    common_lo, common_hi = boot_ci(common_diffs)

    q8_full = qtask_rates("Q8_0", p, all_task_ids)
    q4_full = qtask_rates("Q4_K_M", p, all_task_ids)
    full_diffs = [q8_full.get(t, 0.0) - q4_full.get(t, 0.0) for t in sorted(all_task_ids)]
    full = sum(full_diffs) / len(full_diffs)
    full_lo, full_hi = boot_ci(full_diffs)
    print(
        f"  {p}: own={100*own:+.1f} [{100*own_lo:+.1f},{100*own_hi:+.1f}]  "
        f"common={100*common:+.1f} [{100*common_lo:+.1f},{100*common_hi:+.1f}]  "
        f"full={100*full:+.1f} [{100*full_lo:+.1f},{100*full_hi:+.1f}]"
    )

print("  prompt-wise matched sensitivity analysis:")
for p in PROMPTS:
    prompt_common = sorted(qstable[("Q8_0", p)] & qstable[("Q4_K_M", p)])
    q8 = qtask_rates("Q8_0", p, prompt_common)
    q4 = qtask_rates("Q4_K_M", p, prompt_common)
    diffs = [q8[t] - q4[t] for t in prompt_common]
    mean = sum(diffs) / len(diffs)
    lo, hi = boot_ci(diffs)
    print(
        f"    {p}: n={len(prompt_common)} diff={100*mean:+.1f} "
        f"CI[{100*lo:+.1f},{100*hi:+.1f}]"
    )
print("  config:", json.dumps(Q["config"]))

print("\n== Llama prompt-wise matched analysis ==")
for p in PROMPTS:
    s8 = set(cell("Q8_0", p)["stable_clean_tasks"])
    s4 = set(cell("Q4_K_M", p)["stable_clean_tasks"])
    inter = s8 & s4
    r8 = task_rates("Q8_0", p, inter)
    r4 = task_rates("Q4_K_M", p, inter)
    sh = sorted(set(r8) & set(r4))
    dd = [r8[t] - r4[t] for t in sh]
    lo, hi = boot_ci(dd)
    print(f"  {p}: n={len(sh)} diff={100*sum(dd)/len(dd):+.1f} "
          f"CI[{100*lo:+.1f},{100*hi:+.1f}]")

print("\n== Table 1: per-cell task-mean recovery (both models) ==")
def cellrates(cells, a, p):
    c = cells[f"{a}:{p}:R0"]
    by = defaultdict(list)
    for e in c["episodes"]:
        if e["task_id"] in set(c["stable_clean_tasks"]):
            by[e["task_id"]].append(1 if e.get("success") else 0)
    r = {t_: sum(v) / len(v) for t_, v in by.items() if v}
    return 100 * sum(r.values()) / len(r), c["et_success"], c["et_n"]
for label, cells, arms in [("Llama", C, ["Q8_0", "Q4_K_M", "Q3_K_M"]),
                            ("Qwen", QC, ["Q8_0", "Q4_K_M"])]:
    for a in arms:
        row = []
        for p in PROMPTS:
            tm, k, n = cellrates(cells, a, p)
            row.append(f"{p}:{tm:.1f}%[{k}/{n}]")
        print(f"  {label} {a}: " + "  ".join(row))

print("\n== Per-cell pipeline yield (both models) ==")
def yield20(cells, a, p):
    c = cells[f"{a}:{p}:R0"]
    by = defaultdict(list)
    for e in c["episodes"]:
        by[e["task_id"]].append(1 if e.get("success") else 0)
    r = {t_: sum(v) / len(v) for t_, v in by.items() if v}
    return sum(r.get(t_, 0.0) for t_ in range(1, 21)) / 20
def format_yield(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")
for label, cells, arms in [("Llama", C, ["Q8_0", "Q4_K_M", "Q3_K_M"]),
                            ("Qwen", QC, ["Q8_0", "Q4_K_M"])]:
    for a in arms:
        print(f"  {label} {a}: " + "  ".join(
            f"{p}:{format_yield(yield20(cells, a, p))}" for p in PROMPTS
        ))
