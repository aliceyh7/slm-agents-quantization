"""Strict-vs-tolerant scoring sensitivity for the camera-ready.

Tolerant (as published): an episode succeeds if the submitted amount matches gold,
regardless of whether the final message was pure JSON or the amount was an
arithmetic expression that the harness evaluated.
Strict: success additionally requires strict_json == True and expr_submit == False.
The clean screen is rescored under the same policy, so the clean-passing sets
(and therefore the matched sets and full-pipeline denominators) change too.
"""
import json, random, sys
from collections import defaultdict

random.seed(0)
B = 20000
PROMPTS = ["P0", "P1", "P2", "P3", "P4"]
ARMS = ["Q8_0", "Q4_K_M"]

def ok(e, strict):
    if not e.get("success"):
        return False
    if strict and (not e.get("strict_json", False) or e.get("expr_submit", False)):
        return False
    return True

def load(path):
    d = json.load(open(path))
    task_ids = d["config"]["task_ids"]
    return d["cells"], task_ids

def cell_stats(cell, strict):
    # clean-passing set under policy
    clean = {}
    for e in cell["clean_episodes"]:
        clean.setdefault(e["task_id"], []).append(ok(e, strict))
    S = {t for t, v in clean.items() if all(v)}
    # per-task fault recovery (mean over sites), only for tasks in S
    r = defaultdict(list)
    for e in cell["episodes"]:
        if e["task_id"] in S:
            r[e["task_id"]].append(1.0 if ok(e, strict) else 0.0)
    rt = {t: sum(v) / len(v) for t, v in r.items()}
    # tasks in S with no fault episodes (should not happen) -> treat as 1? skip
    return S, rt

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def paired_boot(d):
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"))
    bs = []
    for _ in range(B):
        s = [d[random.randrange(n)] for _ in range(n)]
        bs.append(mean(s))
    bs.sort()
    return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]

def two_sample_boot(a, b):
    if not a or not b:
        return (float("nan"), float("nan"))
    bs = []
    for _ in range(B):
        sa = [a[random.randrange(len(a))] for _ in range(len(a))]
        sb = [b[random.randrange(len(b))] for _ in range(len(b))]
        bs.append(mean(sa) - mean(sb))
    bs.sort()
    return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]

def analyze(path, label):
    cells, task_ids = load(path)
    out = {}
    for strict in (False, True):
        pol = "strict" if strict else "tolerant"
        for p in PROMPTS:
            s8, r8 = cell_stats(cells[f"Q8_0:{p}:R0"], strict)
            s4, r4 = cell_stats(cells[f"Q4_K_M:{p}:R0"], strict)
            own8 = mean([r8[t] for t in s8]); own4 = mean([r4[t] for t in s4])
            own_ci = two_sample_boot([r8[t] for t in s8], [r4[t] for t in s4])
            M = sorted(s8 & s4)
            md = [r8[t] - r4[t] for t in M]
            m_ci = paired_boot(md)
            full8 = {t: (r8[t] if t in s8 else 0.0) for t in task_ids}
            full4 = {t: (r4[t] if t in s4 else 0.0) for t in task_ids}
            fd = [full8[t] - full4[t] for t in task_ids]
            f_ci = paired_boot(fd)
            # pooled episode counts for reporting
            def pooled(cell, S):
                n = k = 0
                for e in cell["episodes"]:
                    if e["task_id"] in S:
                        n += 1; k += ok(e, strict)
                return k, n
            k8, n8 = pooled(cells[f"Q8_0:{p}:R0"], s8)
            k4, n4 = pooled(cells[f"Q4_K_M:{p}:R0"], s4)
            out[(pol, p)] = dict(
                n8=len(s8), n4=len(s4), own8=own8, own4=own4,
                own_diff=own8 - own4, own_ci=own_ci,
                nM=len(M), m_diff=mean(md), m_ci=m_ci,
                full8=mean(full8.values()), full4=mean(full4.values()),
                f_diff=mean(fd), f_ci=f_ci,
                pooled8=(k8, n8), pooled4=(k4, n4),
            )
    # expression / non-strict usage among tolerant successes
    usage = {}
    for p in PROMPTS:
        for a in ARMS:
            c = cells[f"{a}:{p}:R0"]
            succ = [e for e in c["episodes"] if e.get("success")]
            expr = sum(1 for e in succ if e.get("expr_submit"))
            nonstrict = sum(1 for e in succ if not e.get("strict_json", False))
            cs = [e for e in c["clean_episodes"] if e.get("success")]
            cexpr = sum(1 for e in cs if e.get("expr_submit"))
            cnon = sum(1 for e in cs if not e.get("strict_json", False))
            usage[(a, p)] = dict(fault_succ=len(succ), fault_expr=expr, fault_nonstrict=nonstrict,
                                 clean_succ=len(cs), clean_expr=cexpr, clean_nonstrict=cnon)
    return out, usage

def pct(x):
    return f"{100*x:+.1f}" if x == x else "n/a"

def report(label, out, usage):
    print(f"\n===== {label} =====")
    print("Tolerant successes that used expression submit / non-strict JSON (fault episodes; clean episodes):")
    for p in PROMPTS:
        row = []
        for a in ARMS:
            u = usage[(a, p)]
            row.append(f"{a}: expr {u['fault_expr']}/{u['fault_succ']} nonstrict {u['fault_nonstrict']}/{u['fault_succ']} | clean expr {u['clean_expr']}/{u['clean_succ']} nonstrict {u['clean_nonstrict']}/{u['clean_succ']}")
        print(f"  {p}: " + "   ".join(row))
    for pol in ("tolerant", "strict"):
        print(f"\n--- {pol} ---")
        print(f"{'P':3} {'|S8|':>4} {'|S4|':>4} {'own8':>6} {'own4':>6} {'ownΔ':>7} {'ownCI':>16} {'nM':>3} {'matchedΔ':>9} {'mCI':>16} {'full8':>6} {'full4':>6} {'fullΔ':>7} {'fCI':>16}")
        for p in PROMPTS:
            o = out[(pol, p)]
            print(f"{p:3} {o['n8']:>4} {o['n4']:>4} {100*o['own8']:>6.1f} {100*o['own4']:>6.1f} {pct(o['own_diff']):>7} [{pct(o['own_ci'][0])},{pct(o['own_ci'][1])}] {o['nM']:>3} {pct(o['m_diff']):>9} [{pct(o['m_ci'][0])},{pct(o['m_ci'][1])}] {100*o['full8']:>6.1f} {100*o['full4']:>6.1f} {pct(o['f_diff']):>7} [{pct(o['f_ci'][0])},{pct(o['f_ci'][1])}]")

if __name__ == "__main__":
    res = {}
    for label, path in [("Llama-3.1-8B", "data/confirmatory_sweep_results.json"),
                        ("Qwen2.5-7B", "data/qwen_confirmatory_sweep_results.json")]:
        out, usage = analyze(path, label)
        report(label, out, usage)
        res[label] = {f"{k[0]}:{k[1]}": v for k, v in out.items()}
        res[label + "_usage"] = {f"{k[0]}:{k[1]}": v for k, v in usage.items()}
    json.dump(res, open("strict_vs_tolerant.json", "w"), indent=1, default=str)
