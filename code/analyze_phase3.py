#!/usr/bin/env python3
"""
Phase 3 analysis: conditional (matched-baseline) recovery rates, failure-mode
taxonomy, persistence measures, paired tests vs the reference arm.

Key correction: the matched set is re-derived from the Phase 3 CLEAN condition
(identical prompt/session) rather than Phase 2, because the Phase 3 prompt
perturbs baseline behaviour and that perturbation is itself arm-dependent.
"""
import json, math, os
from collections import defaultdict

ARMS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
REF  = "Q8_0"
CONDS = ["ET", "IT", "EP"]
R = json.load(open(os.environ.get("F", "phase3_results.json")))

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def fisher_exact_2x2(a, b, c, d):
    """two-sided Fisher exact p for [[a,b],[c,d]]"""
    def logC(n, k):
        return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)
    n = a+b+c+d; r1 = a+b; r2 = c+d; c1 = a+c
    def p(x):
        return math.exp(logC(r1, x) + logC(r2, c1-x) - logC(n, c1))
    p0 = p(a); tot = 0.0
    lo = max(0, c1-r2); hi = min(r1, c1)
    for x in range(lo, hi+1):
        px = p(x)
        if px <= p0 * (1 + 1e-9):
            tot += px
    return min(1.0, tot)

def mcnemar(b, c):
    """exact two-sided binomial p on discordant pairs"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    s = sum(math.comb(n, i) for i in range(0, k+1))
    return min(1.0, 2 * s / (2 ** n))

# ---------------------------------------------------------------- index data
ep = {}          # arm -> (task, cond) -> episode
for arm in ARMS:
    rec = R.get(arm, {})
    if "episodes" not in rec: continue
    ep[arm] = {(e["task_id"], e["cond"]): e for e in rec["episodes"]}

# matched set: tasks where CLEAN succeeded for that arm (Phase 3 prompt)
matched = {a: sorted(t for (t, c), e in ep[a].items()
                     if c == "CLEAN" and e.get("success")) for a in ep}

print("=" * 78)
print("MATCHED SETS (task passed CLEAN under the Phase 3 prompt)")
print("=" * 78)
for a in ARMS:
    print(f"  {a:<8} n={len(matched[a]):>3}   (Phase-2 reliable was {len(R[a].get('reliable', []))})")

shared = set.intersection(*[set(matched[a]) for a in ARMS if a in matched])
shared = sorted(shared)
print(f"\n  shared across ALL arms: n={len(shared)}  {shared}")

# ------------------------------------------------------- conditional rates
print("\n" + "=" * 78)
print("CONDITIONAL SUCCESS  (own matched set; 95% Wilson CI)")
print("=" * 78)
print(f"{'arm':<9}" + "".join(f"{c:>22}" for c in CONDS))
cond_rate = defaultdict(dict)
for a in ARMS:
    row = f"{a:<9}"
    for c in CONDS:
        k = sum(1 for t in matched[a] if ep[a].get((t, c), {}).get("success"))
        n = len(matched[a])
        cond_rate[a][c] = (k, n)
        lo, hi = wilson(k, n)
        row += f"{f'{k}/{n} {k/n:.0%} [{lo:.0%},{hi:.0%}]':>22}"
    print(row)

print("\nPRR (ET+IT pooled = recovery given a recoverable fault):")
for a in ARMS:
    k = cond_rate[a]["ET"][0] + cond_rate[a]["IT"][0]
    n = cond_rate[a]["ET"][1] + cond_rate[a]["IT"][1]
    lo, hi = wilson(k, n)
    print(f"  {a:<9} PRR = {k}/{n} = {k/n:.0%}  [{lo:.0%}, {hi:.0%}]")

# ------------------------------------------------------- paired vs reference
print("\n" + "=" * 78)
print(f"PAIRED vs {REF} on the {len(shared)} shared tasks (McNemar exact)")
print("=" * 78)
for c in CONDS:
    print(f"\n  {c}:")
    for a in ARMS:
        if a == REF: continue
        b = sum(1 for t in shared
                if ep[REF][(t, c)]["success"] and not ep[a][(t, c)]["success"])
        d = sum(1 for t in shared
                if not ep[REF][(t, c)]["success"] and ep[a][(t, c)]["success"])
        p = mcnemar(b, d)
        kr = sum(1 for t in shared if ep[REF][(t, c)]["success"])
        ka = sum(1 for t in shared if ep[a][(t, c)]["success"])
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        print(f"    {a:<9} {kr}/{len(shared)} -> {ka}/{len(shared)}   "
              f"ref-only={b} arm-only={d}  p={p:.4f} {star}")

# ---------------------------------------------- interaction: ET deficit vs CLEAN
print("\n" + "=" * 78)
print("C2 INTERACTION: does the injected-fault deficit grow down the ladder?")
print("(all on each arm's own matched set, where CLEAN = 100% by construction)")
print("=" * 78)
for c in CONDS:
    print(f"\n  {c} deficit relative to CLEAN:")
    for a in ARMS:
        k, n = cond_rate[a][c]
        print(f"    {a:<9} {n-k:>2}/{n} failures  ({1-k/n:.0%} deficit)")
    kr, nr = cond_rate[REF][c]
    print(f"    Fisher exact, {REF} vs each arm:")
    for a in ARMS:
        if a == REF: continue
        k, n = cond_rate[a][c]
        p = fisher_exact_2x2(kr, nr-kr, k, n-k)
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        print(f"      {a:<9} p={p:.4f} {star}")

# ------------------------------------------------------- failure taxonomy
print("\n" + "=" * 78)
print("FAILURE MODES on matched set (what happened when it did not succeed)")
print("=" * 78)
print(f"{'arm':<9}{'cond':<7}{'abstain':>9}{'wrongNum':>10}{'noSubmit':>10}"
      f"{'unparsed':>10}{'meanSteps':>11}{'targetCalls':>13}")
for a in ARMS:
    for c in CONDS:
        es = [ep[a][(t, c)] for t in matched[a] if (t, c) in ep[a]]
        fails = [e for e in es if not e.get("success")]
        ab = sum(1 for e in fails if e.get("abstained"))
        wn = sum(1 for e in fails if e.get("answer") is not None and not e.get("abstained"))
        ns = sum(1 for e in fails if e.get("answer") is None and not e.get("abstained")
                 and e.get("parsed_json"))
        up = sum(1 for e in fails if not e.get("parsed_json"))
        ms = sum(e.get("steps", 0) for e in es) / max(len(es), 1)
        tc = sum(e.get("target_calls", 0) for e in es) / max(len(es), 1)
        print(f"{a:<9}{c:<7}{ab:>9}{wn:>10}{ns:>10}{up:>10}{ms:>11.1f}{tc:>13.1f}")

# ------------------------------------------------- persistence / recovery cost
print("\n" + "=" * 78)
print("PERSISTENCE: calls to the injected tool, and step cost of recovery")
print("=" * 78)
print(f"{'arm':<9}{'ET tgt-calls':>14}{'EP tgt-calls':>14}{'ET-abstain':>12}"
      f"{'EP-abstain':>12}{'ET cost':>10}{'IT cost':>10}")
for a in ARMS:
    m = matched[a]
    et_tc = sum(ep[a][(t,"ET")].get("target_calls",0) for t in m)/max(len(m),1)
    epp_tc= sum(ep[a][(t,"EP")].get("target_calls",0) for t in m)/max(len(m),1)
    et_ab = sum(1 for t in m if ep[a][(t,"ET")].get("abstained"))
    ep_ab = sum(1 for t in m if ep[a][(t,"EP")].get("abstained"))
    def cost(c):
        d = [ep[a][(t,c)]["steps"] - ep[a][(t,"CLEAN")]["steps"]
             for t in m if ep[a][(t,c)].get("success")]
        return sum(d)/len(d) if d else float("nan")
    print(f"{a:<9}{et_tc:>14.2f}{epp_tc:>14.2f}{et_ab:>12}{ep_ab:>12}"
          f"{cost('ET'):>10.1f}{cost('IT'):>10.1f}")

print("\nKEY: ET-abstain > 0 means the arm gave up on a RECOVERABLE fault"
      "\n     (premature abandonment). EP tgt-calls high = perseveration on a"
      "\n     dead tool. These two columns carry the persistence story.")

# ------------------------------------------------------- prompt-sensitivity
print("\n" + "=" * 78)
print("PROMPT SENSITIVITY (Phase-2 reliable tasks that FAIL Phase-3 CLEAN)")
print("=" * 78)
for a in ARMS:
    rel = R[a].get("reliable", [])
    lost = [t for t in rel if not ep[a].get((t, "CLEAN"), {}).get("success")]
    print(f"  {a:<9} lost {len(lost):>2}/{len(rel)} = {len(lost)/max(len(rel),1):.0%}   {lost}")
print("\n  -> the resilience/report_failure prompt costs low-bit arms baseline"
      "\n     accuracy even with NO fault injected. Reportable in its own right.")
