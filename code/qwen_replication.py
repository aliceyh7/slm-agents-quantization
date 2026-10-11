#!/usr/bin/env python3
"""
Cross-model replication of the headline ET result on Qwen2.5-7B-Instruct.
Q8_0 vs Q4_K_M, CLEAN + ET conditions, same tasks, same prompts, same harness.
Needs phase1_suite.py and phase3_multi.py in the same directory. ~30-40 min on L4.
"""
import json, math, os, time
import phase1_suite as P1
import phase3_multi as P3

# point the shared harness at the Qwen GGUF repo
P1.REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
P1.STEM = "Qwen2.5-7B-Instruct"

ARMS   = ["Q8_0", "Q4_K_M"]
N_CLEAN_REPS = 3
OUT = "qwen_replication_results.json"

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)

def logC(n,k): return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
def fisher(a,b,c,d):
    n=a+b+c+d; r1=a+b; r2=c+d; c1=a+c
    p=lambda x: math.exp(logC(r1,x)+logC(r2,c1-x)-logC(n,c1))
    p0=p(a); lo=max(0,c1-r2); hi=min(r1,c1)
    return min(1.0,sum(p(x) for x in range(lo,hi+1) if p(x)<=p0*(1+1e-9)))

def main():
    tasks = {t["id"]: t for t in P1.gen_tasks() if t["family"] in ("A", "B")}
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}

    for arm in ARMS:
        if arm in results:
            print(f"{arm}: already done, skipping"); continue
        print(f"\n=== QWEN {arm} ===")
        path = P1.fetch(arm)
        if not path:
            print("download failed - check filename on HF repo"); continue
        proc, url = P1.serve(path)
        if not proc:
            print("server failed"); continue
        t0 = time.time()

        # 1) reliable set: N_CLEAN_REPS clean passes under the Phase-3 prompt
        clean_ok = {}
        for tid, task in tasks.items():
            oks = []
            for _ in range(N_CLEAN_REPS):
                ep = P3.run_episode(url, task, "CLEAN")
                oks.append(bool(ep.get("success")))
            clean_ok[tid] = all(oks)
        R = sorted(t for t, ok in clean_ok.items() if ok)
        print(f"    reliable set: {len(R)}/{len(tasks)}  {R}")

        # 2) ET at every gold-path site on the reliable set
        eps = []
        for tid in R:
            for site in P3.sites(tasks[tid], "ET"):
                ep = P3.run_episode(url, tasks[tid], "ET", site)
                ep.update({"task_id": tid, "site": site})
                eps.append(ep)
            k = sum(e.get("success", False) for e in eps)
            print(f"    task {tid:>2}  ET running: {k}/{len(eps)}", flush=True)
        proc.kill(); proc.wait()

        k = sum(e.get("success", False) for e in eps); n = len(eps)
        results[arm] = {"reliable": R, "et_success": k, "et_n": n,
                        "episodes": eps,
                        "minutes": round((time.time()-t0)/60, 1)}
        json.dump(results, open(OUT, "w"), indent=2)
        d = "/content/drive/MyDrive/quantrec"
        if os.path.isdir(d):
            import shutil; shutil.copy(OUT, d)
        lo, hi = wilson(k, n)
        print(f"  {arm}: ET {k}/{n} = {k/max(n,1):.0%} [{lo:.0%},{hi:.0%}]  "
              f"({results[arm]['minutes']} min)  [saved]")

    print("\n" + "=" * 60)
    print("QWEN2.5-7B REPLICATION - ET recovery, Q8_0 vs Q4_K_M")
    print("=" * 60)
    if all(a in results for a in ARMS):
        k8, n8 = results["Q8_0"]["et_success"], results["Q8_0"]["et_n"]
        k4, n4 = results["Q4_K_M"]["et_success"], results["Q4_K_M"]["et_n"]
        p = fisher(k8, n8-k8, k4, n4-k4)
        for a in ARMS:
            k, n = results[a]["et_success"], results[a]["et_n"]
            lo, hi = wilson(k, n)
            print(f"  {a:<8} {k}/{n} = {k/max(n,1):.0%}  [{lo:.0%}, {hi:.0%}]")
        print(f"  Fisher exact p = {p:.2e}")
        print("\n  Llama-3.1-8B comparison: Q8_0 83% vs Q4_K_M 40%.")
        print("  If Qwen shows the same direction, the finding generalizes across")
        print("  families; if not, that is a reportable boundary condition.")

if __name__ == "__main__":
    main()
