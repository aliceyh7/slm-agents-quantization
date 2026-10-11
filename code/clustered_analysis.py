"""Task-clustered analysis of the same-prompt-matched ET contrast.
Treats tasks, not episodes, as the unit of analysis."""
import json, math, random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
with (DATA_DIR / "phase3_multi_results.json").open() as f:
    R = json.load(f)
HIGH, LOW = ["Q8_0","Q6_K","Q5_K_M"], ["Q4_K_M","Q3_K_M"]

rate, clean = {}, {}
for arm in HIGH+LOW:
    clean[arm] = {e["task_id"] for e in R[arm]["episodes"]
                  if e["cond"] == "CLEAN" and e.get("success")}
    per = defaultdict(lambda: [0,0])
    for e in R[arm]["episodes"]:
        if e["cond"]=="ET":
            per[e["task_id"]][0] += bool(e["success"]); per[e["task_id"]][1] += 1
    rate[arm] = {t: k/n for t,(k,n) in per.items()}

shared = sorted(set.intersection(*[clean[a] for a in HIGH+LOW]))
diffs = [(t, sum(rate[a][t] for a in HIGH)/3, sum(rate[a][t] for a in LOW)/2)
         for t in shared]
pos = sum(1 for _,h,l in diffs if h>l); neg = sum(1 for _,h,l in diffs if h<l)
n = pos+neg
p_sign = min(1.0, sum(math.comb(n,i) for i in range(0,min(pos,neg)+1))*2/2**n)
print(f"tasks={len(shared)}  favor-high={pos}  favor-low={neg}  sign-test p={p_sign:.2e}")
print(f"mean task rate: high {sum(h for _,h,_ in diffs)/len(diffs):.2f}  "
      f"low {sum(l for _,_,l in diffs)/len(diffs):.2f}")
random.seed(0); boot=[]
for _ in range(20000):
    samp=[random.choice(shared) for _ in shared]
    boot.append(sum(sum(rate[a][t] for a in HIGH)/3 - sum(rate[a][t] for a in LOW)/2
                    for t in samp)/len(samp))
boot.sort()
print(f"cluster-bootstrap 95% CI: [{boot[500]:.2f}, {boot[19499]:.2f}]")
