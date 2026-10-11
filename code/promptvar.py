#!/usr/bin/env python3
"""
Prompt-sensitivity check for the ET result. Two reworded system prompts,
same tools/examples/semantics. Q8_0 vs Q4_K_M, CLEAN (1 rep, defines the
solvable set under that variant) then ET at all sites on that set.
Needs phase1_suite.py and phase3_multi.py alongside. ~30 min on L4.
"""
import json, math, os, time
import phase1_suite as P1
import phase3_multi as P3

ARMS = ["Q8_0", "Q4_K_M"]
OUT  = "promptvar_results.json"

def logC(n,k): return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
def fisher(a,b,c,d):
    n=a+b+c+d; r1=a+b; r2=c+d; c1=a+c
    p=lambda x: math.exp(logC(r1,x)+logC(r2,c1-x)-logC(n,c1))
    p0=p(a); lo=max(0,c1-r2); hi=min(r1,c1)
    return min(1.0,sum(p(x) for x in range(lo,hi+1) if p(x)<=p0*(1+1e-9)))

def tool_block(task):
    return "\n".join("  " + P1.SIGS[t] for t in P1.FAMILY_TOOLS[task["family"]])

def prompt_v1(task):
    return """You are an assistant that completes tasks by calling tools.

Available tools (there are no others):
""" + tool_block(task) + """
  submit_answer(amount)      -> finishes the task. Always end this way.
  report_failure(reason)     -> use only when a tool is permanently broken
                                and the task cannot be finished.

No calculator exists, so do any arithmetic yourself. Round money to 2 decimals.

Respond with exactly one JSON object each turn and nothing more. No prose, no
markdown, no code fences, one object only:

  {"tool": "get_order", "args": {"order_id": "O11"}}

Do not repeat a call whose result you already have. Tools sometimes fail or
return incorrect data; if a call fails or a value looks wrong, you may retry
it or re-check an earlier tool. Once the information in hand is enough,
compute the amount and call submit_answer with a plain number, never an
expression:

  {"tool": "submit_answer", "args": {"amount": 123.45}}

A worked example on unrelated data (do not reuse these values):

""" + P1.EXAMPLES[task["family"]] + """

Note how the example finishes with submit_answer as soon as it has what it
needs. Do the same."""

def prompt_v2(task):
    return """Complete the task below using tool calls.

Tools you may call, and only these:
""" + tool_block(task) + """
  submit_answer(amount)      -> submits the final number and ends the episode.
  report_failure(reason)     -> ends the episode; only for a tool that is
                                permanently broken.

Arithmetic is your job; there is no calculator tool. Round money to 2 decimals.

Every reply must be a single JSON object with no other text:

  {"tool": "get_order", "args": {"order_id": "O11"}}

Never write more than one object per reply, and never call a tool again with
arguments you already used successfully. Occasionally a tool errors or gives
bad data. Retrying the call, or re-checking an earlier one, is allowed in that
case. When you can compute the final amount, submit it as a literal number:

  {"tool": "submit_answer", "args": {"amount": 123.45}}

Example episode on different data (values are not yours to reuse):

""" + P1.EXAMPLES[task["family"]] + """

The example stops gathering and submits once it has enough. Follow that
pattern."""

VARIANTS = {"v1": prompt_v1, "v2": prompt_v2}

def main():
    tasks = {t["id"]: t for t in P1.gen_tasks() if t["family"] in ("A","B")}
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm in ARMS:
        path = P1.fetch(arm)
        if not path: print("download failed", arm); continue
        proc, url = P1.serve(path)
        if not proc: print("server failed", arm); continue
        for vname, vfn in VARIANTS.items():
            key = f"{arm}:{vname}"
            if key in results: print(key, "done, skipping"); continue
            P3.system_prompt = lambda task, _f=vfn: _f(task)
            t0 = time.time()
            solvable = []
            for tid, task in tasks.items():
                ep = P3.run_episode(url, task, "CLEAN")
                if ep.get("success"): solvable.append(tid)
            eps = []
            for tid in solvable:
                for site in P3.sites(tasks[tid], "ET"):
                    e = P3.run_episode(url, tasks[tid], "ET", site)
                    eps.append({"task_id": tid, "site": site,
                                "success": bool(e.get("success"))})
            k = sum(e["success"] for e in eps); n = len(eps)
            results[key] = {"solvable": solvable, "et_success": k, "et_n": n,
                            "minutes": round((time.time()-t0)/60,1)}
            json.dump(results, open(OUT,"w"), indent=2)
            d = "/content/drive/MyDrive/quantrec"
            if os.path.isdir(d):
                import shutil; shutil.copy(OUT, d)
            print(f"  {key}: clean-solvable {len(solvable)}/20  "
                  f"ET {k}/{n} = {k/max(n,1):.0%}  ({results[key]['minutes']} min)")
        proc.kill(); proc.wait()

    print("\n" + "="*60)
    print("PROMPT-SENSITIVITY: ET recovery, Q8_0 vs Q4_K_M per variant")
    print("="*60)
    for vname in VARIANTS:
        a = results.get(f"Q8_0:{vname}"); b = results.get(f"Q4_K_M:{vname}")
        if a and b:
            p = fisher(a["et_success"], a["et_n"]-a["et_success"],
                       b["et_success"], b["et_n"]-b["et_success"])
            print(f"  {vname}:  Q8_0 {a['et_success']}/{a['et_n']} "
                  f"({a['et_success']/max(a['et_n'],1):.0%})   "
                  f"Q4_K_M {b['et_success']}/{b['et_n']} "
                  f"({b['et_success']/max(b['et_n'],1):.0%})   p={p:.2e}")
    print("\noriginal prompt:  Q8_0 39/47 (83%)   Q4_K_M 19/47 (40%)")

if __name__ == "__main__":
    main()
