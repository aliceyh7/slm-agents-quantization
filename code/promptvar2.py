#!/usr/bin/env python3
"""
Extended prompt-variant experiment with per-episode logging.
5 prompts (P0 original + P1/P2 existing variants + P3/P4 new variants,
written once, not tuned) x 3 levels (Q8_0, Q4_K_M, Q3_K_M).
Per level+prompt: 1 clean pass over the 20 A/B tasks, then ET at every
gold-path site on the clean-solvable set. A fresh run saves full episode
dicts, including final raw output for failure-mode analysis. Checkpoints per
cell; Drive backup. Existing aggregate-only checkpoints are skipped, so
delete or rename the results file before a complete log-producing rerun.
Needs phase1_suite.py, phase3_multi.py (logging version), promptvar.py.
"""
import json, os, time
import phase1_suite as P1
import phase3_multi as P3
import promptvar as V

ARMS = ["Q8_0", "Q4_K_M", "Q3_K_M"]
OUT  = "promptvar2_results.json"
P0_PROMPT = P3.system_prompt   # capture original before any override

def prompt_v3(task):
    lines = "\n".join("  " + P1.SIGS[t] for t in P1.FAMILY_TOOLS[task["family"]])
    return """You solve tasks by calling tools, one call at a time.

Rules:
1. The only tools that exist are listed below, plus submit_answer and
   report_failure. There is no calculator; do arithmetic yourself and round
   money to 2 decimals.
""" + lines + """
   submit_answer(amount)      -> ends the episode with your final number.
   report_failure(reason)     -> ends the episode; only for a permanently
                                 broken tool.
2. Reply with one JSON object per turn and nothing else. No prose, no
   markdown, no code fences. Example call:
   {"tool": "get_order", "args": {"order_id": "O11"}}
3. Do not repeat a call that already returned a result. If a call fails or a
   value looks wrong, retrying it or re-checking an earlier tool is allowed.
4. When you can compute the final amount, submit it as a plain number, not
   an expression:
   {"tool": "submit_answer", "args": {"amount": 123.45}}

Worked example on unrelated data (do not reuse these values):

""" + P1.EXAMPLES[task["family"]] + """

The example submits as soon as it has enough information. Do the same."""

def prompt_v4(task):
    lines = "\n".join("  " + P1.SIGS[t] for t in P1.FAMILY_TOOLS[task["family"]])
    return """Use the tools below to finish the task. Work step by step, one tool call
per turn.

""" + lines + """
  submit_answer(amount)      -> finish with your answer.
  report_failure(reason)     -> finish by reporting that a permanently broken
                                tool makes the task impossible.

Arithmetic is yours to do; no calculator tool exists. Round money to 2
decimals. Each reply must be exactly one JSON object, for example
{"tool": "get_order", "args": {"order_id": "O11"}}, with no other text.
Never repeat a successful call with the same arguments. A failed call, or a
suspicious value, may be retried or re-checked. Submit a literal number,
never an expression, the moment the answer can be computed:
{"tool": "submit_answer", "args": {"amount": 123.45}}

Example on other data (values not yours to reuse):

""" + P1.EXAMPLES[task["family"]] + """

Notice it stops gathering and submits once it has what it needs."""

PROMPTS = {"P0": lambda t: P0_PROMPT(t), "P1": V.prompt_v1, "P2": V.prompt_v2,
           "P3": prompt_v3, "P4": prompt_v4}

def main():
    tasks = {t["id"]: t for t in P1.gen_tasks() if t["family"] in ("A", "B")}
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm in ARMS:
        path = P1.fetch(arm)
        if not path: print("download failed", arm); continue
        proc, url = P1.serve(path)
        if not proc: print("server failed", arm); continue
        for pname, pfn in PROMPTS.items():
            key = f"{arm}:{pname}"
            if key in results: print(key, "done, skipping"); continue
            P3.system_prompt = pfn
            t0 = time.time()
            clean, eps = {}, []
            for tid, task in tasks.items():
                ep = P3.run_episode(url, task, "CLEAN")
                clean[str(tid)] = bool(ep.get("success"))
            solvable = [int(t) for t, ok in clean.items() if ok]
            for tid in solvable:
                for site in P3.sites(tasks[tid], "ET"):
                    e = P3.run_episode(url, tasks[tid], "ET", site)
                    e.update({"task_id": tid, "site": site})
                    eps.append(e)
            k = sum(bool(e.get("success")) for e in eps); n = len(eps)
            results[key] = {"clean": clean, "solvable": solvable,
                            "et_success": k, "et_n": n, "episodes": eps,
                            "minutes": round((time.time()-t0)/60, 1)}
            json.dump(results, open(OUT, "w"), indent=1)
            d = "/content/drive/MyDrive/quantrec"
            if os.path.isdir(d):
                import shutil; shutil.copy(OUT, d)
            print(f"  {key}: solvable {len(solvable)}/20  ET {k}/{n} = "
                  f"{k/max(n,1):.0%}  ({results[key]['minutes']} min)", flush=True)
        proc.kill(); proc.wait()

    print("\n" + "=" * 66)
    print("ET recovery by level x prompt (own clean-solvable sets)")
    print("=" * 66)
    hdr = f"{'':<9}" + "".join(f"{p:>10}" for p in PROMPTS)
    print(hdr)
    for arm in ARMS:
        row = f"{arm:<9}"
        for p in PROMPTS:
            r = results.get(f"{arm}:{p}")
            row += f"{'--':>10}" if not r else \
                   f"{str(r['et_success'])+'/'+str(r['et_n']):>10}"
        print(row)
    print("\nRun the analysis cell next: common task set, clustered")
    print("cross-prompt inference, and Q3 failure modes.")

if __name__ == "__main__":
    main()
