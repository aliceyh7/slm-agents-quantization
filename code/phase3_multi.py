#!/usr/bin/env python3
"""
Phase 3 - injected-error recovery across the quantization ladder.
Reuses the Phase 1 harness (imports phase1_suite) so task generation,
tool execution, parsing, and scoring are shared across phases.

Conditions per reliable task (ToolMaze taxonomy):
  CLEAN - no injection (re-run under the Phase 3 prompt, which adds report_failure)
  ET    - explicit-transient: first call to target tool errors; retries succeed
  IT    - implicit-transient: get_order's first reply has a corrupted customer_id;
          the fault surfaces one step later at get_customer; re-calling get_order fixes it
  EP    - explicit-permanent: target tool fails every call; correct = report_failure

Reliable set per arm: family A/B tasks solved on ALL clean Phase 2 reps
(read from phase1_results.json). Arms with < MIN_RELIABLE tasks are excluded
from the recovery comparison and reported as below-floor (fixed before data).

Env: ARMS=Q8_0,...   CONDS=CLEAN,ET,IT,EP   RESULTS=phase1_results.json
"""
import json, os, re, subprocess, time, urllib.request
import phase1_suite as P1

PORT         = P1.PORT
MAX_STEPS    = 14                     # +2 headroom: recovery costs steps
MIN_RELIABLE = 8
ARMS  = os.environ.get("ARMS", "Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K").split(",")
CONDS = os.environ.get("CONDS", "CLEAN,ET,IT,EP").split(",")
P2_RESULTS = os.environ.get("RESULTS", "phase1_results.json")

# ------------------------------------------------------------ reliable sets
def reliable_sets():
    d = json.load(open(P2_RESULTS))
    out = {}
    for arm, rec in d.items():
        if "episodes" not in rec:
            out[arm] = []
            continue
        per = {}
        for e in rec["episodes"]:
            if e["family"] in ("A", "B"):
                per.setdefault(e["task_id"], []).append(bool(e.get("correct")))
        out[arm] = sorted(t for t, v in per.items() if v and all(v))
    return out

# ------------------------------------------------------------ injection plan
def sites(task, cond):
    """All injection sites for this (task, cond). Multi-site raises n per arm
    ~2.5x at no design cost - every site is on the gold path, so the fault is
    always encountered."""
    fam = task["family"]
    if cond == "CLEAN":
        return [None]
    if cond in ("ET", "EP"):
        s = ["get_customer", "get_discount"]
        if fam == "B":
            s.append("get_shipping")
        return s
    if cond == "IT":
        # corrupt an upstream field so the fault surfaces one step downstream
        return ["get_order", "get_customer"]
    return [None]

def plan(task, cond, site):
    return site, (None if cond == "CLEAN" else cond)

TRANSIENT_ERR = {"error": "service temporarily unavailable - try again"}
PERMANENT_ERR = {"error": "service permanently unavailable"}
BAD_CUSTOMER  = "C000"
BAD_TIER      = "unknown"

def make_env(task, cond, site):
    """Stateful tool executor implementing the injection at `site`."""
    target, kind = plan(task, cond, site)
    calls = {}
    def exec_tool(name, args):
        calls[name] = calls.get(name, 0) + 1
        if kind == "EP" and name == target:
            return PERMANENT_ERR
        if kind == "ET" and name == target and calls[name] == 1:
            return TRANSIENT_ERR
        if kind == "IT" and name == target and calls[name] == 1:
            real = P1.tool_exec(task, name, args)
            if not isinstance(real, dict):
                return real
            fake = dict(real)
            if "customer_id" in fake:
                fake["customer_id"] = BAD_CUSTOMER          # breaks get_customer
            elif "tier" in fake:
                fake["tier"] = BAD_TIER                     # breaks get_discount
            return fake
        return P1.tool_exec(task, name, args)
    return exec_tool

# ------------------------------------------------------------ prompt
EXTRA_TOOL = ('  report_failure(reason)    -> use ONLY if a required tool is '
              'permanently broken and the task cannot be completed. Ends the episode.')
RESILIENCE = ("""
Tools can occasionally fail or return wrong data. If a call fails or an
observation looks inconsistent, you may retry it or re-call an earlier tool to
verify - a repeated call is allowed when the previous attempt returned an error.
Only call report_failure if repeated attempts show a tool is permanently broken.""")

def system_prompt(task):
    sp = P1.system_prompt(task)
    sp = sp.replace(
        "  submit_answer(amount)      -> ends the episode. THIS IS HOW YOU FINISH.",
        "  submit_answer(amount)      -> ends the episode. THIS IS HOW YOU FINISH.\n"
        + EXTRA_TOOL)
    marker = "There is NO calculator tool. Do all arithmetic yourself."
    sp = sp.replace(marker, marker + RESILIENCE)
    return sp

# ------------------------------------------------------------ episode
def run_episode(url, task, cond, site=None):
    env = make_env(task, cond, site)
    msgs = [{"role": "system", "content": system_prompt(task)},
            {"role": "user", "content": task["question"]}]
    called, strict_all, parsed_ok = [], True, True
    answer, expr_submit, abstained = None, False, False
    last_content = ""
    for _ in range(MAX_STEPS):
        body = {"messages": msgs, "temperature": 0.0, "top_k": 1,
                "seed": 0, "max_tokens": 200}
        req = urllib.request.Request(url + "/v1/chat/completions",
              data=json.dumps(body).encode(),
              headers={"Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=180))
        except Exception as e:
            return {"error": str(e), "cond": cond}
        content = resp["choices"][0]["message"].get("content", "")
        last_content = content
        obj, strict = P1.extract_json(content)
        strict_all &= strict
        if obj is None:
            m = re.search(r'"amount"\s*:\s*([-0-9+*/.()eE ]+)', content or "")
            v = P1.safe_arith(m.group(1)) if m and "submit_answer" in (content or "") else None
            if v is not None:
                answer, expr_submit, strict_all = v, True, False
                break
            parsed_ok = False
            break
        name = obj.get("tool")
        if name == "report_failure":
            abstained = True
            break
        if name == "submit_answer" or "answer" in obj:
            a = (obj.get("args") or {}).get("amount", obj.get("answer"))
            try:
                answer = float(a)
            except Exception:
                answer = P1.safe_arith(str(a))
                if answer is not None:
                    expr_submit = True
            break
        key = (name, json.dumps(obj.get("args") or {}, sort_keys=True))
        obs_raw = env(name, obj.get("args") or {})
        called.append(key)
        msgs.append({"role": "assistant", "content": content})
        left = MAX_STEPS - len(called)
        msgs.append({"role": "user",
                     "content": "OBSERVATION: " + json.dumps(obs_raw) +
                     "\n(" + str(left) + " tool calls remain before the episode "
                     "fails.) If you can compute the amount now, call "
                     "submit_answer. Otherwise call the next tool."})
    low = (last_content or "").lower()
    prose_abstain = (not parsed_ok) and any(
        k in low for k in ("cannot", "can't", "unable", "unavailable",
                           "permanently", "not possible", "报", "failure"))
    correct_num = answer is not None and abs(answer - task["gold"]) < 0.011
    if cond == "EP":
        success = abstained                      # strict: must call report_failure
        success_lenient = abstained or prose_abstain
    else:
        success = correct_num
        success_lenient = correct_num
    tool_names = [k[0] for k in called]
    target = site
    return {"cond": cond, "site": site,
            "task_id": task["id"], "family": task["family"],
            "success": success, "correct_num": correct_num,
            "abstained": abstained,
            "hallucinated": (cond == "EP" and answer is not None),
            "answer": answer, "gold": task["gold"], "steps": len(called),
            "repeat_calls": len(called) - len(set(called)),
            "target_calls": tool_names.count(target) if target else 0,
            "strict_json": strict_all, "parsed_json": parsed_ok,
            "expr_submit": expr_submit,
            "success_lenient": success_lenient,
            "prose_abstain": prose_abstain,
            "last_content": (last_content or "")[:400],
            "submitted": (answer is not None) or abstained}

# ------------------------------------------------------------ main
def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def main():
    tasks = {t["id"]: t for t in P1.gen_tasks()}
    rel = reliable_sets()
    commit = sh("cd llama.cpp && git rev-parse --short HEAD").stdout.strip()
    print("PHASE 3 (multi-site) - injected-error recovery   (llama.cpp", commit or "?", ")")
    for a in ARMS:
        print(f"  {a}: reliable A/B tasks = {len(rel.get(a, []))} {rel.get(a, [])}")

    results = {}
    if os.path.exists("phase3_multi_results.json"):
        results = json.load(open("phase3_multi_results.json"))
        done = [a for a in ARMS if a in results]
        if done:
            print("resuming - done:", ", ".join(done))

    for arm in ARMS:
        if arm in results:
            continue
        R = rel.get(arm, [])
        if len(R) < MIN_RELIABLE:
            print(f"\n=== ARM {arm}: EXCLUDED (reliable set {len(R)} < {MIN_RELIABLE}) ===")
            results[arm] = {"excluded": True, "reliable": R}
            json.dump(results, open("phase3_multi_results.json", "w"), indent=2)
            continue
        per = sum(len(sites(tasks[t], c)) for t in R for c in CONDS)
        print(f"\n=== ARM {arm}  ({len(R)} tasks, {per} episodes) ===")
        path = P1.fetch(arm)
        if not path:
            results[arm] = {"error": "download failed"}; continue
        proc, url = P1.serve(path)
        if not proc:
            results[arm] = {"error": "server failed"}; continue
        t0 = time.time()
        eps = []
        for tid in R:
            for cond in CONDS:
                for site in sites(tasks[tid], cond):
                    eps.append(run_episode(url, tasks[tid], cond, site))
            done_n = sum(e.get("success", False) for e in eps)
            print(f"    task {tid:>2}  cumulative success: {done_n}/{len(eps)}",
                  flush=True)
        proc.kill(); proc.wait()

        summ = {"episodes": eps, "reliable": R, "llama_cpp_commit": commit,
                "minutes": round((time.time() - t0) / 60, 1)}
        for c in CONDS:
            ce = [e for e in eps if e.get("cond") == c]
            n = len(ce)
            summ[c] = {
                "n": n,
                "success": sum(e.get("success", False) for e in ce),
                "abstained": sum(e.get("abstained", False) for e in ce),
                "hallucinated": sum(e.get("hallucinated", False) for e in ce),
                "mean_steps": round(sum(e.get("steps", 0) for e in ce) / max(n, 1), 2),
                "repeats": sum(e.get("repeat_calls", 0) for e in ce),
                "expr": sum(e.get("expr_submit", False) for e in ce),
            }
        results[arm] = summ
        json.dump(results, open("phase3_multi_results.json", "w"), indent=2)
        line = "  ".join(f"{c}:{summ[c]['success']}/{summ[c]['n']}" for c in CONDS)
        print(f"  {arm}: {line}  ({summ['minutes']} min)  [saved]")

    print("\n" + "=" * 78)
    print("PHASE 3 SUMMARY   (success = gold answer; for EP success = correct abstention)")
    print("=" * 78)
    hdr = f"{'quant':<9}" + "".join(f"{c:>12}" for c in CONDS) + f"{'EP-halluc':>11}{'falseAbst':>11}"
    print(hdr)
    for arm in ARMS:
        s = results.get(arm, {})
        if s.get("excluded"):
            print(f"{arm:<9}  EXCLUDED - below reliability floor "
                  f"({len(s.get('reliable', []))} reliable tasks)")
            continue
        if "error" in s:
            print(f"{arm:<9}  ERROR {s['error']}"); continue
        row = f"{arm:<9}"
        for c in CONDS:
            cc = s.get(c, {})
            row += f"{str(cc.get('success','-'))+'/'+str(cc.get('n','-')):>12}"
        ep_h = s.get("EP", {}).get("hallucinated", "-")
        cl_a = s.get("CLEAN", {}).get("abstained", "-")
        row += f"{ep_h:>11}{cl_a:>11}"
        print(row)
    print("\nPRR (ET/IT success given fault encountered) is the headline recovery metric.")
    print("Compare each arm to Q8_0 ON THE SHARED reliable tasks for the paired test.")

if __name__ == "__main__":
    main()
