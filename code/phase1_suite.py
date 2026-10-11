#!/usr/bin/env python3
"""
Phase 1 - task suite for the quantization x agentic-recovery study.

40 deterministic tool-chain tasks in four families of rising difficulty,
generated from a fixed seed. Gold answers and minimal trajectories are computed
programmatically, so scoring is exact.

  A  depth-3 chain     order -> customer -> discount            (10 tasks)
  B  depth-4 chain     + shipping fee by zone                   (10 tasks)
  C  depth-5 chain     two orders, summed, then discount        (10 tasks)
  D  distractors       depth-3/4 chain + 3 irrelevant tools     (10 tasks)

GATE (pre-registered): the reference arm (Q8_0) must solve >= 70% at temp 0.

Metrics per episode: correct, steps used vs minimal, repeated calls,
strict/parsed JSON. Results saved incrementally per arm to phase1_results.json.

Env knobs:  ARMS=Q8_0,Q4_K_M   N_TASKS=10   N_REPEATS=2
"""
import ast as _ast
import json, os, random, re, shutil, subprocess, time, urllib.request

def safe_arith(expr):
    """Evaluate a plain arithmetic expression (numbers, + - * / and parens) safely."""
    try:
        def ev(n):
            if isinstance(n, _ast.Expression):
                return ev(n.body)
            if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            if isinstance(n, _ast.BinOp) and isinstance(
                    n.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div)):
                a, b = ev(n.left), ev(n.right)
                return {_ast.Add: a + b, _ast.Sub: a - b,
                        _ast.Mult: a * b, _ast.Div: a / b}[type(n.op)]
            if isinstance(n, _ast.UnaryOp) and isinstance(n.op, (_ast.USub, _ast.UAdd)):
                v = ev(n.operand)
                return -v if isinstance(n.op, _ast.USub) else v
            raise ValueError("disallowed node")
        return ev(_ast.parse(expr.strip(), mode="eval"))
    except Exception:
        return None


MODEL_DIR    = "models"
PORT         = 8090
KEEP_WEIGHTS = True
SEED         = 20260826
N_REPEATS    = int(os.environ.get("N_REPEATS", "3"))
N_TASKS      = int(os.environ.get("N_TASKS", "40"))   # subset for smoke tests
MAX_STEPS    = 12

REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
STEM = "Meta-Llama-3.1-8B-Instruct"
ARMS = os.environ.get("ARMS", "Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K").split(",")

# Revision pinning: if pinned_revisions.json exists (written once, at the
# start of a run, by resolving each repo's current commit sha), every
# download in this run and any rerun uses that exact frozen revision
# instead of silently tracking whatever "main" happens to point to later.
_REV_FILE = "pinned_revisions.json"
REPO_REVISIONS = {}
if os.path.exists(_REV_FILE):
    try:
        REPO_REVISIONS = json.load(open(_REV_FILE))
    except Exception:
        REPO_REVISIONS = {}

FAMILY_TOOLS = {
    "A": ["get_order", "get_customer", "get_discount"],
    "B": ["get_order", "get_customer", "get_discount", "get_shipping"],
    "C": ["get_order", "get_customer", "get_discount"],
    "D": ["get_order", "get_customer", "get_discount", "get_shipping",
          "get_inventory", "get_weather"],
}

TIERS = ["gold", "silver", "bronze", "basic"]
ZONES = ["north", "south", "east", "west"]

# ------------------------------------------------------------ task generator
def gen_tasks():
    rng = random.Random(SEED)
    tasks = []
    fam_counts = {"A": 10, "B": 10, "C": 10, "D": 10}
    tid = 0
    for fam, n in fam_counts.items():
        for _ in range(n):
            tid += 1
            rates = {t: round(rng.choice([10, 20, 25, 50]) / 100, 2)
                     for t in TIERS}
            ship  = {z: round(rng.choice([5, 10, 15, 20]) * 1.0, 2)
                     for z in ZONES}
            orders, customers = {}, {}
            def mk_order(cid):
                oid = f"O{rng.randint(10, 99)}{tid}"
                orders[oid] = {"customer_id": cid,
                               "subtotal": float(rng.choice(
                                   [100, 200, 300, 400, 500, 600, 700, 800])),
                               "zone": rng.choice(ZONES)}
                return oid
            def mk_cust():
                cid = f"C{rng.randint(10, 99)}{tid}"
                customers[cid] = {"tier": rng.choice(TIERS)}
                return cid

            if fam == "A":
                cid = mk_cust(); oid = mk_order(cid)
                o = orders[oid]; t = customers[cid]["tier"]
                gold = round(o["subtotal"] * (1 - rates[t]), 2)
                q = (f"For order {oid}: find the ordering customer's tier and that tier's "
                     f"discount rate, then compute the amount the customer must PAY after "
                     f"the discount is deducted from the subtotal. Submit that amount.")
                minimal = 3
            elif fam == "B":
                cid = mk_cust(); oid = mk_order(cid)
                o = orders[oid]; t = customers[cid]["tier"]
                gold = round(o["subtotal"] * (1 - rates[t]) + ship[o["zone"]], 2)
                q = (f"For order {oid}: apply the customer's tier discount to the subtotal, "
                     f"then ADD the shipping fee for the order's zone. Submit the total the "
                     f"customer must pay.")
                minimal = 4
            elif fam == "C":
                cid = mk_cust(); o1 = mk_order(cid); o2 = mk_order(cid)
                t = customers[cid]["tier"]
                s = orders[o1]["subtotal"] + orders[o2]["subtotal"]
                gold = round(s * (1 - rates[t]), 2)
                q = (f"Orders {o1} and {o2} belong to the same customer. Sum their subtotals, "
                     f"then apply the customer's tier discount to the combined amount. Submit "
                     f"the discounted total.")
                minimal = 4
            else:  # D - same as A or B but with distractor tools available
                cid = mk_cust(); oid = mk_order(cid)
                o = orders[oid]; t = customers[cid]["tier"]
                if rng.random() < 0.5:
                    gold = round(o["subtotal"] * (1 - rates[t]), 2); minimal = 3
                    q = (f"For order {oid}: compute the amount the customer must PAY after "
                         f"their tier discount is applied to the subtotal. Submit it. "
                         f"Ignore any information you do not need.")
                else:
                    gold = round(o["subtotal"] * (1 - rates[t]) + ship[o["zone"]], 2); minimal = 4
                    q = (f"For order {oid}: apply the tier discount to the subtotal and add "
                         f"the zone shipping fee. Submit the total. Ignore any information "
                         f"you do not need.")
            tasks.append({"id": tid, "family": fam, "question": q,
                          "orders": orders, "customers": customers,
                          "rates": rates, "ship": ship,
                          "gold": gold, "minimal_steps": minimal,
                          "distractors": fam == "D"})
    return tasks[:N_TASKS] if N_TASKS < 40 else tasks

# --------------------------------------------------------------- tool layer
def tool_exec(task, name, args):
    allowed = FAMILY_TOOLS[task["family"]]
    if name not in allowed:
        return {"error": "no such tool: " + str(name) + ". Available tools: "
                + ", ".join(allowed) + ", submit_answer. Do arithmetic yourself."}
    try:
        if name == "get_order":
            return task["orders"][args["order_id"]]
        if name == "get_customer":
            return task["customers"][args["customer_id"]]
        if name == "get_discount":
            return {"rate": task["rates"][args["tier"]]}
        if name == "get_shipping":
            return {"fee": task["ship"][args["zone"]]}
        if name == "get_inventory":
            return {"in_stock": True, "warehouse": "W4"}      # distractor
        if name == "get_weather":
            return {"forecast": "clear", "temp_c": 21}        # distractor
    except Exception as e:
        return {"error": f"lookup failed: {e}"}
    return {"error": "unknown tool"}


EXAMPLES = {
"A": """{"tool": "get_order", "args": {"order_id": "OX1"}}
OBSERVATION: {"customer_id": "CX1", "subtotal": 100.0, "zone": "east"}
{"tool": "get_customer", "args": {"customer_id": "CX1"}}
OBSERVATION: {"tier": "basic"}
{"tool": "get_discount", "args": {"tier": "basic"}}
OBSERVATION: {"rate": 0.1}
{"tool": "submit_answer", "args": {"amount": 90.0}}""",

"B": """{"tool": "get_order", "args": {"order_id": "OX1"}}
OBSERVATION: {"customer_id": "CX1", "subtotal": 100.0, "zone": "east"}
{"tool": "get_customer", "args": {"customer_id": "CX1"}}
OBSERVATION: {"tier": "basic"}
{"tool": "get_discount", "args": {"tier": "basic"}}
OBSERVATION: {"rate": 0.1}
{"tool": "get_shipping", "args": {"zone": "east"}}
OBSERVATION: {"fee": 20.0}
{"tool": "submit_answer", "args": {"amount": 110.0}}""",

"C": """{"tool": "get_order", "args": {"order_id": "OX1"}}
OBSERVATION: {"customer_id": "CX1", "subtotal": 100.0, "zone": "east"}
{"tool": "get_order", "args": {"order_id": "OX2"}}
OBSERVATION: {"customer_id": "CX1", "subtotal": 300.0, "zone": "east"}
{"tool": "get_customer", "args": {"customer_id": "CX1"}}
OBSERVATION: {"tier": "basic"}
{"tool": "get_discount", "args": {"tier": "basic"}}
OBSERVATION: {"rate": 0.1}
{"tool": "submit_answer", "args": {"amount": 360.0}}""",

"D": """{"tool": "get_order", "args": {"order_id": "OX1"}}
OBSERVATION: {"customer_id": "CX1", "subtotal": 100.0, "zone": "east"}
{"tool": "get_customer", "args": {"customer_id": "CX1"}}
OBSERVATION: {"tier": "basic"}
{"tool": "get_discount", "args": {"tier": "basic"}}
OBSERVATION: {"rate": 0.1}
{"tool": "get_shipping", "args": {"zone": "east"}}
OBSERVATION: {"fee": 20.0}
{"tool": "submit_answer", "args": {"amount": 110.0}}""",
}

SIGS = {
 "get_order":     'get_order(order_id)        -> {"customer_id": str, "subtotal": float, "zone": str}',
 "get_customer":  'get_customer(customer_id)  -> {"tier": str}',
 "get_discount":  'get_discount(tier)         -> {"rate": float}',
 "get_shipping":  'get_shipping(zone)         -> {"fee": float}',
 "get_inventory": 'get_inventory(order_id)    -> stock info',
 "get_weather":   'get_weather(zone)          -> weather info',
}

def system_prompt(task):
    fam = task["family"]
    lines = "\n".join("  " + SIGS[t] for t in FAMILY_TOOLS[fam])
    n_calls = EXAMPLES[fam].count('{"tool"') - 1
    return """You are a tool-using agent solving a task step by step.

These are the ONLY tools that exist:
""" + lines + """
  submit_answer(amount)      -> ends the episode. THIS IS HOW YOU FINISH.

There is NO calculator tool. Do all arithmetic yourself.

Each turn, write exactly ONE tool call - a single JSON object - and then STOP
writing. The environment will reply with an OBSERVATION line. NEVER write
OBSERVATION lines yourself, and NEVER write more than one tool call per message:
each call after the first would be ignored.

Gather ONLY the observations you need, then STOP and submit. Do not keep calling
tools once you can compute the answer. Calling a tool you have already called
with the same arguments is an error.

The amount you submit must be a single literal number you have already computed,
like 123.45 - NEVER an arithmetic expression. {"amount": 300 * 0.8} is invalid;
compute it and write {"amount": 240.0}.

This task needs about """ + str(n_calls) + """ tool call(s) before you submit.

WORKED EXAMPLE on DIFFERENT data (do not reuse these values):

""" + EXAMPLES[fam] + """

Notice the example ends with submit_answer as soon as the needed observations
are in hand. Your episode must end the same way."""

# ------------------------------------------------------------- json parsing
def extract_json(text):
    """Return (obj, strict). strict=True iff the reply was exactly one clean JSON
    object. If the model wrote several objects or extra text, take the FIRST
    complete object (its first action) and mark non-strict."""
    t = (text or "").strip()
    try:
        return json.loads(t), True
    except Exception:
        pass
    t2 = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    i = t2.find("{")
    while i >= 0:
        try:
            obj, end = json.JSONDecoder().raw_decode(t2[i:])
            if isinstance(obj, dict):
                return obj, False
        except Exception:
            pass
        i = t2.find("{", i + 1)
    return None, False

# --------------------------------------------------------------- episodes
def run_episode(url, task):
    msgs = [{"role": "system", "content": system_prompt(task)},
            {"role": "user", "content": task["question"]}]
    called, strict_all, parsed_ok = [], True, True
    answer, expr_submit = None, False
    for _ in range(MAX_STEPS):
        body = {"messages": msgs, "temperature": 0.0, "top_k": 1,
                "seed": 0, "max_tokens": 200}
        req = urllib.request.Request(url + "/v1/chat/completions",
              data=json.dumps(body).encode(),
              headers={"Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=180))
        except Exception as e:
            return {"error": str(e), "called": called}
        content = resp["choices"][0]["message"].get("content", "")
        obj, strict = extract_json(content)
        strict_all &= strict
        if obj is None:
            # rescue: submit_answer whose amount is an arithmetic expression
            m = re.search(r'"amount"\s*:\s*([-0-9+*/.()eE ]+)', content or "")
            v = safe_arith(m.group(1)) if m and "submit_answer" in (content or "") else None
            if v is not None:
                answer, expr_submit, strict_all = v, True, False
                break
            parsed_ok = False
            break
        name = obj.get("tool")
        if name == "submit_answer" or "answer" in obj:
            a = (obj.get("args") or {}).get("amount", obj.get("answer"))
            try:
                answer = float(a)
            except Exception:
                answer = safe_arith(str(a))
                if answer is not None:
                    expr_submit = True
            break
        key = (name, json.dumps(obj.get("args") or {}, sort_keys=True))
        repeated = key in called
        called.append(key)
        msgs.append({"role": "assistant", "content": content})
        if repeated:
            obs = json.dumps({"error": "You already called " + str(name) +
                " with those arguments and received the result above. Repeating "
                "calls is not allowed. If your observations are sufficient, "
                "compute the amount and call submit_answer now."})
        else:
            obs = json.dumps(tool_exec(task, name, obj.get("args") or {}))
        left = MAX_STEPS - len(called)
        msgs.append({"role": "user",
                     "content": "OBSERVATION: " + obs +
                     "\n(" + str(left) + " tool calls remain before the episode "
                     "fails.) If you can compute the amount now, call "
                     "submit_answer. Otherwise call the next tool."})
    correct = answer is not None and abs(answer - task["gold"]) < 0.011
    return {"correct": correct, "answer": answer, "gold": task["gold"],
            "steps": len(called), "minimal_steps": task["minimal_steps"],
            "repeat_calls": len(called) - len(set(called)),
            "distractor_calls": sum(1 for n, _ in called
                                    if n in ("get_inventory", "get_weather")),
            "strict_json": strict_all, "parsed_json": parsed_ok,
            "expr_submit": expr_submit,
            "submitted": answer is not None}

# --------------------------------------------------------------- plumbing
def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

def fetch(tag):
    os.makedirs(MODEL_DIR, exist_ok=True)
    fn = f"{STEM}-{tag}.gguf"
    path = os.path.join(MODEL_DIR, fn)
    if os.path.exists(path):
        return path
    print(f"    downloading {fn} ...")
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=REPO, filename=fn, local_dir=MODEL_DIR,
                                revision=REPO_REVISIONS.get(REPO))
    except Exception as e:
        print(f"    FAILED {fn}: {e}")
        return None

def serve(path):
    sh("pkill -f llama-server"); time.sleep(2)
    proc = subprocess.Popen(
        ["llama.cpp/build/bin/llama-server", "-m", path, "-ngl", "99",
         "-c", "8192", "--port", str(PORT), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{PORT}"
    for _ in range(180):
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            return proc, url
        except Exception:
            time.sleep(1)
    proc.kill()
    return None, None

def main():
    if not os.path.exists("llama.cpp/build/bin/llama-server"):
        print("llama-server missing - build it first (see phase0)."); return
    tasks = gen_tasks()
    print(f"PHASE 1: {len(tasks)} tasks x {N_REPEATS} reps, arms: {', '.join(ARMS)}")
    r = sh("nvidia-smi --query-gpu=name --format=csv,noheader")
    print("GPU:", r.stdout.strip() or "NONE")

    all_results = {}
    if os.path.exists("phase1_results.json"):
        all_results = json.load(open("phase1_results.json"))
        done = [a for a in ARMS if a in all_results]
        if done:
            print("resuming - already done:", ", ".join(done))

    for tag in ARMS:
        if tag in all_results:
            continue
        print(f"\n=== ARM {tag} ===")
        path = fetch(tag)
        if not path:
            all_results[tag] = {"error": "download failed"}; continue
        proc, url = serve(path)
        if not proc:
            all_results[tag] = {"error": "server failed"}; continue
        t0 = time.time()
        episodes = []
        for task in tasks:
            for rep in range(N_REPEATS):
                ep = run_episode(url, task)
                ep.update({"task_id": task["id"], "family": task["family"], "rep": rep})
                episodes.append(ep)
            nc = sum(e.get("correct", False) for e in episodes)
            print(f"    task {task['id']:>2} ({task['family']})  "
                  f"running correct: {nc}/{len(episodes)}", flush=True)
        proc.kill(); proc.wait()
        if not KEEP_WEIGHTS:
            try: os.remove(path)
            except Exception: pass

        n = len(episodes)
        summ = {
            "episodes": episodes,
            "n": n,
            "correct": sum(e.get("correct", False) for e in episodes),
            "submitted": sum(e.get("submitted", False) for e in episodes),
            "strict_json": sum(e.get("strict_json", False) for e in episodes),
            "parsed_json": sum(e.get("parsed_json", False) for e in episodes),
            "repeat_calls_total": sum(e.get("repeat_calls", 0) for e in episodes),
            "expr_submits": sum(e.get("expr_submit", False) for e in episodes),
            "distractor_calls_total": sum(e.get("distractor_calls", 0) for e in episodes),
            "mean_steps": round(sum(e.get("steps", 0) for e in episodes) / max(n,1), 2),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        by_fam = {}
        for e in episodes:
            f = e["family"]; by_fam.setdefault(f, [0, 0])
            by_fam[f][0] += bool(e.get("correct")); by_fam[f][1] += 1
        summ["by_family"] = {f: f"{c}/{t}" for f, (c, t) in sorted(by_fam.items())}
        all_results[tag] = summ
        json.dump(all_results, open("phase1_results.json", "w"), indent=2)
        print(f"  {tag}: correct {summ['correct']}/{n}  by family {summ['by_family']}  "
              f"({summ['minutes']} min)  [saved]")

    print("\n" + "=" * 76)
    print("PHASE 1 SUMMARY")
    print("=" * 76)
    print(f"{'quant':<9}{'correct':>10}{'submit':>8}{'strict':>8}{'repeats':>9}"
          f"{'distract':>10}{'expr':>6}{'steps':>7}  by_family")
    for tag in ARMS:
        s = all_results.get(tag, {})
        if "error" in s:
            print(f"{tag:<9}  ERROR: {s['error']}"); continue
        print(f"{tag:<9}{str(s['correct'])+'/'+str(s['n']):>10}"
              f"{s['submitted']:>8}{s['strict_json']:>8}{s['repeat_calls_total']:>9}"
              f"{s['distractor_calls_total']:>10}{s.get('expr_submits',0):>6}"
              f"{s['mean_steps']:>7}  {s['by_family']}")

    ref = all_results.get("Q8_0", {})
    if ref and "error" not in ref:
        rate = ref["correct"] / max(ref["n"], 1)
        print(f"\nGATE (pre-registered): reference >= 70% -> "
              f"{'PASS' if rate >= 0.70 else 'FAIL'} ({rate:.0%})")
        if rate >= 0.995:
            print("  NOTE: reference at ceiling. Fine for the gate, but if the WHOLE ladder")
            print("  is at ceiling, Phase 3's error injection carries all the discrimination.")

if __name__ == "__main__":
    main()
