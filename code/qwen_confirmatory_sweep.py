#!/usr/bin/env python3
"""Five-prompt Qwen2.5-7B sweep with complete episode logs."""

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import phase1_suite as P1
import phase3_multi as P3
import promptvar as V


P1.REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
P1.STEM = "Qwen2.5-7B-Instruct"
QWEN_MODEL_REVISION = os.environ.get("QWEN_MODEL_REVISION")
if QWEN_MODEL_REVISION:
    P1.REPO_REVISIONS[P1.REPO] = QWEN_MODEL_REVISION

ARMS = os.environ.get("SWEEP_ARMS", "Q8_0,Q4_K_M").split(",")
RUNS = int(os.environ.get("SWEEP_RUNS", "3"))
CLEAN_REPEATS = int(os.environ.get("CLEAN_REPEATS", "1"))
FAULT_REPEATS = int(os.environ.get("FAULT_REPEATS", "1"))
OUT = Path("../data/qwen_confirmatory_sweep_results.json")
DRIVE_DIR = Path("/content/drive/MyDrive/quantrec")

P0_PROMPT = P3.system_prompt


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


PROMPTS = {
    "P0": lambda task: P0_PROMPT(task),
    "P1": V.prompt_v1,
    "P2": V.prompt_v2,
    "P3": prompt_v3,
    "P4": prompt_v4,
}


def atomic_save(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, OUT)
    if DRIVE_DIR.is_dir():
        shutil.copy2(OUT, DRIVE_DIR / OUT.name)


def main():
    tasks = {t["id"]: t for t in P1.gen_tasks() if t["family"] in ("A", "B")}
    config = {
        "arms": ARMS,
        "prompts": list(PROMPTS),
        "runs": RUNS,
        "clean_repeats": CLEAN_REPEATS,
        "fault_repeats": FAULT_REPEATS,
        "task_ids": sorted(tasks),
        "qwen_model_revision": QWEN_MODEL_REVISION,
        "llama_cpp_commit": os.environ.get("LLAMA_CPP_COMMIT"),
    }
    if OUT.exists():
        payload = json.loads(OUT.read_text())
        if payload.get("config") != config:
            raise RuntimeError(
                "Existing confirmatory results use a different configuration. "
                "Rename the file or restore the recorded environment before resuming."
            )
    else:
        payload = {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "cells": {},
        }
        atomic_save(payload)

    for arm in ARMS:
        model_path = P1.fetch(arm)
        if not model_path:
            raise RuntimeError(f"Model download failed for {arm}")
        for prompt_name, prompt_fn in PROMPTS.items():
            P3.system_prompt = prompt_fn
            for run in range(RUNS):
                key = f"{arm}:{prompt_name}:R{run}"
                if key in payload["cells"]:
                    print(key, "complete; skipping")
                    continue

                proc, url = P1.serve(model_path)
                if not proc:
                    raise RuntimeError(f"llama-server failed for {key}")
                started = time.time()
                clean_episodes = []
                fault_episodes = []
                try:
                    clean_by_task = {tid: [] for tid in tasks}
                    for clean_repeat in range(CLEAN_REPEATS):
                        for tid, task in tasks.items():
                            episode = P3.run_episode(url, task, "CLEAN")
                            episode.update({
                                "task_id": tid,
                                "run": run,
                                "clean_repeat": clean_repeat,
                            })
                            clean_episodes.append(episode)
                            clean_by_task[tid].append(bool(episode.get("success")))

                    stable_tasks = sorted(
                        tid for tid, outcomes in clean_by_task.items()
                        if len(outcomes) == CLEAN_REPEATS and all(outcomes)
                    )
                    for fault_repeat in range(FAULT_REPEATS):
                        for tid in stable_tasks:
                            for site in P3.sites(tasks[tid], "ET"):
                                episode = P3.run_episode(url, tasks[tid], "ET", site)
                                episode.update({
                                    "task_id": tid,
                                    "site": site,
                                    "run": run,
                                    "fault_repeat": fault_repeat,
                                })
                                fault_episodes.append(episode)
                finally:
                    proc.kill()
                    proc.wait()

                successes = sum(bool(e.get("success")) for e in fault_episodes)
                payload["cells"][key] = {
                    "arm": arm,
                    "prompt": prompt_name,
                    "run": run,
                    "stable_clean_tasks": stable_tasks,
                    "clean_episodes": clean_episodes,
                    "episodes": fault_episodes,
                    "et_success": successes,
                    "et_n": len(fault_episodes),
                    "minutes": round((time.time() - started) / 60, 2),
                }
                atomic_save(payload)
                rate = successes / max(1, len(fault_episodes))
                print(
                    f"{key}: stable clean {len(stable_tasks)}/{len(tasks)}, "
                    f"ET {successes}/{len(fault_episodes)}={rate:.1%}"
                )

    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_save(payload)
    print("Complete:", OUT)


if __name__ == "__main__":
    main()
