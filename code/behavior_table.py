"""Behavioral breakdown of ET/EP episodes (behavioral breakdown)."""
import json
R = json.load(open("../data/phase3_multi_results.json"))
for arm in ["Q8_0","Q6_K","Q5_K_M","Q4_K_M","Q3_K_M"]:
    eps = [e for e in R[arm]["episodes"] if e["cond"]=="ET"]
    rec  = sum(1 for e in eps if e["success"])
    loop = sum(1 for e in eps if not e["success"] and e.get("answer") is None
               and not e.get("abstained") and e.get("parsed_json"))
    ab   = sum(1 for e in eps if e.get("abstained"))
    wr   = sum(1 for e in eps if e.get("answer") is not None and not e["success"])
    mal  = sum(1 for e in eps if not e.get("parsed_json"))
    nore = sum(1 for e in eps if e.get("target_calls",0) <= 1)
    print(f"{arm}: n={len(eps)} recovered={rec} loop={loop} abstain={ab} "
          f"wrong={wr} malformed={mal} never-retried={nore}")
    epp = [e for e in R[arm]["episodes"] if e["cond"]=="EP"]
    mt = sum(e.get("target_calls",0) for e in epp)/len(epp)
    prose = sum(1 for e in epp if not e["success"] and not e.get("parsed_json"))
    print(f"      EP: mean calls to dead tool {mt:.2f}, prose-format failures {prose}")
