# Quantization Effects on Tool-Failure Recovery Vary Across Prompts and Evaluation Designs

Code and data for the paper (NeurIPS 2026 Workshop on Small Language Models for Agentic
Systems, SLM-Agents). Paper: https://openreview.net/forum?id=QZdcb68DRP

Everything needed to reproduce all experiments, figures, and statistics.

## Contents
```
code/
  phase1_suite.py        Task generator + clean-baseline evaluation
                         (Sec. 4, "Tasks"; feeds the P0 ladder)
  phase3_multi.py        P0 fault-injection ladder: CLEAN/ET/IT/EP at every
                         gold-path site (Appendix A)
  analyze_phase3.py      Ladder analysis: Wilson CIs, Fisher/McNemar,
                         failure taxonomy
  same_prompt_matched_analysis.py
                         Same-prompt-matched ladder table, pooled ET tests,
                         task-level robustness check (Appendix A)
  clustered_analysis.py  Task-clustered analysis of the pooled P0 contrast
  qwen_replication.py    Earlier one-prompt Qwen2.5-7B audit run
  promptvar.py           First prompt-variant experiment (P1/P2)
  promptvar2.py          Exploratory five-prompt sweep (P0-P4 x 3 levels)
  confirmatory_sweep.py  CONFIRMATORY sweep: frozen prompts/plan, pinned
                         llama.cpp commit and model revision, complete
                         per-episode logs for all 15 Llama cells
  qwen_confirmatory_sweep.py
                         Five-prompt Qwen2.5-7B extension using the frozen
                         tasks, prompts, harness, and analysis (10 cells)
  confirmatory_analysis.py
                         The frozen primary-outcome analysis
  confirmatory_paper_stats.py
                         Reproduces the cross-prompt estimates and appendix
                         tables from both confirmatory data files; validates all 25 cells
                         and verifies the 15 Llama exploratory/confirmatory
                         cells match (no GPU)
  strict_vs_tolerant.py  Camera-ready addition: rescores the released logs
                         under strict scoring (strict JSON + literal amount,
                         clean screen rescored) and prints all three
                         evaluation targets under both policies (Table 2)
data/
  confirmatory_sweep_results.json  Confirmatory Llama sweep, complete episode
                                   logs for all 15 prompt x precision cells;
                                   source of Figs. 1-2 and the result tables
  qwen_confirmatory_sweep_results.json
                                   Complete logs for all 10 Qwen prompt x
                                   precision cells; source of Fig. 1 and
                                   the result tables
  phase1_results.json              Full-ladder clean baselines
  phase3_multi_results.json        P0 ladder injection episodes (Appendix A)
  qwen_replication_results.json    Earlier one-prompt Qwen audit episodes
  promptvar_results.json           First prompt-variant run (aggregate)
  promptvar2_results.json          Exploratory sweep (aggregate counts for
                                   all cells; episode logs for 3 cells) --
                                   retained as the exploratory record; its
                                   counts match the confirmatory run exactly
PROMPTS.md               All five complete system prompts (P0-P4)
figures/
  make_confirmatory_figures.py  Fig. 1 (separate-screen, prompt-matched,
                                and full-pipeline estimates for two models)
                                and Fig. 2 (exploratory Llama-only P0/P1),
                                computed from the two confirmatory data files
  make_figures.py, make_overview.py, make_promptfig.py  earlier figures,
                                computed from their JSONs
```

## Environment
- GPU: single NVIDIA L4 (24 GB); any >=12 GB CUDA GPU works.
- Python 3.10+; `pip install -r requirements.txt`
- llama.cpp built with CUDA at pinned commit `b10f9ca`:
  ```
  git clone https://github.com/ggml-org/llama.cpp
  git -C llama.cpp checkout b10f9ca
  cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON
  cmake --build llama.cpp/build --config Release -j --target llama-server
  ```
  The `llama.cpp/` directory must sit next to the scripts.
- Model files are pinned by Hugging Face revision
  (`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` at revision
  `bf5b95e96dac0462e2a09145ec66cae9a3f12067`; see the `config` block of
  `confirmatory_sweep_results.json`, and
  `bartowski/Qwen2.5-7B-Instruct-GGUF` at revision
  `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`; see
  `qwen_confirmatory_sweep_results.json`).

## Reproducing the paper's main results
All decoding is temperature 0, top-k 1, fixed seed; task generation uses a
fixed seed (20260826).

1. Statistics only (no GPU, uses the shipped data):
   `python confirmatory_paper_stats.py`
2. Full Llama sweep (~55 min per run on the recorded L4):
   ```
   export LLAMA_MODEL_REVISION=bf5b95e96dac0462e2a09145ec66cae9a3f12067
   export SWEEP_RUNS=1 CLEAN_REPEATS=1 FAULT_REPEATS=1
   export SWEEP_ARMS=Q8_0,Q4_K_M,Q3_K_M LLAMA_CPP_COMMIT=b10f9ca
   python confirmatory_sweep.py     -> ../data/confirmatory_sweep_results.json
   python confirmatory_analysis.py  -> ../data/confirmatory_analysis.json
   ```
3. Qwen extension (~20 min on the recorded L4):
   ```
   export QWEN_MODEL_REVISION=8911e8a47f92bac19d6f5c64a2e2095bd2f7d031
   export SWEEP_RUNS=1 CLEAN_REPEATS=1 FAULT_REPEATS=1
   export SWEEP_ARMS=Q8_0,Q4_K_M LLAMA_CPP_COMMIT=b10f9ca
   python qwen_confirmatory_sweep.py
   ```
4. Figures (no GPU): `python figures/make_confirmatory_figures.py`
5. Appendix A ladder (~2.5 h): `python phase1_suite.py`, then
   `python phase3_multi.py`, then `python same_prompt_matched_analysis.py`.
6. Earlier one-prompt Qwen audit (~35 min): `python qwen_replication.py`.

## Determinism note
Every aggregate count in the 15-cell Llama sweep is identical between the two
executions documented in this artifact: the exploratory run
(promptvar2_results.json) and the confirmatory run from the pinned build
(confirmatory_sweep_results.json, whose config block records the llama.cpp
commit and model revision). Intermediate repeats during development also
matched, but their logs were not retained and they are not claimed as
released evidence. Repeats under deterministic decoding are reproducibility
checks, not independent samples; the paper reports unique counts (e.g.,
31/32 failed episodes) and treats tasks as the independent unit.
The 10-cell Qwen extension is a separate model-family run under the frozen
protocol. It was planned after the Llama result and is not presented as a
preregistered replication.

## Notes
- Tool calls use a text protocol rather than the server's native tool-call
  path; parser tolerances are logged per episode (`strict_json`,
  `parsed_json`, `expr_submit`).
- The exploratory `promptvar2_results.json` is retained for the audit trail.
  Its per-episode logs cover only Q8_0/P0 and Q3_K_M/P0-P1; the confirmatory
  file has complete logs for all 15 cells and is the paper's data source.
- `qwen_confirmatory_sweep_results.json` has 20 clean episodes per cell and
  complete fault episodes for every clean-passing task in all 10 cells. Its
  configuration records both the model and llama.cpp revisions.
- Scripts write a Drive backup if /content/drive/MyDrive/quantrec exists
  (Colab); harmless elsewhere.


## Citation

```bibtex
@inproceedings{hu2026quantization,
  title     = {Quantization Effects on Tool-Failure Recovery Vary Across Prompts and Evaluation Designs},
  author    = {Hu, Yuhe},
  booktitle = {NeurIPS 2026 Workshop on Small Language Models for Agentic Systems (SLM-Agents)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
