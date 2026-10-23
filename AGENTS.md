# AGENTS.md

This file provides guidance to coding agents (Codex, Claude Code) when working with code in this repository.

## Two code trees, only one is live

- **Top level** (`train.py`, `models/`, `data/`, `trainers/`, `utils/`, `configs/`): the official upstream UniGraph2 release (commits by Yufei He, up to `901691f`). It does not run as-is (broken relative imports, mismatched widths, masking at inference, etc. — see "Why a separate implementation" in `reproduction/README.md`). Treat it as read-only reference; do not fix or edit it.
- **`reproduction/`**: the active, paper-guided reimplementation (plain PyTorch + scipy, no DGL/PyG/Lightning). All new work goes here. `reproduction/README.md` is the experiment log: results tables, Slurm job IDs, fidelity-vs-paper table, and the representation-collapse investigation. Keep it current when results or defaults change.

## Running (HPC cluster)

The login node prohibits compute scripts — submit through Slurm from `reproduction/`. Python is the existing conda env `/home/pusit.see/.conda/envs/mllm-xai/bin/python` (override with `UNIGRAPH_PYTHON`); never install packages into it.

```bash
cd /home/pusit.see/UniGraph2/reproduction
sbatch run.sbatch                       # unit tests, then run.py -> results/run-$JOBID, then summarize.py
sbatch run.sbatch --expert attention --seed 1   # extra args pass straight to run.py
sbatch report.sbatch results/run-83924  # diagnose.py + summarize.py on a finished run (CPU partition)
sbatch trace.sbatch --variants ...      # trace_collapse.py: per-stage effective rank during pre-training
sbatch factorial.sbatch                 # 2x2x2 lr/warmup/standardize sweep x 3 encoder seeds
```

Tests (`unittest`, not pytest), run from `reproduction/` on a compute node (`run.sbatch` and `factorial.sbatch` run them first anyway):

```bash
PY=/home/pusit.see/.conda/envs/mllm-xai/bin/python
$PY -m unittest -v test_core
$PY -m unittest test_core.CoreTests.test_spd_disconnected   # single test
```

Logs go to `reproduction/logs/<kind>-$JOBID.log`; artifacts to `reproduction/results/` (both gitignored, as is the `reference-mm-graph/` dataset-code checkout). Data is read in place from `/home/pusit.see/dataset/raw/paper_2602_00067/mm_graph/<domain>/`; never copy it.

## Reproduction architecture

Pipeline: `graph_data.py` → `model.py` → `run.py` → `diagnose.py` / `summarize.py`. `trace_collapse.py` imports `seed_all`/`probe` from `run.py`.

- `graph_data.read_graph` loads one MM-Graph domain: `clip_feat.pt` is `[N,1024]` = concatenated 512-d text + image CLIP halves, split and averaged over available (finite, nonzero) modalities by `fuse_features`. NC domains use `nc_edges-nodeid.pt`/`labels-w-missing.pt`/`split.pt`; LP domains use train edges from `lp-edge-split.pt`. `limit` takes the first N nodes' induced graph and intersects splits. `ppr_nodes` (local push PPR) picks ≤256-node subgraphs; `spd_targets` gives finite shortest-path pairs inside the induced subgraph.
- `model.UniGraph2.encode`: frozen standardization buffers (`feature_mean`/`feature_std`) → optional mask-token replacement → `MoE` (dense softmax by default, `top_k>0` sparse; experts `mlp` or `attention`) → stack of residual `GATLayer`s (scatter-based softmax, self loops added by `Graph.subgraph`). `objective` = masked-only scaled cosine reconstruction against the **raw** CLIP feature, via a decoder keyed by **domain name**, + `lambda_spd` × MSE from a shared pair-concatenation SPD head.
- `run.py`: builds a fixed seeded pool of `--steps` PPR subgraphs per pre-training domain, reshuffles/remasks it each "epoch" (an epoch is a pool pass, not a dataset pass), AdamW + warmup/cosine, saves `history.json`/`checkpoint.pt` every epoch. Then encodes the target graph unmasked and linear-probes three frozen representations: `CLIP`, `NoPretrain` (same encoder before training), `UniGraph2`. Writes `results.json` with args, env, data-file metadata and sha256 of every `*.py`.
- `diagnose.py` reports collapse metrics (mean cosine to centroid, effective rank) on `embeddings.pt`; `summarize.py` writes `REPORT.md` and refuses runs without `status: completed`.

## Invariants to preserve

- Pre-training is label-free and never touches the evaluation domain's graph or labels; standardization statistics come from pre-training domains only.
- Choose hyperparameters/fixes by validation accuracy only; test accuracy is computed after probe-epoch selection and never used for selection.
- Defaults (`--lr 1e-4 --warmup 0.1 --standardize`) are the collapse fix. Paper Table 9 settings (`--lr 1e-3 --warmup 0 --no-standardize`) collapse embeddings to rank ~1–3; that is a known result, not a bug to rediscover.
- A run counts only when `results.json` has `status: completed` and all three probe results. Report cross-seed results as mean ± population std across encoder seeds, each the mean over probe seeds, with job IDs.
- Changes to `model.py` options should extend `test_core.test_inference_and_gradients` (it loops over device × expert × top_k and checks gradients, domain-decoder isolation, deterministic unmasked inference and permutation equivariance).

## Git

- Commit after every completed change without asking. Stage only the files you changed. Write a descriptive message; for experiments, include job IDs and the resulting numbers, as earlier commits do. Do not push unless asked.
