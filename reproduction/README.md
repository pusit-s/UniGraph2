# UniGraph2 reproduction

A runnable, paper-guided implementation of **UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs** (WWW 2025), with a bounded real-data experiment. This is a method reproduction, not a replication of the published benchmark scores.

The latest [shared-attention comparison](#shared-attention-results-array-84392) includes **135 completed runs across five datasets**, with an archived [full report](experiments/shared-84392/REPORT.md), [performance chart](experiments/shared-84392/performance.png), and per-dataset Figure-3-style visualizations. The earlier [RoMA experiment](#roma-comparison-results-array-84359) remains archived separately.

## Completed experiment

Defaults use the collapse fix. Paper Table 9 settings, which collapse (job 81737, below), are `sbatch run.sbatch --lr 1e-3 --warmup 0 --no-standardize`. Slurm jobs **83924, 83926, 83928** (encoder seeds 0, 1, 2) completed on one GPU. Five correctness tests passed. Mean Ele-Fashion subset test accuracy across encoder seeds (each the mean of three probe seeds), ± population std across encoder seeds:

| Frozen representation | Test accuracy (%) |
| --- | ---: |
| CLIP | 85.61 ± 0.00 |
| NoPretrain | 86.32 ± 0.23 |
| UniGraph2 | 84.30 ± 0.54 |

Representation collapse is fixed: pre-trained embeddings now have mean cosine to centroid 0.90–0.91 and effective rank 28–42 (previously 0.99999994 and 2.84), and UniGraph2 accuracy rose from 28.95% to 84.30%. Pre-training still **does not beat the untrained encoder** (−2.0 points) in this bounded run: 160 optimizer steps on 32 Amazon-Cloth subgraphs. This does not test the paper's large-scale claims.

Artifacts per seed: `results/run-JOBID/` ([seed 0 report](results/run-83924/REPORT.md)), logs in `logs/run-JOBID.log`.

### Representation collapse (job 81737) and its fix

The first run (job 81737, paper Table 9 lr 1e-3, raw CLIP input) gave **UniGraph2 28.95%** vs NoPretrain 83.79%: embeddings were near-constant (cosine to centroid 0.99999994, effective rank 2.84). `trace_collapse.py` (`sbatch trace.sbatch --variants ...`; logs `trace-83915`…`trace-83923`) traces per-stage effective rank on the unmasked Ele-Fashion graph during pre-training. Pre-training is label-free; fixes were chosen by validation accuracy only, never test. It found two independent causes:

1. **Tiny raw input scale.** MM-Graph CLIP rows have norm ≈0.8 (entries ≈0.036), so default bias terms dominate the input-dependent signal: the untrained MoE output already has cosine 0.98 to its centroid. Bias updates then swamp node-specific signal even under SGD with negligible weight change. Fix: frozen per-dimension standardization of the encoder input, using pre-training-domain statistics (default on, `--no-standardize` disables; model buffers `feature_mean`/`feature_std`, identity by default; target-domain statistics are never used). The mask token lives in the standardized space; the reconstruction target stays the raw CLIP feature, as in the paper.
2. **AdamW lr 1e-3 at width 1024.** Even with standardized input, every lr 1e-3 run collapsed to rank 1–3 within 20 steps, whether or not the GAT, LayerNorm, MoE, SPD loss, dropout or cosine loss were removed. Adam's sign-like steps on coherent gradients give a low-rank update whose spectral norm exceeds the initial weights' within ~10 steps. Fix: `--lr 1e-4 --warmup 0.1` (10% linear warmup, cosine decay), the values in the official `configs/default.yaml`.

Neither fix alone suffices: raw input at lr 1e-4 reached only 73% validation accuracy; standardized input at lr 1e-3 still collapsed. Even plain autoencoding without masking collapsed under the old settings, so the objective itself was not the cause.

Sources: [paper](https://arxiv.org/abs/2502.00806), [paper HTML](https://arxiv.org/html/2502.00806v1), [official code](https://github.com/yf-he/UniGraph2), [MM-Graph benchmark](https://github.com/mm-graph-benchmark/mm-graph-benchmark).

The official repository is preserved outside this directory at commit `901691ffd256611e1a1592a5549f561c6b99b3be`. The dataset reference checkout is `reference-mm-graph/`, commit `0769fa670c69a6dc0756a2d7fc697b980e494775`.

## Run on this cluster

The login node prohibits compute scripts. Submit with Slurm:

```bash
cd /home/pusit.see/UniGraph2/reproduction
sbatch run.sbatch
```

The script uses the existing `mllm-xai` environment and runs correctness checks before training. It requests one GPU, four CPUs and 24 GB RAM with a one-hour limit. Change the Python executable with `UNIGRAPH_PYTHON`. No dependencies are installed into your environment.

To override the experiment:

```bash
sbatch run.sbatch --pretrain cloth-copurchase sports-copurchase --hidden 128 --epochs 5 --steps 32
```

Outside this cluster, install `requirements.txt`, then run:

```bash
python -m unittest -v test_core
python run.py --data-root /path/to/mm_graph --output results/my-run --device cuda
```

Data stays in `/home/pusit.see/dataset/raw/paper_2602_00067/mm_graph`. No raw data is copied. The default experiment pre-trains on Amazon-Cloth (125,839 nodes) and probes Ele-Fashion on its first 4,096 nodes and their induced edges, intersecting the original train/validation/test node splits. It never uses Ele-Fashion labels or graph data for encoder pre-training. Only the probe uses training labels. Validation accuracy chooses the probe epoch; test accuracy is computed after selection.

## What is implemented

- Frozen, cached CLIP text and image features. MM-Graph's `[N,1024]` tensor stores concatenated 512-dimensional text/image halves (see `reference-mm-graph/nc/MMGCN/run.py`, lines 197–199). Split and average available modalities into `[N,512]`, following Eq. 8. Missing zero/nonfinite modality vectors are excluded; nodes without any valid modality fail validation.
- Learned node mask token, random masking without replacement and masked-only scaled cosine reconstruction loss (Eq. 15). Labels never enter this objective.
- Shared MoE alignment, followed by four GAT layers. Domain-specific reconstruction decoders and a shared concatenated-pair shortest-path regressor.
- Approximate PPR subgraphs, up to 256 nodes, with precomputed shortest-path targets and sampled finite node pairs.
- Unmasked frozen inference; node-classification linear probes and symmetric Hadamard-product linear edge probes for fused CLIP, the identical encoder before pre-training (`NoPretrain`), and the trained encoder.
- Checkpoints, embeddings, sampled node IDs, loss history, trained probe weights, seed-level results and runtime environment metadata.

## Fidelity and limits

| Item | Paper | This experiment |
| --- | --- | --- |
| Pre-training domains | Products, Papers100M, Goodreads-LP, Amazon-Cloth | Amazon-Cloth by default; supports multiple local MM-Graph domains |
| Hidden / GNN layers / experts | 1024 / 4 / 8 | Same defaults |
| Mask / dropout / weight decay | .8 / .4 / .01 | Same defaults, from Table 9 |
| Optimizer / epochs / PPR top-k / SPD coefficient | AdamW / 5 / 256 / .1 | Same defaults; each epoch is a **bounded subgraph-pool pass**, not a full dataset pass |
| Learning rate schedule | Table 9: 1e-3; official config: 1e-4, warmup, cosine | Default **1e-4, 10% warmup, cosine** (official config); Table 9: `--lr 1e-3 --warmup 0` (collapses here) |
| Encoder input | CLIP features | Default **standardized per dimension** with pre-training-domain statistics; `--no-standardize` for raw CLIP; reconstruction target unchanged |
| Routing | Eqs. 10–12 show dense softmax; prose says selected experts | Dense softmax by default; `--top-k 2` supports the official code's sparse variant |
| GAT details | Backbone named; heads, residuals and normalization not fully specified | 4 concatenated heads, residual connections, LayerNorm and ELU; explicit reconstruction choice |
| Feature loss gamma | At least 1; no value in Table 9 | 2, following released model default |
| SPD targets | Precomputed graph shortest paths | Shortest paths **inside induced PPR subgraphs**; unreachable pairs omitted, up to 4096 finite pairs |
| PPR parameters | Top-k 256; remaining parameters unspecified | Local residual push, alpha .15, epsilon 1e-4, at most 50,000 pushes |
| Subgraph schedule | Web-scale pre-training | Fixed seeded pool of 32 subgraphs/domain, shuffled and freshly masked each epoch |
| Downstream graph | Full target dataset | First 4096 nodes and induced edges; `--eval-limit 0` enables full graph subject to memory |
| Probe | Adam, lr .01, up to 5000 epochs, early stopping | Same optimizer/lr/maximum; patience 100 by default; unit-normalized embeddings |
| Repetitions | 10 or 20 depending on benchmark | Three probe seeds, one encoder seed by default |
| Hardware/software | 8 A100 40GB, PyTorch 2.0.1, DGL 1.1.2 | One allocated GPU, PyTorch 2.5.1, native scatter-based GAT; device recorded in results |
| Tasks | Linear probes, link prediction, few-shot, generation | Node-classification and bounded link-prediction probing; few-shot/generation are not implemented |

The default run is a limited transfer experiment. It cannot substantiate the paper's large-scale or cross-domain generalization claims. Accuracy must not be compared directly with the published Ele-Fashion score because the pre-training corpus and evaluation graph differ. `--pretrain` supports domain-specific decoders but pre-training on Sports changes it into a target-domain exposure condition for Sports evaluation.

## Why a separate implementation

The released source contains two incompatible model/training paths. The entry point imports packages through broken relative imports; its main model averages features of different widths without projection, constructs its MoE for a mismatching width, masks even when returning inference embeddings, creates masks on CPU, loops through every SPD pair in Python and admits infinite unreachable distances into MSE. Its decoders are keyed by modality rather than graph domain. The alternate graph encoder pools node embeddings into one graph vector, which does not provide the required node representations. This reproduction implements the paper's objective in a separate directory, leaving the official source untouched.

## Validation

`test_core.py` compares sparse GAT against a dense reference, verifies available-modality fusion, frozen input standardization (without mutating the caller's tensor), finite SPD targets, PPR connectivity, gradients through the mask/MoE/GNN/both decoders, domain decoder isolation, deterministic unmasked inference and node-permutation equivariance. CPU and allocated CUDA devices are exercised.

Every completed run writes `results/run-JOBID/results.json`. `history.json` and `checkpoint.pt` are saved each epoch. A submitted or partially trained job does not count as a completed reproduction; require `status: completed` and all three probe results.

Generate a Markdown report and inspect saved embedding variation on a compute node:

```bash
sbatch report.sbatch results/run-83924
```

Full benchmark replication still requires the complete Products, Papers100M and Goodreads pre-training data, full downstream graphs, independent encoder repetitions, full-candidate link prediction, and implementations of few-shot and generative evaluations.

## RoMA routing and attention comparison

[RoMA (Routing Manifold Alignment)](https://arxiv.org/html/2511.07419v2), [reference repository](https://github.com/tianyi-lab/RoMA), is router-only post-training rather than a new expert architecture. `roma.py` implements Eqs. 5–10 using live differentiable routing probabilities, frozen successful-neighbor routes, and normalized Gaussian kNN weights. Experts, GNN, reconstruction/SPD heads, input statistics and mask token remain frozen; dropout is disabled during this stage.

This is an explicit **label-free graph adaptation**: frozen unit-normalized raw CLIP features replace task-text embeddings; the best half of distinct pre-training nodes by mean masked reconstruction error, selected separately per domain, replace correctly answered training examples. Errors are scored on one fixed masked pass over the existing subgraph pool, averaging repeated observations of each node. Neighbors exclude the same domain/node identity. Reference and online routes are unmasked, since all masked nodes receive one identical mask token. No target-domain graph, features, labels, or statistics enter the bank or router post-training. This surrogate differs from RoMA's supervised correct-answer criterion.

Defaults for the optional stage: 3 additional pool passes, router lr `7e-5`, λ `0.8`, k `5`, σ `0.4`, success fraction `0.5`; lr uses the existing warmup/cosine policy and λ warms up over the same initial fraction. Baseline defaults are unchanged. Sparse top-1 router tuning is rejected because its normalized routing weights have zero derivative; dense and top-k ≥2 are supported.

```bash
sbatch run.sbatch --router-posttrain roma
sbatch run.sbatch --expert attention --router-posttrain roma
sbatch roma_compare.sbatch                 # three encoder seeds; six conditions per seed
sbatch roma_report.sbatch results/roma-JOBID # validate paired runs and generate figures
```

The six conditions are MLP and attention experts, each with no post-training, task-loss-only router tuning, and RoMA. Task-only and RoMA branch from the exact same saved baseline checkpoint and use identical post-training masks/order. Checkpoint lineage, input metadata, source hashes, complete probes and matched settings are checked. CUDA scatter operations may differ by a few ulps between processes; checkpoint resumption checks numerical agreement and reuses the saved untrained embeddings for exactly paired reference probes.

`plot_roma.py` writes PNG/PDF/SVG Figure-3-style UMAP plots, a performance plot, per-probe CSV, inspectable coordinates/routes and JSON provenance, plus `COMPARISON.md`. Every panel uses the same randomly sampled nodes and four unsupervised CLIP groups. The fourth panel shows the earlier attention method instead of the paper's label-optimized oracle. Independent UMAP panels cannot establish alignment by their appearance: original-space CLIP/routing kNN overlap, silhouette, routing spread and effective rank are also reported. The first declared encoder seed is always plotted; no seed or view is chosen by test accuracy or apparent cluster separation.

### RoMA comparison results (array 84359)

All 18 runs completed on NVIDIA A100 80GB GPUs: encoder seeds 0/1/2 use Slurm task IDs **84360/84361/84359**, respectively (array **84359**, tasks 0–2). Every condition has all three frozen representations and three probe seeds. Figures and matched-run validation completed in CPU jobs **84362** and **84364**; the latter corrects the supplementary legend layout. Mean ± population std across encoder seeds, each averaged over probe seeds:

| Trained encoder | Test accuracy (%) | Seed 0 / 1 / 2 (%) |
| --- | ---: | --- |
| UniGraph2 / original MLP experts | 84.30 ± 0.54 | 83.77 / 84.10 / 85.05 |
| Earlier attention experts | 80.60 ± 0.42 | 80.84 / 80.95 / 80.00 |
| MLP + task-only router tuning | 84.17 ± 0.47 | 83.74 / 83.93 / 84.83 |
| MLP + RoMA | **84.52 ± 0.47** | 84.18 / 84.18 / 85.19 |
| Attention + task-only router tuning | 79.40 ± 3.03 | 81.34 / 81.73 / 75.12 |
| Attention + RoMA | 78.61 ± 3.99 | 81.26 / 81.59 / 72.97 |

MLP + RoMA improves the original trained encoder by **0.22 percentage points**, and its equal-budget task-only control by **0.353 ± 0.080 points** (paired population std across encoder seeds). This is a small observed gain, not evidence of statistical significance or full-benchmark superiority. It remains below frozen CLIP (**85.61%**) and the untrained MLP encoder (**86.32 ± 0.23%**).

Attention is unstable under router post-training: RoMA averages **1.99 points below** the earlier attention baseline and **0.790 ± 0.960 points below** the matched task-only control. Seed 2 deteriorates in both post-training conditions; no seed was excluded and no settings were selected from test results.

The Figure-3-style plot uses the same 1,500 randomly sampled Ele-Fashion nodes in all panels (encoder seed 0, sampling/UMAP seed 42). It does **not** reproduce the paper's sharply separated RoMA clusters: original-space semantic/routing kNN overlap@15 is **0.1797 → 0.1861** for MLP baseline → RoMA. The task-only control reaches 0.2207 despite lower probe accuracy. CLIP-cluster silhouette also decreases under MLP RoMA (0.0795 → 0.0634), so these results do not establish the paper's strong manifold-alignment claim on this graph transfer task.

Tracked evidence: [comparison report](experiments/roma-84359/COMPARISON.md), [performance chart](experiments/roma-84359/figures/performance.png), [Figure 3 PNG](experiments/roma-84359/figures/figure3-routing.png) / [PDF](experiments/roma-84359/figures/figure3-routing.pdf) / [SVG](experiments/roma-84359/figures/figure3-routing.svg), [all routing conditions](experiments/roma-84359/figures/routing-controls.png), [per-probe scores](experiments/roma-84359/scores.csv), [complete source-backed results](experiments/roma-84359/comparison.json), and [plot settings and diagnostics](experiments/roma-84359/figures/visualization.json). Raw source CLIP features are read in place; archived plot data contains node IDs, cluster assignments, learned routing weights and UMAP coordinates.

Validation: nine CPU/CUDA tests passed, covering the manifold formula, self-exclusion with domain/node identity, live gate gradients with detached teacher routes, frozen non-router state, RNG-isolated label-free bank construction, complete paired-run validation and the existing inference/gradient invariants. Six-condition integration smoke test: job **84358**. Initial array 84355 stopped on an overly strict bitwise CUDA inference check and is excluded from results; the final array uses the corrected numerical check and exactly reused baseline reference embeddings. No environment packages were installed, and the upstream code tree was not edited.

## All-dataset NC / link-prediction workflow

`download_mmgraph.py` downloads only missing CLIP features and task files into the canonical data root and verifies every required file against its SHA256 in MM-Graph release `c707a3203b77a153f11370a05d2a9be6571f7670`. Download job **84374** completed for **Ele-Fashion, Books-NC, Books-LP, Amazon-Sports, and Amazon-Cloth**. The two CoDEx folders in that release contain relational triples without cached CLIP node features and are recorded as unsupported by this feature/task contract. Raw images and unused feature families are not downloaded.

`read_graph` memory-maps tensor files when supported and slices target CLIP rows before fusion, so the bounded Books cohorts do not require materializing multi-gigabyte feature copies. Labels and original split IDs remain aligned. LP target message passing uses training edges only.

`link_probe.py` preserves official positive split membership within the first 4,096 nodes. It deduplicates undirected positives and excludes held-out edges already present in earlier splits, including reversed training edges. Fixed seed 1729 chooses at most 4,096 training positives and 1,024 positives each for validation/test. Each held-out query has 100 fixed distinct sampled non-neighbors from that cohort, excluding self loops and every known positive. Training uses one fixed negative per positive. Candidate arrays and their logical SHA256 are shared across all conditions and encoder/probe seeds. This is **sampled MRR**, with a different candidate population from the official full-graph benchmark. Accuracy and MRR must not be pooled into one average.

The frozen edge probe is a linear classifier on Hadamard products of unit-normalized endpoint embeddings. It uses Adam lr .01, up to 5000 epochs and patience 100, checking validation MRR every five epochs. Only the selected head is evaluated on test; MRR and Hits@1/3/10 use average optimistic/pessimistic ranks for ties. Existing node-classification defaults are unchanged.

```bash
sbatch download_mmgraph.sbatch
sbatch all_datasets.sbatch # 5 datasets × 3 encoder seeds × 6 conditions = 90 runs
sbatch all_datasets_report.sbatch results/all-datasets-JOBID \
  --download-manifest results/download-84374.json \
  --archive experiments/all-datasets-JOBID
```

Each target is excluded from encoder pre-training and the RoMA bank. Four targets use Amazon-Cloth as source; Amazon-Cloth evaluation uses Amazon-Sports as source. All target/method pairs retain the prior 160 source updates and optional 96 gate-only updates. Every run probes CLIP, NoPretrain and UniGraph2 using three probe seeds. Reports include all six conditions, paired deltas, frozen references, per-probe scores, split/candidate provenance, job IDs and Figure-3-style visualizations for every dataset. Correctness checks now include link split isolation, negative validity, ranking ties, frozen edge-probe inputs, reproducibility, bounded memory-mapped loading, and matching NC/LP reporting protocols.

## Shared attention before the MoE

The optional `--shared-attention` architecture follows the shared-attention / feed-forward-expert placement in the [Transformer versus MoE diagram](https://www.dailydoseofds.com/p/transformer-vs-mixture-of-experts-in-llms/). It is separate from `--expert attention`, whose experts each own an attention layer. The default remains the original MLP MoE with no shared attention.

```mermaid
flowchart LR
    X[Node CLIP features] --> S[Source standardization and optional mask]
    S --> A[Shared multihead self-attention]
    A --> N[Output projection + residual + LayerNorm]
    N --> R[Learned router]
    R --> E[Weighted selected MLP experts]
    E --> G[Existing GAT backbone]
```

`SharedSelfAttention` uses separate learned Q/K/V projections, scaled dot-product attention per head, concatenation, an output projection and post-residual LayerNorm: `u = LayerNorm(z + W_out Attention(z))`. The router and each selected MLP consume `u`. At the default CLIP width 512, four heads have width 128 each; the attention output remains width 512 and the existing MLP experts map 512 → 1024 → 1024. `--attention-heads` is independent of GAT `--heads`. The input width must divide evenly by the attention head count.

Each call attends only within its current graph/PPR subgraph. Nodes are unordered, so this graph adaptation uses neither position encodings nor a causal attention mask. Attention dropout is zero; existing GAT dropout is disabled by evaluation mode. Masking remains before attention, and unmasked inference does not use the learned mask token. Both routing inspection and encoding share exactly the same preprocessing. Shared attention requires `--expert mlp`; combining it with the old per-expert attention option is rejected.

```bash
sbatch run.sbatch --shared-attention --attention-heads 4
sbatch run.sbatch --shared-attention --top-k 2
sbatch run.sbatch --expert attention             # old per-expert attention
sbatch run.sbatch --shared-attention --router-posttrain roma
sbatch shared_datasets.sbatch                    # 5 targets × 3 encoder seeds × 9 conditions
```

Dense routing remains the default; top-2 probabilities are learned and renormalized over the two selected MLP experts. The shared block adds parameters, so architectural comparisons match data, seeds, widths and update budgets rather than total parameter count. Result files record both total and shared-attention parameter counts.

RoMA freezes the shared attention together with experts/GAT/decoders and updates only gate parameters. Its contextual teacher routes are computed on each complete original unmasked source PPR pool and averaged per successful node over the observations used to score reconstruction quality. They are never recomputed with only anchor nodes as the attention context. Online routing uses the complete current unmasked PPR subgraph. No evaluation features or graph enter this bank.

Seventeen CPU/CUDA tests passed, including explicit dense attention references; Q/K/V/output/normalization and gate gradients; dense/top-2 routing; input immutability; domain decoder isolation; deterministic unmasked evaluation; node permutation equivariance; call-local graph context; contextual teacher averaging; and the existing NC/LP checks. A separate regression against the prior model verified exact initialization/state tensors/RNG and matching CPU/CUDA inference for both existing architectures over seeds 0/1/2. End-to-end checks completed for all nine dense cases (**84389**) and shared-attention top-2 LP baseline/task/RoMA cases (**84388**). The nine-condition report and routing figures were rendered and inspected on the smoke run.

The earlier incomplete sweep **84381** and dependent report **84384** were cancelled when the requested architecture changed; its artifacts are retained and excluded from the new cross-seed comparison. `shared_datasets.sbatch` runs MLP, old per-expert attention, and shared-attention MLP, each with no post-training, task-only gate tuning, and RoMA: **135 runs**, with three probe seeds for every representation. The main comparison is dense throughout; top-2 has separate gradient and end-to-end smoke checks. Candidate populations and source holdouts follow the all-dataset protocol above, and the existing defaults remain unchanged.

### Shared-attention results (array 84392)

All **135 runs completed**, covering five targets × three encoder seeds × three architectures × three training stages. All 1,215 representation/probe records are present. Array task jobs **84394/84395/84392** cover encoder seeds 0/1/2; worker **84396** completed seed 2's Books-LP and Amazon-Cloth cases on the freed second GPU. Each target/seed directory uses an exclusive lock, and the main task reused those 18 cases only after matching exact commands, core hashes and unchanged inputs. The completed-case reuse smoke check reused all nine cases without retraining. No model or probe setting changed during scheduling.

Report/figure job **84393** completed successfully. Mean ± population standard deviation across the three encoder seeds, each the mean over probe seeds 0/1/2:

| Target | Metric (%) | Original MLP | Old per-expert attention | Shared attention + MLP | Shared attention + RoMA |
| --- | --- | ---: | ---: | ---: | ---: |
| Ele-Fashion | Accuracy | 84.30 ± 0.54 | 80.60 ± 0.42 | 84.54 ± 0.31 | 84.64 ± 0.55 |
| Books-NC | Accuracy | 66.62 ± 0.41 | 46.19 ± 5.74 | 67.13 ± 1.53 | 66.73 ± 1.53 |
| Amazon-Sports | Sampled MRR | 80.61 ± 1.27 | 49.04 ± 1.61 | 82.08 ± 0.16 | 81.83 ± 0.24 |
| Amazon-Cloth | Sampled MRR | 65.65 ± 1.03 | 47.70 ± 2.04 | 65.38 ± 0.80 | 65.54 ± 0.95 |
| Books-LP | Sampled MRR | 15.51 ± 2.75 | 11.70 ± 3.48 | 19.24 ± 1.50 | 19.11 ± 1.21 |

The new shared-attention model has higher mean scores than the old attention experts on **all five targets**, and than the original trained MLP on **four of five**. Differences versus MLP are +0.23, +0.51, +1.47, −0.27 and +3.73 points in the table's dataset order. These are observed means, not significance claims. Shared attention adds **1,051,648 parameters** (22,073,353 total versus 21,021,705 for MLP), so this comparison matches training budgets and widths rather than parameter counts.

RoMA's additional effect on the new architecture is mixed: +0.10, −0.40, −0.25, +0.16 and −0.13 points versus its own baseline. Its paired effects versus equal-budget task-only gate tuning are +0.130 ± 0.332, +0.438 ± 0.335, +0.081 ± 0.086, +0.278 ± 0.425 and −0.202 ± 0.146 points. All nine architecture/stage conditions and per-seed values are in each dataset report; no test-driven model or seed selection was performed.

Pre-training with shared attention improves over its own untrained encoder only on Books-NC in this bounded experiment (65.89 → 67.13). The corresponding untrained/pre-trained values are 85.72 → 84.54 on Ele-Fashion, 83.19 → 82.08 on Sports, 65.58 → 65.38 on Cloth and 20.17 → 19.24 on Books-LP. Better scores than the older trained architecture therefore do not establish that pre-training is beneficial on every target.

All targets use the first **4,096 nodes**. LP uses 100 fixed non-neighbors per query after excluding observed/reversed positives; these sampled-MRR values use a different population from official full-graph rankings. CoDEx-S/M remain outside the runnable contract because their release folders lack cached CLIP node features. The tests support both dense and top-2 routing; this performance table uses dense routing throughout.

An independent compute-node audit passed for all 135 completed results, 1,215 unique probe rows, recalculated means/population standard deviations, validation-selected epoch bounds, all **81 saved LP candidate hashes**, and byte-identical archived exports. All five primary routing figures and the combined chart were visually inspected; the routing and performance PDFs were rendered and checked. [Validation receipt](experiments/shared-84392/VALIDATION.json).

Archived evidence: [full report](experiments/shared-84392/REPORT.md), [performance PNG](experiments/shared-84392/performance.png) / [PDF](experiments/shared-84392/performance.pdf), [structured results](experiments/shared-84392/summary.json), [all probe scores](experiments/shared-84392/scores.csv), and [download checksums](experiments/shared-84392/download-manifest.json). Figure-3-style routing views: [Ele-Fashion](experiments/shared-84392/ele-fashion/figures/figure3-routing.png), [Books-NC](experiments/shared-84392/books-nc/figures/figure3-routing.png), [Sports](experiments/shared-84392/sports-copurchase/figures/figure3-routing.png), [Cloth](experiments/shared-84392/cloth-copurchase/figures/figure3-routing.png), [Books-LP](experiments/shared-84392/books-lp/figures/figure3-routing.png). Each dataset folder includes vector exports, all-nine-condition views, routing arrays, node IDs and source/protocol provenance. No packages were installed or upstream-tree files changed.
