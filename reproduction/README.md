# UniGraph2 reproduction

A runnable, paper-guided implementation of **UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs** (WWW 2025), with a bounded real-data experiment. This is a method reproduction, not a replication of the published benchmark scores.

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
- Unmasked frozen inference; linear probes for fused CLIP, the identical encoder before pre-training (`NoPretrain`), and the trained encoder.
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
| Tasks | Linear probes, link prediction, few-shot, generation | Node-classification linear probing; other tasks are not implemented |

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

Full benchmark replication still requires the complete Products, Papers100M and Goodreads pre-training data, full downstream graphs, independent encoder repetitions and implementations of link-prediction, few-shot and generative evaluations. The present code does not implement those additional tasks.
