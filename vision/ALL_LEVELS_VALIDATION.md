# Scalable VAE and all-level sanity run

Validated 2026-11-22 on the local Mac with PyTorch 2.15.0. Training used Apple MPS/FP32. No remote machine was launched and no CUDA/NCCL performance claim is made.

## Dataset

Collected **96 distinct episodes**, three each for every authored stage 1-1 through 8-4, using the seeded random-action policy, seed 22000 plus 100 per stage, and a 360-frame cap. Whole-episode preparation used seed 42 and stride 48. No episodes were removed as duplicates. Verified all 32 levels appear in every split:

| Split | Episodes | Sampled frames |
| --- | ---: | ---: |
| Train | 32 | 208 |
| Validation | 32 | 213 |
| Test | 32 | 190 |

The sampler visits every training frame and balances levels by oversampling to eight frames each (256 sampled frames per epoch). Original recordings are retained for future dense export. These short random-policy episodes cover many opening scenes and some subsequent movement; they do not represent comprehensive traversal, power-up, enemy, or late-level coverage. Similar initial screens occur across trajectories and levels.

## Run and results

The final local run used `vision/configs/local-sanity.json`: spatial VAE, width 16, one residual block per stage, bottleneck attention, factor-8 downsampling, eight latent channels (8×30×32 grid, 7,680 flattened coordinates), **550,067 parameters**. Batch size 4 with two-step accumulation, 12 epochs / 384 optimizer updates, KL beta 0.0001, 100-step KL warmup, 20-step LR warmup/cosine decay, and EMA 0.99 with initial decay ramp. The cloud starting configuration uses width 64 and two residual blocks, **11,784,083 parameters**, the same latent grid, and activation checkpointing.

| Metric | Value |
| --- | ---: |
| First-epoch validation MSE | 0.073136 |
| Final validation MSE | 0.034039 |
| Final equal-weight per-level validation MSE | 0.034106 |
| Test MSE | 0.033652 |
| Test MAE | 0.105474 |
| Test PSNR | 14.73 dB |
| Test dynamic-region MSE | 0.055074 |

An earlier factor-16 compact spatial run on the same splits completed eight epochs and reached validation MSE 0.042801. Different training duration and capacity mean this is not a controlled architecture comparison. The test split was evaluated only for the final factor-8 model. Validation and test use posterior-mean decoding and EMA weights; train metrics use stochastic samples and raw weights.

Inspected reconstruction sheets, including late-world stages. The model recovers broad layout and Mario's approximate location in many examples. Bricks, HUD text, exact sprite details, and thin collision geometry remain blurred. **This is an operational and learning sanity check, not sufficient reconstruction quality for a playable or expert world model.** Further data, longer training, and held-out fidelity/control assessment are required. The larger cloud model has been instantiated for parameter verification but has not been trained on CUDA here.

## Tests

**15 tests passed** across the full 14-test suite and the added dense spatial export test. Coverage includes:

- Native vector/spatial shapes, encoder/decoder gradients, finite losses, analytic KL, deterministic mean reconstructions, and tiny-example learning.
- Both factor-8 and factor-16 activation checkpointing match uncheckpointed gradients.
- Episode-level splitting, visual deduplication, corruption detection, portable data roots, packed arrays, and dense frame/action alignment.
- Level-balanced sampling covers every frame; distributed shards reconstruct the global sequence.
- Exact CPU epoch-boundary and interrupted mid-epoch continuation, including accumulation, LR schedule, optimizer, and EMA weights.
- Actual two-process CPU/Gloo training and restart with per-rank RNG restoration, unequal final accumulation groups, and periodic checkpoints.
- Actual CLI spatial export with finite posterior arrays, grid-shape metadata, and terminal null action preservation.

Local elastic `--standalone` rendezvous timed out before starting workers; an explicit `--master-addr=127.0.0.1 --master-port=...` launch passed. CUDA AMP/NCCL and multi-node deployment remain untested on this hardware.

## Artifacts and commands

- Dataset: `mario-recreation/datasets/all-levels-sanity/`
- Manifest: `vision/artifacts/all-levels-split.json`
- Packed cache: `vision/artifacts/all-levels-cache/`
- Final run/checkpoints/history: `vision/runs/all-levels-spatial-f8/`
- Test metrics/provenance/all-level sheets: `vision/artifacts/all-levels-f8-evaluation/`
- Portable HTML report: `vision/artifacts/all-levels-report/index.html`

Weights, recordings, and generated artifacts are git-ignored. The run is **complete**, not running in the background. To continue it locally (with the fixed schedule already at its minimum LR):

```sh
python -m vision train --config vision/configs/local-sanity.json --manifest vision/artifacts/all-levels-split.json --cache vision/artifacts/all-levels-cache --output vision/runs/all-levels-spatial-f8 --resume vision/runs/all-levels-spatial-f8/last.pt --epochs 24
```

For a real longer experiment, use a fresh run and intentionally choose the schedule/dataset budget; see [README.md](README.md) for cloud commands. Do not tune repeatedly on this sanity test split.

## Longer local continuation — 2026-11-23

Preserved the original run and resumed a copy in `vision/runs/all-levels-spatial-f8-extended/` from epoch 12 through **epoch 36 / 1,152 optimizer updates**. Architecture, dataset, losses, and optimizer state were unchanged. The learning rate remained at the original schedule's minimum, 0.00003; this was a continuation, not a restarted higher-rate schedule.

Compared the original epoch-12 and final epoch-36 EMA checkpoints on the same 213 validation frames across all 32 levels:

| Metric | Epoch 12 | Epoch 36 | Change |
| --- | ---: | ---: | ---: |
| Validation MSE | 0.034039 | 0.030282 | 11.0% lower |
| Dynamic-region MSE | 0.054870 | 0.046490 | 15.3% lower |
| PSNR | 14.68 dB | 15.19 dB | +0.51 dB |

Inspected before/after examples: Mario's silhouette and some terrain boundaries become clearer, but texture and HUD reconstruction remain poor, with visible striping in castle surfaces. This supports further training as useful; it does not establish that capacity or data coverage are sufficient. The dataset was not expanded, and the test split was not reused for this continuation assessment.

The mobile-friendly comparison report is `vision/artifacts/extended-comparison/index.html`. It contains one identical held-out frame per level with original/epoch-12/epoch-36 images (96 images), stacked vertically on narrow screens. This is a report for mobile viewing; training ran on the Mac, not on a phone. Both runs are complete.

Reproduce the comparison with:

```sh
python -m vision.tools.compare --manifest vision/artifacts/all-levels-split.json --cache vision/artifacts/all-levels-cache --before vision/runs/all-levels-spatial-f8/best.pt --after vision/runs/all-levels-spatial-f8-extended/best.pt --split val --output vision/artifacts/another-comparison
```
