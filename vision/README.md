# Mario vision model (V)

A native-resolution variational autoencoder and training pipeline for the vision stage of VMC. One model handles all 32 stages. It consumes RGB; actions are preserved for future dynamics training. This package does not yet implement M or C.

## Architecture

The default **spatial residual VAE** maps RGB `[B,3,240,256]` to a diagonal Gaussian on an `[8,30,32]` grid and decodes back to native RGB. Spatial structure gives small objects more representational capacity than the original 256-dimensional vector bottleneck. The API flattens the grid to **7,680 dimensions**; exports retain the grid shape in metadata. This increases the input/output cost of the future memory model. Use `--latent-channels` and `--downsample-factor 8|16` to explore that fidelity/compression tradeoff.

Residual convolution blocks, GroupNorm, SiLU, and bottleneck self-attention are followed by a nearest-neighbor upsampling/convolution decoder. There are no encoder-to-decoder skip connections: reconstruction must pass through the latent. Training samples the posterior; evaluation uses its mean. Width, residual depth, attention, spatial compression, and activation checkpointing are configurable. `--architecture vector --base-channels 16` retains the first architecture; its existing checkpoints still load for inference.

| Configuration | Width / residual blocks | Parameters | Latent shape |
| --- | --- | ---: | --- |
| Local sanity | 16 / 1 | 550,067 | 8×30×32 |
| CLI default | 32 / 2 | 2,957,267 | 8×30×32 |
| Cloud starting configuration | 64 / 2 | 11,784,083 | 8×30×32 |

The objective combines pixel MSE, L1, horizontal/vertical gradient error, and dynamic-object-region MSE, plus a warm-started KL term. Object boxes come from simulator metadata **only as loss/evaluation supervision**, never as encoder inputs. The ROI term focuses gradient on Mario, enemies, items, and projectiles. All terms use mean reductions; KL is averaged over latent coordinates, with optional free nats per coordinate after averaging over the microbatch. This free-nats reduction is microbatch-local under accumulation/DDP. `--roi-weight 0` disables object supervision. Coefficients are experimental starting points, not proven optima.

Training alone cannot guarantee the best representation: compare held-out sprite/geometry reconstructions and, once M/C exist, predictive/control performance. Increasing capacity or training time does not fix missing gameplay coverage.

## Setup and tests

From the repository root, using Python 3.10+:

```sh
source mario-recreation/.venv/bin/activate
python -m pip install -r vision/requirements-dev.txt
python -m pytest vision/tests -q
```

For a server, create a fresh environment and install an appropriate CUDA-enabled PyTorch build, then these requirements. [Game build and collector dependencies](../mario-recreation/README.md) are needed only when collecting on that machine. Tests include exact CPU mid-epoch resume with accumulation/EMA, spatial gradients, activation checkpointing, cache relocation, and actual two-process CPU training/resume. CUDA/NCCL and BF16 require validation on the target GPU; they cannot be hardware-tested on this Mac.

## Small all-level experiment

```sh
python vision/tools/collect_all.py --output mario-recreation/datasets/all-levels --episodes 3 --frames 360 --workers 4
python -m vision prepare --data mario-recreation/datasets/all-levels --output vision/artifacts/split.json --stride 48
python -m vision pack --manifest vision/artifacts/split.json --output vision/artifacts/cache
python -m vision train --config vision/configs/local-sanity.json --manifest vision/artifacts/split.json --cache vision/artifacts/cache --output vision/runs/local
python -m vision evaluate --manifest vision/artifacts/split.json --cache vision/artifacts/cache --checkpoint vision/runs/local/best.pt --split test --output vision/artifacts/evaluation
python vision/tools/report.py --run vision/runs/local --evaluation vision/artifacts/evaluation --output vision/artifacts/report
```

Open the report's `index.html` for a loss curve, all-level metrics, and reconstruction/error sheets. The completed local experiment and its exact paths are in [ALL_LEVELS_VALIDATION.md](ALL_LEVELS_VALIDATION.md). It is deliberately tiny and not a production training set.

Prepare verifies image hashes, dimensions, and frame continuity, deduplicates exact visual episodes, and splits whole episodes by starting level. Each level needs at least three distinct episodes to appear in all three splits. Roughly 15% each are reserved for validation and test, with a minimum of one when available. Different episodes can share initial observations, so this measures held-out trajectory reconstruction, not unseen-level generalization. For production, collect many more trajectories reaching later sections, different power-ups, enemies, deaths, and transitions; the random collector is not an expert solver.

Balanced training includes every sampled frame each epoch and oversamples shorter levels to match the longest. This guarantees all represented levels are visited, unlike replacement-only random sampling. `--uniform-sampling` uses each frame once. DDP partitions the same seeded global order across ranks, padding a few repeated examples to equalize rank lengths. `--max-batches` truncates coverage and is only for smoke checks.

## Long runs on a server

Transfer the repository, split manifest, packed cache, and optionally the original recordings. New manifests use paths relative to a recorded data root; pass `--data-root /new/location` after moving recordings. The fingerprint stays unchanged. A packed cache alone is sufficient for train/evaluate. Dense latent export requires original recordings. Old absolute-path manifests should be prepared again before a new server run.

`pack` verifies source PNGs once and creates uint8 memory-mapped `.npy` arrays. This avoids repeatedly decoding/hashing thousands of small PNGs. Cache metadata validates the manifest fingerprint and arrays' shapes/dtypes on load; treat the packed files as immutable trusted artifacts. For large corpora, put the cache on local NVMe/EBS rather than reading per-frame files directly from S3. Native frames use about **184 GB per million sampled frames**, uncompressed. The manifest/index and sampler still live in RAM; this implementation is not a streaming billion-frame dataset engine.

Single CUDA GPU:

```sh
python -m vision train --config vision/configs/cloud.json --manifest /data/split.json --cache /data/cache --output /checkpoints/mario-v --device cuda
```

Four GPUs on one server:

```sh
torchrun --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=4 -m vision train --config vision/configs/cloud.json --manifest /data/split.json --cache /data/cache --output /checkpoints/mario-v --device cuda
```

The same trainer uses DDP with NCCL on CUDA and Gloo on CPU. Effective batch size is `batch_size × accum_steps × world_size` (smaller at an epoch's final partial group). The example cloud configuration yields 32 frames per GPU per optimizer step; four GPUs yield 128. Adjust the microbatch for GPU memory and tune learning rate rather than assuming larger batches improve results.

The trainer supports CUDA BF16/FP16 autocast, FP16 gradient scaling, activation checkpointing, gradient accumulation with DDP synchronization only on updates, gradient clipping, EMA weights, warmup/cosine learning rate, KL warmup, configurable workers/prefetch, and pinned CUDA transfers. MPS/CPU use FP32. Learning-rate decay is set by **optimizer steps**, independently of the requested epoch count. `decay_steps` in the cloud configuration is a starting budget: choose it to match the actual dataset and intended duration. No AWS instance is provisioned by these commands.

## Checkpoints and resume

`config.json` records the resolved architecture/options; `history.jsonl` records training components, per-level validation, posterior statistics, and epoch times. `last.pt` is saved atomically at configurable optimizer-step intervals and each epoch. `best.pt` is selected by mean per-level validation MSE (equal weight per level), using EMA weights when enabled. Training does not read test data. Reconstruction previews are paginated so every level is included.

Resume the cloud run with the same launch command and add:

```sh
--resume /checkpoints/mario-v/last.pt --epochs 200
```

`--epochs` is the total target. Keep the same architecture, options, dataset fingerprint, and world size. Paths may change: copy the run directory, pass its copied `last.pt` and matching `--output`, and provide the relocated manifest/cache. Explicit CLI flags override JSON configuration values. Checkpoints include raw and EMA weights, optimizer, scaler, step/epoch/batch position, per-rank RNG states, and partial epoch statistics. Mid-epoch resume reconstructs the sampler and skips already consumed batches. Preemption loses work since the last completed checkpoint. A crash between writing history and saving the epoch checkpoint can leave a repeated final history entry on replay; checkpoints are the authority for progress.

Old initial-trainer checkpoints remain inference-compatible but cannot resume with the new optimizer/options schema. CPU continuation is tested for exact weight equality; device changes and GPU kernels can change numerical trajectories. Changing world size intentionally requires a new training run; exact resharded resume is not implemented. Keep checkpoint storage persistent and copy it off an instance before terminating it.

## Evaluate and export for M

Evaluation produces aggregate/per-level MSE, MAE, PSNR, dynamic-region MSE, posterior KL/std/active dimensions, checkpoint provenance, and original/reconstruction/error sheets. High background PSNR alone is not evidence that Mario survives compression. Use the validation split for tuning and reserve test data for a final assessment.

```sh
python -m vision encode --manifest /data/split.json --data-root /data/recordings --checkpoint /checkpoints/mario-v/best.pt --split train --output /data/train-latents
```

Exports preserve **every original frame**, even when V training used a frame stride: float32 `mu.npy`/`logvar.npy`, row-aligned `index.jsonl`, and metadata with grid shape and checkpoint/data fingerprints. Arrays are written with bounded memory and can be read with `numpy.load(..., mmap_mode='r')`. Default inference/export uses EMA when available. Flattening order is channel, height, width. Use `z = mu` deterministically or `mu + exp(0.5*logvar)*epsilon` to sample.

Freeze V before preparing M's data. Row t's action advances frame t to t+1; final observations have null actions. Never create transitions across episode boundaries. Original recordings retain termination/truncation and full state; the export is not a complete RL replay buffer. Spatial latents are larger: the default mu/logvar pair occupies about **61 GB per million dense frames**.

Implementation references: [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [AMP and accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html). The model is a custom VMC-style baseline; no claim is made that these hyperparameters are optimal.
