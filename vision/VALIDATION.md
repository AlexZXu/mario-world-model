# Initial validation

Validated on 2026-11-21 with Python 3.12, PyTorch 2.15.0, Apple MPS for the real-data experiment and CPU for automated tests.

## Automated checks

`python -m pytest vision/tests -q`: **7 passed**. Checks cover native shapes and gradients; deterministic inference; analytic KL; learning on a tiny example; whole-episode splits/deduplication; integrity failures; dense action/frame alignment; exact CPU epoch-boundary resume; and evaluation artifacts.

## Real game experiment

Collected three random-policy episodes each for levels 1-1, 1-2, 1-4, and 2-2 (420-frame cap, per-level seeds 101, 201, 401, 301). This is a small pipeline validation dataset, not comprehensive gameplay coverage. Preparation used stride 12 and seed 42:

| Split | Episodes | Sampled frames |
| --- | ---: | ---: |
| Train | 4 | 111 |
| Validation | 4 | 106 |
| Test | 4 | 115 |

Trained the default 256-dimensional, 6,981,587-parameter VAE on MPS for 12 epochs, then successfully resumed `last.pt` for epoch 13. Batch size 8, beta 0.001, warmup 100 steps, learning rate 0.0003, balanced level sampling. The final validation MSE was 0.035120 (PSNR 14.54 dB). Held-out test evaluation of the best checkpoint:

| Metric | Value |
| --- | ---: |
| MSE | 0.029018 |
| MAE | 0.099359 |
| PSNR | 15.37 dB |
| Dynamic object region MSE | 0.129420 |

Visually inspected the original/reconstruction/error contact sheet. Broad scene colors and layout reconstruct, but sprites, text, bricks, and fine collision geometry are blurred or absent. Dynamic-region error is much worse than aggregate error. **These bootstrap weights are not yet suitable for a playable world model or expert controller.** More diverse data and training are required; reconstruction quality must be reassessed, and the bottleneck/objective may need adjustment if small objects remain missing. This run establishes operational correctness, not sufficient representation quality or generalization across all 32 levels.

Dense train export produced `mu.npy` and `logvar.npy` of shape **(1292, 256)** despite the training stride of 12. Checked matching index rows, consecutive frame numbers within each episode, finite values, and null terminal actions.

Local generated artifacts (git-ignored):

- `vision/artifacts/vision-split.json`
- `vision/runs/initial-vae/best.pt` and `last.pt`
- `vision/artifacts/initial-evaluation/metrics.json`, `provenance.json`, `reconstructions.png`
- `vision/artifacts/train-latents/`
- `mario-recreation/datasets/vision-bootstrap/`

To reproduce evaluation of the existing local checkpoint:

```sh
python -m vision evaluate --manifest vision/artifacts/vision-split.json --checkpoint vision/runs/initial-vae/best.pt --output vision/artifacts/recheck
```

Output paths must be new. For future model selection, use validation data and reserve a fresh test set for the final assessment rather than repeatedly tuning against this bootstrap test set.
