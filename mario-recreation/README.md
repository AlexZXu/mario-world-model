# Mario recreation for world-model data

A local, playable recreation of **NES Super Mario Bros. (1985)**, with all 32 authored stages and a deterministic, frame-stepped Python interface. Built on Gold872's MIT-licensed C++/SDL recreation, with local rendering, physics, determinism, and recording changes. No ROM download is required.

This is a recreation, **not cycle-accurate NES emulation**. See [the validation report](validation/REPORT.md) for measured matches, screenshot comparisons, and remaining differences. Treat generated trajectories as synthetic data, not as ground-truth NES trajectories.

## Play

The executable is already built in this workspace:

```sh
cd mario-recreation
./build/mario
./build/mario --level 2-2
```

| Input | Action |
|---|---|
| Left/right arrows or A/D | Move |
| Space or Z | Jump; hold for a higher jump |
| Shift or X | Run / throw fireballs when powered up |
| Down arrow or S | Duck / enter a compatible pipe |
| Q | Throw fireball without running |
| Escape | Pause |
| Close window | Quit |

The fixed 512×480 window displays the 256×240 pixel scene at 2× scale. `--mute` disables sound. The game starts at the requested stage after its intro card, and advances through subsequent stages.

## Build on another machine

macOS:

```sh
brew install cmake pkg-config sdl2 sdl2_image sdl2_ttf sdl2_mixer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Ubuntu/Debian:

```sh
sudo apt install build-essential cmake pkg-config libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Resource paths are resolved from the source location compiled into the executable, so launch it from any working directory. Rebuild after moving the project. Linux instructions are provided but the current validation was performed on macOS arm64.

## Generate training data

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tools/collect.py --output datasets/run-001 --episodes 10 --frames 1800 --seed 42
.venv/bin/python tools/replay.py datasets/run-001/episode-00000
```

In this workspace `.venv` is already installed. Use `--policy random` for seeded random action bursts, and `--level 4-1` to select another stage. The default exploratory policy is deliberately simple and can die; it is not a solver. Each worker/environment owns its native process, so environments can run independently in parallel.

Every episode contains:

- `rgb/000000.png`, …: native RGB observations, with no debug overlays or rescaling.
- `steps.jsonl`: observation path/hash, action bitmask, terminal/truncation flags, and structured state. Row `t` contains `s_t` and the action that produces `s_(t+1)`. The last observation has `action: null`.
- The dataset `manifest.json`: dimensions, tick rate, seeds, policy, action mapping, and executable/resource hashes.

One action advances exactly one 60 Hz simulation tick by default. No wall-clock time, dropped render frames, or audio callbacks drive the simulation. RGB frames include the normal in-game HUD. State includes world/stage, camera, player position/velocity/power, score, timer, and visible collision objects, enemies, and items. Positions use **native world pixels**, not the 2× window coordinates. Structured state is privileged supervision and should not be passed to a model intended to learn from pixels alone.

Replay verifies both state equality and PNG hashes. Use the same executable, resources, and SDL/image libraries for byte-exact replay. The seeded PRNG is fixed; cross-platform floating-point identity has not been certified. Episodes end on first death or final game completion; stage transitions are retained. Time limits use `truncated`, not `terminated`.

## Python API

```python
from mario_env import MarioEnv, Action

with MarioEnv(level="1-1", seed=7, max_frames=1800) as env:
    rgb, info = env.reset()
    rgb, reward, terminated, truncated, info = env.step(
        Action.RIGHT | Action.RUN | Action.JUMP
    )
    env.capture("jump.png")
```

The API follows Gym's return convention but has no Gym dependency and is not a `gymnasium.Env` subclass. `render=False` skips reading pixels on every step; call `capture` or `render` explicitly. `step(action, frames=4)` holds the action for up to four ticks and stops early on death. For frame-by-frame training, retain the default of one. The auxiliary reward combines horizontal progress, score changes, and a death penalty; it is not an original game mechanic and is not used by the recorder.

Action bits: right=1, left=2, jump=4, run=8, down=16, fire=32. To model the original controller's B button in data generation, set **both run and fire**. Reset starts a fresh process to clear all timers, singleton state, and queued callbacks.

## Validation and reference material

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python tools/validate_visuals.py
```

See [validation/REPORT.md](validation/REPORT.md), [the comparison sheet](validation/comparison.png), and [captured gameplay](validation/moments.png). Reference PNGs were extracted from actual NES footage; they are not renders of this recreation. The report distinguishes footage review from pixel comparison with a static level map.

## Source and asset attribution

See [UPSTREAM.md](UPSTREAM.md) and [vendor/LICENSE](vendor/LICENSE). The engine source's MIT license does not transfer ownership of Nintendo's characters, graphics, sounds, or game design. The included artwork/audio retain their original ownership. No claim of Nintendo affiliation is made.
