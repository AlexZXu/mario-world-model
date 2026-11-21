"""Write frame-aligned RGB/action/state trajectories, with bounded memory."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mario_env import MarioEnv, Action, ROOT, fingerprint


def policy(state, rng, frame):
    # Intentionally varied exploratory behavior; this is not an optimal agent.
    # Pulse jump with release frames, so landing doesn't imply automatic jumping.
    action = Action.RIGHT
    if frame % 300 < 230:
        action |= Action.RUN | Action.FIRE
    if frame % 65 < 35:
        action |= Action.JUMP
    if frame % 480 in range(400, 425):
        action = Action.LEFT
    if frame % 480 in range(425, 440):
        action = Action.DOWN
    if frame % 90 == 0:
        action = rng.choice([action, Action.NOOP, Action.RIGHT, Action.RIGHT | Action.JUMP])
    return int(action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/demo")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--policy", choices=["explore", "random"], default="explore")
    args = parser.parse_args()
    if args.episodes < 1 or args.frames < 1:
        parser.error("episodes and frames must be positive")
    # Never silently overwrite an existing dataset.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "mario-trajectories-v1", "level": args.level,
        "fps": 60, "rgb_shape": [240, 256, 3], "seed": args.seed,
        "policy": args.policy, "episodes": args.episodes, "max_frames": args.frames,
        "action_bits": {a.name: int(a) for a in Action},
        "alignment": "row t: observation=s_t; action advances to s_(t+1); terminal row has action=null",
        **fingerprint(),
        "fidelity": "recreation, not cycle-accurate NES emulation; see validation/REPORT.md",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for episode in range(args.episodes):
        folder = args.output / f"episode-{episode:05}"
        folder.mkdir()
        (folder / "rgb").mkdir()
        rng = random.Random(args.seed + episode)
        with MarioEnv(args.level, args.seed + episode, args.frames, render=False) as env, (folder / "steps.jsonl").open("w") as rows:
            _, info = env.reset()
            terminal = truncated = False
            held_action = 0
            while True:
                t = info["frame"]
                relative = f"rgb/{t:06}.png"
                image = env.capture(folder / relative)
                if args.policy == "random":
                    if t % 20 == 0:
                        held_action = rng.choice([0, 1, 2, 5, 9, 13, 41, 45, 16])
                    action = held_action
                else:
                    action = policy(info, rng, t)
                rows.write(json.dumps({"frame": t, "rgb": relative,
                    "rgb_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "action": None if terminal or truncated else action,
                    "terminated": terminal, "truncated": truncated, "state": info}, separators=(",", ":")) + "\n")
                if terminal or truncated:
                    break
                _, _, terminal, truncated, info = env.step(action)
        print(f"{folder.name}: {t} transitions, {'death/win' if terminal else 'time limit'}", flush=True)


if __name__ == "__main__":
    main()
