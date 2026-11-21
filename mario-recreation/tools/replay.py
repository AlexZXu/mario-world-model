"""Replay a collected episode and verify every image hash and structured state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mario_env import MarioEnv, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.episode.parent / "manifest.json").read_text())
    for key, value in fingerprint().items():
        if key in manifest and manifest[key] != value:
            raise RuntimeError(f"{key} differs from the recorded dataset; use its original build/resources")
    index = int(args.episode.name.split("-")[-1])
    with MarioEnv(manifest["level"], manifest["seed"] + index, manifest["max_frames"], render=False) as env:
        _, state = env.reset()
        for count, line in enumerate((args.episode / "steps.jsonl").read_text().splitlines()):
            row = json.loads(line)
            if state != row["state"]:
                raise AssertionError(f"State diverged at frame {count}")
            frame = env.capture(env._frame_path)
            if hashlib.sha256(frame.read_bytes()).hexdigest() != row["rgb_sha256"]:
                raise AssertionError(f"Pixels diverged at frame {count}")
            if row["action"] is not None:
                _, _, _, _, state = env.step(row["action"])
        print(f"Verified {count + 1} observations: exact state and PNG hashes.")


if __name__ == "__main__":
    main()
