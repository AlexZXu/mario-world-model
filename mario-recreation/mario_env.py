"""Process-isolated, deterministic Mario environment. No ROM or browser required."""
from __future__ import annotations

import enum
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def fingerprint(binary=None):
    """Identify both executable and data; map edits need not change the binary."""
    binary = Path(binary or os.environ.get("MARIO_BINARY", ROOT / "build/mario"))
    resources = hashlib.sha256()
    for path in sorted((ROOT / "vendor/res").rglob("*")):
        if path.is_file():
            resources.update(path.relative_to(ROOT / "vendor/res").as_posix().encode())
            resources.update(b"\0")
            resources.update(hashlib.sha256(path.read_bytes()).digest())
    return {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "resources_sha256": resources.hexdigest()}


class Action(enum.IntFlag):
    NOOP = 0
    RIGHT = 1
    LEFT = 2
    JUMP = 4
    RUN = 8
    DOWN = 16
    FIRE = 32


class MarioEnv:
    """Gym-style API (not a gymnasium.Env subclass).

    Observations: uint8 RGB [240,256,3]. State coordinates are native pixels,
    velocities pixels/tick. One tick is 1/60 s. `reset` starts a fresh process,
    including RNG, timers, static engine state and pending callbacks.
    An episode terminates on first death or game completion. Level transitions
    stay within the episode. `max_frames` is a truncation, never a death.
    """

    def __init__(self, level="1-1", seed=0, max_frames=18000, render=True, binary=None):
        if level not in {f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)}:
            raise ValueError("level must be 1-1 through 8-4")
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self.level, self.seed, self.max_frames = level, seed, max_frames
        self.render_enabled = render
        self.binary = Path(binary or os.environ.get("MARIO_BINARY", ROOT / "build/mario")).resolve()
        self.process = None
        self.info = None
        self.truncated = False
        self._temp = tempfile.TemporaryDirectory(prefix="mario-env-")
        self._frame_path = Path(self._temp.name) / "frame.png"

    def _read(self):
        # The native engine writes exactly one JSON line for each command.
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(timeout=30):
                self.close()
                raise TimeoutError("Mario did not respond within 30 seconds")
            line = self.process.stdout.readline()
        finally:
            selector.close()
        if not line:
            self.process.wait(timeout=5)
            self._stderr.seek(0)
            raise RuntimeError(f"Mario exited ({self.process.returncode}): {self._stderr.read()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def _request(self, command):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Call reset() before using the environment")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self._read()

    def reset(self, *, seed=None):
        self._stop()
        if seed is not None:
            self.seed = seed
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if not self.binary.is_file():
            raise FileNotFoundError(f"Build the game first: cmake -S {ROOT} -B {ROOT / 'build'} && cmake --build {ROOT / 'build'}")
        self._stderr = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            [str(self.binary), "--headless", "--level", self.level, "--seed", str(self.seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1,
        )
        self.info = self._read()
        self.truncated = False
        return self.render() if self.render_enabled else None, self.info

    def step(self, action=Action.NOOP, frames=1):
        if self.info is None:
            raise RuntimeError("Call reset() before step()")
        if self.info["terminated"] or self.truncated:
            raise RuntimeError("Episode finished; call reset()")
        if not isinstance(action, (int, Action)) or not 0 <= int(action) <= 63:
            raise ValueError("action must be a six-bit integer (0..63)")
        if not isinstance(frames, int) or not 1 <= frames <= 3600:
            raise ValueError("frames must be 1..3600")
        frames = min(frames, self.max_frames - self.info["frame"])
        before = self.info
        self.info = self._request(f"step {int(action)} {frames}")
        terminal = self.info["terminated"]
        self.truncated = not terminal and self.info["frame"] >= self.max_frames
        # Explicit auxiliary reward; raw score and state are always preserved.
        progress = self.info["player"]["x"] - before["player"]["x"] if self.info["level"] == before["level"] else 0
        reward = progress + (self.info["score"] - before["score"]) / 100 - 100 * self.info["dead"]
        return self.render() if self.render_enabled else None, reward, terminal, self.truncated, self.info

    def capture(self, path):
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # std::quoted recognizes escaped quotes and backslashes.
        escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
        if "\n" in escaped or "\r" in escaped:
            raise ValueError("capture path cannot contain newlines")
        self._request(f'capture "{escaped}"')
        return path

    def render(self):
        self.capture(self._frame_path)
        with Image.open(self._frame_path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)

    def _stop(self):
        if self.process:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write("quit\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
            self._stderr.close()
            self.process = None
        self.info = None

    def close(self):
        self._stop()
        self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
