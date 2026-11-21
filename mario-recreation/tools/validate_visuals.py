"""Run gameplay, capture moments, and compare with retained NES reference frames."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mario_env import MarioEnv, ROOT

OUT = ROOT / "validation"
SHOTS = OUT / "screenshots"


def capture_route():
    targets = {430: "pipes", 610: "pipe-jump", 700: "goombas"}
    states = {}
    with MarioEnv(render=False) as env:
        _, state = env.reset()
        env.capture(SHOTS / "opening.png")
        states["opening"] = state
        jump_ticks, previous_y, last_ground_jump = 0, 192, False
        for frame in range(650):
            p = state["player"]
            x, y = p["x"], p["y"]
            grounded = p["vy"] == 0 and y >= previous_y
            previous_y = y
            enemy = any(o["kind"] == "enemy" and -4 < o["x"] - x < 52 and abs(o["y"] - y) < 30 for o in state["objects"])
            wall = any(o["kind"] == "solid" and 10 < o["x"] - x < 45 and o["y"] < y+16 and o["y"]+o["height"] > y for o in state["objects"])
            gap = y >= 175 and not any(o["kind"] == "solid" and o["x"] <= x+45 < o["x"]+o["width"] and o["y"] == 208 for o in state["objects"])
            if grounded and not last_ground_jump and jump_ticks == 0 and (enemy or wall or gap):
                jump_ticks = 34
            action = 9 | (4 if jump_ticks else 0)
            jump_ticks = max(0, jump_ticks-1)
            last_ground_jump = grounded and bool(action & 4)
            _, _, terminal, _, state = env.step(action)
            for target, name in targets.items():
                if name not in states and state["camera"][0] >= target:
                    env.capture(SHOTS / f"{name}.png")
                    states[name] = state
            if terminal or len(states) == 4:
                break
    if len(states) != 4:
        raise AssertionError("Validation route failed to reach the comparison scenes")
    with MarioEnv(render=False) as env:
        _, state = env.reset()
        while state["player"]["x"] < 220:
            _, _, _, _, state = env.step(1)
        env.step(0, 42)
        _, _, _, _, state = env.step(4, 15)
        assert state["coins"] == 1 and state["score"] == 200
        env.capture(SHOTS / "coin-block.png")
        states["coin-block"] = state
    for level, name, actions in [
        ("1-2", "underground", [(0, 420)]),
        ("2-2", "underwater", [(1, 240), (5, 20)]),
        ("1-4", "castle", [(1, 35), (5, 15)]),
    ]:
        with MarioEnv(level=level, render=False) as env:
            env.reset()
            for action, frames in actions:
                _, _, terminal, _, state = env.step(action, frames)
                if terminal:
                    break
            env.capture(SHOTS / f"{name}.png")
            states[name] = state
    (OUT / "capture-states.json").write_text(json.dumps(states, indent=2) + "\n")
    return states


def sheet(images, columns, path, scale=2, crop=False):
    width, height = 256*scale, (224 if crop else 240)*scale
    canvas = Image.new("RGB", (columns*width, ((len(images)+columns-1)//columns)*(height+34)), "#15191f")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(images):
        if crop and image.height == 240:
            image = image.crop((0, 8, 256, 232))
        x, y = i % columns*width, i//columns*(height+34)
        canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.NEAREST), (x, y+34))
        draw.text((x+10, y+10), label, fill="#eff4ff")
    canvas.save(path)


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    states = capture_route()
    reference = Image.open(OUT / "reference/world1-1-map.png").convert("RGB").crop((0, 0, 256, 240))
    opening = Image.open(SHOTS / "opening.png").convert("RGB")
    a, b = np.asarray(reference), np.asarray(opening)
    mask = np.ones((240, 256), dtype=bool)
    mask[:40] = False  # HUD isn't part of a background map.
    mask[192:208, 32:48] = False  # Player isn't part of a background map.
    equal = np.all(a == b, axis=2)
    metrics = {"opening_static_region": {
        "compared_pixels": int(mask.sum()), "matching_pixels": int(equal[mask].sum()),
        "exact_match_fraction": float(equal[mask].mean()),
        "excluded": "top 40 rows (HUD); x=32..47,y=192..207 (Mario)",
        "scope": "Only the opening scene's static geometry/palette, not whole-game accuracy",
    }, "footage": [], "binary_sha256": hashlib.sha256((ROOT / "build/mario").read_bytes()).hexdigest()}
    pairs = []
    # The encoder removes 8 rows above and below the native 240-row image.
    for t, camera, name in [(12,430,"pipes"), (14,610,"pipe-jump"), (15,700,"goombas")]:
        pairs.append((Image.open(OUT / f"reference/nes-{t}.png"), f"NES footage at {t}s | camera ~{camera}px"))
        pairs.append((Image.open(SHOTS / f"{name}.png"), f"Recreation tick {states[name]['frame']} | camera {states[name]['camera'][0]:.1f}px"))
        metrics["footage"].append({"timestamp_seconds": t, "reference_camera_inferred": camera,
            "capture": name, "recreation_camera": states[name]["camera"][0],
            "note": "Similar location, different actions and animation phase. Visual comparison, not a synchronized trajectory equivalence test."})
    sheet(pairs, 2, OUT / "comparison.png", crop=True)
    sheet([(reference, "NES static map"), (opening, "Recreation: static scenery matches")], 2, OUT / "opening-comparison.png")
    names = ["opening", "coin-block", "pipe-jump", "underground", "underwater", "castle"]
    sheet([(Image.open(SHOTS / f"{name}.png"), name) for name in names], 3, OUT / "moments.png")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    assert metrics["opening_static_region"]["exact_match_fraction"] == 1.0
    print(json.dumps(metrics["opening_static_region"], indent=2))
    print("Wrote comparison.png, opening-comparison.png, moments.png, capture-states.json, metrics.json")


if __name__ == "__main__":
    main()
