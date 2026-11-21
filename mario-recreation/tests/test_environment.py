import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mario_env import MarioEnv, Action


def test_seeded_reset_and_replay_are_identical():
    with MarioEnv(seed=123) as env:
        sequences = []
        for _ in range(2):
            rgb, state = env.reset()
            assert rgb.shape == (240, 256, 3) and rgb.dtype == np.uint8
            trajectory = [(rgb, state)]
            for action, frames in [(1, 40), (5, 28), (1, 30), (2, 12), (0, 20)]:
                rgb, _, term, trunc, state = env.step(action, frames)
                assert not term and not trunc
                trajectory.append((rgb, state))
            sequences.append(trajectory)
        for (a, sa), (b, sb) in zip(*sequences):
            assert sa == sb
            np.testing.assert_array_equal(a, b)


def test_walk_run_acceleration_and_release():
    with MarioEnv(render=False) as env:
        env.reset()
        _, _, _, _, start = env.step(Action.RIGHT, 1)
        assert start["player"]["vx"] == pytest.approx(0x98 / 4096)
        _, _, _, _, walk = env.step(Action.RIGHT, 59)
        assert walk["player"]["vx"] == 1.5
        _, _, _, _, stopped = env.step(Action.NOOP, 45)
        assert stopped["player"]["vx"] == 0
        assert stopped["player"]["x"] > walk["player"]["x"]
        env.reset()
        _, _, _, _, run = env.step(Action.RIGHT | Action.RUN, 60)
        assert run["player"]["vx"] == 2.5
        assert run["player"]["x"] > walk["player"]["x"]


def test_variable_jump_height_no_double_jump_and_landing():
    def jump(hold):
        with MarioEnv(render=False) as env:
            env.reset()
            env.step(0)  # Establish ground contact.
            ys = []
            for f in range(95):
                action = Action.JUMP if f < hold or f == 8 else Action.NOOP
                _, _, term, _, state = env.step(action)
                assert not term
                ys.append(state["player"]["y"])
                if f == 8:
                    assert state["player"]["vy"] > -4  # No new mid-air impulse.
            return ys
    short, long = jump(1), jump(35)
    assert min(long) < min(short) - 20
    assert short[-1] == long[-1] == 192


def test_terminal_is_not_truncation():
    with MarioEnv(max_frames=5, render=False) as env:
        env.reset()
        _, _, terminal, truncated, state = env.step(1, 30)
        assert state["frame"] == 5
        assert truncated and not terminal
        with pytest.raises(RuntimeError):
            env.step(1)
    with MarioEnv(render=False) as env:
        env.reset()
        # Walking into the first goomba ends the episode, before automatic respawn.
        _, _, terminal, truncated, state = env.step(1, 300)
        assert terminal and not truncated and state["dead"]
        assert state["frame"] < 300


def test_input_validation():
    with pytest.raises(ValueError):
        MarioEnv(level="9-9")
    with MarioEnv(render=False) as env:
        env.reset()
        for action in [-1, 64, 1.5]:
            with pytest.raises(ValueError):
                env.step(action)
        for frames in [0, -1, 3601]:
            with pytest.raises(ValueError):
                env.step(1, frames)


def test_question_block_dispenses_one_coin_and_becomes_used():
    with MarioEnv(render=False) as env:
        _, state = env.reset()
        while state["player"]["x"] < 220:
            _, _, _, _, state = env.step(Action.RIGHT)
        env.step(Action.NOOP, 42)
        _, _, terminal, _, state = env.step(Action.JUMP, 15)
        assert not terminal
        assert state["coins"] == 1
        assert state["score"] == 200


def test_flying_fish_callback_survives_stomp_regression():
    with MarioEnv(level="2-3", render=False) as env:
        env.reset()
        for frame in range(600):
            _, _, terminal, _, state = env.step(13 if frame % 65 < 35 else 9)
            if terminal:
                break
        assert state["frame"] > 423


@pytest.mark.parametrize("level", [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)])
def test_every_stage_loads_and_advances(level):
    with MarioEnv(level=level) as env:
        rgb, state = env.reset()
        assert state["level"] == level
        assert np.unique(rgb.reshape(-1, 3), axis=0).shape[0] > 5
        rgb, _, _, _, state = env.step(Action.RIGHT, 8)
        assert state["frame"] == 8
        assert np.isfinite([state["player"]["x"], state["player"]["y"]]).all()
