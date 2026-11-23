"""Collect independently seeded trajectories from every authored stage."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--frames', type=int, default=480)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=22000)
    a = p.parse_args()
    if min(a.episodes, a.frames, a.workers) < 1:
        p.error('episodes, frames, workers must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    collector = Path(__file__).resolve().parents[2]/'mario-recreation/tools/collect.py'
    levels = [f'{w}-{s}' for w in range(1,9) for s in range(1,5)]
    def collect(item):
        i, level = item
        command = [sys.executable, str(collector), '--output', str(a.output/level),
                   '--level', level, '--episodes', str(a.episodes), '--frames', str(a.frames),
                   '--policy', 'random', '--seed', str(a.seed+i*100)]
        result = subprocess.run(command, capture_output=True, text=True)
        (a.output/f'{level}.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'Collection failed for {level}: {result.stderr}')
        return level
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(collect,item) for item in enumerate(levels)]):
            print(f'Collected {future.result()}', flush=True)
    (a.output/'collection.json').write_text(json.dumps({
        'levels':levels, 'episodes_per_level':a.episodes, 'max_frames':a.frames,
        'seed':a.seed, 'policy':'random'},indent=2)+'\n')


if __name__ == '__main__':
    main()
