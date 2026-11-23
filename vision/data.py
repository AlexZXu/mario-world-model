"""Validated episode manifests and leakage-resistant episode-level splits."""
import hashlib
import io
import json
import os
from pathlib import Path
import random
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_json(path):
    return json.loads(Path(path).read_text())


def evaluation_state(state):
    """Keep only the box metadata used by ROI supervision/evaluation."""
    def box(obj):
        return {k:obj[k] for k in ('x','y','width','height')}
    return {'camera':state['camera'], 'player':box(state['player']),
            'objects':[dict(box(o),kind=o['kind']) for o in state.get('objects',[])
                       if o['kind'] in ('enemy','item','projectile')]}


def prepare(roots, output, seed=42, stride=4):
    if stride < 1:
        raise ValueError('stride must be positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = sorted({p.resolve() for root in roots for p in Path(root).rglob('steps.jsonl')})
    if not paths:
        raise ValueError('No recorded episodes found (expected steps.jsonl)')
    data_root = Path(os.path.commonpath([str(Path(r).resolve()) for r in roots]))
    groups, seen, duplicates = defaultdict(list), set(), []
    for path in paths:
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if not rows:
            raise ValueError(f'Empty episode: {path}')
        frames, hasher = [], hashlib.sha256()
        for i, row in enumerate(rows):
            if row['frame'] != i:
                raise ValueError(f'Non-consecutive frame index in {path}: {i}')
            image_path = (path.parent / row['rgb']).resolve()
            if not image_path.is_relative_to(path.parent):
                raise ValueError(f'Image path escapes episode: {image_path}')
            checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if row['rgb_sha256'] != checksum:
                raise ValueError(f'Image checksum mismatch: {image_path}')
            with Image.open(image_path) as image:
                if image.size != (256, 240) or image.mode not in ('RGB', 'RGBA'):
                    raise ValueError(f'Expected native RGB/RGBA 256x240: {image_path}')
            hasher.update(checksum.encode())
            # A vision split deduplicates identical visual episodes regardless of action labels.
            frames.append({'path': str(image_path.relative_to(data_root)), 'frame': i, 'level': row['state']['level'],
                           'state': evaluation_state(row['state']), 'action': row['action'], 'sha256': checksum})
        identity = hasher.hexdigest()
        if identity in seen:
            duplicates.append(str(path.parent))
            continue
        seen.add(identity)
        groups[rows[0]['state']['level']].append({'id': identity, 'source': str(path.relative_to(data_root)),
                                                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                                'frames': frames[::stride]})
    splits = {'train': [], 'val': [], 'test': []}
    rng = random.Random(seed)
    for level, episodes in sorted(groups.items()):
        rng.shuffle(episodes)
        n = len(episodes)
        # Reserve at least one complete episode per available held-out split.
        ntest = max(1, int(n * .15)) if n >= 3 else 0
        nval = max(1, int(n * .15)) if n >= 2 else 0
        splits['test'] += episodes[:ntest]
        splits['val'] += episodes[ntest:ntest+nval]
        splits['train'] += episodes[ntest+nval:]
    if any(not episodes for episodes in splits.values()):
        raise ValueError('Need train/val/test episodes; record at least 3 visually distinct episodes for a level')
    result = {'schema': 'mario-vision-split-v1', 'data_root': str(data_root), 'seed': seed, 'stride': stride,
              'duplicate_episodes_removed': duplicates, 'splits': splits}
    result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return {name: {'episodes': len(eps), 'frames': sum(len(e['frames']) for e in eps)} for name, eps in splits.items()}


class FrameDataset(Dataset):
    def __init__(self, manifest, split, full_sequence=False, data_root=None, cache=None):
        self.manifest = read_json(manifest)
        content = dict(self.manifest)
        expected = content.pop('fingerprint')
        if hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != expected:
            raise ValueError('Dataset manifest fingerprint mismatch')
        root = Path(data_root or self.manifest.get("data_root", "."))
        self.records = []
        for episode in self.manifest['splits'][split]:
            frames = [dict(f, path=str(root/f['path'])) for f in episode['frames']]
            if full_sequence:
                source = root/episode['source']
                raw = source.read_bytes()
                if hashlib.sha256(raw).hexdigest() != episode.get('source_sha256'):
                    raise ValueError('Source trajectory changed or lacks provenance; prepare a new manifest')
                rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
                frames = [{'path': str((source.parent/r['rgb']).resolve()), 'frame': r['frame'],
                           'level': r['state']['level'], 'state': evaluation_state(r['state']), 'action': r['action'],
                           'sha256': r['rgb_sha256']} for r in rows]
            self.records.extend(dict(f, episode=episode['id']) for f in frames)
        if not self.records:
            raise ValueError(f'Empty {split} split')

        self.cache_path = None
        self._cache = None
        if cache and not full_sequence:
            metadata = read_json(Path(cache)/"metadata.json")
            if metadata["fingerprint"] != expected:
                raise ValueError("Packed cache and manifest differ")
            self.cache_path = str(Path(cache)/f"{split}.npy")
            array = np.load(self.cache_path, mmap_mode="r")
            if array.shape != (len(self.records),240,256,3) or array.dtype != np.uint8:
                raise ValueError("Invalid packed cache shape/dtype")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.cache_path:
            if self._cache is None:
                self._cache = np.load(self.cache_path, mmap_mode="r")
            rgb = np.array(self._cache[index],dtype=np.float32)/255
            return torch.from_numpy(rgb).permute(2,0,1), index
        record = self.records[index]
        raw = Path(record['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f"Image changed after preparation: {record['path']}")
        with Image.open(io.BytesIO(raw)) as image:
            rgb = np.array(image.convert('RGB'), dtype=np.float32) / 255
        return torch.from_numpy(rgb).permute(2, 0, 1), index


def pack(manifest, output, data_root=None):
    """Verify PNGs once and pack sampled frames into portable memory-mapped arrays."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    counts={}
    for split in ('train','val','test'):
        dataset=FrameDataset(manifest,split,data_root=data_root)
        array=np.lib.format.open_memmap(output/f'{split}.npy',mode='w+',dtype='uint8',
                                       shape=(len(dataset),240,256,3))
        for i in range(len(dataset)):
            rgb,_=dataset[i]
            array[i]=(rgb.permute(1,2,0).numpy()*255).round().astype('uint8')
        array.flush();counts[split]=len(dataset)
    (output/'metadata.json').write_text(json.dumps({'fingerprint':dataset.manifest['fingerprint'],
        'frames':counts,'format':'native-rgb-uint8-v1'},indent=2)+'\n')
    return counts
