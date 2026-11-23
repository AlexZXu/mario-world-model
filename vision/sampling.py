"""Deterministic epoch sampling with identical global order across DDP ranks."""
import math
from collections import Counter
import torch
from torch.utils.data import Sampler


class EpochSampler(Sampler):
    def __init__(self, records, seed, epoch, rank=0, world_size=1, balanced=True):
        self.records, self.seed, self.epoch = records, seed, epoch
        self.rank, self.world_size, self.balanced = rank, world_size, balanced

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        if self.balanced:
            # Include every frame once, oversample smaller levels to the largest.
            groups = {}
            for i,r in enumerate(self.records): groups.setdefault(r['level'],[]).append(i)
            size=max(map(len,groups.values()))
            ids=[]
            for group in groups.values():
                ids.extend(group)
                extra=size-len(group)
                ids.extend(group[i] for i in torch.randint(len(group),(extra,),generator=generator).tolist())
            ids=[ids[i] for i in torch.randperm(len(ids),generator=generator).tolist()]
        else:
            ids=torch.randperm(len(self.records),generator=generator).tolist()
        total=math.ceil(len(ids)/self.world_size)*self.world_size
        ids += (ids*math.ceil((total-len(ids))/max(1,len(ids))))[:total-len(ids)]
        return iter(ids[self.rank:total:self.world_size])

    def __len__(self):
        if self.balanced:
            counts=Counter(r['level'] for r in self.records)
            n=max(counts.values())*len(counts)
        else: n=len(self.records)
        return math.ceil(n/self.world_size)
