import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .model import VisionVAE, VAEConfig


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    if name == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.device(name)


def load_model(path, device, use_ema=True):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != 'mario-vision-vae-v1':
        raise ValueError('Unsupported checkpoint schema')
    model = VisionVAE(VAEConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['ema'] if use_ema and checkpoint.get('ema') is not None else checkpoint['model'])
    model.eval()
    return model, checkpoint


def save_checkpoint(path, model, optimizer, **metadata):
    path = Path(path)
    payload = {'schema': 'mario-vision-vae-v1', 'config': asdict(model.config),
               'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'optimizer': optimizer.state_dict(), **metadata}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
