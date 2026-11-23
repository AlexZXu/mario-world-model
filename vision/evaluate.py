"""Deterministic posterior-mean reconstructions, level/ROI metrics, and previews."""
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader


def roi_mask(record):
    mask = torch.zeros((240, 256), dtype=torch.bool)
    state = record['state']
    cx, cy = state['camera']
    boxes = [state['player']] + [o for o in state.get('objects', []) if o['kind'] in ('enemy', 'item', 'projectile')]
    for box in boxes:
        x0, y0 = math.floor(box['x']-cx), math.floor(box['y']-cy)
        x1, y1 = math.ceil(box['x']+box['width']-cx), math.ceil(box['y']+box['height']-cy)
        if x1 > 0 and y1 > 0 and x0 < 256 and y0 < 240:
            mask[max(0,y0):min(240,y1), max(0,x0):min(256,x1)] = True
    return mask


def image_from_tensor(x):
    return Image.fromarray((x.detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8'))


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=8, output=None, preview=True):
    model.eval()
    totals = defaultdict(lambda: dict(n=0, mse=0., mae=0., roi_sse=0., roi_values=0))
    sum_mu = torch.zeros(model.config.encoded_dim, dtype=torch.float64)
    sum_mu2 = sum_mu.clone()
    std_sum = kl_sum = 0.
    for images, indices in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images = images.to(device)
        reconstruction, mu, logvar = model(images, sample=False)
        error = (reconstruction-images).cpu()
        mu_cpu = mu.cpu().double()
        sum_mu += mu_cpu.sum(0)
        sum_mu2 += mu_cpu.square().sum(0)
        std_sum += float((logvar*.5).exp().mean(1).sum().cpu())
        kl_sum += float((-.5*(1+logvar-mu.square()-logvar.exp())).mean(1).sum().cpu())
        for i, index in enumerate(indices.tolist()):
            record = dataset.records[index]
            mse, mae = float(error[i].square().mean()), float(error[i].abs().mean())
            mask = roi_mask(record)
            values = error[i][:, mask]
            for key in ['all', record['level']]:
                t = totals[key]
                t['n'] += 1; t['mse'] += mse; t['mae'] += mae
                t['roi_sse'] += float(values.square().sum()); t['roi_values'] += values.numel()
    metrics = {}
    for key, t in totals.items():
        mse = t['mse']/t['n']
        metrics[key] = {'frames': t['n'], 'mse': mse, 'mae': t['mae']/t['n'],
                        'psnr_db': -10*math.log10(max(mse, 1e-12)),
                        'dynamic_roi_mse': t['roi_sse']/t['roi_values'] if t['roi_values'] else None,
                        'dynamic_roi_channel_values': t['roi_values']}
    n = totals['all']['n']
    variance = (sum_mu2/n-(sum_mu/n).square()).clamp_min(0)
    metrics['latent'] = {'dimensions': model.config.encoded_dim,
        'active_dimensions_var_gt_0.01': int((variance > .01).sum()),
        'mean_posterior_std': std_sum/n, 'kl_per_dimension': kl_sum/n,
        'mu_mean': (sum_mu/n).tolist(), 'mu_std': variance.sqrt().tolist()}
    if output:
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        if not preview:
            return metrics
        chosen = []
        for level in sorted({r['level'] for r in dataset.records}):
            ids = [i for i,r in enumerate(dataset.records) if r['level']==level]
            chosen += [ids[0], ids[len(ids)//2]]
        for page in range(0, len(chosen), 16):
            page_ids = chosen[page:page+16]
            canvas = Image.new('RGB', (256*3, 266*len(page_ids)), '#15191f')
            draw = ImageDraw.Draw(canvas)
            for row, index in enumerate(page_ids):
                image, _ = dataset[index]
                reconstructed, _, _ = model(image.unsqueeze(0).to(device), sample=False)
                reconstructed = reconstructed[0].cpu()
                error = (reconstructed-image).abs().mean(0).numpy()
                heat = np.zeros((240,256,3), dtype=np.uint8)
                heat[:,:,0] = np.minimum(error*4, 1)*255
                y = row*266
                label = dataset.records[index]
                for col, (im, title) in enumerate([(image_from_tensor(image),'Original'),
                    (image_from_tensor(reconstructed),'Reconstructed'), (Image.fromarray(heat),'Absolute error x4')]):
                    draw.text((col*256+6,y+6), f"{title} | {label['level']} t={label['frame']}", fill='white')
                    canvas.paste(im, (col*256,y+26))
            canvas.save(output/('reconstructions.png' if page == 0 else f'reconstructions-{page//16+1}.png'))
    return metrics
