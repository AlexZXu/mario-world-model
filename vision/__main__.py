import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import prepare, FrameDataset, pack
from .runtime import load_model, device_for
from .evaluate import evaluate
from .train import train


def main():
    parser = argparse.ArgumentParser(description='Shared Mario vision VAE: prepare, train, evaluate, encode')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Validate recordings, deduplicate episodes, split by episode')
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stride', type=int, default=4)
    p = sub.add_parser('pack', help='Verify and pack sampled PNGs for fast memory-mapped loading')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-root')
    p = sub.add_parser('train')
    train_parser = p
    p.add_argument('--config', help='JSON hyperparameters; explicit CLI flags override')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--latent-dim', type=int, default=256)
    p.add_argument('--base-channels', type=int, default=32)
    p.add_argument('--architecture', choices=['vector','spatial'], default='spatial')
    p.add_argument('--latent-channels', type=int, default=8)
    p.add_argument('--residual-blocks', type=int, default=2)
    p.add_argument('--no-attention', action='store_true')
    p.add_argument('--downsample-factor',type=int,choices=[8,16],default=8)
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--precision', choices=['fp32','bf16','fp16'], default='fp32')
    p.add_argument('--lr-warmup-steps', type=int, default=100)
    p.add_argument('--decay-steps', type=int, default=100000)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--l1-weight', type=float, default=.1)
    p.add_argument('--edge-weight', type=float, default=.1)
    p.add_argument('--roi-weight', type=float, default=1.)
    p.add_argument('--free-nats', type=float, default=.01)
    p.add_argument('--ema-decay', type=float, default=.999)
    p.add_argument('--checkpoint-every', type=int, default=250)
    p.add_argument('--preview-every', type=int, default=5)
    p.add_argument('--data-root')
    p.add_argument('--cache')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--beta', type=float, default=1e-3)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-batches', type=int, default=0, help='Smoke-test cap per epoch; 0 uses all training frames')
    p.add_argument('--resume')
    p.add_argument('--uniform-sampling', action='store_true', help='Use uniform frames instead of balancing levels')
    p.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    for command in ['evaluate','encode']:
        p = sub.add_parser(command)
        p.add_argument('--manifest', required=True)
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--data-root')
        p.add_argument('--cache')
        p.add_argument('--split', choices=['train','val','test'], default='test')
        p.add_argument('--batch-size', type=int, default=8)
        p.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    args = parser.parse_args()
    if args.command == 'train' and args.config:
        settings = json.loads(Path(args.config).read_text())
        allowed = {a.dest for a in train_parser._actions} - {'help','config','manifest','output','resume','data_root','cache'}
        if not isinstance(settings,dict) or settings.keys()-allowed:
            parser.error('Config must contain only recognized training hyperparameters')
        train_parser.set_defaults(**settings)
        args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.data,args.output,args.seed,args.stride),indent=2))
    elif args.command == 'pack':
        print(json.dumps(pack(args.manifest,args.output,args.data_root),indent=2))
    elif args.command == 'train':
        train(args)
    else:
        device = device_for(args.device)
        model, checkpoint = load_model(args.checkpoint, device)
        dataset = FrameDataset(args.manifest, args.split, full_sequence=args.command == 'encode',data_root=args.data_root,cache=args.cache)
        if checkpoint['data_fingerprint'] != dataset.manifest['fingerprint']:
            raise ValueError('Checkpoint and dataset manifest differ')
        if args.command == 'evaluate':
            metrics = evaluate(model,dataset,device,args.batch_size,args.output)
            provenance = {'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                          'split': args.split, 'epoch': checkpoint['epoch'],
                          'data_fingerprint': dataset.manifest['fingerprint']}
            (Path(args.output)/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
            print(json.dumps({k:v for k,v in metrics.items() if k!='latent'},indent=2))
        else:
            output = Path(args.output); output.mkdir(parents=True,exist_ok=False)
            # Write directly to disk: no trajectory-sized RAM accumulation.
            shape = (len(dataset),model.config.encoded_dim)
            mu_file = np.lib.format.open_memmap(output/'mu.npy',mode='w+',dtype='float32',shape=shape)
            logvar_file = np.lib.format.open_memmap(output/'logvar.npy',mode='w+',dtype='float32',shape=shape)
            offset = 0
            with torch.inference_mode():
                for images, _ in DataLoader(dataset,batch_size=args.batch_size,shuffle=False):
                    mu, logvar = model.encode(images.to(device))
                    n=len(images)
                    mu_file[offset:offset+n]=mu.cpu().numpy()
                    logvar_file[offset:offset+n]=logvar.cpu().numpy()
                    offset += n
            mu_file.flush(); logvar_file.flush()
            with (output/'index.jsonl').open('w') as index:
                for record in dataset.records:
                    index.write(json.dumps({k:record[k] for k in ['episode','frame','level','path','action']})+'\n')
            (output/'metadata.json').write_text(json.dumps({'shape':shape,'split':args.split,
                'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                'data_fingerprint':dataset.manifest['fingerprint'],'stride':1,
                'vision_training_stride':dataset.manifest['stride'],
                'latent_shape':[model.config.latent_channels,240//model.config.downsample_factor,256//model.config.downsample_factor] if model.config.architecture=='spatial' else [model.config.latent_dim],
                'weights':'ema' if checkpoint.get('ema') is not None else 'raw',
                'note':'Dense original frames, including terminal observations. Never connect transitions across episode boundaries; action[t] advances to frame[t+1].'},indent=2)+'\n')
            print(f'Exported posterior mu/logvar {shape}; row alignment is in index.jsonl')


if __name__ == '__main__':
    main()
