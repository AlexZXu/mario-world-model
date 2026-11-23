"""Single-device/torchrun trainer with restartable optimizer-step checkpoints."""
import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, BatchSampler
from .model import VisionVAE, VAEConfig, detail_loss
from .data import FrameDataset
from .runtime import device_for, seed_everything, load_model, save_checkpoint
from .evaluate import evaluate, roi_mask
from .sampling import EpochSampler


EXTRA_DEFAULTS = dict(uniform_sampling=False, accum_steps=1, workers=0, precision='fp32',
    lr_warmup_steps=0, decay_steps=0, min_lr_ratio=.1, l1_weight=0., edge_weight=0.,
    roi_weight=0., free_nats=0., ema_decay=0., checkpoint_every=0, preview_every=0)


def learning_rate(step, options):
    warmup = options['lr_warmup_steps']
    if warmup and step < warmup:
        return options['lr']*(step+1)/warmup
    decay = options['decay_steps']
    progress = min(1., max(0., (step-warmup)/max(1,decay-warmup))) if decay else 0.
    return options['lr']*(options['min_lr_ratio']+(1-options['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))


def rng_state(device):
    state={'torch':torch.get_rng_state()}
    if device.type == 'mps': state['device']=torch.mps.get_rng_state()
    if device.type == 'cuda': state['device']=torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    if device.type == 'mps' and 'device' in state: torch.mps.set_rng_state(state['device'])
    if device.type == 'cuda' and 'device' in state: torch.cuda.set_rng_state(state['device'],device)


def train(args):
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if world > 1:
        if args.device in ('auto','cuda') and torch.cuda.is_available():
            torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
        elif args.device == 'cpu': device=device_for('cpu')
        else: raise ValueError('Distributed training requires CUDA or explicit CPU')
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    else: device=device_for(args.device)
    try:
        return _train(args,device,rank,world)
    finally:
        if world > 1: dist.destroy_process_group()


def _train(args, device, rank, world):
    options={k:getattr(args,k) for k in ['seed','batch_size','lr','beta','warmup_steps','max_batches']}
    options.update({k:getattr(args,k,v) for k,v in EXTRA_DEFAULTS.items()})
    if (args.epochs < 1 or min(args.batch_size,options['accum_steps']) < 1 or args.lr <= 0
        or min(args.beta,args.warmup_steps,args.max_batches,options['workers'],options['free_nats'],
               options['lr_warmup_steps'],options['decay_steps'],options['checkpoint_every'],
               options['preview_every'],options['l1_weight'],options['edge_weight'],options['roi_weight']) < 0
        or not 0 <= options['ema_decay'] < 1 or not 0 <= options['min_lr_ratio'] <= 1):
        raise ValueError('Invalid training hyperparameters')
    precision=options['precision']
    if precision not in ('fp32','bf16','fp16'): raise ValueError('Invalid precision')
    if precision!='fp32' and device.type!='cuda': raise ValueError('Mixed precision requires CUDA; use fp32 on MPS/CPU')
    if precision=='bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('CUDA device does not support bf16')
    seed_everything(args.seed+rank)
    dataset_options=dict(data_root=getattr(args,'data_root',None),cache=getattr(args,'cache',None))
    training=FrameDataset(args.manifest,'train',**dataset_options)
    validation=FrameDataset(args.manifest,'val',**dataset_options)
    config=VAEConfig(args.latent_dim,args.base_channels,getattr(args,'architecture','vector'),
        getattr(args,'latent_channels',8),getattr(args,'residual_blocks',2),not getattr(args,'no_attention',False),
        getattr(args,'downsample_factor',16),getattr(args,'gradient_checkpointing',False))
    output=Path(args.output)
    start_epoch=step=next_batch=0;best=float('inf');partial=None
    if args.resume:
        model,ckpt=load_model(args.resume,device,use_ema=False)
        if model.config!=config or ckpt['options']!=options:
            raise ValueError('Resume requires identical architecture and training options')
        if ckpt['data_fingerprint']!=training.manifest['fingerprint']:
            raise ValueError('Dataset split changed since checkpoint')
        if ckpt.get('world_size',1)!=world: raise ValueError('Resume requires the same world size')
        if Path(args.resume).resolve().parent!=output.resolve():
            raise ValueError('Resume into the checkpoint’s original output directory')
        start_epoch=ckpt['epoch'];next_batch=ckpt.get('next_batch',0)
        step=ckpt['global_step'];best=ckpt['best_val_mse'];partial=ckpt.get('partial')
    else:
        if rank==0: output.mkdir(parents=True,exist_ok=False)
        if world>1: dist.barrier()
        model=VisionVAE(config).to(device)
    if start_epoch>=args.epochs: raise ValueError('--epochs must exceed the completed checkpoint epoch')
    if world>1:
        wrapped=DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None)
    else: wrapped=model
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    ema=copy.deepcopy(model).eval().requires_grad_(False) if options['ema_decay'] else None
    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer']);scaler.load_state_dict(ckpt.get('scaler',{}))
        if ema is not None: ema.load_state_dict(ckpt['ema'])
        restore_rng(ckpt['rng_states'][rank],device)
    run_config={'model':asdict(config),'options':options,'device':str(device),'world_size':world,
        'effective_batch_size':args.batch_size*options['accum_steps']*world,
        'manifest':str(Path(args.manifest).resolve()),'data_fingerprint':training.manifest['fingerprint'],
        'parameters':sum(p.numel() for p in model.parameters()),'torch_version':str(torch.__version__),
        'latent_shape':[config.latent_channels,240//config.downsample_factor,256//config.downsample_factor] if config.architecture=='spatial' else [config.latent_dim]}
    if rank==0:
        if not args.resume: (output/'config.json').write_text(json.dumps(run_config,indent=2)+'\n')
        print(json.dumps(run_config),flush=True)
    def checkpoint(epoch, batch, stats, improved=False):
        state=rng_state(device)
        states=[None]*world
        if world>1: dist.all_gather_object(states,state)
        else: states=[state]
        if rank==0:
            metadata=dict(epoch=epoch,next_batch=batch,partial=stats,global_step=step,
                best_val_mse=best,options=options,data_fingerprint=training.manifest['fingerprint'],
                world_size=world,rng_states=states,scaler=scaler.state_dict(),
                ema={k:v.detach().cpu() for k,v in ema.state_dict().items()} if ema is not None else None)
            save_checkpoint(output/'last.pt',model,optimizer,**metadata)
            if improved: save_checkpoint(output/'best.pt',model,optimizer,**metadata)
        if world>1: dist.barrier()
    for epoch in range(start_epoch,args.epochs):
        started=time.monotonic();wrapped.train()
        sampler=EpochSampler(training.records,args.seed,epoch,rank,world,not options['uniform_sampling'])
        batches=list(BatchSampler(sampler,args.batch_size,drop_last=False))
        if args.max_batches: batches=batches[:args.max_batches]
        offset=next_batch if epoch==start_epoch else 0
        if offset>len(batches): raise ValueError('Invalid checkpoint batch position')
        loader=DataLoader(training,batch_sampler=batches[offset:],num_workers=options['workers'],
            pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(args.seed+epoch),
            **({'prefetch_factor':2} if options['workers'] else {}))
        # Partial totals are globally reduced in checkpoints, not rank-local.
        sums={k:0. for k in ['loss','mse','l1','edge','roi','kl','n']}
        prior=partial if epoch==start_epoch and partial else dict(sums)
        optimizer.zero_grad(set_to_none=True)
        for local_batch,(images,indices) in enumerate(loader):
            batch=offset+local_batch
            group_start=(batch//options['accum_steps'])*options['accum_steps']
            group_end=min(group_start+options['accum_steps'],len(batches))
            count_in_group=sum(len(b) for b in batches[group_start:group_end])
            update=batch+1==group_end
            beta=args.beta*min(1.,(step+1)/max(1,args.warmup_steps))
            images=images.to(device,non_blocking=True)
            masks=torch.stack([roi_mask(training.records[i]) for i in indices.tolist()]).to(device) if options['roi_weight'] else None
            sync=wrapped.no_sync() if world>1 and not update else nullcontext()
            with sync:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision!='fp32'):
                    reconstruction,mu,logvar=wrapped(images)
                    loss,metrics=detail_loss(reconstruction,images,mu,logvar,beta,
                        **{k:options[k] for k in ['l1_weight','edge_weight','roi_weight','free_nats']},masks=masks)
                if not torch.isfinite(loss): raise FloatingPointError(f'Non-finite loss at step {step}')
                scaler.scale(loss*(len(images)/count_in_group)).backward()
            for k,v in dict(metrics,loss=loss).items(): sums[k]+=float(v.detach())*len(images)
            sums['n']+=len(images)
            if not update: continue
            lr=learning_rate(step,options)
            for group in optimizer.param_groups: group['lr']=lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=precision!='fp16')
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            optimizer.zero_grad(set_to_none=True)
            skipped=scaler.get_scale()<old_scale
            if not skipped:
                step+=1
                if ema is not None:
                    decay=min(options['ema_decay'],(1+step)/(10+step))
                    with torch.no_grad():
                        for e,p in zip(ema.parameters(),model.parameters()): e.lerp_(p,1-decay)
            if rank==0 and (step%25==0 or batch+1==len(batches)):
                print(f'epoch={epoch+1} batch={batch+1}/{len(batches)} step={step} mse={float(metrics["mse"].detach()):.6f} lr={lr:.3g}',flush=True)
            if options['checkpoint_every'] and step%options['checkpoint_every']==0 and not skipped:
                totals=_totals(sums,prior,device,world)
                checkpoint(epoch,batch+1,totals)
        totals=_totals(sums,prior,device,world)
        if rank==0:
            scores=evaluate(ema or model,validation,device,args.batch_size,
                output/f'validation-{epoch+1:04}' if options['preview_every'] and (epoch+1)%options['preview_every']==0 and not (output/f'validation-{epoch+1:04}').exists() else None)
            # Equal-weight level score prevents easy/frequent backgrounds dominating selection.
            selection=sum(v['mse'] for k,v in scores.items() if k not in ('all','latent'))/(len(scores)-2)
            improved=selection<best;best=min(best,selection)
            row={'epoch':epoch+1,'global_step':step,'lr':learning_rate(max(0,step-1),options),
                'train':{k:v/totals['n'] for k,v in totals.items() if k!='n'},
                'validation':scores,'best_val_mse':best,'selection':'macro_level_mse',
                'seconds':time.monotonic()-started}
            with (output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(f'epoch={epoch+1} train_mse={row["train"]["mse"]:.6f} val_mse={scores["all"]["mse"]:.6f} macro_mse={selection:.6f} psnr={scores["all"]["psnr_db"]:.2f}',flush=True)
        else: improved=False
        if world>1:
            values=[best,improved];dist.broadcast_object_list(values,src=0);best,improved=values
        checkpoint(epoch+1,0,None,improved)
        next_batch=0;partial=None
    return best


def _totals(sums, prior, device, world):
    values=torch.tensor(list(sums.values()),device=device,dtype=torch.float32)
    if world>1:dist.all_reduce(values)
    return {k:float(v)+prior[k] for k,v in zip(sums,values.cpu())}
