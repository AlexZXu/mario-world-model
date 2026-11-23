import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from vision.model import VisionVAE, VAEConfig, vae_loss
from vision.data import prepare, FrameDataset
from vision.runtime import load_model, device_for
from vision.train import train
from vision.evaluate import evaluate, roi_mask


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def recordings(tmp_path):
    root = tmp_path/'data'
    for ep in range(4):
        folder = root/f'episode-{ep:05}'
        (folder/'rgb').mkdir(parents=True)
        rows=[]
        for f in range(3):
            image = np.zeros((240,256,3),dtype='uint8')
            image[:] = (40+ep*20, 80, 180)
            image[192:208,32+f:48+f] = (255, 100, 10)
            path=folder/'rgb'/f'{f:06}.png'; Image.fromarray(image).save(path)
            rows.append({'frame':f,'rgb':f'rgb/{f:06}.png','action':1 if f<2 else None,
                'rgb_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'state':{'level':'1-1','camera':[0,0],
                         'player':{'x':32+f,'y':192,'width':16,'height':16},'objects':[]}})
        (folder/'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    return root


def test_native_shapes_gradients_and_deterministic_eval():
    torch.manual_seed(7)
    model = VisionVAE(VAEConfig(16,4))
    x = torch.rand(2,3,240,256)
    y, mu, lv = model(x)
    assert y.shape == x.shape and mu.shape == lv.shape == (2,16)
    loss, _, kl = vae_loss(y,x,mu,lv,.001)
    assert kl >= 0 and torch.isfinite(loss)
    loss.backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.decoder[-1].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        a=model(x)[0];b=model(x)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError):model(torch.rand(1,3,64,64))


def test_vae_kl_matches_standard_normal():
    zeros=torch.zeros(2,8)
    image=torch.zeros(2,3,240,256)
    loss,mse,kl=vae_loss(image,image,zeros,zeros,1.)
    assert float(loss)==float(mse)==float(kl)==0.
    _,_,kl=vae_loss(image,image,torch.ones(2,8),zeros,1.)
    assert float(kl)==.5


def test_learning_reduces_reconstruction_error():
    torch.manual_seed(1)
    model=VisionVAE(VAEConfig(8,4))
    x=torch.zeros(1,3,240,256);x[:,2]=.6;x[:,:,192:208,32:48]=1
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    before=float((model(x,sample=False)[0]-x).square().mean().detach())
    for _ in range(20):
        opt.zero_grad();y,mu,lv=model(x,sample=False)
        loss,_,_=vae_loss(y,x,mu,lv,0)
        loss.backward();opt.step()
    after=float((model(x,sample=False)[0]-x).square().mean().detach())
    assert after < before*.5


def test_episode_splits_and_duplicate_detection(recordings,tmp_path):
    import shutil
    shutil.copytree(recordings/'episode-00000',recordings/'duplicate')
    manifest=tmp_path/'split.json'
    prepare([recordings],manifest,stride=1)
    split=json.loads(manifest.read_text())
    assert len(split['duplicate_episodes_removed'])==1
    ids=[{e['id'] for e in split['splits'][name]} for name in ['train','val','test']]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    assert sum(map(len,ids))==4
    data=FrameDataset(manifest,'train')
    image,index=data[0]
    assert image.shape==(3,240,256) and image.dtype==torch.float32
    assert image.min()>=0 and image.max()<=1
    assert roi_mask(data.records[0]).sum()==256


def test_corrupt_recording_rejected(recordings,tmp_path):
    path=recordings/'episode-00000/rgb/000000.png'
    Image.new('RGB',(256,240)).save(path)
    with pytest.raises(ValueError,match='checksum'):
        prepare([recordings],tmp_path/'split.json')


def test_dense_export_preserves_action_alignment_and_detects_edits(recordings,tmp_path):
    manifest=tmp_path/'split.json';prepare([recordings],manifest,stride=2)
    sampled=FrameDataset(manifest,'test')
    dense=FrameDataset(manifest,'test',full_sequence=True)
    assert len(sampled)==2 and len(dense)==3
    assert [r['frame'] for r in dense.records]==[0,1,2]
    assert [r['action'] for r in dense.records]==[1,1,None]
    image=Path(dense.records[0]['path']);Image.new('RGB',(256,240)).save(image)
    with pytest.raises(ValueError,match='changed'):dense[0]
    content=json.loads(manifest.read_text());content['stride']=9
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='fingerprint'):FrameDataset(manifest,'test')


def test_train_resume_checkpoint_and_evaluation(recordings,tmp_path):
    manifest=tmp_path/'split.json';prepare([recordings],manifest,stride=1)
    def args(out,epochs,resume=None):
        return SimpleNamespace(manifest=str(manifest),output=str(out),epochs=epochs,
            batch_size=2,lr=.001,beta=.001,warmup_steps=4,max_batches=1,seed=3,
            latent_dim=8,base_channels=4,device='cpu',resume=resume)
    out=tmp_path/'run';train(args(out,1));train(args(out,2,str(out/'last.pt')))
    full=tmp_path/'full';train(args(full,2))
    a,ca=load_model(out/'last.pt',device_for('cpu'))
    b,cb=load_model(full/'last.pt',device_for('cpu'))
    assert ca['epoch']==2 and ca['global_step']==2
    for name,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
    test=FrameDataset(manifest,'test')
    metrics=evaluate(a,test,torch.device('cpu'),2,tmp_path/'evaluation')
    assert metrics['all']['frames']==3
    assert metrics['all']['dynamic_roi_mse'] is not None
    assert (tmp_path/'evaluation/reconstructions.png').is_file()
    # Existing run protection avoids silently replacing trained weights.
    with pytest.raises(FileExistsError):train(args(out,1))


def test_spatial_latent_and_detail_loss():
    from vision.model import detail_loss
    torch.manual_seed(9)
    model=VisionVAE(VAEConfig(base_channels=4,architecture='spatial',latent_channels=2,residual_blocks=1))
    x=torch.rand(2,3,240,256)
    y,mu,lv=model(x)
    assert mu.shape==lv.shape==(2,480) and y.shape==x.shape
    loss,terms=detail_loss(y,x,mu,lv,.001,edge_weight=.1,roi_weight=1.,masks=torch.ones(2,240,256))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert terms['roi']==terms['mse']
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.decode(model.encode(x)[0]),model(x)[0])
        assert (model.decode(mu)-model.decode(torch.zeros_like(mu))).abs().mean()>.001
    exact,terms=detail_loss(x,x,torch.zeros_like(mu),torch.zeros_like(lv),.1,edge_weight=1.)
    assert exact==0 and terms['edge']==0


def test_portable_packed_frames(recordings,tmp_path):
    import shutil
    from vision.data import pack
    manifest=tmp_path/'split.json';prepare([recordings],manifest,stride=2)
    moved=tmp_path/'moved';shutil.move(recordings,moved)
    raw=FrameDataset(manifest,'train',data_root=moved)
    packed=tmp_path/'cache';pack(manifest,packed,data_root=moved)
    cached=FrameDataset(manifest,'train',cache=packed)
    torch.testing.assert_close(raw[0][0],cached[0][0],atol=0,rtol=0)
    assert len(FrameDataset(manifest,'test',full_sequence=True,data_root=moved))==3
    metadata=json.loads((packed/'metadata.json').read_text());metadata['fingerprint']='wrong'
    (packed/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='differ'):FrameDataset(manifest,'train',cache=packed)


def test_balanced_distributed_sampler_covers_levels():
    from vision.sampling import EpochSampler
    records=[{'level':'a'}]*2+[{'level':'b'}]*7
    whole=list(EpochSampler(records,42,0))
    assert set(whole)==set(range(9))
    assert sum(records[i]['level']=='a' for i in whole)==7
    shards=[list(EpochSampler(records,42,0,rank,2)) for rank in range(2)]
    assert len(shards[0])==len(shards[1])==7
    assert [i for pair in zip(*shards) for i in pair]==whole
    assert whole==list(EpochSampler(records,42,0))
    assert whole!=list(EpochSampler(records,42,1))


def test_mid_epoch_resume_with_accumulation_and_ema(recordings,tmp_path,monkeypatch):
    import importlib
    module=importlib.import_module('vision.train')
    manifest=tmp_path/'split.json';prepare([recordings],manifest,stride=1)
    def args(output,resume=None):
        return SimpleNamespace(manifest=str(manifest),output=str(output),epochs=2,
            batch_size=1,lr=.001,beta=.001,warmup_steps=4,max_batches=0,seed=3,
            latent_dim=8,base_channels=4,device='cpu',resume=resume,
            accum_steps=2,checkpoint_every=1,ema_decay=.99,lr_warmup_steps=2,decay_steps=10)
    out=tmp_path/'interrupted'
    original=module.save_checkpoint
    def interrupt(path,model,optimizer,**kwargs):
        original(path,model,optimizer,**kwargs)
        if kwargs['global_step']==1:raise RuntimeError('simulated interruption')
    with monkeypatch.context() as m:
        m.setattr(module,'save_checkpoint',interrupt)
        with pytest.raises(RuntimeError,match='interruption'):train(args(out))
    train(args(out,str(out/'last.pt')))
    full=tmp_path/'full';train(args(full))
    _,a=load_model(out/'last.pt',torch.device('cpu'))
    _,b=load_model(full/'last.pt',torch.device('cpu'))
    assert a['global_step']==b['global_step']==6
    for kind in ['model','ema']:
        for name,value in a[kind].items():torch.testing.assert_close(value,b[kind][name],rtol=0,atol=0)
    assert a['scaler']==b['scaler']


def test_two_process_cpu_training_and_resume(recordings,tmp_path):
    import os
    import subprocess
    import sys
    manifest=tmp_path/'split.json';prepare([recordings],manifest,stride=1)
    output=tmp_path/'ddp'
    import socket
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    command=[sys.executable,'-m','torch.distributed.run','--master-addr=127.0.0.1',f'--master-port={port}','--nproc-per-node=2',
        '-m','vision','train','--manifest',str(manifest),'--output',str(output),
        '--device','cpu','--architecture','vector','--base-channels','4','--latent-dim','8',
        '--batch-size','1','--accum-steps','2','--checkpoint-every','1','--preview-every','0']
    for epochs in (1,2):
        args=command+['--epochs',str(epochs)]
        if epochs==2:args+=['--resume',str(output/'last.pt')]
        result=subprocess.run(args,env=dict(os.environ,OMP_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
    _,checkpoint=load_model(output/'last.pt',torch.device('cpu'))
    assert checkpoint['world_size']==2 and checkpoint['epoch']==2
    assert checkpoint['global_step']==4 and len(checkpoint['rng_states'])==2
    assert len((output/'history.jsonl').read_text().splitlines())==2


@pytest.mark.parametrize('factor',[8,16])
def test_activation_checkpointing_preserves_spatial_gradients(factor):
    from dataclasses import replace
    torch.manual_seed(123)
    config=VAEConfig(base_channels=4,architecture='spatial',latent_channels=2,
                     residual_blocks=1,downsample_factor=factor)
    original=VisionVAE(config)
    checkpointed=VisionVAE(replace(config,gradient_checkpointing=True))
    checkpointed.load_state_dict(original.state_dict())
    x=torch.rand(1,3,240,256)
    for model in (original,checkpointed):
        y,mu,lv=model(x,sample=False)
        assert mu.shape==(1,config.encoded_dim) and y.shape==x.shape
        (y.square().mean()+mu.square().mean()+lv.square().mean()).backward()
    for a,b in zip(original.parameters(),checkpointed.parameters()):
        torch.testing.assert_close(a.grad,b.grad)


def test_spatial_cli_export_preserves_dense_alignment(recordings,tmp_path):
    import subprocess
    import sys
    from vision.runtime import save_checkpoint
    manifest=tmp_path/'split.json';prepare([recordings],manifest,stride=2)
    model=VisionVAE(VAEConfig(base_channels=4,architecture='spatial',latent_channels=2,
                             residual_blocks=1,downsample_factor=8))
    optimizer=torch.optim.Adam(model.parameters())
    checkpoint=tmp_path/'model.pt'
    save_checkpoint(checkpoint,model,optimizer,data_fingerprint=json.loads(manifest.read_text())['fingerprint'])
    output=tmp_path/'latents'
    result=subprocess.run([sys.executable,'-m','vision','encode','--manifest',str(manifest),
        '--checkpoint',str(checkpoint),'--output',str(output),'--device','cpu'],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    mu=np.load(output/'mu.npy');lv=np.load(output/'logvar.npy')
    assert mu.shape==lv.shape==(3,1920)
    assert np.isfinite(mu).all() and np.isfinite(lv).all()
    metadata=json.loads((output/'metadata.json').read_text())
    assert metadata['latent_shape']==[2,30,32] and metadata['stride']==1
    rows=[json.loads(s) for s in (output/'index.jsonl').read_text().splitlines()]
    assert [r['action'] for r in rows]==[1,1,None]
