"""Compare two vision checkpoints on identical held-out frames in a mobile-friendly report."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import torch
from vision.data import FrameDataset
from vision.runtime import load_model, device_for
from vision.evaluate import evaluate, image_from_tensor


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--before',required=True)
    p.add_argument('--after',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cache')
    p.add_argument('--data-root')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--device',default='auto',choices=['auto','mps','cuda','cpu'])
    a=p.parse_args()
    device=device_for(a.device)
    data=FrameDataset(a.manifest,a.split,data_root=a.data_root,cache=a.cache)
    before,bck=load_model(a.before,device)
    after,ack=load_model(a.after,device)
    if any(c['data_fingerprint']!=data.manifest['fingerprint'] for c in [bck,ack]):
        raise ValueError('Both checkpoints must match the dataset manifest')
    a.output.mkdir(parents=True,exist_ok=False)
    bm=evaluate(before,data,device,batch_size=4)
    am=evaluate(after,data,device,batch_size=4)
    metrics={'before':bm,'after':am,'split':a.split,
        'before_epoch':bck['epoch'],'after_epoch':ack['epoch'],
        'data_fingerprint':data.manifest['fingerprint'],
        'before_sha256':hashlib.sha256(Path(a.before).read_bytes()).hexdigest(),
        'after_sha256':hashlib.sha256(Path(a.after).read_bytes()).hexdigest()}
    (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    cards=[]
    levels=sorted({r['level'] for r in data.records})
    for level in levels:
        ids=[i for i,r in enumerate(data.records) if r['level']==level]
        index=ids[len(ids)//2]
        x,_=data[index]
        b=before(x[None].to(device),sample=False)[0][0]
        y=after(x[None].to(device),sample=False)[0][0]
        figures=[]
        for label,tensor,kind in [('Original',x,'original'),(f'Epoch {bck["epoch"]}',b,'before'),(f'Epoch {ack["epoch"]}',y,'after')]:
            filename=f'{level}-{kind}.png'
            image_from_tensor(tensor).save(a.output/filename)
            figures.append(f'<figure><figcaption>{label}</figcaption><img loading="lazy" src="{filename}" alt="{label}, level {html.escape(level)}"></figure>')
        cards.append(f'<section><h2>Level {html.escape(level)} · frame {data.records[index]["frame"]}</h2><div class="frames">{"".join(figures)}</div></section>')
    rows=[]
    for title,key in [('Pixel MSE','mse'),('Dynamic-region MSE','dynamic_roi_mse'),('PSNR (dB)','psnr_db')]:
        rows.append(f'<tr><td>{title}</td><td>{bm["all"][key]:.5f}</td><td>{am["all"][key]:.5f}</td></tr>')
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Mario: longer training comparison</title>
<style>body{{font:16px system-ui;background:#151922;color:#eef;margin:0 auto;padding:20px;max-width:1000px}}h1{{font-size:1.7rem}}h2{{font-size:1.2rem}}section{{margin:32px 0}}.frames{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}figure{{margin:0}}figcaption{{padding:8px 0}}img{{width:100%;height:auto;image-rendering:pixelated}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:10px 4px;border-bottom:1px solid #445}}@media(max-width:600px){{.frames{{grid-template-columns:1fr}}figure{{max-width:360px;width:100%;margin:auto}}}}</style></head><body>
<h1>Mario: longer local training</h1><p>Epoch {bck['epoch']} vs {ack['epoch']}. Same architecture, dataset, and frames. Posterior-mean reconstructions with EMA weights.</p>
<p>Evaluation: {html.escape(a.split)} · {len(data)} frames · {len(levels)} levels. The tiny dataset mostly covers opening scenes; this is not a test of full-game generalization.</p>
<table><tr><th>Metric</th><th>Before</th><th>After</th></tr>{''.join(rows)}</table>
<p>Lower MSE is better; higher PSNR is better. Check Mario, enemies, and platform edges as well as backgrounds. On narrow screens, each example stacks original → before → after.</p>
{''.join(cards)}</body></html>'''
    (a.output/'index.html').write_text(page)
    print(json.dumps({k:metrics[k]['all'] for k in ['before','after']},indent=2))
    print(a.output/'index.html')


if __name__=='__main__':main()
