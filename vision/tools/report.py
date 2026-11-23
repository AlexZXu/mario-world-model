"""Create a portable, offline HTML training report from existing artifacts."""
import argparse
import html
import json
from pathlib import Path
import shutil


def chart(history):
    series=[('train','#5cc8ff',[r['train']['mse'] for r in history]),
            ('validation','#ffa65c',[r['validation']['all']['mse'] for r in history])]
    maximum=max(max(v) for _,_,v in series)*1.1
    parts=['<svg viewBox="0 0 800 270" role="img" aria-label="Reconstruction MSE by epoch">']
    for i in range(5):
        y=20+i*50;value=maximum*(1-i/4)
        parts.append(f'<path d="M60 {y} H780" stroke="#39404d"/><text x="0" y="{y+5}" fill="#ccd">{value:.3f}</text>')
    for name,color,values in series:
        points=' '.join(f'{60+i*720/max(1,len(values)-1):.1f},{220-v/maximum*200:.1f}' for i,v in enumerate(values))
        parts.append(f'<polyline points="{points}" stroke="{color}" stroke-width="3" fill="none"/>')
    parts.append(f'<text x="60" y="250" fill="#ccd">Epoch 1</text><text x="690" y="250" fill="#ccd">Epoch {history[-1]["epoch"]}</text></svg>')
    return ''.join(parts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    history=[json.loads(s) for s in (a.run/'history.jsonl').read_text().splitlines()]
    config=json.loads((a.run/'config.json').read_text())
    metrics=json.loads((a.evaluation/'metrics.json').read_text())
    provenance=json.loads((a.evaluation/'provenance.json').read_text())
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["frames"]}</td><td>{v["mse"]:.5f}</td><td>{v["psnr_db"]:.2f}</td><td>{v["dynamic_roi_mse"] if v["dynamic_roi_mse"] is not None else "n/a"}</td></tr>'
                 for k,v in metrics.items() if k!='latent')
    images=[]
    for image in sorted(a.evaluation.glob('reconstructions*.png')):
        shutil.copy2(image,a.output/image.name)
        images.append(f'<details><summary>{html.escape(image.name)}</summary><img loading="lazy" src="{image.name}" alt="Original, reconstructed, and error frames"/></details>')
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Mario vision training</title>
<style>body{{font:16px system-ui;background:#171b23;color:#eef;max-width:1000px;margin:40px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #445;text-align:left}}img{{max-width:100%}}summary{{padding:16px;cursor:pointer}}pre{{white-space:pre-wrap}}svg{{width:100%}}</style>
<h1>Mario vision training</h1><p>{len(history)} completed epochs · {config['parameters']:,} parameters · evaluation split: {html.escape(provenance['split'])} · checkpoint epoch {provenance['epoch']}</p>
<p>Small all-level sanity dataset. This demonstrates learning and pipeline operation, not sufficient quality for a controller. Inspect small sprites and collision geometry.</p>
<h2>Reconstruction MSE</h2><p style="color:#5cc8ff">Blue: training (sampled posterior, raw weights). <span style="color:#ffa65c">Orange: validation (posterior mean, EMA when enabled).</span></p>{chart(history)}
<h2>Held-out metrics</h2><table><tr><th>Level</th><th>Frames</th><th>MSE</th><th>PSNR (dB)</th><th>Dynamic ROI MSE</th></tr>{rows}</table>
<h2>Reconstructions: all levels</h2>{''.join(images)}
<details><summary>Configuration and provenance</summary><pre>{html.escape(json.dumps({'config':config,'evaluation':provenance},indent=2))}</pre></details></html>'''
    (a.output/'index.html').write_text(page)
    print(a.output/'index.html')


if __name__=='__main__':main()
