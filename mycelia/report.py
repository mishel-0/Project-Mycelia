from __future__ import annotations

import html
import json
from pathlib import Path


def svg(organism):
    env = organism.environment
    h,w = env.shape
    scale = 9
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w*scale} {h*scale}" role="img" aria-label="Mycelial network on nutrient substrate">',
             f'<rect width="{w*scale}" height="{h*scale}" fill="#071713"/>']
    maximum = max(float(env.nutrient.max()),1e-12)
    for y in range(h):
        for x in range(w):
            value = float(env.nutrient[y,x])/maximum
            color = '#314840' if env.blocked[y,x] else f'rgb({int(7+25*value)},{int(23+65*value)},{int(19+30*value)})'
            parts.append(f'<rect x="{x*scale}" y="{y*scale}" width="{scale+.1}" height="{scale+.1}" fill="{color}"/>')
    for e in organism.segments.values():
        if not e.alive:
            continue
        a,b = organism.nodes[e.a],organism.nodes[e.b]
        color = '#efbc73' if e.fusion else '#a5e7ca'
        width = max(.6,e.radius*4)
        title = html.escape(f'Segment {e.id}; flow {e.flow:.5f}; septal opening {e.pore_open:.2f}')
        parts.append(f'<line x1="{(a.x+.5)*scale:.3f}" y1="{(a.y+.5)*scale:.3f}" x2="{(b.x+.5)*scale:.3f}" y2="{(b.y+.5)*scale:.3f}" stroke="{color}" stroke-width="{width:.2f}" stroke-linecap="round"><title>{title}</title></line>')
    for t in organism.tips.values():
        if t.active:
            n = organism.nodes[t.node]
            parts.append(f'<circle cx="{(n.x+.5)*scale:.3f}" cy="{(n.y+.5)*scale:.3f}" r="2.5" fill="#fff2bd"/>')
    parts.append('</svg>')
    return '\n'.join(parts)


def write_report(organism,directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    summary = organism.summary()
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    organism.save(directory/'state.json')
    drawing = svg(organism)
    (directory/'network.svg').write_text(drawing,encoding='utf-8')
    metrics = ''.join(f'<div class="metric"><span>{label}</span><strong>{value}</strong></div>' for label,value in (
        ('Compartments',summary['nodes']),('Active tips',summary['active_tips']),('Branches',summary['events'].get('branch',0)),('Fusions',summary['events'].get('fusion',0)),
        ('Total flow',f"{summary['total_flow']:.5f}"),('Carbon error',f"{summary['carbon_error']:.2e}")))
    history = list(organism.history)
    def chart(key,color):
        values = [float(s[key]) for s in history]
        peak = max(values,default=1) or 1
        points = ' '.join(f'{i*480/max(1,len(values)-1):.1f},{90-v/peak*80:.1f}' for i,v in enumerate(values))
        return f'<svg viewBox="0 0 480 100" role="img" aria-label="{key} over time"><polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/></svg>'
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MYCELIA — simulation report</title>
<style>body{margin:0;background:#07110f;color:#e4eee9;font:16px system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:36px 24px}h1{font-size:42px;letter-spacing:-2px;margin:0}p{color:#a9bbb0;line-height:1.6}.badge{color:#afdbbe;font-size:12px;letter-spacing:2px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:28px 0}.metric{padding:18px;background:#11231c;border:1px solid #284336;border-radius:12px}.metric span{display:block;color:#a9bbb0;font-size:13px}.metric strong{display:block;font-size:28px;margin-top:8px}.layout{display:grid;grid-template-columns:1.5fr 1fr;gap:24px}.panel{background:#101d18;border:1px solid #284336;border-radius:16px;padding:20px}.panel svg{width:100%;display:block}h2{font-size:18px}a{color:#b5e5c6}small{color:#a9bbb0}button{background:#254b38;color:white;border:0;border-radius:8px;padding:10px;cursor:pointer}.note{font-size:13px}@media(max-width:750px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}h1{font-size:34px}}</style><main>
<span class="badge">EXPERIMENTAL DIGITAL MYCELIUM</span><h1>MYCELIA</h1><p>Local physiology creates a living transport graph. This report shows one reproducible simulation run.</p>'''
    document += f'<div class="metrics">{metrics}</div><div class="layout"><section class="panel"><h2>Network and substrate</h2>{drawing}<p class="note">Green: nutrient availability · pale lines: hyphae · amber: fusion · bright points: active tips · grey: obstacles</p></section><section class="panel"><h2>Compartments over time</h2>{chart("nodes","#a5e7ca")}<h2>Transport activity</h2>{chart("total_flow","#efbc73")}<h2>Run details</h2><p>{summary["step"]} steps · time {summary["time"]:.2f} model units<br>Water budget error: {summary["water_error"]:.2e}<br>Pressure residual: {summary["pressure_residual"]:.2e}</p><p><a href="summary.json">Summary JSON</a> · <a href="state.json">Resumable state</a> · <a href="network.svg">Export SVG</a></p></section></div>'
    document += '<p class="note">This is a compartment-level research model with explicit assumptions. It is not a quantitatively calibrated real fungus or a medical classifier. Parameters and mechanism coverage are documented in the README.</p></main></html>'
    (directory/'report.html').write_text(document,encoding='utf-8')
    return directory/'report.html'

