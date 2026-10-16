"""Lossless check: full-resolution test MRIs rebuilt from trained receptor responses."""
import sys, pickle, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from mycelia.receptor_memory import ReceptorField, ReceptorConfig
dataset, dictionary, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
m = pickle.load(open(dictionary, 'rb'))
field = ReceptorField(ReceptorConfig()); field.mu, field.whiten, field.receptors = m['mu'], m['Wz'], m['D']
files = [next((dataset / 'Testing' / c).glob('*.jpg')) for c in ('glioma', 'meningioma', 'notumor', 'pituitary')]
rows = []
for f in files:
    img = np.asarray(Image.open(f).convert('L'), dtype=np.float64) / 255
    s = field.sense(img)
    full = field.reconstruct(s); part = field.reconstruct(s, keep=10)
    err = int(np.abs(np.round(full * 255) - np.round(img * 255)).max())
    psnr = 10 * np.log10(1 / np.mean((img - part) ** 2))
    print(f'{f.name} {img.shape}: all receptors max error {err} gray levels (lossless={err == 0}); 10 receptors PSNR {psnr:.1f} dB')
    rows.append([img, full, part, f.parent.name])
W = 256; canvas = Image.new('L', (3 * W, 4 * (W + 18)), 255); dr = ImageDraw.Draw(canvas)
for r, (a, b, c, name) in enumerate(rows):
    for col, im in enumerate((a, b, c)):
        canvas.paste(Image.fromarray(np.round(im * 255).astype('uint8')).resize((W, W)), (col * W, r * (W + 18) + 18))
    dr.text((4, r * (W + 18) + 3), f'{name}: original (full res) | rebuilt, all receptors (lossless) | 10 receptors', fill=0)
canvas.save(out)
