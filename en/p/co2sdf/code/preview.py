"""Preview an svg_figure snippet: the PNG with its labels drawn by PIL (approximation)."""
import sys, re
from PIL import Image, ImageDraw, ImageFont
src = open(sys.argv[1]).read()
png = re.search(r'\{\{img figs/([^}]+)\}\}', src).group(1)
vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', src).group(1).split()]
im = Image.open(png).convert('RGB')
sc = im.width / vb[2]
d = ImageDraw.Draw(im)
for m in re.finditer(r'<line x1="([\d.-]+)" y1="([\d.-]+)" x2="([\d.-]+)" y2="([\d.-]+)" stroke="([^"]+)"', src):
    x1, y1, x2, y2 = [float(m.group(i)) * sc for i in range(1, 5)]
    d.line([(x1, y1), (x2, y2)], fill=m.group(5), width=3)
for m in re.finditer(r'<text x="([\d.-]+)" y="([\d.-]+)" font-size="([\d.]+)"[^>]*fill="([^"]+)" text-anchor="(\w+)"[^>]*>([^<]*)</text>', src):
    x, y, fs = float(m.group(1)) * sc, float(m.group(2)) * sc, float(m.group(3)) * sc
    f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', int(fs))
    w = d.textlength(m.group(6), font=f)
    x0 = x - (w if m.group(5) == 'end' else w / 2 if m.group(5) == 'middle' else 0)
    d.text((x0, y - fs * 0.8), m.group(6), font=f, fill=m.group(4), stroke_width=3, stroke_fill='white')
im.save(sys.argv[2])
