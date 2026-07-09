"""Shared plot style: reference dataviz palette, light mode."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = '#fcfcfb'
PAGE = '#f9f9f7'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASE = '#c3c2b7'

BLUE = '#2a78d6'
AQUA = '#1baf7a'
YELLOW = '#eda100'
GREEN = '#008300'
VIOLET = '#4a3aa7'
RED = '#e34948'
ORANGE = '#eb6834'

SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
            '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
            '#0d366b']
CMAP_BLUE = LinearSegmentedColormap.from_list('seqblue', SEQ_BLUE)


def save_axes(fig, axes, path, pad=0.10, dpi=200):
    """Save one panel (single Axes or a list of Axes, e.g. plot + its colorbar)
    of a multi-panel figure as a standalone image, cropped to its own content
    (titles, tick labels, legends and annotations included). Neighbouring
    panels are hidden while saving so their labels never bleed into the crop."""
    from matplotlib.transforms import Bbox
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    keep = set(map(id, axes))
    hidden = [a for a in fig.get_axes() if id(a) not in keep and a.get_visible()]
    for a in hidden:
        a.set_visible(False)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = Bbox.union([a.get_tightbbox(r) for a in axes])
    bb = bb.transformed(fig.dpi_scale_trans.inverted())
    bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
    fig.savefig(path, bbox_inches=bb, dpi=dpi)
    for a in hidden:
        a.set_visible(True)
    print('saved', path)


def apply_style():
    mpl.rcParams.update({
        'figure.facecolor': PAGE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': PAGE,
        'axes.edgecolor': BASE,
        'axes.labelcolor': INK2,
        'axes.titlecolor': INK,
        'axes.titlesize': 10.5,
        'axes.labelsize': 9,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'text.color': INK,
        'font.family': 'DejaVu Sans',
        'legend.frameon': False,
        'legend.fontsize': 8.5,
        'figure.dpi': 170,
    })
