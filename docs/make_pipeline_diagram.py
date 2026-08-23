"""Render docs/pipeline.png — the block diagram used in the README.

Kept in the repo so the diagram can be regenerated rather than re-drawn by hand:

    python3 docs/make_pipeline_diagram.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe                             # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

INK = '#1c2430'
MUTED = '#6b7684'
WAND = '#c2410c'
CAM = '#1d4ed8'
GAME = '#047857'
BG = '#ffffff'


def box(ax, x, y, w, h, title, sub=None, colour=INK, fill='#ffffff'):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.10,rounding_size=0.16',
        linewidth=1.6, edgecolor=colour, facecolor=fill, zorder=3))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 0.16, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=INK, zorder=4)
        ax.text(x + w / 2, cy - 0.19, sub, ha='center', va='center',
                fontsize=8.5, color=MUTED, zorder=4)
    else:
        ax.text(x + w / 2, cy, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=INK, zorder=4)


def arrow(ax, p0, p1, colour=MUTED, style='-', lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=13, linewidth=lw,
        linestyle=style, color=colour, connectionstyle=f'arc3,rad={rad}',
        zorder=2, shrinkA=2, shrinkB=2))


def label(ax, x, y, text, colour=MUTED, size=8.4):
    ax.text(x, y, text, ha='center', va='center', fontsize=size, color=colour,
            zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground=BG)])


fig, ax = plt.subplots(figsize=(12, 4.4), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.4)
ax.axis('off')

ax.text(1.35, 4.08, 'W A N D', fontsize=9, fontweight='bold', color=WAND,
        ha='center')
ax.text(7.6, 4.08, 'J E T S O N   O R I N   N A N O', fontsize=9,
        fontweight='bold', color=CAM, ha='center')
ax.plot([2.72, 2.72], [0.45, 3.92], color='#e3e7ec', lw=1.2, zorder=1)

# --- wand
box(ax, 0.35, 2.95, 2.0, 0.85, 'AprilTag', 'on the wand', WAND, '#fff7ed')
box(ax, 0.35, 1.60, 2.0, 0.85, 'IMU  500 Hz', 'detects the tap', WAND, '#fff7ed')

# --- one camera image feeding both detectors: that shared frame is what lets a
#     tag pixel and a bounding box be compared without any registration
box(ax, 2.95, 2.28, 1.6, 0.85, 'RGB camera', '1280x720', CAM, '#eff4ff')
box(ax, 5.00, 2.95, 2.1, 0.85, 'AprilTag', 'GPU detector', CAM, '#eff4ff')
box(ax, 5.00, 1.60, 2.1, 0.85, 'YOLOv8', 'TensorRT', CAM, '#eff4ff')

# --- association + game
box(ax, 7.75, 2.30, 3.9, 0.95, 'WHICH OBJECT?',
    'tag position at the tap instant', GAME, '#ecfdf5')
box(ax, 7.75, 0.70, 3.9, 0.85, 'GAME', 'prompt  ·  score', GAME, '#ecfdf5')

# tag is seen by the camera
arrow(ax, (2.35, 3.30), (2.90, 2.85), CAM, rad=-0.1)
label(ax, 2.68, 3.32, 'sees it', CAM)

# camera image to both detectors
arrow(ax, (4.55, 2.88), (4.95, 3.30), CAM)
arrow(ax, (4.55, 2.52), (4.95, 2.10), CAM)

# detectors to association
arrow(ax, (7.10, 3.35), (7.70, 3.02), CAM, rad=-0.1)
label(ax, 7.42, 3.42, 'tag pixel', CAM)
arrow(ax, (7.10, 2.03), (7.70, 2.52), CAM, rad=0.1)
label(ax, 7.40, 1.94, 'objects + names', CAM)

# tap event: routed under the perception boxes so it crosses nothing
ax.plot([1.35, 1.35], [1.58, 1.15], color=WAND, lw=1.5, ls=(0, (5, 3)), zorder=2)
ax.plot([1.35, 7.45], [1.15, 1.15], color=WAND, lw=1.5, ls=(0, (5, 3)), zorder=2)
ax.plot([7.45, 7.45], [1.15, 2.78], color=WAND, lw=1.5, ls=(0, (5, 3)), zorder=2)
arrow(ax, (7.45, 2.78), (7.72, 2.78), WAND, style=(0, (5, 3)))
label(ax, 4.30, 1.32, 'tap event  ·  micro-ROS / WiFi  ·  stamped in the '
                      "Jetson's clock", WAND)

# association to game
arrow(ax, (9.70, 2.26), (9.70, 1.60), GAME, lw=1.7)

ax.text(6.0, 0.22,
        'the IMU says WHEN  ·  the camera says WHERE  ·  two clocks, one answer',
        fontsize=9.4, color=MUTED, ha='center', style='italic')

fig.savefig('docs/pipeline.png', dpi=200, bbox_inches='tight',
            facecolor=BG, pad_inches=0.2)
print('wrote docs/pipeline.png')
