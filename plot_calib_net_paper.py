"""The single-column paper figures: pose mAA@10 against runtime, 5 x 5 ray map only, a legend below the plot.

Color = the network or solver, line style = the pose pipeline. Curves as in `plot_calib_net.py`.

    python plot_calib_net_paper.py results/calib_net/cathedral results/calib_net/cathedral_eq --out figs/calib_net
"""
import argparse
import glob
import os

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from plot_calib_net import BASELINES, COLORS, HIDDEN, STYLES, curve

NETWORKS = [('ours_ray_vitl', 'Ours, ViT-L'), ('ours_ray_cnxt', 'Ours, ConvNeXt-T'),
            ('anycalib_dist', 'AnyCalib$_\\mathrm{dist}$'), ('geocalib', 'GeoCalib'), ('unik3d', 'UniK3D')]
PIPELINES = [('solid', '5pt'), ('dotted', '5pt + LO'), ('dashdot', '6pt'), ((0, (5, 1, 1, 1, 1, 1)), '6pt + LO')]
BLANK = Line2D([], [], linestyle='none', label=' ')

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.6})


def draw(directory, out):
    name = os.path.basename(directory.rstrip('/\\'))
    eq = name.endswith('_eq')
    fig, ax = plt.subplots(figsize=(3.25, 2.95 if eq else 2.8))
    baselines = []
    for path in sorted(glob.glob(os.path.join(directory, '*.npz'))):
        exp = os.path.basename(path)[:-4]
        if 'ours_log' in exp or exp in HIDDEN:
            continue
        xs, ys, _ = curve(path)
        if '+' in exp:
            solver, net = exp.split('+')
            method, use = net.rsplit('_', 1)
            ax.semilogx(xs, ys, color=COLORS[method], linestyle=STYLES[f'{solver}_{use}'], linewidth=1.1)
        else:
            label, color, marker = BASELINES[exp]
            ax.semilogx(xs, ys, color=color, marker=marker, markersize=3.5, linewidth=1.1)
            baselines.append(Line2D([], [], color=color, marker=marker, markersize=3.5, linewidth=1.1, label=label))

    ax.set_xlabel('Mean runtime per pair (ms)', labelpad=1)
    ax.set_ylabel('Pose mAA@10$^\\circ$ (%)', labelpad=1)
    ax.grid(True, which='major', color='#e5e4e0', linewidth=0.5)
    ticks = [t for t in (20, 50, 100, 200, 500) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.tick_params(length=2, width=0.6, pad=2)
    ax.spines[['top', 'right']].set_visible(False)

    pipelines = PIPELINES if eq else PIPELINES[:2]
    columns = [
        [Line2D([], [], color=COLORS[m], linewidth=1.3, label=label) for m, label in NETWORKS],
        [Line2D([], [], color='#333333', linestyle=s, linewidth=1.1, label=label) for s, label in pipelines],
        baselines,
    ]
    rows = max(len(c) for c in columns)
    handles = [h for c in columns for h in c + [BLANK] * (rows - len(c))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7.5,
               handlelength=2.6, columnspacing=1.2, labelspacing=0.25, handletextpad=0.5, borderpad=0)
    fig.tight_layout(pad=0.2)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'pose_{"equal" if eq else "different"}.pdf')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print('saved', path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--out', default='figs/calib_net')
    a = p.parse_args()
    for d in a.dirs:
        draw(d, a.out)
