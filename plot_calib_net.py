"""Pose mAA against runtime from `eval_calib_net.py`, one figure per graph (equal / different intrinsics).

mAA@10 = the mean over thresholds of 1, ..., 10 deg of the fraction of pairs with max(R_err, t_err) below it
(rdnet's "AUC@10"). x = mean (RANSAC + the network's time per pair), one point per RANSAC budget; a failure or
NaN counts as 180 deg.

    python plot_calib_net.py results/calib_net/cathedral results/calib_net/cathedral_eq --out figs
    python plot_calib_net.py <dirs> --keep E_5pt+ _VLO      # only some curves, by substring
"""
import argparse
import csv
import glob
import os

import numpy as np
from matplotlib import pyplot as plt

# rdnet's palette (`utils/vis.py`): seaborn's tab10, the solvers keeping their own hues (6pt green, 7pt red,
# 9pt purple, 10pt brown); the networks take the hues rdnet's network curves had, then the unused ones
COLORS = {
    'ours_ray_vitl': '#1f77b4', 'ours_log_vitl': '#17becf', 'ours_ray_cnxt': '#ff7f0e', 'ours_log_cnxt': '#bcbd22', 'ours_ray_vitb': '#ff7f0e',
    'anycalib_dist': '#e377c2', 'anycalib_ours': '#9467bd', 'geocalib': '#7f7f7f', 'unik3d': '#bcbd22',
}
LABELS = {
    'anycalib_ours': 'AnyCalib$_' + r'\mathrm{ours}$',
    'ours_ray_vitl': 'Ours ray map ViT-L', 'ours_log_vitl': 'Ours log map ViT-L',
    'ours_ray_cnxt': 'Ours ray map ConvNeXt-T', 'ours_log_cnxt': 'Ours log map ConvNeXt-T', 'ours_ray_vitb': 'Ours ray map ViT-B',
    'anycalib_dist': 'AnyCalib$_\\mathrm{dist}$', 'geocalib': 'GeoCalib', 'unik3d': 'UniK3D',
}
# as rdnet's `get_color_style`: 'o' the non-minimal solvers, '*' the sampling ones
BASELINES = {
    'kFk_9pt': ('9pt $\\mathbf{F}_\\lambda$', '#9467bd', 'o'), 'k2Fk1_10pt': ('10pt $\\mathbf{F}_{\\lambda_1\\lambda_2}$', '#8c564b', 'o'),
    'Efeq_6pt_s3': ('6pt $\\mathbf{E}_f$, 3 $\\lambda$', '#2ca02c', '*'), 'F_7pt_s3': ('7pt $\\mathbf{F}$, 3 $\\lambda$', '#d62728', '*'),
}
# only lambda = 0: run, not plotted (user)
HIDDEN = ('F_7pt', 'Efeq_6pt')
# pipeline -> line style, the method's color stays
STYLES = {'E_5pt_Veq': (0, (3, 1)), 'E_5pt_Vfix': (0, (1, 3)), 'E_5pt_V': 'solid', 'E_5pt_VLO': 'dotted', 'E_5pt_VLOeq': 'dotted',  # rdnet: LO dotted
          'Efeq_6pt_V': 'dashdot', 'Efeq_6pt_VLO': (0, (5, 1, 1, 1, 1, 1)),
          'E_3pt_V': 'dashed', 'E_3pt_VLO': (0, (1, 1)), 'E_3pt_VLOeq': (0, (1, 1))}
PIPELINE = {'E_5pt_Veq': '5pt, shared final', 'E_5pt_Vfix': '5pt, fixed', 'E_5pt_V': '5pt', 'E_5pt_VLO': '5pt + LO', 'E_5pt_VLOeq': '5pt + LO', 'Efeq_6pt_V': '6pt',
            'Efeq_6pt_VLO': '6pt + LO', 'E_3pt_V': '3pt', 'E_3pt_VLO': '3pt + LO', 'E_3pt_VLOeq': '3pt + LO'}


def curve(path):
    d = np.load(path)
    errs = np.maximum(d['R_err'], d['t_err'])
    errs[np.isnan(errs)] = 180.0
    xs, ys, ns = [], [], []
    for it in np.unique(d['iterations']):
        m = d['iterations'] == it
        xs.append(np.mean(d['ransac_ms'][m] + d['base_ms'][m]))
        ys.append(100 * np.mean([np.mean(errs[m] < t) for t in range(1, 11)]))
        ns.append(int(it))
    return np.array(xs), np.array(ys), ns


def draw(directory, out, keep, ylim=None, exclude=None, suffix=''):
    name = os.path.basename(directory.rstrip('/\\'))
    plt.figure(figsize=(7.5, 4.2))
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, '*.npz'))):
        exp = os.path.basename(path)[:-4]
        if keep and not all(k in exp for k in keep):
            continue
        if exclude and any(k in exp for k in exclude):
            continue
        if exp in HIDDEN:
            continue
        xs, ys, ns = curve(path)
        rows.extend([exp, n, x, y] for n, x, y in zip(ns, xs, ys))
        if '+' in exp:
            solver, net = exp.split('+')
            method, use = net.rsplit('_', 1)
            pipeline = f'{solver}_{use}'
            plt.semilogx(xs, ys, color=COLORS[method], linestyle=STYLES[pipeline], linewidth=1.6,
                         label=f'{LABELS[method]}, {PIPELINE[pipeline]}')
        else:
            label, color, marker = BASELINES[exp]
            plt.semilogx(xs, ys, color=color, marker=marker, markersize=5, linewidth=1.6, label=label)
    plt.xlabel('Mean runtime per pair (ms), log scale')
    plt.ylabel('Pose mAA@10$^\\circ$ (%)')
    if ylim:
        plt.ylim(ylim)
    plt.grid(True, which='major', color='#e5e4e0', linewidth=0.7)
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000) if plt.xlim()[0] <= t <= plt.xlim()[1]]
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.minorticks_off()
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(out, f'calib_net_{name}{suffix}.{ext}'), bbox_inches='tight', pad_inches=0.02, dpi=200)
    plt.close()
    with open(os.path.join(out, f'calib_net_{name}{suffix}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'iterations', 'runtime_ms', 'mAA10'])
        writer.writerows(rows)
    print('saved', name, len(rows), 'points')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--out', default='figs')
    p.add_argument('--keep', nargs='*', default=None)
    p.add_argument('--ylim', nargs=2, type=float, default=None)
    p.add_argument('--exclude', nargs='*', default=None, help='leave out curves containing any of these')
    p.add_argument('--suffix', default='', help='added to the output file names')
    a = p.parse_args()
    for d in a.dirs:
        draw(d, a.out, a.keep, a.ylim, a.exclude, a.suffix)
