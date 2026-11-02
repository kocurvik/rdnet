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

# one fixed color per method (validated: `dataviz` skill's categorical order), baselines in gray
COLORS = {
    'ours_ray_vitl': '#2a78d6', 'ours_log_vitl': '#6250d6', 'ours_ray_cnxt': '#1baf7a', 'ours_log_cnxt': '#008300',
    'anycalib_dist': '#eb6834', 'geocalib': '#eda100', 'unik3d': '#e87ba4',
}
LABELS = {
    'ours_ray_vitl': 'Ours ray map ViT-L', 'ours_log_vitl': 'Ours log map ViT-L',
    'ours_ray_cnxt': 'Ours ray map ConvNeXt-T', 'ours_log_cnxt': 'Ours log map ConvNeXt-T',
    'anycalib_dist': 'AnyCalib$_\\mathrm{dist}$', 'geocalib': 'GeoCalib', 'unik3d': 'UniK3D',
}
BASELINES = {
    'kFk_9pt': ('9pt $\\mathbf{F}_\\lambda$', '#0b0b0b', 'o'), 'k2Fk1_10pt': ('10pt $\\mathbf{F}_{\\lambda_1\\lambda_2}$', '#52514e', 's'),
    'Efeq_6pt': ('6pt $\\mathbf{E}_f$, $\\lambda{=}0$', '#8a8984', '^'), 'Efeq_6pt_s3': ('6pt $\\mathbf{E}_f$, 3 $\\lambda$', '#8a8984', '*'),
    'F_7pt': ('7pt $\\mathbf{F}$, $\\lambda{=}0$', '#8a8984', '^'), 'F_7pt_s3': ('7pt $\\mathbf{F}$, 3 $\\lambda$', '#8a8984', '*'),
}
# pipeline -> line style, the method's color stays
STYLES = {'E_5pt_V': 'solid', 'E_5pt_VLO': 'dashed', 'E_5pt_VLOeq': 'dashed',
          'Efeq_6pt_V': 'dashdot', 'Efeq_6pt_VLO': (0, (5, 1, 1, 1, 1, 1)),
          'E_3pt_V': 'dotted', 'E_3pt_VLO': (0, (1, 1)), 'E_3pt_VLOeq': (0, (1, 1))}
PIPELINE = {'E_5pt_V': '5pt', 'E_5pt_VLO': '5pt + LO', 'E_5pt_VLOeq': '5pt + LO', 'Efeq_6pt_V': '6pt',
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


def draw(directory, out, keep, ylim=None):
    name = os.path.basename(directory.rstrip('/\\'))
    plt.figure(figsize=(7.5, 4.2))
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, '*.npz'))):
        exp = os.path.basename(path)[:-4]
        if keep and not all(k in exp for k in keep):
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
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(out, f'calib_net_{name}.{ext}'), bbox_inches='tight', pad_inches=0.02, dpi=200)
    plt.close()
    with open(os.path.join(out, f'calib_net_{name}.csv'), 'w', newline='') as f:
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
    a = p.parse_args()
    for d in a.dirs:
        draw(d, a.out, a.keep, a.ylim)
