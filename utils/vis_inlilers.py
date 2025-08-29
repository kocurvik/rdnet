import json
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

from utils.vis import large_size, small_size, font

inlier_ratios = [1.0, 0.9, 0.8, 0.75, 0.5, 0.25]


basenames = ['E_3pt', 'E_5pt', 'Efeq_6pt_s3', 'F_7pt_s3', 'kFk_9pt', 'k2Fk1_10pt', 'kFk_8pt', 'empty', 'k2k1_9pt']

def get_color_style(experiment):
    color_dict = {exp: sns.color_palette("tab10")[i] for i, exp in enumerate(basenames)}
    color_dict['Efeq_6pt'] = color_dict['Efeq_6pt_s3']

    color = color_dict[experiment.split('+')[0]]

    if '+' in experiment and 'VLO' in experiment:
        style = 'dotted'
    elif '+' in experiment:
        style = 'solid'
    else:
        style = 'solid'

    if '+' in experiment:
        marker = None
    elif '_s3' in experiment:
        marker = '*'
    else:
        marker = 'o'

    return color, style, marker

def get_dict(eq_str, experiments):
    d = {k: {} for k in experiments}

    for i in inlier_ratios:
        with open(os.path.join('results', f'focal-cathedral-{i:.2f}inliers-pairs-features_superpoint_noresize_2048-LG{eq_str}.json'), 'r') as f:
            results = json.load(f)

        for exp in experiments:
            exp_results = [x for x in results if x['experiment'] == exp]
            p_errs = np.array([r['P_err'] for r in exp_results])
            p_errs[np.isnan(p_errs)] = 180
            AUC10 = np.mean(np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 11)]))

            k_errs = np.array([0.5 * (np.abs(r['k1'] - r['k1_gt']) + np.abs(r['k2'] - r['k2_gt'])) for r in exp_results])
            k_errs[np.isnan(k_errs)] = 2.0
            med_k = np.median(k_errs)

            f_errs = np.array(
                [0.5 * (np.abs(r['f1'] - r['f1_gt']) / r['f1_gt'] + np.abs(r['f2'] - r['f2_gt']) / r['f2_gt']) for r in
                 exp_results])
            f_errs[np.isnan(f_errs)] = 2.0
            med_f = np.median(f_errs)

            d[exp][i] = {'med_k': med_k, 'med_f': med_f, 'AUC10': AUC10}

    # for exp in experiments:
    #     if 'Geo_V' in exp and 'VLO' not in exp:
    #         for i in inlier_ratios[1:]:
    #             d[exp][i]['med_k'] = d[exp][inlier_ratios[0]]['med_k']
    #             d[exp][i]['med_f'] = d[exp][inlier_ratios[0]]['med_f']

    return d

def get_experiments(eq, geo_iters=(1,2,5,30)):
    if eq:
        experiments = ['Efeq_6pt_s3', 'kFk_9pt', 'kFk_8pt', 'Efeq_6pt+Geo_VLO', 'E_5pt+Geo_VLO', 'E_3pt+Geo_VLO',
                       'Efeq_6pt+Geo_V', 'E_5pt+Geo_V', 'E_3pt+Geo_V']
    else:
        experiments = ['F_7pt_s3', 'k2Fk1_10pt', 'k2k1_9pt','E_5pt+Geo_VLO', 'E_5pt+Geo_V', 'E_3pt+Geo_VLO', 'E_3pt+Geo_V']
    return experiments


def plot(experiments, d, key, title, label):
    plt.figure(frameon=False, figsize=(0.5 * 8, 0.5 * 5))

    all_ys = []

    for exp in experiments:
        xs = inlier_ratios
        ys = [d[exp][i][key] for i in inlier_ratios]
        all_ys.extend(ys)
        color, style, marker = get_color_style(exp)
        plt.plot(xs, ys, label=exp, color=color, linestyle=style, marker=marker)

    plt.xlabel('Inlier Ratio', fontsize=large_size, **font)
    plt.ylabel(label, fontsize=large_size, **font)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    plt.xticks([0.25, 0.5, 0.75, 1.0])

    y_min = np.floor(np.percentile(all_ys, 5) * 10) / 10
    if y_min < 0.2:
        y_min = 0.0

    y_max = np.ceil(np.percentile(all_ys, 90) * 10) / 10 + 0.1

    plt.ylim([y_min, y_max])

    if y_max - y_min > 0.3:
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    else:
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))

    plt.savefig(f'figs/{title}.pdf', bbox_inches='tight', pad_inches=0.0)
    print(f'saved pose: {title}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'figs/{title}.png', bbox_inches='tight', pad_inches=0.0)
    print(f'saved pose: {title}')
    plt.show()

def draw_inlier_graph(eq=True, load=False):
    eq_str = '_eq' if eq else ''
    experiments = get_experiments(eq)

    if load:
        with open(f'fig_data/focal_inliers_cathedral{eq_str}.json', 'r') as f:
            d = json.load(f)
        for exp in experiments:
            for i in inlier_ratios:
                d[exp][i] = d[exp][str(i)]
    else:
        d = get_dict(eq_str, experiments)
        with open(f'fig_data/focal_inliers_cathedral{eq_str}.json', 'w') as f:
            json.dump(d, f)

    plot(experiments, d, 'AUC10', f'inliers{eq_str}_pose', 'AUC@10$^\\circ$')
    plot(experiments, d, 'med_k', f'inliers{eq_str}_k', 'Median ε(λ)')
    plot(experiments, d, 'med_f', f'inliers{eq_str}_f', 'Median ξ(f)')


if __name__ == '__main__':
    draw_inlier_graph(eq=True, load=False)
    draw_inlier_graph(eq=False, load=False)