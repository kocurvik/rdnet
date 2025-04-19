import json
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.data import experiments, iterations_list, colors

large_size = 24
small_size = 20

print(colors)

print(sns.color_palette("tab10").as_hex())

basenames = ['E_3pt', 'E_5pt', 'Efeq_6pt_s3', 'F_7pt_s3', 'kFk_9pt', 'k2Fk1_10pt']

def get_color_style(experiment):
    color_dict = {exp: sns.color_palette("tab10")[i] for i, exp in enumerate(basenames)}

    color = color_dict[experiment.split('+')[0]]

    if '+' in experiment and 'VLO' in experiment:
        style = 'dashed'
    elif '+' in experiment:
        style = 'dotted'
    else:
        style = 'solid'

    if '+' in experiment:
        marker = None
    elif '_s3' in experiment:
        marker = 'star'
    else:
        marker = 'circle'

    return color, style, marker





# import matplotlib.font_manager as fm
# print(sorted(fm.get_font_names()))
# font = {'fontname' : 'Latin Modern Math'}
# plt.rcParams.update({
#     "text.usetex": True
# })
font = {}


def find_boundary(x_values, y_values, lowest_y=True):
    xs = np.array(x_values)
    ys = np.array(y_values)

    new_ys = np.empty_like(ys)

    for i in range(len(xs)):
        if lowest_y:
            new_ys[i] = np.min(ys[xs <= xs[i]])
        else:
            new_ys[i] = np.max(ys[xs <= xs[i]])

    new_ys = new_ys[xs.argsort()]
    xs = xs[xs.argsort()]

    return xs.tolist(), new_ys.tolist()

def process_curves(d, experiments, use_lowest=True):
    new_d = {exp: d[exp] for exp in experiments if '+' not in exp}
    net_experiments = [exp for exp in experiments if '+' in exp]

    base_exp_names = sorted(list({item[:item.rindex('_')] if '_' in item else item for item in net_experiments}))
    lm_iters = sorted(list({item[item.rindex('_') + 1:] if '_' in item else "" for item in net_experiments}))

    for net_exp in base_exp_names:
        xs = []
        ys = []
        for i in lm_iters:
            xs.extend(d[f'{net_exp}_{i}']['xs'])
            ys.extend(d[f'{net_exp}_{i}']['ys'])

        new_xs, new_ys = find_boundary(xs, ys, lowest_y=use_lowest)

        new_d[net_exp] = {'xs': new_xs, 'ys': new_ys}

    return new_d


def draw_results_pose_auc_10(results, experiments, iterations_list, title=None):
    plt.figure(frameon=True)

    d = {}

    for experiment in experiments:
        experiment_results = [x for x in results if x['experiment'] == experiment]

        d[experiment] = {'xs': [], 'ys': []}

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([r['P_err'] for r in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))

            d[experiment]['xs'].append(mean_runtime)
            d[experiment]['ys'].append(AUC10)

        # color, style = get_color_style(experiment, experiments)
        # plt.semilogx(xs, ys, label=experiment, marker='*', color=color, linestyle=style)

    d = process_curves(d, experiments, use_lowest=False)

    for experiment, vals in d.items():
        color, style, marker = get_color_style(experiment)
        plt.semilogx(vals['xs'], vals['ys'], label=experiment, color=color, linestyle=style, marker=marker)

    # plt.xlim([5.0, 1.9e4])
    plt.xlabel('Mean runtime (ms)', fontsize=large_size, **font)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size, **font)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        plt.title(title)
        # plt.legend()
        plt.savefig(f'figs/{title}_pose.pdf')#, bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()
        plt.savefig(f'figs/{title}_pose.png', bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()


def draw_results_k_med(results, experiments, iterations_list, title=None):
    plt.figure(frameon=True)

    d = {}

    for experiment in experiments:
        experiment_results = [x for x in results if x['experiment'] == experiment]

        d[experiment] = {'xs': [], 'ys': []}

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([0.5 * (np.abs(r['k1'] - r['k1_gt']) + np.abs(r['k2'] - r['k2_gt'])) for r in iter_results])
            # errs.extend([np.abs(r['k2'] - r['k2_gt']) for r in iter_results])
            # errs = np.array(errs)
            errs[np.isnan(errs)] = 2.0
            med = np.median(errs)

            d[experiment]['xs'].append(mean_runtime)
            d[experiment]['ys'].append(med)

    d = process_curves(d, experiments, use_lowest=True)

    for experiment, vals in d.items():
        color, style, marker = get_color_style(experiment)
        plt.semilogx(vals['xs'], vals['ys'], label=experiment, color=color, linestyle=style, marker=marker)

    plt.xlabel('Mean runtime (ms)', fontsize=large_size, **font)
    # plt.ylabel('Median absolute $\\lambda$ error', fontsize=large_size)
    # plt.ylabel('Mean $\\epsilon(\\lambda)$', fontsize=large_size, **font)
    plt.ylabel('Median ε(λ)', fontsize=large_size, **font)
    plt.ylim([0.0, 0.5])
    # plt.xlim([5.0, 1.9e4])
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        plt.title(title)
        # plt.legend()
        plt.savefig(f'figs/{title}_k.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved k: {title}')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'figs/{title}_k.png', bbox_inches='tight', pad_inches=0)
    else:
        plt.legend()
        plt.show()


def draw_results_f_med(results, experiments, iterations_list, title=None):
    plt.figure(frameon=True)

    d = {}

    for experiment in experiments:
        experiment_results = [x for x in results if x['experiment'] == experiment]

        d[experiment] = {'xs': [], 'ys': []}

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([0.5 * (np.abs(r['f1'] - r['f1_gt'])/r['f1_gt'] + np.abs(r['f2'] - r['f2_gt'])/r['f2_gt']) for r in iter_results])
            # errs.extend([np.abs(r['k2'] - r['k2_gt']) for r in iter_results])
            # errs = np.array(errs)
            errs[np.isnan(errs)] = 2.0
            med = np.median(errs)

            d[experiment]['xs'].append(mean_runtime)
            d[experiment]['ys'].append(med)

    d = process_curves(d, experiments, use_lowest=True)

    for experiment, vals in d.items():
        color, style, marker = get_color_style(experiment)
        plt.semilogx(vals['xs'], vals['ys'], label=experiment, color=color, linestyle=style, marker=marker)

    plt.xlabel('Mean runtime (ms)', fontsize=large_size, **font)
    # plt.ylabel('Median absolute $\\lambda$ error', fontsize=large_size)
    # plt.ylabel('Mean $\\epsilon(\\lambda)$', fontsize=large_size, **font)
    plt.ylabel('Median ξ(f)', fontsize=large_size, **font)
    plt.ylim([0.0, 0.5])
    # plt.xlim([5.0, 1.9e4])
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        plt.title(title)
        plt.savefig(f'figs/{title}_f.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved k: {title}')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'figs/{title}_f.png', bbox_inches='tight', pad_inches=0)
    else:
        plt.legend()
        plt.show()



def get_experiments(eq, geo_iters=(1,2,5,30)):
    if eq:
        experiments = ['Efeq_6pt', 'Efeq_6pt_s3', 'kFk_9pt']
        experiments.extend([f'Efeq_6pt+Geo_VLO_{i}' for i in geo_iters])
        experiments.extend([f'E_5pt+Geo_VLO_{i}' for i in geo_iters])
        experiments.extend([f'E_3pt+Geo_VLO_{i}' for i in geo_iters])
        experiments.extend([f'Efeq_6pt+Geo_V_{i}' for i in geo_iters])
        experiments.extend([f'E_5pt+Geo_V_{i}' for i in geo_iters])
        experiments.extend([f'E_3pt+Geo_V_{i}' for i in geo_iters])
    else:
        experiments = ['F_7pt', 'F_7pt_s3', 'k2Fk1_10pt']
        experiments.extend([f'E_5pt+Geo_VLO_{i}' for i in geo_iters])
        experiments.extend([f'E_5pt+Geo_V_{i}' for i in geo_iters])
        experiments.extend([f'E_3pt+Geo_VLO_{i}' for i in geo_iters])
        experiments.extend([f'E_3pt+Geo_V_{i}' for i in geo_iters])
    return experiments

def draw_graphs(name):
    with open(os.path.join('results', f'{name}.json'), 'r') as f:
        results = json.load(f)

    experiments = get_experiments('eq' in name)
    # experiments = sorted(list(set([x['experiment'] for x in results])))

    draw_results_pose_auc_10(results, experiments, iterations_list, title=name)
    draw_results_k_med(results, experiments, iterations_list, title=name)
    draw_results_f_med(results, experiments, iterations_list, title=name)

if __name__ == '__main__':
    draw_graphs('focal-graph-cathedral-pairs-features_superpoint_noresize_2048-LG')
    draw_graphs('focal-graph-cathedral-pairs-features_superpoint_noresize_2048-LG_eq')
    draw_graphs('focal-graph-rotunda_new-pairs-features_superpoint_noresize_2048-LG')
    draw_graphs('focal-graph-rotunda_new-pairs-features_superpoint_noresize_2048-LG_eq')