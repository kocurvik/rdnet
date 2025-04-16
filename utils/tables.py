import json
import os

import numpy as np

from utils.data import basenames_pt, basenames_eth

eq_order = ['k2k1_9pt', 'k2Fk1_10pt', 'kFk_8pt', 'kFk_9pt',
            'Fns_7pt', 'F_7pt', 'F_7pt_s3', 'Efeq_6pt', 'Efeq_6pt_s3', 'F_7pt+Geo_V', 'F_7pt+Geo_VLO',
            'Efeq_6pt+Geo_V', 'Efeq_6pt+Geo_VLO'
            'E_5pt+Geo_V', 'E_5pt+Geo_VLO', 'E_3pt+Geo_V', 'E_3pt+Geo_VLO']

neq_order = ['k2k1_9pt', 'k2Fk1_10pt',
             'Fns_7pt', 'F_7pt', 'F_7pt_s3',  'F_7pt+Geo_V', 'F_7pt+Geo_VLO',
             'E_5pt+Geo_V', 'E_5pt+Geo_VLO', 'E_3pt+Geo_V', 'E_3pt+Geo_VLO']


incdec = [1, 1, -1, -1, -1, 1, 1, 1]

def table_text(dataset_name, eq_rows, neq_rows, sarg):
    leq = len(eq_rows)
    lneq = len(neq_rows)

    rd_val = '0'
    if sarg == 3:
        rd_val = '-0.9'
    if sarg < 3:
        rd_vals = '0.0, -0.6, -1.2'
    elif sarg == 3:
        rd_vals ='-0.6, -0.9, -1.2'
    if sarg == 2:
        comment = '%'
        leq -= 2
        lneq -= 1
    else:
        comment = ''



    table_f_string = (
        f'\\begin{{tabular}}{{ c | r c c | c c c | c c | c c | c}}\n'
        f'    \\toprule\n'
        f'    & & & & \\multicolumn{{7}}{{c}}{{Poselib - {dataset_name}}} \\\\\n'
        f'    \\midrule\n'
        f'    & Minimal & Refinement & Sample & AVG $(^\\circ)$ $\\downarrow$ & MED $(^\\circ)$ $\\downarrow$ & AUC@10 $\\uparrow$ & AVG $\\epsilon(\\lambda)$ $\\downarrow$ & MED $\\epsilon(\\lambda)$ $\\downarrow$  & AVG $\\xi(f)$ $\\downarrow$ & MED $\\xi(f)$ $\\downarrow$ & Time (ms) $\\downarrow$ \\\\\n'
        f'    \\midrule\n'
        f'    \\multirow{{{leq}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{$\\lambda_1 = \\lambda_2$}}}} '
        f'    & 9pt \\Fkk & $\\R,\\t,f,\\lambda$ & \\ding{{55}} & {eq_rows[0]} \\\\\n'
        f'    & 10pt \\Fkk & $\\R,\\t,f,\\lambda$ & \\ding{{55}} & {eq_rows[1]} \\\\\n'
        f'    & 8pt \\Fk & $\\R,\\t,f,\\lambda$ & \\ding{{55}} & {eq_rows[2]} \\\\\n'
        f'    & 9pt \\Fk & $\\R,\\t,f,\\lambda$ & \\ding{{55}} & {eq_rows[3]} \\\\\n'        
        f'    & 7pt \\F & $\\R,\\t,f$ & $\\lambda$ = 0 & {eq_rows[4]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f,\\lambda$ & $\\lambda = {rd_val}$ & {eq_rows[5]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f,\\lambda$ & $\\lambda \\in \\{{{rd_vals}\\}}$ & {eq_rows[6]} \\\\\n'
        f'    & 6pt \\Ef & $\\R,\\t,f,\\lambda$ & $\\lambda = {rd_val}$ & {eq_rows[7]} \\\\\n'
        f'    & 6pt \\Ef & $\\R,\\t,f,\\lambda$ & $\\lambda \\in \\{{{rd_vals}\\}}$ & {eq_rows[8]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f$ & GeoCalib - $\\lambda$ & {eq_rows[9]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f,\\lambda$ &  GeoCalib - $\\lambda$ & {eq_rows[10]} \\\\\n'
        f'    & 6pt \\Ef & $\\R,\\t,f$ & GeoCalib - $\\lambda$ & {eq_rows[11]} \\\\\n'
        f'    & 6pt \\Ef & $\\R,\\t,f,\\lambda$ & GeoCalib - $\\lambda$ & {eq_rows[12]} \\\\\n'
        f'    & 5pt \\E & $\\R$ & GeoCalib - $\\lambda,f$ & {eq_rows[13]} \\\\\n'
        f'    & 5pt \\E & $\\R,\\t,f,\\lambda$ & GeoCalib - $\\lambda,f$ & {eq_rows[14]} \\\\\n'
        f'    & 3pt \\E & $\\R,\\t$ & GeoCalib - $\\lambda,f,\\g$ & {eq_rows[15]} \\\\\n'
        f'    & 3pt \\E & $\\R,\\t,f,\\lambda$ & GeoCalib - $\\lambda,f,\\g$ & {eq_rows[16]} \\\\\n'

        f'    \\midrule\n'
        f'    \\midrule\n'
        f'    \\multirow{{{lneq}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{$\\lambda_1 \\neq \\lambda_2$}}}} '
        f'    & 9pt \\Fkk & $\\R,\\t,f_1,f_2,\\lambda_1,\\lambda_2$ & \\ding{{55}} & {neq_rows[0]} \\\\\n'
        f'    & 10pt \\Fkk & $\\R,\\t,f_1,f_2,\\lambda_1, \\lambda_2$ & \\ding{{55}} & {neq_rows[1]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f_1, f_2$ & $\\lambda_1 = \\lambda_2$ = 0 & {neq_rows[2]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f_1, f_2, \\lambda_1, \\lambda_2$ & $\\lambda_1 = \\lambda_2 = {rd_val}  $ & {neq_rows[3]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f_1, f_2, \\lambda_1, \\lambda_2$ & $\\lambda_1, \\lambda_2 \\in \\{{{rd_vals}\\}} $ & {neq_rows[4]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f_1, f_2$ & GeoCalib - $\\lambda_1, \\lambda_2$ & {neq_rows[5]} \\\\\n'
        f'    & 7pt \\F & $\\R,\\t,f_1,f_2,\\lambda_1, \\lambda_2$ &  GeoCalib - $\\lambda_1, \\lambda_2$ & {neq_rows[6]} \\\\\n'
        f'    & 5pt \\E & $\\R$ & GeoCalib - $\\lambda_1, \\lambda_2,f_1, f_2$ & {neq_rows[7]} \\\\\n'
        f'    & 5pt \\E & $\\R,\\t,f_1, f_2,\\lambda_1, \\lambda_2$ & GeoCalib - $\\lambda_1,\\lambda_2,f_1,f_2$ & {neq_rows[8]} \\\\\n'
        f'    & 3pt \\E & $\\R,\\t$ & GeoCalib - $\\lambda_1, \\lambda_2,f_1, f_2,\\g_1,\\g_2$ & {neq_rows[9]} \\\\\n'
        f'    & 3pt \\E & $\\R,\\t,f_1,f_2, \\lambda_1, \\lambda_2 $ & GeoCalib - $\\lambda_1, \\lambda_2, f_1, f_2, \\g_1, \\g_2$ & {neq_rows[10]} \\\\\n'
        f'    \\bottomrule\n'
        f'\\end{{tabular}}'
    )
    return table_f_string




def get_rows(results, order):
    num_rows = []

    for experiment in order:
        exp_results = [x for x in results if x['experiment'] == experiment]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])
        p_auc_10 = np.mean(p_res[:10])
        p_avg = np.mean(p_errs)
        p_med = np.median(p_errs)

        k_errs = np.array([0.5 * (np.abs(r['k1'] - r['k1_gt']) + np.abs(r['k2'] - r['k2_gt'])) for r in exp_results])
        k_errs[np.isnan(k_errs)] = 4.0
        k_avg = np.mean(k_errs)
        k_med = np.median(k_errs)

        f_errs = np.array([0.5 * (np.abs(r['f1'] - r['f1_gt'])/r['f1_gt'] + np.abs(r['f2'] - r['f2_gt'])/r['f2_gt']) for r in exp_results])
        f_errs[np.isnan(f_errs)] = 1.0
        f_avg = np.mean(f_errs)
        f_med = np.median(f_errs)
        f_res = np.array([np.sum(p_errs < t / 100) / len(p_errs) for t in range(1, 21)])
        f_auc_10 = np.mean(f_res[:10])

        times = [r['info']['runtime'] for r in exp_results]
        time_avg = np.mean(times)

        num_rows.append([p_avg, p_med, p_auc_10, k_avg, k_med, f_avg, f_med, time_avg])

    text_rows = [[f'{x:0.2f}' for x in y] for y in num_rows]
    lens = np.array([[len(x) for x in y] for y in text_rows])
    arr = np.array(num_rows)
    for j in range(len(text_rows[0])):
        idxs = np.argsort(incdec[j] * arr[:, j])
        text_rows[idxs[0]][j] = '\\textbf{' + text_rows[idxs[0]][j] + '}'
        text_rows[idxs[1]][j] = '\\underline{' + text_rows[idxs[1]][j] + '}'

    max_len = np.max(lens, axis=0)
    phantoms = max_len - lens
    for i in range(len(text_rows)):
        for j in range(len(text_rows[0])):
            if phantoms[i, j] > 0:
                text_rows[i][j] = '\\phantom{' + (phantoms[i, j] * '1') + '}' + text_rows[i][j]

    return [' & '.join(row) for row in text_rows]

def generate_table(dataset, i, feat):
    if dataset == 'pt':
        basenames = basenames_pt
        name = '\\Phototourism'
    elif dataset == 'eth3d':
        basenames = basenames_eth
        name = '\\ETH'
    elif dataset == 'rotunda':
        basenames = ['rotunda_new']
        name = '\\ROTUNDA'
    elif dataset == 'cathedral':
        basenames = ['cathedral']
        name = '\\CATHEDRAL'

    else:
        raise ValueError

    if i > 0:
        synth_char = "XABC"[i]
        name = name + f' - Synth {synth_char}'

    if i > 0:
        neq_results_type = f'focal-synth{synth_char}-uneq-final-pairs-features_{feat}_noresize_2048-LG-synth{i}'
        eq_results_type = f'focal-synth{synth_char}-eq-final-pairs-features_{feat}_noresize_2048-LG-synth{i}'
    else:
        neq_results_type = f'pairs-features_{feat}_noresize_2048-LG'
        eq_results_type = f'pairs-features_{feat}_noresize_2048-LG_eq'

    # results_type = 'graph-SIFT_triplet_correspondences'

    neq_results = []
    eq_results = []
    for basename in basenames:
        json_path = os.path.join('results', f'{basename}-{neq_results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            neq_results.extend(json.load(f))
        json_path = os.path.join('results', f'{basename}-{eq_results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            eq_results.extend(json.load(f))

    print("Data loaded")

    print(30 * '*')
    print(30 * '*')
    print(30 * '*')
    print("Printing: ", name)
    print(30 * '*')

    neq_rows = get_rows(neq_results, neq_order)
    eq_rows = get_rows(eq_results, eq_order)
    print(table_text(name, eq_rows, neq_rows, i))

if __name__ == '__main__':
    # for features in ['superpoint', 'sift']:
    #     generate_table('rotunda', 0, features)
    #     generate_table('cathedral', 0, features)

    # for i in range(1, 4):
    #     generate_table('pt', i, 'superpoint')
    for i in range(1, 4):
        generate_table('eth3d', i, 'superpoint')
