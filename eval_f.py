import argparse
import json
import os
from multiprocessing import Pool
from time import perf_counter

import h5py
import numpy as np
import poselib
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

from utils.geometry import rotation_angle, angle, k_err, f_err, normalize
from utils.geometry import get_camera_dicts, undistort, distort, pose_from_F
from utils.rand import get_random_rd_distribution
from utils.vis import draw_results_pose_auc_10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-s', '--synth', type=int, default=0)
    parser.add_argument('-e', '--eq', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()

def get_pairs(file):
    return [tuple(x.split('-'))[:2] for x in file.keys() if 'feat' not in x and 'desc' not in x and 'uneq' not in x]


def get_result_dict(info, image_pair, k1_gt, k2_gt, R_gt, t_gt, K1, K2, T1, T2):
    out = {}

    mean_scale = (T1[0, 0] + T2[0, 0]) / 2

    f1_gt = (K1[0, 0] + K1[1, 1]) / (2 * mean_scale)
    f2_gt = (K2[0, 0] + K2[1, 1]) / (2 * mean_scale)


    out['K1_gt'] = K1.tolist()
    out['K2_gt'] = K2.tolist()
    
    R_est = image_pair.pose.R
    t_est = image_pair.pose.t
    
    k1_est = image_pair.camera1.params[-1]
    k2_est = image_pair.camera2.params[-1]
    
    f1_est = image_pair.camera1.params[0]
    f2_est = image_pair.camera2.params[0]

    out['R_err'] = rotation_angle(R_est.T @ R_gt)
    out['t_err'] = angle(t_est, t_gt)
    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = R_est.tolist()
    out['t_gt'] = R_gt.tolist()

    out['P_err'] = max(out['R_err'], out['t_err'])
    out['k1_err'] = k_err(k1_gt, k1_est)
    out['k1'] = k1_est
    out['k1_gt'] = k1_gt

    out['k2_err'] = k_err(k2_gt, k2_est)
    out['k2'] = k2_est
    out['k2_gt'] = k2_gt
    
    out['f1_err'] = f_err(f1_gt, f1_est)
    out['f1'] = f1_est
    out['f1_gt'] = f1_gt

    out['f2_err'] = f_err(f2_gt, f2_est)
    out['f2'] = f2_est
    out['f2_gt'] = f2_gt

    info['inliers'] = []
    out['info'] = info

    return out



def eval_experiment(x):
    iters, experiment, kp1_distorted, kp2_distorted, k1, k2, R_gt, t_gt, T1, T2, K1, K2, sarg = x

    solver = experiment.split('_')[0]
    mean_scale = (T1[0, 0] + T2[0,0]) / 2
    
    if iters is None:
        ransac_dict = {'max_iterations': 10000, 'max_epipolar_error': 3.0 / mean_scale, 'progressive_sampling': False,
                       'min_iterations': 100, 'lo_iterations': 25}
    else:
        ransac_dict = {'max_iterations': iters, 'max_epipolar_error': 3.0 / mean_scale, 'progressive_sampling': False,
                       'min_iterations': iters}
    
    shared_intrinsics = 'kFk' in experiment or 'Efeq' in experiment
    use_minimal = 'k2k1_9pt' in experiment or 'kFk_8pt' in experiment

    bundle_dict = {'refine_pose': True, 'refine_focal_length': True, 'refine_principal_point': False,
                   'refine_extra_params': '_ns' not in experiment, 'max_iterations': 100}
    
    opt_dict = {'max_error': 3.0 / 1000, 'ransac': ransac_dict, 'bundle': bundle_dict,
                'shared_intrinsics': shared_intrinsics, 'use_minimal': use_minimal}

    if 'Efeq' in experiment or 'F_7pt' in experiment:
        rd_vals = [0.0]
        if sarg == 3:
            rd_vals = [-0.9]
        if 's3' in experiment:
            if sarg < 2:
                rd_vals = [0.0, -0.6, -1.2]
            elif sarg == 3:
                rd_vals = [-0.6, -0.9, -1.2]
    else:
        # Rd vals empty means we use nonminimal solvers
        rd_vals = []

    start = perf_counter()
    image_pair, info = poselib.estimate_focal_rd_relpose(kp1_distorted, kp2_distorted, rd_vals, opt_dict)
    info['runtime'] = 1000 * (perf_counter() - start)
    
    result_dict = get_result_dict(info, image_pair, k1, k2, R_gt, t_gt, K1, K2, T1, T2)
    result_dict['experiment'] = experiment

    return result_dict


def print_results(experiments, results, eq_only=False):
    tab = PrettyTable(['solver', 'LO',
                       'median pose err', 'Pose AUC@10',
                       'median k err', 'k AUC@0.1',
                       'median f err', 'f AUC@0.1',
                       'median time', 'mean time', 'median inliers', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        if eq_only:
            exp_results = [x for x in exp_results if x['k1_gt'] == x['k2_gt']]# and x['K1_gt'] == x['K2_gt']]
        else:
            exp_results = [x for x in exp_results if x['k1_gt'] != x['k2_gt']]  # and x['K1_gt'] == x['K2_gt']]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])

        k_errs = [k_err(r['k1_gt'], r['k1']) for r in exp_results]
        k_errs.extend([k_err(r['k2_gt'], r['k2']) for r in exp_results])
        k_errs = np.array(k_errs)
        k_errs[np.isnan(k_errs)] = 1.0
        k_res = np.array([np.sum(k_errs < t / 100) / len(k_errs) for t in range(1, 21)])

        f_errs = [f_err(r['f1_gt'], r['f1']) for r in exp_results]
        f_errs.extend([f_err(r['f2_gt'], r['f2']) for r in exp_results])
        f_errs = np.array(f_errs)
        f_errs[np.isnan(f_errs)] = 1.0
        f_res = np.array([np.sum(f_errs < t / 100) / len(f_errs) for t in range(1, 21)])

        times = np.array([x['info']['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        lo = 'kFk' if 'kFk' in exp or 'eq' in exp else 'k2Fk1'
        lo = exp.split('_')[0] if '_ns' in exp else lo
        exp_name = exp.replace('_', ' ').replace('eq','')


        tab.add_row([exp_name, lo,
                     np.median(p_errs), np.mean(p_res[:10]),
                     np.median(k_errs), np.mean(k_res[:10]),
                     np.median(f_errs), np.mean(f_res[:10]),
                     np.median(times), np.mean(times),
                     np.median(inliers), np.mean(inliers)])
    print(tab)

    print('latex')

    print(tab.get_formatted_string('latex'))


def eval(args):
    if args.synth:
        chars = ['', 'A', 'B', 'C']
        synth_string = f'synth{chars[args.synth]}'
        assert synth_string in args.feature_file
        S_file = h5py.File(os.path.join(args.dataset_path, f'{synth_string}v3-distortion.h5'))

    if args.eq:
        if args.synth != 2:
            experiments = ['Efeq_6pt', 'Efeq_6pt_s3', 'Efeq_6pt_ns',
                           'kFk_8pt', 'kFk_9pt',
                           'k2k1_9pt', 'k2Fk1_10pt',
                           'F_7pt', 'F_7pt_s3', 'F_7pt_ns']
        else:
            experiments = ['Efeq_6pt', 'Efeq_6pt_ns', 'kFk_8pt', 'kFk_9pt',
                           'k2k1_9pt', 'k2Fk1_10pt',
                           'F_7pt', 'F_7pt_ns']
    else:
        if args.synth != 2:
            experiments = ['k2k1_9pt', 'k2Fk1_10pt',
                           'F_7pt', 'F_7pt_s3', 'F_7pt_ns']
        else:
            experiments = ['k2k1_9pt', 'k2Fk1_10pt',
                           'F_7pt', 'F_7pt_ns']

    dataset_path = args.dataset_path
    basename = os.path.basename(dataset_path)

    matches_basename = os.path.basename(args.feature_file)

    if args.graph:
        basename = f'{basename}-graph'
        iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    else:
        iterations_list = [None]


    s_string = ""
    if args.synth:
        s_string = f"-synth{args.synth}"
        if args.eq:
            s_string = f"-syntheq{args.synth}"
    json_string = f'focal-{basename}-{matches_basename}{s_string}.json'

    if args.load:
        print("Loading: ", json_string)
        with open(os.path.join('results', json_string), 'r') as f:
            results = json.load(f)

    else:
        R_file = h5py.File(os.path.join(dataset_path, 'R.h5'))
        T_file = h5py.File(os.path.join(dataset_path, 'T.h5'))
        P_file = h5py.File(os.path.join(dataset_path, 'parameters_rd.h5'))
        C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))

        R_dict = {k: np.array(v) for k, v in R_file.items()}
        t_dict = {k: np.array(v) for k, v in T_file.items()}
        w_dict = {k.split('-')[0]: v[0, 0] for k, v in P_file.items()}
        h_dict = {k.split('-')[0]: v[1, 1] for k, v in P_file.items()}
        k_dict = {k.split('-')[0]: v[2, 2] for k, v in P_file.items()}
        camera_dicts = get_camera_dicts(os.path.join(dataset_path, 'K.h5'))

        pairs = get_pairs(C_file)

        if args.first is not None:
            pairs = pairs[:args.first]

        def gen_data():
            for img_name_1, img_name_2 in pairs:
                R1 = R_dict[img_name_1]
                t1 = t_dict[img_name_1]
                R2 = R_dict[img_name_2]
                t2 = t_dict[img_name_2]
                K1 = camera_dicts[img_name_1]
                K2 = camera_dicts[img_name_2]

                R_gt = np.dot(R2, R1.T)
                t_gt = t2 - np.dot(R_gt, t1)

                if args.synth:
                    if args.eq:
                        matches = np.array(C_file[f'{img_name_1}-{img_name_2}-eq'])
                    else:
                        matches = np.array(C_file[f'{img_name_1}-{img_name_2}-uneq'])
                else:
                    matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])

                kp1 = matches[:, :2]
                kp2 = matches[:, 2:4]

                # F, info = poselib.estimate_fundamental(kp1, kp2)
                # R, t = pose_from_F(F, K1, K2, kp1, kp2)
                #
                # print(angle(t,t_gt), info['inlier_ratio'])

                # use only when sift matches are used
                # if 'sift' in args.feature_file.lower():
                #     kp1 = kp1[matches[:, 4] <= 0.8]
                #     kp2 = kp2[matches[:, 4] <= 0.8]
                # kp1 = kp1[matches[:, 4] > 0.5]
                # kp2 = kp2[matches[:, 4] > 0.5]

                if len(kp1) < 10:
                    continue

                kp1_distorted, T1 = normalize(kp1, w_dict[img_name_1], h_dict[img_name_1])
                kp2_distorted, T2 = normalize(kp2, w_dict[img_name_2], h_dict[img_name_2])

                if args.synth:
                    k1 = S_file[f'{img_name_1}-{img_name_2}-k1'][()]
                    if args.eq:
                        k2 = k1
                    else:
                        k2 = S_file[f'{img_name_1}-{img_name_2}-k2'][()]
                else:
                    k1 = k_dict[img_name_1]
                    k2 = k_dict[img_name_2]

                # if k1 > -0.1 or k2 > -0.1:
                #     continue

                for experiment in experiments:
                    for iterations in iterations_list:
                        yield iterations, experiment, np.copy(kp1_distorted), np.copy(kp2_distorted), k1, k2, R_gt, t_gt, T1, T2, K1, K2, args.synth


        total_length = len(experiments) * len(pairs) * len(iterations_list)

        print(f"Total runs: {total_length} for {len(pairs)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        os.makedirs('results', exist_ok=True)

        if args.append:
            print(f"Appending from: {os.path.join('results', json_string)}")
            with open(os.path.join('results', json_string), 'r') as f:
                prev_results = json.load(f)
            results.extend(prev_results)

        with open(os.path.join('results', json_string), 'w') as f:
            json.dump(results, f)

        print("Done")

    print("Printing results for all combinations")
    print_results(experiments, results)
    # draw_cumplots(experiments, results)

    print("Printing results for pairs with equal intrinsics")
    print_results(experiments, results, eq_only=True)
    # draw_cumplots(experiments, results, eq_only=True)

    if args.graph:
        draw_results_pose_auc_10(results, experiments, iterations_list)

if __name__ == '__main__':
    args = parse_args()
    eval(args)