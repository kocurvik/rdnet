import argparse
import json
import os
from multiprocessing import Pool
from time import perf_counter

import h5py
import numpy as np
import poselib
from tqdm import tqdm

from utils.data import get_pairs
from utils.geometry import rotation_angle, angle, k_err, f_err, normalize
from utils.geometry import get_camera_dicts
from utils.tables import print_results
from utils.vis import draw_results_pose_auc_10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-t', '--threshold', type=float, default=3.0)
    parser.add_argument('-e', '--eq', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def get_result_dict(info, image_pair, k1_gt, k2_gt, R_gt, t_gt, K1, K2, T1, T2, base_runtime):
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
    info['ransac_runtime'] = info['runtime']
    info['base_runtime'] = base_runtime
    info['runtime'] = info['runtime'] + base_runtime
    out['info'] = info

    return out


def eval_experiment(x):
    iters, experiment, kp1_distorted, kp2_distorted, k1, k2, R_gt, t_gt, T1, T2, K1, K2, net_dict, sarg, t, base_runtime = x

    solver = experiment.split('_')[0]
    mean_scale = (T1[0, 0] + T2[0, 0]) / 2

    if iters is None:
        ransac_dict = {'max_iterations': 10000, 'max_epipolar_error': t / mean_scale, 'progressive_sampling': False,
                       'min_iterations': 100, 'lo_iterations': 25}
    else:
        ransac_dict = {'max_iterations': iters, 'max_epipolar_error': t / mean_scale, 'progressive_sampling': False,
                       'min_iterations': iters}

    shared_intrinsics = 'kFk' in experiment or 'eq' in experiment
    use_minimal = 'k2k1_9pt' in experiment or 'kFk_8pt' in experiment

    bundle_dict = {'refine_pose': True, 'refine_focal_length': True, 'refine_principal_point': False,
                   'refine_extra_params': '_ns' not in experiment, 'max_iterations': 100}

    opt_dict = {'max_error': 3.0 / mean_scale, 'ransac': ransac_dict, 'bundle': bundle_dict,
                'shared_intrinsics': shared_intrinsics, 'use_minimal': use_minimal, 'tangent_sampson': True}

    if 'E_5pt' in experiment or 'E_3pt' in experiment:
        net = experiment.split('+')[1]
        net_name, net_use, net_iters = net.split('_')
        net_iters = int(net_iters)

        if 'E_3p' in experiment:
            g1 = net_dict[net_iters][f'g1']
            g2 = net_dict[net_iters][f'g2']
        else:
            g1 = np.zeros(3)
            g2 = np.zeros(3)

        camera1 = {'model': "SIMPLE_DIVISION", 'width': -1, 'height': -1,
                   'params': [net_dict[net_iters]['f1'], 0.0, 0.0, net_dict[net_iters]['k1']]}
        camera2 = {'model': "SIMPLE_DIVISION", 'width': -1, 'height': -1,
                   'params': [net_dict[net_iters]['f2'], 0.0, 0.0, net_dict[net_iters]['k2']]}

        if 'V' == net_use:
            start = perf_counter()
            pose, info = poselib.estimate_relative_pose(kp1_distorted, kp2_distorted, camera1, camera2, opt_dict, g1,
                                                        g2)
            info['runtime'] = 1000 * (perf_counter() - start)

            camera1 = poselib.Camera("SIMPLE_DIVISION",
                                     [net_dict[net_iters]['f1'], 0.0, 0.0, net_dict[net_iters]['k1']], -1, -1)
            camera2 = poselib.Camera("SIMPLE_DIVISION",
                                     [net_dict[net_iters]['f2'], 0.0, 0.0, net_dict[net_iters]['k2']], -1, -1)
            image_pair = poselib.ImagePair(pose, camera1, camera2)

            result_dict = get_result_dict(info, image_pair, k1, k2, R_gt, t_gt, K1, K2, T1, T2, base_runtime)
            result_dict['experiment'] = experiment

            return result_dict
        if 'VLO' in net_use:
            start = perf_counter()
            image_pair, info = poselib.estimate_relative_pose_lo(kp1_distorted, kp2_distorted, camera1, camera2,
                                                                 opt_dict, g1, g2)
            info['runtime'] = 1000 * (perf_counter() - start)
            result_dict = get_result_dict(info, image_pair, k1, k2, R_gt, t_gt, K1, K2, T1, T2, base_runtime)
            result_dict['experiment'] = experiment
            return result_dict

    if 'Efeq' in experiment or 'F_7pt' in experiment:
        if '+' in experiment:
            net = experiment.split('+')[1]
            net_name, net_use, net_iters = net.split('_')
            net_iters = int(net_iters)
            rd_pred_1 = net_dict[net_iters]['k1']
            rd_pred_2 = net_dict[net_iters]['k2']

            if net_use == 'V':
                bundle_dict['refine_extra_params'] = False

            rd_vals_1 = [rd_pred_1]
            rd_vals_2 = [rd_pred_2]
        else:
            rd_vals_1 = [0.0]
            if sarg == 3:
                rd_vals_1 = [-0.9]
            if 's3' in experiment:
                if sarg < 2:
                    rd_vals_1 = [0.0, -0.6, -1.2]
                elif sarg == 3:
                    rd_vals_1 = [-0.6, -0.9, -1.2]
            rd_vals_2 = rd_vals_1
    else:
        # Rd vals empty means we use nonminimal solvers
        rd_vals_1 = []
        rd_vals_2 = []

    start = perf_counter()
    image_pair, info = poselib.estimate_focal_rd_relpose(kp1_distorted, kp2_distorted, rd_vals_1, rd_vals_2, opt_dict)
    info['runtime'] = 1000 * (perf_counter() - start)

    result_dict = get_result_dict(info, image_pair, k1, k2, R_gt, t_gt, K1, K2, T1, T2, base_runtime)
    result_dict['experiment'] = experiment

    return result_dict


def geo_iter_runtime(i, eq=False):
    if eq:
        return 10 * i + 80

    return 10 * i + 80


def eval(args):
    # geo_iters = [1, 5, 10, 15, 20, 30]
    geo_iters = [1, 2, 5, 30]
    if args.eq:
        experiments = ['Efeq_6pt', 'Efeq_6pt_s3', 'k2Fk1_10pt']
        base_runtimes = [0, 0, 0]
        experiments.extend([f'Efeq_6pt+Geo_VLO_{i}' for i in geo_iters])
        base_runtimes.extend([geo_iter_runtime(i, eq=True) for i in geo_iters])
        experiments.extend([f'E_5pt+Geo_VLO_{i}' for i in geo_iters])
        base_runtimes.extend([geo_iter_runtime(i, eq=True) for i in geo_iters])
        experiments.extend([f'E_3pt+Geo_VLO_{i}' for i in geo_iters])
        base_runtimes.extend([geo_iter_runtime(i, eq=True) for i in geo_iters])
    else:
        experiments = ['F_7pt', 'F_7pt_s3', 'k2Fk1_10pt']
        base_runtimes = [0, 0, 0]
        experiments.extend([f'E_5pt+Geo_VLO_{i}' for i in geo_iters])
        base_runtimes.extend([geo_iter_runtime(i) for i in geo_iters])
        experiments.extend([f'E_5pt+Geo_V_{i}' for i in geo_iters])
        base_runtimes.extend([geo_iter_runtime(i) for i in geo_iters])
        experiments.extend([f'E_3pt+Geo_VLO_{i}' for i in geo_iters])
        base_runtimes.extend([geo_iter_runtime(i) for i in geo_iters])
        experiments.extend([f'E_3pt+Geo_V_{i}' for i in geo_iters])
        base_runtimes.extend([geo_iter_runtime(i) for i in geo_iters])

    dataset_path = args.dataset_path
    basename = os.path.basename(dataset_path)

    matches_basename = os.path.basename(args.feature_file)

    # iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    iterations_list = [10, 20, 50, 100, 200, 500, 1000]

    t_string = "" if args.threshold == 3.0 else f'-{args.threshold}t'
    json_string = f'focal-graph-{basename}-{matches_basename}{t_string}.json'

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

        geo_dicts = {i : {} for i in geo_iters}
        for i in geo_iters:
            Geo_file = h5py.File(os.path.join(dataset_path, f'GeoCalibPredictions_kfg_iter{i}.h5'))
            geo_k_dict = {k.split('-')[0]: v[()] for k, v in Geo_file.items() if '-k' in k}
            geo_f_dict = {k.split('-')[0]: np.mean(v[()]) for k, v in Geo_file.items() if '-f' in k}
            geo_g_dict = {k.split('-')[0]: np.array(v) for k, v in Geo_file.items() if '-g' in k}
            # scale geo dict
            geo_k_dict = {k: v * (max(h_dict[k], w_dict[k]) / geo_f_dict[k]) ** 2 for k, v in geo_k_dict.items()}
            geo_f_dict = {k: v / max(h_dict[k], w_dict[k]) for k, v in geo_f_dict.items()}

            geo_dicts[i]['f'] = geo_f_dict
            geo_dicts[i]['k'] = geo_k_dict
            geo_dicts[i]['g'] = geo_g_dict

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

                # if args.synth:
                #     if args.eq:
                #         # matches = np.array(C_file[f'{img_name_1}-{img_name_2}-eq'])
                #         matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])
                #     else:
                #         # matches = np.array(C_file[f'{img_name_1}-{img_name_2}-uneq'])
                #         matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])
                # else:
                matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])

                kp1 = matches[:, :2]
                kp2 = matches[:, 2:4]

                if len(kp1) < 10:
                    continue

                kp1_distorted, T1 = normalize(kp1, w_dict[img_name_1], h_dict[img_name_1])
                kp2_distorted, T2 = normalize(kp2, w_dict[img_name_2], h_dict[img_name_2])

                k1 = k_dict[img_name_1]
                k2 = k_dict[img_name_2]

                net_dict = {i: {} for i in geo_iters}

                for i in geo_iters:
                    net_dict[i]['k1'] = geo_dicts[i]['k'][img_name_1]
                    net_dict[i]['k2'] = geo_dicts[i]['k'][img_name_2]
                    net_dict[i]['f1'] = geo_dicts[i]['f'][img_name_1]
                    net_dict[i]['f2'] = geo_dicts[i]['f'][img_name_2]
                    net_dict[i]['g1'] = geo_dicts[i]['g'][img_name_1]
                    net_dict[i]['g2'] = geo_dicts[i]['g'][img_name_2]

                for experiment, base_runtime in zip(experiments, base_runtimes):
                    for iterations in iterations_list:
                        yield iterations, experiment, np.copy(kp1_distorted), np.copy(kp2_distorted), k1, k2, R_gt, t_gt, T1, T2, K1, K2, net_dict, 0, args.threshold, base_runtime

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

    draw_results_pose_auc_10(results, experiments, iterations_list)


if __name__ == '__main__':
    args = parse_args()
    eval(args)