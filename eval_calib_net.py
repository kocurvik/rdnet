"""Relative pose from calibration networks against the non-minimal and sampling solvers.

Reads the predictions radcalib's `eval/rd_predict.py` writes (one `<label>.h5` per network) and
runs PoseLib over RANSAC budgets, one file per experiment, so a killed job resumes:

    <out>/<dataset>[_eq]/<experiment>.npz

Experiments, as in `eval_graph.py`:

    different intrinsics   k2Fk1_10pt, F_7pt, F_7pt_s3;
                           E_5pt+<net>_{V,VLO}
    equal intrinsics (-e)  kFk_9pt, k2Fk1_10pt, Efeq_6pt, Efeq_6pt_s3;
                           E_5pt+<net>_{V,VLOeq}; Efeq_6pt+<net>_{V,VLO}

Vfix keeps the network's camera throughout. V: the same in RANSAC and local optimization, but the final
refinement also frees f and k (as `eval_graph.py`'s `_V`), and only its pose is kept. VLO refines f and k in
local optimization too, VLOeq refines one shared camera. Veq: V with the final refinement on one shared camera
(V on the equal graph frees each image's f and k separately). On the equal graph a network gives one (f, lambda) per pair, its joint fit.

A network's time per pair goes in `base_ms`, measured in one job by `rd_predict.py`:
2 (net + fit) on different intrinsics, 2 net + joint fit on equal ones.

    python eval_calib_net.py --preds <dir> -nw 64 <feature_file> <dataset_path>
    python eval_calib_net.py --preds <dir> -nw 64 -e <feature_file>_eq <dataset_path>
"""
import argparse
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from multiprocessing import Pool
from time import perf_counter

import h5py
import numpy as np
import poselib
from tqdm import tqdm

from utils.data import get_pairs
from utils.geometry import rotation_angle, angle, normalize, get_camera_dicts

METHODS = ['geocalib', 'unik3d', 'anycalib_dist', 'ours_ray_vitl', 'ours_log_vitl', 'ours_ray_cnxt', 'ours_log_cnxt', 'ours_ray_vitb']
ITERATIONS = [10, 20, 50, 100, 200, 500, 1000]
GRAVITY_METHODS = []  # E_3pt curves: dropped (user), the code stays

COLUMNS = ['pair', 'iterations', 'R_err', 't_err', 'k1', 'k2', 'k1_gt', 'k2_gt', 'f1', 'f2', 'f1_gt', 'f2_gt',
           'ransac_ms', 'base_ms', 'inlier_ratio', 'num_inliers']

DATA = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', required=True, help='the directory of rd_predict.py\'s <label>.h5')
    parser.add_argument('--methods', nargs='+', default=METHODS)
    parser.add_argument('--out', default='results/calib_net')
    parser.add_argument('-f', '--first', type=int, default=None, help='first N pairs (smoke test)')
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-t', '--threshold', type=float, default=3.0)
    parser.add_argument('-e', '--eq', action='store_true', default=False)
    parser.add_argument('--iterations', type=int, nargs='+', default=ITERATIONS)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')
    return parser.parse_args()


def get_experiments(eq, methods):
    if eq:
        experiments = ['kFk_9pt', 'k2Fk1_10pt', 'Efeq_6pt', 'Efeq_6pt_s3']
    else:
        experiments = ['k2Fk1_10pt', 'F_7pt', 'F_7pt_s3']
    lo = 'VLOeq' if eq else 'VLO'
    for method in methods:
        experiments.extend([f'E_5pt+{method}_V', f'E_5pt+{method}_{lo}', f'E_5pt+{method}_Vfix'])
        if eq:
            experiments.extend([f'E_5pt+{method}_Veq', f'Efeq_6pt+{method}_V', f'Efeq_6pt+{method}_VLO'])
        if method in GRAVITY_METHODS:
            experiments.extend([f'E_3pt+{method}_V', f'E_3pt+{method}_{lo}'])
    return experiments


def parse_experiment(experiment):
    """'E_5pt+ours_ray_vitl_VLOeq' -> ('E_5pt', 'ours_ray_vitl', 'VLOeq'); no net: (name, None, None)."""
    if '+' not in experiment:
        return experiment, None, None
    solver, net = experiment.split('+')
    method, use = net.rsplit('_', 1)
    return solver, method, use


def load_predictions(path, eq):
    """{key: dict(f, lam, g, base_ms)}: per image, or per pair on the equal graph. f = f / max(w, h),
    as `utils.geometry.normalize` scales the points; lam needs no conversion."""
    with h5py.File(path) as f:
        attrs = dict(f.attrs)
        get = lambda k: f[k][()]
        side = np.maximum(get('w'), get('h')).astype(np.float64)
        names = [x.decode() if isinstance(x, bytes) else x for x in get('names')]
        sides = dict(zip(names, side))
        if eq:
            pairs = [tuple(x.decode() if isinstance(x, bytes) else x for x in p) for p in get('pair_names')]
            pf, pl = get('pair_f'), get('pair_lam')
            g1 = get('pair_g1') if 'pair_g1' in f else np.zeros((len(pairs), 3))
            g2 = get('pair_g2') if 'pair_g2' in f else np.zeros((len(pairs), 3))
            preds = {}
            for i, (a, b) in enumerate(pairs):
                preds[(a, b)] = dict(f1=pf[i] / sides[a], f2=pf[i] / sides[b], k1=pl[i], k2=pl[i], g1=g1[i], g2=g2[i])
            base_ms = 2 * attrs['net_ms'] + attrs['joint_fit_ms']
        else:
            fs, ls = get('f'), get('lam')
            g = get('g') if 'g' in f else np.zeros((len(names), 3))
            preds = {n: dict(f=fs[i] / sides[n], k=ls[i], g=g[i]) for i, n in enumerate(names)}
            base_ms = 2 * (attrs['net_ms'] + attrs['fit_ms'])
    return preds, float(base_ms)


def pair_prediction(preds, a, b, eq):
    if eq:
        return preds[(a, b)]
    p1, p2 = preds[a], preds[b]
    return dict(f1=p1['f'], f2=p2['f'], k1=p1['k'], k2=p2['k'], g1=p1['g'], g2=p2['g'])


def run(task):
    experiment, i, iters = task
    d = DATA['pairs'][i]
    solver, method, use = parse_experiment(experiment)
    kp1, kp2, T1, T2 = d['kp1'], d['kp2'], d['T1'], d['T2']
    mean_scale = (T1[0, 0] + T2[0, 0]) / 2
    t = DATA['threshold']

    ransac_dict = {'max_iterations': iters, 'max_epipolar_error': t / mean_scale, 'progressive_sampling': False,
                   'min_iterations': iters}
    shared = solver in ('kFk_9pt', 'Efeq_6pt', 'Efeq_6pt_s3') or use in ('VLOeq', 'Veq')
    free = use != 'Vfix'  # Vfix: the final refinement on the pose alone
    bundle_dict = {'refine_pose': True, 'refine_focal_length': free, 'refine_principal_point': False,
                   'refine_extra_params': free, 'max_iterations': 100}
    opt_dict = {'max_error': 3.0 / mean_scale, 'ransac': ransac_dict, 'bundle': bundle_dict,
                'shared_intrinsics': shared, 'use_minimal': False, 'tangent_sampson': True}

    base_ms = 0.0
    if method is not None:
        p = pair_prediction(DATA['preds'][method], d['name1'], d['name2'], DATA['eq'])
        base_ms = DATA['base_ms'][method]

    if solver in ('E_5pt', 'E_3pt'):
        g1, g2 = (p['g1'], p['g2']) if solver == 'E_3pt' else (np.zeros(3), np.zeros(3))
        cam1 = [p['f1'], 0.0, 0.0, p['k1']]
        cam2 = [p['f2'], 0.0, 0.0, p['k2']]
        dict1 = {'model': 'SIMPLE_DIVISION', 'width': -1, 'height': -1, 'params': cam1}
        dict2 = {'model': 'SIMPLE_DIVISION', 'width': -1, 'height': -1, 'params': cam2}
        start = perf_counter()
        if use in ('V', 'Veq', 'Vfix'):
            pose, info = poselib.estimate_relative_pose(kp1, kp2, dict1, dict2, opt_dict, g1, g2)
            ransac_ms = 1000 * (perf_counter() - start)
            image_pair = poselib.ImagePair(pose, poselib.Camera('SIMPLE_DIVISION', cam1, -1, -1),
                                           poselib.Camera('SIMPLE_DIVISION', cam2, -1, -1))
        else:
            image_pair, info = poselib.estimate_relative_pose_lo(kp1, kp2, dict1, dict2, opt_dict, g1, g2)
            ransac_ms = 1000 * (perf_counter() - start)
    else:
        if method is not None:
            rd_1, rd_2 = [p['k1']], [p['k2']]
            bundle_dict['refine_extra_params'] = use != 'V'
        elif solver in ('kFk_9pt', 'k2Fk1_10pt'):  # empty: the non-minimal solvers
            rd_1, rd_2 = [], []
        elif solver.endswith('_s3'):
            rd_1 = rd_2 = [0.0, -0.6, -1.2]
        else:
            rd_1 = rd_2 = [0.0]
        start = perf_counter()
        image_pair, info = poselib.estimate_focal_rd_relpose(kp1, kp2, rd_1, rd_2, opt_dict)
        ransac_ms = 1000 * (perf_counter() - start)

    R, t_est = image_pair.pose.R, image_pair.pose.t
    return (i, iters, rotation_angle(R.T @ d['R_gt']), angle(t_est, d['t_gt']),
            image_pair.camera1.params[-1], image_pair.camera2.params[-1], d['k1_gt'], d['k2_gt'],
            image_pair.camera1.params[0], image_pair.camera2.params[0], d['f1_gt'], d['f2_gt'],
            ransac_ms, base_ms, info['inlier_ratio'], info['num_inliers'])


def load_pairs(args):
    path = args.dataset_path
    R_file = h5py.File(os.path.join(path, 'R.h5'))
    T_file = h5py.File(os.path.join(path, 'T.h5'))
    P_file = h5py.File(os.path.join(path, 'parameters_rd.h5'))
    C_file = h5py.File(os.path.join(path, f'{args.feature_file}.h5'))
    R_dict = {k: np.array(v) for k, v in R_file.items()}
    t_dict = {k: np.array(v) for k, v in T_file.items()}
    P_dict = {k.split('-')[0]: np.array(v) for k, v in P_file.items()}
    cameras = get_camera_dicts(os.path.join(path, 'K.h5'))

    names = get_pairs(C_file)
    if args.first is not None:
        names = names[:args.first]
    out = []
    for a, b in names:
        matches = np.array(C_file[f'{a}-{b}'])
        if len(matches) < 10:
            continue
        R_gt = R_dict[b] @ R_dict[a].T
        t_gt = t_dict[b] - R_gt @ t_dict[a]
        kp1, T1 = normalize(matches[:, :2], P_dict[a][0, 0], P_dict[a][1, 1])
        kp2, T2 = normalize(matches[:, 2:4], P_dict[b][0, 0], P_dict[b][1, 1])
        out.append(dict(name1=a, name2=b, kp1=kp1, kp2=kp2, T1=T1, T2=T2, R_gt=R_gt, t_gt=t_gt,
                        k1_gt=P_dict[a][2, 2], k2_gt=P_dict[b][2, 2],
                        f1_gt=cameras[a][0, 0] / T1[0, 0], f2_gt=cameras[b][0, 0] / T2[0, 0]))
    return out


def main():
    args = parse_args()
    DATA['pairs'] = load_pairs(args)
    DATA['threshold'] = args.threshold
    DATA['eq'] = args.eq
    DATA['preds'], DATA['base_ms'] = {}, {}
    for method in args.methods:
        DATA['preds'][method], DATA['base_ms'][method] = load_predictions(os.path.join(args.preds, f'{method}.h5'), args.eq)
    print('network ms per pair:', DATA['base_ms'])

    dataset = os.path.basename(args.dataset_path.rstrip('/'))
    out_dir = os.path.join(args.out, f'{dataset}_eq' if args.eq else dataset)
    os.makedirs(out_dir, exist_ok=True)

    experiments = get_experiments(args.eq, args.methods)
    print(f'{len(experiments)} experiments, {len(DATA["pairs"])} pairs, {len(args.iterations)} budgets')
    # forked after DATA is filled: the workers read the pairs, only indices are sent
    with Pool(args.num_workers) as pool:
        for experiment in experiments:
            path = os.path.join(out_dir, f'{experiment}.npz')
            if os.path.exists(path):
                print('done already:', experiment)
                continue
            tasks = [(experiment, i, it) for i in range(len(DATA['pairs'])) for it in args.iterations]
            rows = list(tqdm(pool.imap(run, tasks, chunksize=8), total=len(tasks), desc=experiment, mininterval=60))
            rows = np.array(rows, dtype=np.float64)
            tmp = path + '.tmp.npz'
            np.savez_compressed(tmp, **{c: rows[:, j] for j, c in enumerate(COLUMNS)})
            os.replace(tmp, path)


if __name__ == '__main__':
    main()
