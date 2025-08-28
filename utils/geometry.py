import h5py
import numpy as np
import cv2

def angle(x, y):
    x = x.ravel()
    y = y.ravel()
    return np.rad2deg(np.arccos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))

def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def get_camera_dicts(K_file_path):
    K_file = h5py.File(K_file_path)

    d = {}

    # Treat data from Charalambos differently since it is in pairs
    if 'K1_K2' in K_file_path:

        for k, v in K_file.items():
            key1, key2 = k.split('-')
            if key1 not in d.keys():
                K1 = np.array(v)[0, 0]
                # d[key1] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K1[0, 2]), 'height': int(2 * K1[1,2]), 'params': [K1[0, 0], K1[0, 2], K1[1, 2]]}
                d[key1] = K1
            if key2 not in d.keys():
                K2 = np.array(v)[0, 1]
                d[key2] = K2
                # d[key2] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K2[0, 2]), 'height': int(2 * K2[1,2]), 'params': [K2[0, 0], K2[0, 2], K2[1, 2]]}

        return d

    for key, v in K_file.items():
        K = np.array(v)
        d[key] = K
        # d[key] = {'model': 'PINHOLE', 'width': int(2 * K[0, 2]), 'height': int(2 * K[1,2]), 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}

    return d


def undistort(x, k):
    if k == 0.0:
        return x
    return x / (1 + k * (x[:, :1] ** 2 + x[:, 1:] ** 2))


def distort(u, k):
    if k == 0.0:
        return u

    x_u = u[:, :1]
    y_u = u[:, 1:]

    ru2 = x_u ** 2 + y_u ** 2
    ru = np.sqrt(ru2)

    dist_sign = np.sign(k)

    rd = (0.5 / k) / ru - dist_sign * np.sqrt((0.25 / (k * k)) /ru2 - 1 / k)
    rd /= ru

    return rd * u


def recover_pose_from_fundamental(F, K1, K2, pts1, pts2):
    # Compute the essential matrix E from the fundamental matrix F
    E = K2.T @ F @ K1

    # Decompose the essential matrix into rotation and translation
    # This function returns the possible rotation matrices and translation vectors
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # Ensure t is a column vector
    t = t.reshape(3, 1)

    # Four possible solutions: (R1, t), (R1, -t), (R2, t), (R2, -t)
    possible_solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    # To determine the correct solution, we use the cheirality condition
    # We need to triangulate points and check which solution has points in front of both cameras
    correct_solution = None
    max_positive_depths = -1

    for R, t in possible_solutions:
        # Create projection matrices for both cameras
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, t))

        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert homogeneous coordinates to 3D
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        # Check the number of points in front of both cameras
        num_positive_depths = np.sum((points_3d[2, :] > 0))

        if num_positive_depths > max_positive_depths:
            max_positive_depths = num_positive_depths
            correct_solution = (R, t)

    R, t = correct_solution
    return R, t


def bougnoux_original(F, p1=np.array([0, 0]), p2=np.array([0, 0])):
    ''' Returns squared focal losses estimated using the Bougnoux formula with given principal points

    :param F: 3 x 3 Fundamental matrix
    :param p1: 2-dimensional coordinates of the principal point of the first camera
    :param p2: 2-dimensional coordinates of the principal point of the first camera
    :return: the estimated squared focal lengths for the two cameras
    '''
    p1 = np.append(p1, 1).reshape(3, 1)
    p2 = np.append(p2, 1).reshape(3, 1)
    try:
        e2, _, e1 = np.linalg.svd(F)
    except Exception:
        return np.nan, np.nan

    e1 = e1[2, :]
    e2 = e2[:, 2]


    s_e2 = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    s_e1 = np.array([
        [0, -e1[2], e1[1]],
        [e1[2], 0, -e1[0]],
        [-e1[1], e1[0], 0]
    ])

    II = np.diag([1, 1, 0])

    f1 = (-p2.T @ s_e2 @ II @ F @ (p1 @ p1.T) @ F.T @ p2) / (p2.T @ s_e2 @ II @ F @ II @ F.T @ p2)
    f2 = (-p1.T @ s_e1 @ II @ F.T @ (p2 @ p2.T) @ F @ p1) / (p1.T @ s_e1 @ II @ F.T @ II @ F @ p1)

    return np.sqrt(f1[0, 0]), np.sqrt(f2[0, 0])


def get_K(f, p=np.array([0, 0])):
    return np.array([[f, 0, p[0]], [0, f, p[1]], [0, 0, 1]])


def pose_from_F(F, K1, K2, kp1, kp2):
    try:
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)

        E = K2.T @(F @ K1)

        # print(np.linalg.svd(E)[1])

        kp1 = np.column_stack([kp1, np.ones(len(kp1))])
        kp2 = np.column_stack([kp2, np.ones(len(kp1))])

        kp1_unproj = (K1_inv @ kp1.T).T
        kp1_unproj = kp1_unproj[:, :2] / kp1_unproj[:, 2, np.newaxis]
        kp2_unproj = (K2_inv @ kp2.T).T
        kp2_unproj = kp2_unproj[:, :2] / kp2_unproj[:, 2, np.newaxis]

        _, R, t, mask = cv2.recoverPose(E, kp1_unproj, kp2_unproj)
    except:
        print("Pose exception!")
        return np.eye(3), np.ones(3)

    return R, t


def normalize(kp, width, height):
    new_kp = np.copy(kp)

    scale = max(width, height)
    new_kp -= np.array([[width / 2, height / 2]])
    new_kp /= scale

    T = np.array([[scale, 0.0, width / 2], [0.0, scale, height / 2], [0, 0, 1]])

    return new_kp, T


def k_err(k_gt, k_est):
    # return abs((1 / (1 + k_gt)) - (1 /(1 + k_est))) / abs(( 1 / (1 + k_gt)))
    return np.abs(k_gt - k_est)

def f_err(f_gt, f_est):
    # return abs((1 / (1 + k_gt)) - (1 /(1 + k_est))) / abs(( 1 / (1 + k_gt)))
    return np.abs(f_gt - f_est) / f_gt

def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def add_rand_pts(x, multiplier):
    x_new = np.random.rand(int(multiplier * len(x)), 2)
    x_new[:, 0] -= 0.5
    x_new[:, 1] -= 0.5
    return np.row_stack([x, x_new])


def unproject_with_jac(x, k):
    r2 = x[0] ** 2 + x[1] ** 2
    X = np.array([x[0], x[1], 1 + k * r2])
    inv_norm = 1.0 / np.linalg.norm(X)
    X = inv_norm * X

    jac = np.empty([3, 2])

    jac[0, 0] = 1 - X[0] * X[0] - 2 * X[0] * X[2] * k * x[0]
    jac[0, 1] = -X[0] * (X[1] + 2 * X[2] * k * x[1])
    jac[1, 0] = -X[1] * (X[0] + 2 * X[2] * k * x[0])
    jac[1, 1] = 1 - X[1] * X[1] - 2 * X[1] * X[2] * k * x[1]
    jac[2, 0] = -X[0] * X[2] + 2 * k * (-x[0]) * (X[2] * X[2] - 1)
    jac[2, 1] = -X[1] * X[2] + 2 * k * (-x[0]) * (X[2] * X[2] - 1)

    return X, jac



def get_inliers_tsamp(kp1, kp2, F, k1, k2, t):
    sq_t = t**2
    out = np.zeros(len(kp1))

    for i in range(len(kp1)):
        d1, M1 = unproject_with_jac(kp1[i], k1)
        d2, M2 = unproject_with_jac(kp2[i], k2)


        C = d2.dot(F @ d1)
        denom2 = np.linalg.norm(M2.T @ F @ d1)**2 + np.linalg.norm(M1.T @ F.T @ d2)**2
        r2 = C * C / denom2

        if (r2 < sq_t):
            out[i] = 1.0
        else:
            out[i] = 0.0

    return out == 1.0


def add_rand_gs(g, multiplier):
    g_new = np.random.randn(int(multiplier * len(g)), 3)
    g_new /= np.linalg.norm(g_new, axis=1)
    return np.row_stack([g, g_new])


def force_inliers(kp1, kp2, R_gt, t_gt, k1, k2, K1, K2, T1, T2, ratio, t):
    mean_scale = (T1[0, 0] + T2[0, 0]) / 2

    f1_gt = (K1[0, 0] + K1[1, 1]) / (2 * mean_scale)
    f2_gt = (K2[0, 0] + K2[1, 1]) / (2 * mean_scale)

    K1 = np.diag([f1_gt, f1_gt, 1])
    K2 = np.diag([f2_gt, f2_gt, 1])

    F = np.linalg.inv(K2).T @ skew(t_gt.ravel()) @ R_gt @ np.linalg.inv(K1)

    l = get_inliers_tsamp(kp1, kp2, F, k1, k2, t / mean_scale)

    multiplier = (1 - ratio) / ratio

    kp1, kp2 = kp1[l], kp2[l]

    kp1 = add_rand_pts(kp1, multiplier)
    kp2 = add_rand_pts(kp2, multiplier)

    return kp1, kp2