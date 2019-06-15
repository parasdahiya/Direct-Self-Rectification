import numpy as np
from metrics import get_vert_align_acc


def get_temp_hy(master_pts, slave_pts):
    total_pts = len(master_pts)
    A = np.zeros((total_pts, 5))
    for i in range(total_pts):
        A[i, :] = [slave_pts[i, 0], slave_pts[i, 1], 1, -1*slave_pts[i, 0]*master_pts[i, 1], -1*slave_pts[i, 1]*master_pts[i, 1]]
    A = np.array(A)
    # print(A)
    y = master_pts[:, 1]
    A_pinv = np.linalg.pinv(A)
    H_params = np.matmul(A_pinv, y)
    # print(A_pinv.shape, y.shape, H_params.shape)
    Hy_temp = [[1., 0., 0.],
               [H_params[0], H_params[1], H_params[2]],
               [H_params[3], H_params[4], 1]]

    # print(Hy_temp)

    return np.array(Hy_temp)


def get_best_hy(master_pts, slave_pts, num_trials, sample_size, threshold = 1):
    max_perc_inliers = 0
    best_Hy = np.identity(3)
    total_pts = len(master_pts)
    for i in range(num_trials):
        rn = np.random.randint(0, total_pts, sample_size)
        ran_samples_master = master_pts[rn, :]
        ran_samples_slave = slave_pts[rn, :]
        # print(master_pts[0:3])
        # print(slave_pts[0:3])
        Hy_temp = get_temp_hy(ran_samples_master, ran_samples_slave)
        # print(Hy_temp.shape)
        perc_inliers = get_vert_align_acc(Hy_temp, master_pts, slave_pts, threshold)

        if perc_inliers > max_perc_inliers:
            max_perc_inliers = perc_inliers
            best_Hy = Hy_temp

    return best_Hy, max_perc_inliers
