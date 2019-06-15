import numpy as np


def get_hk(H_temp, master_pts, slave_pts):
    total_pts = len(master_pts)
    slave_pts_homo = np.concatenate((slave_pts, np.ones(total_pts).reshape(total_pts, 1)), axis=1)
    slave_pts_t = np.transpose(slave_pts_homo)
    # print(slave_pts_t)
    transformed_pts_t = np.matmul(H_temp, slave_pts_t)
    transformed_pts = np.transpose(transformed_pts_t)
    transformed_pts = transformed_pts/transformed_pts[:, 2].reshape(total_pts, 1)  # Making transformation homogeneous
    # print(transformed_pts.shape)

    horizontal_shifts = master_pts[:, 0] - transformed_pts[:, 0]
    k = max(horizontal_shifts)

    # print(horizontal_shifts)

    Hk = [[1, 0, k],
          [0, 1, 0],
          [0, 0, 1]]
    return np.array(Hk, dtype=float)