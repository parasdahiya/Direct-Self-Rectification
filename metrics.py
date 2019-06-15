import numpy as np


def get_NVD(H, w, h):
    end_pt_matrix = np.array([[1, 1, 1],
                              [w, 1, 1],
                              [1, h, 1],
                              [w, h, 1]], dtype=float)
    end_pt_matrix_t = np.transpose(end_pt_matrix)
    transformed_pts_t = np.matmul(H, end_pt_matrix_t)
    transformed_pts = np.transpose(transformed_pts_t)
    transformed_pts = transformed_pts / transformed_pts[:, 2].reshape(4, 1)
    diff_v = np.square(end_pt_matrix - transformed_pts)

    NVD_err = np.sqrt(diff_v[:, 0] + diff_v[:, 1])
    NVD_err = sum(NVD_err)
    norm = pow(pow(w,2) + pow(h,2), 0.5)
    NVD = NVD_err/norm

    return NVD


def get_vert_align_acc(Hy_temp, master_pts, slave_pts, threshold):
    total_pts = len(master_pts)
    # print(total_pts)
    y = np.ones(total_pts).reshape(total_pts, 1)
    # print(y.shape)
    # print(master_pts.shape)
    master_pts = np.concatenate((master_pts, y), axis=1)
    slave_pts = np.concatenate((slave_pts, y), axis=1)
    slave_pts_T = np.transpose(slave_pts)
    # print(slave_pts_T[:, 0])
    tranformed_slave_T = np.matmul(Hy_temp, slave_pts_T)
    tranformed_slave = np.transpose(tranformed_slave_T)
    tranformed_slave = tranformed_slave/(tranformed_slave[:, 2].reshape(total_pts, 1))
    # print(tranformed_slave)
    vertical_error = np.abs(master_pts[:, 1] - tranformed_slave[:, 1])
    # print(sum(vertical_error < threshold))
    inliers_perc = sum(vertical_error < threshold)/(total_pts)

    return inliers_perc
