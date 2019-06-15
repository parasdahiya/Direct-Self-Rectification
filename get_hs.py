import numpy as np


def get_hs(Hy, w, h):
    mid_pt_matrix = np.array([[w/2, 0, 1],
                              [w, h/2, 1],
                              [w/2, h, 1],
                              [0, h/2, 1]
                              ], dtype=float)
    mid_pt_matrix = np.transpose(mid_pt_matrix)
    # print(mid_pt_matrix)
    transformed_pts_t = np.matmul(Hy, mid_pt_matrix)
    transformed_pts = np.transpose(transformed_pts_t)
    transformed_pts = transformed_pts/transformed_pts[:, 2].reshape(4, 1)
    u = transformed_pts[1, :] - transformed_pts[3, :]
    v = transformed_pts[0, :] - transformed_pts[2, :]
    ux, uy, vx, vy = u[0], u[1], v[0], v[1]

    sa = (pow(h, 2)*pow(uy, 2) + pow(w, 2)*pow(vy, 2)) / (h*w*(uy*vx - ux*vy))
    sb = (pow(h, 2)*ux*uy + pow(w, 2)*vx*vy) / (h*w*(ux*vy - uy*vx))

    Hs = [[sa, sb, 0],
          [0, 1, 0],
          [0, 0, 1]]
    return np.array(Hs)