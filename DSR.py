import cv2
import numpy as np
from feature_match import get_matching_inliers
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def test_func(img_master, img_slave, master_pts, slave_pts):
    fig = plt.figure(figsize=(100, 100))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    x, y = np.random.rand(100), np.random.rand(100)

    ax1.imshow(img_master)
    ax2.imshow(img_slave)
    #
    i = 10
    print("yo")
    print(master_pts[10])
    print(slave_pts[10])
    # xy = (x[i], y[i])
    con = ConnectionPatch(xyA=slave_pts[10], xyB=master_pts[10], coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="red")
    ax2.add_artist(con)

    ax1.plot(x[i], y[i], 'ro', markersize=10)
    ax2.plot(x[i], y[i], 'ro', markersize=10)

    plt.show()


def get_perc_inliers(Hy_temp, master_pts, slave_pts, threshold):
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
    # tranformed_slave = tranformed_slave/(tranformed_slave[:, 2].reshape(total_pts, 1))
    # print(tranformed_slave)
    vertical_error = np.abs(master_pts[:, 1] - tranformed_slave[:, 1])
    # print(sum(vertical_error < threshold))
    inliers_perc = sum(vertical_error < threshold)/(total_pts)

    return inliers_perc


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


img_master = cv2.imread('image0_s.png', cv2.IMREAD_GRAYSCALE)
img_slave = cv2.imread('image1_s.png', cv2.IMREAD_GRAYSCALE)
img_master = cv2.resize(img_master, (720, 960))
img_slave = cv2.resize(img_slave, (720, 960))

master_pts, slave_pts = get_matching_inliers(img_master, img_slave, show_matches=False)

# print(master_pts[10])
# print(slave_pts[10])

# test_func(img_master, img_slave, master_pts, slave_pts)
num_trials = 200
sample_size = 20
threshold = 1

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
    perc_inliers = get_perc_inliers(Hy_temp, master_pts, slave_pts, threshold)

    if perc_inliers > max_perc_inliers:
        max_perc_inliers = perc_inliers
        best_Hy = Hy_temp

print("Score :", max_perc_inliers)
print(best_Hy)



