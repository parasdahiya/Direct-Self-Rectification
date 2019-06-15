import cv2
import numpy as np
from feature_match import get_matching_inliers
from get_hy import get_best_hy
from get_hs import get_hs
from get_hk import get_hk


width = 720
height = 960
img_master = cv2.imread('image0_s.png', cv2.IMREAD_GRAYSCALE)
img_slave = cv2.imread('image1_s.png', cv2.IMREAD_GRAYSCALE)
img_master = cv2.resize(img_master, (720, 960))
img_slave = cv2.resize(img_slave, (720, 960))

master_pts, slave_pts = get_matching_inliers(img_master, img_slave, show_matches=False)

num_trials = 200
sample_size = 20
threshold = 1

Hy, PAP_acc = get_best_hy(master_pts, slave_pts, num_trials, sample_size, threshold)
# print(Hy)
print("Score :", PAP_acc)

Hs = get_hs(Hy, width, height)
H_temp = np.matmul(Hs, Hy)
Hk = get_hk(H_temp, master_pts, slave_pts)
H = np.matmul(Hk, H_temp)
print(H)
# print(Hs)



