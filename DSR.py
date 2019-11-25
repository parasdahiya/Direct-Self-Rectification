import cv2
import numpy as np
from feature_match import get_matching_inliers
from get_hy import get_best_hy
from get_hs import get_hs
from get_hk import get_hk
from metrics import get_vert_align_acc
from metrics import get_NVD
import matplotlib.pyplot as plt


width = 720
height = 960
img_master = cv2.imread('image0_s.png', cv2.IMREAD_GRAYSCALE)
img_slave = cv2.imread('image1_s.png', cv2.IMREAD_GRAYSCALE)
img_master = cv2.resize(img_master, (720, 960))
img_slave = cv2.resize(img_slave, (720, 960))

master_pts, slave_pts = get_matching_inliers(img_master, img_slave, show_matches=True)

num_trials = 200
sample_size = 20
threshold = 1

## Vertical alignment matrix
Hy, PAP_acc = get_best_hy(master_pts, slave_pts, num_trials, sample_size, threshold)
print("Score :", PAP_acc)

## Shearing matrix alignment
Hs = get_hs(Hy, width, height)
H_temp = np.matmul(Hs, Hy)

## Horizontal alignement
Hk = get_hk(H_temp, master_pts, slave_pts)
H = np.matmul(Hk, H_temp)
# print(H)

pap1 = get_vert_align_acc(H, master_pts, slave_pts, threshold=1)
pap2 = get_vert_align_acc(H, master_pts, slave_pts, threshold=2)
pap3 = get_vert_align_acc(H, master_pts, slave_pts, threshold=3)
nvd_error = get_NVD(H, width, height)

print("Vertical Alignment Accuracy (PAP) with pixel threshold of 1: {0}, 2:{1}, 3: {2}".format(pap1,pap2,pap3))
print("Geometric distortion error (NVD): ", nvd_error)

warped_img = cv2.warpPerspective(img_slave, H, (800, 1000))

cv2.imshow('img', warped_img)
# plt.subplot(121),plt.imshow(img_master, cmap='gray'),plt.title('Input')
# plt.subplot(122),plt.imshow(warped_img, cmap='gray'),plt.title('Output')
# plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
