import cv2
import numpy as np
from feature_match import get_matching_inliers


img_master = cv2.imread('image0_s.png', cv2.IMREAD_GRAYSCALE)
img_slave = cv2.imread('image1_s.png', cv2.IMREAD_GRAYSCALE)
img_master = cv2.resize(img_master, (720, 960))
img_slave = cv2.resize(img_slave, (720, 960))

master_pts, slave_pts = get_matching_inliers(img_master, img_slave, show_matches=True)

cv2.waitKey(0)
cv2.destroyAllWindows()