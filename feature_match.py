import cv2
import numpy as np


def get_matching_inliers(img_master, img_slave, show_matches = True):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_master, None)
    kp2, des2 = sift.detectAndCompute(img_slave, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    print("Initially detected interest points: ", len(matches))

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    F, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    inliers = []
    for i, m in enumerate(good):
        if matchesMask[i] != 0:
            inliers.append([m])
    print("Interest points after RANSAC - Inliers: ", len(inliers))

    if show_matches:
        img3 = cv2.drawMatchesKnn(img_master, kp1, img_slave, kp2, inliers, None, flags=2)
        cv2.imshow('Image', img3)

    master_pts = np.float32([kp1[m.queryIdx].pt for [m] in inliers])
    slave_pts = np.float32([kp2[m.queryIdx].pt for [m] in inliers])

    return master_pts, slave_pts