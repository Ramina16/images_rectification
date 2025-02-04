import numpy as np
import cv2
import os

from typing import List, Tuple


FLANN_INDEX_KDTREE = 1


def add_lines(img: np.ndarray, line_thickness: int = 1, line_spacing: int = 75, line_color: tuple = (255, 255, 255)):
    """
    Draw horizontal lines on the image

    :param img: image
    :param line_thickness: thickness of lines
    :param line_spacing: num of px between lines
    :param line_color: lines' color
    """
    h, w = img.shape[:2]
    num_lines = h // line_spacing - 1

    for i in range(1, num_lines + 1):
        y = i * line_spacing
        cv2.line(img, (0, y), (w, y), line_color, thickness=line_thickness)


def kp_detection(img1: np.ndarray, img2: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray, List[cv2.KeyPoint], np.ndarray]:
    """
    Calculate keypoints on provided images using SIFT

    :param img1: first image
    :param img2: second image
    :return: keypoints and descriptor for first image, keypoints and descriptor for second image
    """
    sift = cv2.SIFT_create()

    kp1, ds1 = sift.detectAndCompute(img1, None)
    kp2, ds2 = sift.detectAndCompute(img2, None)

    return kp1, ds1, kp2, ds2


def flann_matcher(kp1: List[cv2.KeyPoint], ds1: np.ndarray, kp2: List[cv2.KeyPoint], ds2: np.ndarray):
    """
    Find matches between keypoints on two images using FLANN matcher

    :param kp1: list of keypoints for first image
    :param ds1: descriptor for first image
    :param kp2: list of keypoints for second image
    :param ds2: descriptor for second image
    :return: matches, matches_mask, "good" matches, matched points on first image, matched points on second image
    """
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(ds1, ds2, k=2)

    # Keep good matches: calculate distinctive image features
    matches_mask = [[0, 0] for _ in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            matches_mask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    return matches, matches_mask, good, pts1, pts2


def draw_matches(matches_mask: list, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], img1: np.ndarray, img2: np.ndarray, matches) -> np.ndarray:
    """
    Draw matches on the images

    :param matches_mask: list of values indicating which matches to draw.
    :param kp1: list of keypoints detected in the first image.
    :param kp2: list of keypoints detected in the second image.
    :param img1: first image.
    :param img2: second image.
    :param matches: list of matches found between the keypoints of the two images

    :return: image with the matches drawn between img1 and img2
    """
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matches_mask[300:500],
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    
    keypoint_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[300:500], None, **draw_params)
    
    return keypoint_matches


def calc_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate fundamental matrix from the matched keypoints

    :param pts1: array of 2D points from the first image
    :param pts2: array of 2D points from the second image
    :return: filtered inlier points from the first image, filtered inlier points from the second image, computed fundamental matrix
    """
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    return pts1, pts2, fundamental_matrix


def stereo_rectification_unclb(fundamental_matrix: np.ndarray, pts1_in: np.ndarray, pts2_in: np.ndarray, 
                               img1: np.ndarray, img2: np.ndarray, out_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectify and save images using algorithm w/o known intrinsics

    :param fundamental_matrix: fundamental matrix computed from matched keypoints
    :param pts1_in: array of 2D points from the first image
    :param pts2_in: array of 2D points from the second image
    :param img1: first image
    :param img2: second image
    :return: rectified first and second images
    """
    h, w = img1.shape[:2]

    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1_in), np.float32(pts2_in), fundamental_matrix, imgSize=(w, h))

    img1_rect = cv2.warpPerspective(img1, H1, (w, h))
    img2_rect = cv2.warpPerspective(img2, H2, (w, h))

    cv2.imwrite(os.path.join(out_path, 'left_img_rect_unclb.png'), img1_rect)
    cv2.imwrite(os.path.join(out_path, 'right_img_rect_unclb.png'), img2_rect)

    return img1_rect, img2_rect


def stereo_rectification(R: np.ndarray, T: np.ndarray, intrinsics: np.ndarray, 
                         img1: np.ndarray, img2: np.ndarray, out_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectify and save images using algorithm with known intrinsics

    :param R: rotation matrix between the two camera views
    :param T: translation vector between the two camera views
    :param intrinsics: camera intrinsic matrix
    :param img1: first image
    :param img2: second image
    :return: rectified first image, rectified second image
    """
    h, w = img1.shape[:2]
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(intrinsics, None, intrinsics, None, (w, h), R, T)
    map1_x, map1_y = cv2.initUndistortRectifyMap(intrinsics, None, R1, P1, (w, h), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(intrinsics, None, R2, P2, (w, h), cv2.CV_32FC1)

    rectified_img1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_img2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

    cv2.imwrite(os.path.join(out_path, 'left_img_rect.png'), rectified_img1)
    cv2.imwrite(os.path.join(out_path, 'right_img_rect.png'), rectified_img2)

    return rectified_img1, rectified_img2


def show_image(name: str, img: np.ndarray):
    """
    Show image

    :param name: name of the window where image will be shown
    :param img: image to show
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
