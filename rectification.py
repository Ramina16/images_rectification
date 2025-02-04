import numpy as np
import cv2
import argparse

from utils import add_lines, calc_fundamental_matrix, draw_matches, flann_matcher, kp_detection, show_image, stereo_rectification, stereo_rectification_unclb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img1', '-path_img1', type=str, help='path to left image for rectification')
    parser.add_argument('--path_img2', '-path_img2', type=str, help='path to right image for rectification')
    args = parser.parse_args()


    left_img = cv2.imread(args.path_img1)
    right_img = cv2.imread(args.path_img2)

    h, w = left_img.shape[:2]

    # resize images to reduce computations
    left_img = cv2.resize(left_img, (w // 8, h // 8), left_img)
    right_img = cv2.resize(right_img, (w // 8, h // 8), right_img)

    h, w = left_img.shape[:2]

    combined_img = cv2.hconcat([left_img, right_img])
    add_lines(combined_img)

    show_image('Combined Image with lines', combined_img)

    # Calculate keypoints using SIFT
    kp1, ds1, kp2, ds2 = kp_detection(left_img, right_img)
    imgSift = cv2.drawKeypoints(left_img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    show_image("SIFT Keypoints", imgSift)

    # Calculate matches for detected keypoints
    matches, matches_mask, good, pts1, pts2 = flann_matcher(kp1, ds1, kp2, ds2)
    keypoint_matches = draw_matches(matches_mask, kp1, kp2, left_img, right_img, matches)

    show_image("Keypoint matches", keypoint_matches)

    # Calculate the fundamental matrix for the cameras
    pts1_in, pts2_in, fundamental_matrix = calc_fundamental_matrix(pts1, pts2)

    # -------------------------------------------------------------------------------------------------------------------
    # Stereo rectification for case with unknown intrinsics parameters
    # Perform stereo rectification
    left_img_rect, right_img_rect = stereo_rectification_unclb(fundamental_matrix, pts1_in, pts2_in, left_img, right_img, out_path='images')

    combined_img_rect = cv2.hconcat([left_img_rect, right_img_rect])
    add_lines(combined_img_rect)

    show_image('Combined Rectified Image with lines', combined_img_rect)
    # -------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------
    # Stereo rectification for case with known intrinsics parameters
    intrinsics = np.array([[5.5 * h / 8.152, 0, 510],
                           [0, 5.5 * w / 6.115, 382.5],
                           [0, 0, 1]])

    essential_matrix = intrinsics.T @ fundamental_matrix @ intrinsics

    pts1_in = np.float32(pts1_in)
    pts2_in = np.float32(pts2_in)

    _, R, T, _ = cv2.recoverPose(essential_matrix, pts1_in, pts2_in, intrinsics)

    left_img_rect, right_img_rect = stereo_rectification(R, T, intrinsics, left_img, right_img, out_path='images')

    combined_img_rect = cv2.hconcat([left_img_rect, right_img_rect])
    add_lines(combined_img_rect)

    show_image('Combined Rectified Image with lines', combined_img_rect)
    # --------------------------------------------------------------------------------------------------------------------

