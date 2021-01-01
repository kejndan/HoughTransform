import cv2
import matplotlib.pyplot as plt
import numpy as np
from detect import *

from utils import *


def draw_ellipse(store, img):
    height, width, _ = img.shape
    # plt.imshow(img)

    for ellipse in store:
        x, y, a, b, alph, v = ellipse
        if v >= 30:
            print(x, y, a, b, alph)
            alph = np.deg2rad(np.rad2deg(alph))
            rot_mat = np.array([[np.cos(alph), -np.sin(alph)],
                                [np.sin(alph), np.cos(alph)]])
            t = np.linspace(0, 2 * np.pi, 100)
            vec = np.array([a * np.cos(t), b * np.sin(t)])
            vec_rot = np.round(rot_mat @ vec, 0).astype(np.int)
            vec_rot[1] = vec_rot[1] + y
            vec_rot[0] = vec_rot[0] + x
            vec_rot[1] = np.where(vec_rot[1] >= 0, vec_rot[1], 0)
            vec_rot[1] = np.where(vec_rot[1] < height, vec_rot[1], height - 1)

            vec_rot[0] = np.where(vec_rot[0] >= 0, vec_rot[0], 0)
            vec_rot[0] = np.where(vec_rot[0] < width, vec_rot[0], width - 1)

            # plt.plot(vec_rot[0], vec_rot[1])
            img[vec_rot[1], vec_rot[0]] = [255, 0, 0]


def draw_lines(accum, img):
    height, width, _ = img.shape
    diagonal_dist = int(np.sqrt((height - 1) ** 2 + (width - 1) ** 2))
    rhos, thetas = np.where(accum != 0)
    for rho, theta in zip(rhos, thetas):
        rho -= diagonal_dist
        theta -= 90
        cos_line = np.cos(np.deg2rad(theta))
        sin_line = np.sin(np.deg2rad(theta))
        x1 = rho * cos_line
        y1 = rho * sin_line
        point_0 = (int(x1 + 2000 * -sin_line), int(y1 + 2000 * cos_line))
        point_2 = (int(x1 - 1000 * -sin_line), int(y1 - 1000 * cos_line))
        cv2.line(img, point_0, point_2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Final', img)


def draw_circle(accum, img):
    height, width, _ = img.shape
    a_s, b_s, r_s = np.where(accum != 0)

    for a, b, r in zip(a_s, b_s, r_s):
        for theta in range(360):
            rad_theta = np.deg2rad(theta)
            x = a + r * np.cos(rad_theta)
            x = int(x) if int(x) < img.shape[1] else img.shape[1] - 1
            x = int(x) if int(x) >= 0 else 0
            y = b + r * np.sin(rad_theta)
            y = int(y) if int(y) < img.shape[0] else img.shape[0] - 1
            y = int(y) if int(y) >= 0 else 0
            img[y, x] = [0, 0, 255]





def show_3d_matrix(matrix):
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    slices = [_ for _ in range(0, len(matrix[0, 0]), len(matrix[0, 0]) // 25)]
    q = 0
    for row in range(0, 5):
        for column in range(0, 5):
            axes[row, column].imshow(matrix[:, :, slices[q]])
            axes[row, column].set_title(f'Radius = {slices[q]}')
            q += 1
    plt.show()
