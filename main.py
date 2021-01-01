import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect import *
from show import *
from utils import *










def start_detect_lines():
    optimization_types = [None, 1, 2, 3]
    optimization_type = 3
    size_filter = 301  # Размер фильтра в NMS
    tau = 0  # От 0 до 100. Порог отсеивания в NMS

    path_to_images = 'C:\\Users\\adels\PycharmProjects\data\Lines'
    path_img = os.path.join(path_to_images, 'track1.jpg')
    orig_img = np.array(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB), dtype=np.float64)
    show_image(orig_img, title='Original')
    gray_img = cv2.imread(path_img, 0)
    show_image(gray_img, title='Gray', cmap='gray')
    edges_img = cv2.Canny(gray_img, 150, 200)
    show_image(edges_img, title='Edges', cmap='gray')
    hough_matrix = lines_hough(edges_img, opt_type=optimization_type)
    hough_matrix = cv2.GaussianBlur(hough_matrix, (3, 3), 10)
    # show_3d_matrix(hough_matrix)
    show_image(hough_matrix, title='Accum matrix')
    hough_matrix = hough_matrix / np.max(hough_matrix) * 100
    nmsed_hough_matrix = nms_2d(hough_matrix, size_filter=size_filter, tau=tau)
    orig_with_draw = orig_img.copy()
    draw_lines(nmsed_hough_matrix, orig_with_draw)
    show_image(orig_with_draw, 'Final')

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0, 0].imshow(np.uint8(orig_img))
    axes[0, 0].set_title('Original Image')

    axes[0, 1].imshow(edges_img, cmap='gray')
    axes[0, 1].set_title('Binary Image')

    axes[1, 0].imshow(hough_matrix)
    axes[1, 0].set_title(f'Accum Matrix')

    axes[1, 1].imshow(np.uint8(orig_with_draw))
    axes[1, 1].set_title(f'Final Image Opt type {optimization_type}')
    plt.show()


def start_detect_circles():
    optimization_types = [None, 1, 2, 3]
    optimization_type = 3
    size_filter = 141  # Размер фильтра в NMS
    tau = 0  # От 0 до 100. Порог отсеивания в NMS

    path_to_images = 'C:\\Users\\adels\PycharmProjects\data\Coins'
    path_img = os.path.join(path_to_images, 'coins_37.png')
    orig_img = np.array(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB), dtype=np.float64)
    show_image(orig_img, title='Original')
    gray_img = cv2.imread(path_img, 0)
    show_image(gray_img, title='Gray', cmap='gray')
    edges_img = cv2.Canny(gray_img, 150, 200)
    show_image(edges_img, title='Edges', cmap='gray')
    hough_matrix = circle_hough(edges_img, opt_type=optimization_type)
    hough_matrix = GaussianBlur_3d(hough_matrix, (3, 3), 10)
    show_3d_matrix(hough_matrix)
    hough_matrix = hough_matrix / np.max(hough_matrix) * 100
    nmsed_hough_matrix = nms_3d(hough_matrix, size_filter=size_filter, tau=tau)
    orig_with_draw = orig_img.copy()
    draw_circle(nmsed_hough_matrix, orig_with_draw)
    show_image(orig_with_draw, 'Final')

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0, 0].imshow(np.uint8(orig_img))
    axes[0, 0].set_title('Original Image')

    axes[0, 1].imshow(edges_img, cmap='gray')
    axes[0, 1].set_title('Binary Image')
    r = np.random.randint(0, hough_matrix.shape[2])
    axes[1, 0].imshow(hough_matrix[:, :, r])
    axes[1, 0].set_title(f'Accum Matrix R = {r}')

    axes[1, 1].imshow(np.uint8(orig_with_draw))
    axes[1, 1].set_title(f'Final Image Opt type {optimization_type}')
    plt.show()


def start_detect_ellipse():
    path_to_images = 'C:\\Users\\adels\PycharmProjects\data\Coins'
    path_img = os.path.join(path_to_images, 'ellipse.png')
    orig_img = np.array(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB), dtype=np.float64)
    show_image(orig_img, title='Original')
    gray_img = cv2.imread(path_img, 0)
    show_image(gray_img, title='Gray', cmap='gray')
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 15)

    edges_img = cv2.Canny(gray_img, 140, 205)
    show_image(edges_img, title='Edges', cmap='gray')

    store = ellipse_hough(edges_img)
    for el in store:
        print(f'X0 = {el[0]} Y0 = {el[1]} a = {el[2]} b = {el[3]} Theta = {np.rad2deg(el[4]), el[4]} vote = {el[5]}')
    orig_with_draw = orig_img.copy()
    draw_ellipse(store, orig_with_draw)
    show_image(orig_with_draw)
    # plt.show()


if __name__ == '__main__':
    # start_detect_lines()
    start_detect_circles()
    # start_detect_ellipse()
