import numpy as np
from show import *
from utils import *

def lines_hough(img, opt_type=None):
    height, width = img.shape
    thetas = np.linspace(-90.0, 180.0, 271)
    diagonal_dist = int(np.sqrt((height - 1) ** 2 + (width - 1) ** 2))
    rhos = np.linspace(-diagonal_dist, diagonal_dist, 2 * diagonal_dist + 1)
    accum = np.zeros((len(rhos), len(thetas)))
    coords = np.where(img != 0)
    idxs_coords = np.arange(len(coords[0]))
    if opt_type is None or opt_type == 1:
        if opt_type == 1:
            idxs_coords = np.random.choice(idxs_coords, int(len(idxs_coords) * 0.2), replace=False)
        for i in idxs_coords:
            for theta_idx in range(len(thetas)):
                rad_theta = np.deg2rad(thetas[theta_idx])
                rho = coords[1][i] * np.cos(rad_theta) + coords[0][i] * np.sin(rad_theta)
                rho_idx = np.argmin(np.abs(rhos - rho))
                accum[rho_idx, theta_idx] += 1
    elif opt_type == 2:
        nums_coords = len(idxs_coords) if len(idxs_coords) % 2 == 0 else len(idxs_coords) - 1
        nums_coords *= 4
        idxs_coords = np.random.choice(idxs_coords, nums_coords, replace=True)
        idxs_coords = zip(idxs_coords[:nums_coords // 2], idxs_coords[nums_coords // 2:])

        for point_1, point_2 in idxs_coords:
            if coords[1][point_1] == coords[1][point_2]:
                if coords[1][point_1] < img.shape[1] - 1:
                    coords[1][point_1] += 1
                else:
                    coords[1][point_1] -= 1
            if coords[0][point_1] == coords[0][point_2]:
                if coords[0][point_1] < img.shape[0] - 1:
                    coords[0][point_1] += 1
                else:
                    coords[0][point_1] -= 1
            tan_theta = (coords[1][point_1] - coords[1][point_2]) / (coords[0][point_2] - coords[0][point_1])
            theta = np.arctan(tan_theta)

            deg_theta = np.rad2deg(theta)
            theta_idx = np.argmin(np.abs(thetas - deg_theta))
            rho = coords[1][point_1] * np.cos(theta) + coords[0][point_1] * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos - rho))
            accum[rho_idx, theta_idx] += 1
    elif opt_type == 3:
        grad = calc_grad(img)
        for i in idxs_coords:
            theta = int(grad[coords[0][i], coords[1][i]] * 180 / np.pi)
            theta_idx = np.argmin(np.abs(thetas - theta))
            rad_theta = np.deg2rad(thetas[theta_idx])
            rho = coords[1][i] * np.cos(rad_theta) + coords[0][i] * np.sin(rad_theta)
            rho_idx = np.argmin(np.abs(rhos - rho))
            accum[rho_idx, theta_idx] += 1

    return accum


def circle_hough(img, opt_type=None):
    height, width = img.shape
    diagonal_dist = int(np.sqrt((height - 1) ** 2 + (width - 1) ** 2))
    r_s = np.linspace(0, diagonal_dist, diagonal_dist)
    a_s = np.linspace(0, width, width)
    b_s = np.linspace(0, height, height)
    accum = np.zeros((len(a_s), len(b_s), len(r_s)))
    coords = np.where(img != 0)

    idxs_coords = np.arange(len(coords[0]))
    if opt_type is None or opt_type == 1:
        if opt_type == 1:
            idxs_coords = np.random.choice(idxs_coords, int(len(idxs_coords) * 0.2), replace=False)
        iters = len(a_s) * len(idxs_coords) * len(b_s)
        print(f'Iterations:{iters}')
        step = 1000000
        iter = 0
        for i in idxs_coords:
            for a_idx in range(len(a_s)):
                for b_idx in range(len(b_s)):
                    r = np.sqrt((coords[1][i] - a_s[a_idx]) ** 2 + (coords[0][i] - b_s[b_idx]) ** 2)
                    r_idx = np.argmin(np.abs(r_s - r))
                    accum[a_idx, b_idx, r_idx] += 1
                    iter += 1
                    if iter % step == 0:
                        print(f'Iteration {iter}')
    elif opt_type == 2:
        d_len = len(idxs_coords) if len(idxs_coords) % 2 == 0 else len(idxs_coords) - 1
        d_len *= 4
        idxs_coords = np.random.choice(idxs_coords, d_len, replace=True)
        print(f'Iters {len(idxs_coords[:d_len // 2]) * len(a_s)}')
        idxs_coords = zip(idxs_coords[:d_len // 2], idxs_coords[d_len // 2:])

        count = 0

        for point_1, point_2 in idxs_coords:
            point_1 = np.array([coords[0][point_1], coords[1][point_1]])
            point_2 = np.array([coords[0][point_2], coords[1][point_2]])

            if point_1[1] == point_2[1]:
                if point_1[1] < img.shape[1] - 1:
                    point_1[1] += 1
                else:
                    point_1[1] -= 1
            if point_1[0] == point_2[0]:
                if point_1[0] < img.shape[0] - 1:
                    point_1[0] += 1
                else:
                    point_1[0] -= 1
            point_med = (point_1 + point_2) / 2
            k = (point_2[0] - point_1[0]) / (point_2[1] - point_1[1])
            for a_idx in range(len(a_s)):
                y_normal_line = -1 / k * (a_idx - point_med[1]) + point_med[0]
                b_idx = np.argmin(np.abs(b_s - y_normal_line))
                k1 = ((point_med - point_1) ** 2).sum()
                k2 = ((np.array([b_idx, a_idx]) - point_med) ** 2).sum()
                r = np.sqrt(k1 + k2)

                r_idx = np.argmin(np.abs(r_s - r))

                accum[a_idx, b_idx, r_idx] += 1
                count += 1
                if count % 10000 == 0:
                    print(count)
            # print('-------------------')
    elif opt_type == 3:
        count = 0
        grad = calc_grad(img)
        for i in idxs_coords:
            k = np.tan(grad[coords[0][i], coords[1][i]])
            b = coords[0][i] - k * coords[1][i]
            for a_idx in range(len(a_s)):
                y_normal_line = k * a_s[a_idx] + b
                b_idx = np.argmin(np.abs(b_s - y_normal_line))
                r = np.sqrt((coords[1][i] - a_s[a_idx]) ** 2 + (coords[0][i] - b_s[b_idx]) ** 2)
                r_idx = np.argmin(np.abs(r_s - r))

                accum[a_idx, b_idx, r_idx] += 1
                count += 1
                if count % 10000 == 0:
                    print(count)

    return accum





def ellipse_hough(img):
    height, width = img.shape
    diagonal_dist = int(np.sqrt((height - 1) ** 2 + (width - 1) ** 2))
    b_s = np.linspace(0, diagonal_dist, diagonal_dist + 1)
    ellipse_store = []

    coords = np.where(img != 0)

    least_dist_1 = 10
    black_list = []
    for i in range(len(coords[0])):
        if i not in black_list:
            for j in range(len(coords[0])):
                store_b = {i: [] for i in range(len(b_s))}
                if j not in black_list:
                    x0 = (coords[1][i] + coords[1][j]) / 2
                    y0 = (coords[0][i] + coords[0][j]) / 2
                    a = np.sqrt((np.power(coords[1][j] - coords[1][i],2) + np.power(coords[0][j] - coords[0][i], 2))) / 2
                    if 2*a < least_dist_1:
                        continue
                    alpha = np.arctan2((coords[0][j] - coords[0][i]), (coords[1][j] - coords[1][i]))
                    accum = np.zeros(len(b_s))
                    for k in range(len(coords[0])):
                        if k not in black_list and coords[0][k] > y0:
                            d = np.power(np.power(x0 - coords[1][k], 2) + np.power(y0 - coords[0][k],2),1/2)
                            if d >= least_dist_1:
                                f = np.power((np.power(coords[1][j] - coords[1][k],2) + np.power(coords[0][j] - coords[0][k],2)), 1 / 2)
                                cosT_sqr = ((a * a + d * d - f * f) / (2 * a * d)) ** 2
                                sinT_sqr = 1 - cosT_sqr
                                if np.sqrt(cosT_sqr) >= 0:
                                    b = ((a * a * d * d * sinT_sqr) / (a * a - d * d * cosT_sqr)) ** (1 / 2)
                                    if np.isnan(b):
                                        print()
                                    if not np.isnan(b) and 0 < b < diagonal_dist:
                                        store_b[np.argmin(np.abs(b_s - b))].append(k)
                                        accum[np.argmin(np.abs(b_s - b))] += 1
                    b = np.argmax(accum)
                    if accum[b] >= 30:
                        black_list.extend(store_b[b])
                        black_list.append(i)
                        black_list.append(j)
                        ellipse_store.append((x0, y0, a, b, alpha, np.max(accum)))

    return ellipse_store