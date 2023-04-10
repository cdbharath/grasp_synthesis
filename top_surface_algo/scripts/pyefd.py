import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from itertools import combinations

def elliptic_fourier_descriptors(contour, order=10):
    """Calculate elliptical Fourier descriptors for a contour.
    :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
    :param int order: The order of Fourier coefficients to calculate.
    :param bool normalize: If the coefficients should be normalized;
        see references for details.
    :param bool return_transformation: If the normalization parametres should be returned.
        Default is ``False``.
    :return: A ``[order x 4]`` array of Fourier coefficients and optionally the
        transformation parametres ``scale``, ``psi_1`` (rotation) and ``theta_1`` (phase)
    :rtype: ::py:class:`numpy.ndarray` or (:py:class:`numpy.ndarray`, (float, float, float))
    """
    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    t = np.concatenate([([0.0]), np.cumsum(dt)])
    T = t[-1]

    phi = (2 * np.pi * t) / T

    orders = np.arange(1, order + 1)
    consts = T / (2 * orders * orders * np.pi * np.pi)
    phi = phi * orders.reshape((order, -1))

    d_cos_phi = np.cos(phi[:, 1:]) - np.cos(phi[:, :-1])
    d_sin_phi = np.sin(phi[:, 1:]) - np.sin(phi[:, :-1])

    a = consts * np.sum((dxy[:, 0] / dt) * d_cos_phi, axis=1)
    b = consts * np.sum((dxy[:, 0] / dt) * d_sin_phi, axis=1)
    c = consts * np.sum((dxy[:, 1] / dt) * d_cos_phi, axis=1)
    d = consts * np.sum((dxy[:, 1] / dt) * d_sin_phi, axis=1)

    coeffs = np.concatenate(
        [
            a.reshape((order, 1)),
            b.reshape((order, 1)),
            c.reshape((order, 1)),
            d.reshape((order, 1)),
        ],
        axis=1,
    )

    return coeffs
    
def calculate_dc_coefficients(contour):
    """Calculate the :math:`A_0` and :math:`C_0` coefficients of the elliptic Fourier series.
    :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
    :return: The :math:`A_0` and :math:`C_0` coefficients.
    :rtype: tuple
    """
    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    t = np.concatenate([([0.0]), np.cumsum(dt)])
    T = t[-1]

    xi = np.cumsum(dxy[:, 0]) - (dxy[:, 0] / dt) * t[1:]
    A0 = (1 / T) * np.sum(((dxy[:, 0] / (2 * dt)) * np.diff(t ** 2)) + xi * dt)
    delta = np.cumsum(dxy[:, 1]) - (dxy[:, 1] / dt) * t[1:]
    C0 = (1 / T) * np.sum(((dxy[:, 1] / (2 * dt)) * np.diff(t ** 2)) + delta * dt)

    # A0 and CO relate to the first point of the contour array as origin.
    # Adding those values to the coefficients to make them relate to true origin.
    return contour[0, 0] + A0, contour[0, 1] + C0

def get_curve(coeffs, locus=(0.0, 0.0), n=300):
    """Populate xt and yt using the given Fourier coefficient array.
    :param numpy.ndarray coeffs: ``[N x 4]`` Fourier coefficient array.
    :param list, tuple or numpy.ndarray locus:
        The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
    :param int n: Number of points to use for plotting of Fourier series.
    :return: Tuple of populated xt and yt arrays.
    """
    t = np.linspace(0, 1.0, n)
    xt = np.ones((n,)) * locus[0]
    yt = np.ones((n,)) * locus[1]
    for n in range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )
    return xt, yt

def compute_tangents_normals(xt, yt):
    # compute first and second derivatives of x(t) and y(t)
    dxdt = np.gradient(xt)
    dydt = np.gradient(yt)
    d2xdt2 = np.gradient(dxdt)
    d2ydt2 = np.gradient(dydt)
    
    # compute unit-length tangents and normals
    tangents = np.stack([dxdt, dydt], axis=-1)
    # tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True)
    
    normals = np.stack([d2xdt2, d2ydt2], axis=-1)
    # normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    
    return tangents, normals

def compute_curvature(normals):
    return np.linalg.norm(normals, axis=-1)
    
def find_concave(tangent):
    # Take the dot product of consecutive normal vectors
    dot_products = np.cross(tangent[:-1], tangent[1:])
    
    # Handle the last normal vector
    dot_products = np.concatenate([dot_products, [np.cross(tangent[-1], tangent[0])]])
    return dot_products

def plot_random_lines(xt, yt, tangents, random_indices=None, color='red'):
    n_arrows = 20
    
    if random_indices is None:
        random_indices = np.random.choice(len(xt), size=n_arrows, replace=False)
    for i in random_indices:
        start = [xt[i], yt[i]]
        end = start + 20*tangents[i]
        plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], 
                  head_width=0.02, head_length=0.02, fc=color, ec=color)
    
def find_local_max_min_indices(arr):            
    diff = np.diff(arr)
    maxima = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
    minima = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0] + 1
    return np.array(maxima), np.array(minima)

def have_common_indices(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    return list(set1.intersection(set2))
    
if __name__ == "__main__":
    im = cv.imread('test3.jpg')
    assert im is not None, "file could not be read, check with os.path.exists()"
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    ret, thresh = cv.threshold(imgray,230,255,cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea).squeeze()
    
    contours_image = np.ones(im.shape, np.uint8)*255
    contours_image = cv.drawContours(contours_image, [largest_contour], -1, (0,255,0), 3)
    cv.imshow('contours', contours_image)
    cv.waitKey(0)
        
    coeffs = elliptic_fourier_descriptors(largest_contour, order=10)
    a0, c0 = calculate_dc_coefficients(largest_contour)
    xt, yt = get_curve(coeffs, locus=(a0,c0), n=300)
    
    tangents, normals = compute_tangents_normals(xt, yt)
    normals_norm = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
 
    curvature = compute_curvature(normals)
    maxima, minima = find_local_max_min_indices(curvature)
    maxima_minima = np.concatenate([maxima, minima])
    
    concavities = find_concave(tangents)
    concave_indices = np.where(concavities >= 0)[0]    
    concave_curvature = curvature[concave_indices]
    concave_maxima_minima = have_common_indices(concave_indices, maxima_minima)
    
    candidate_points = np.array([xt[concave_maxima_minima], yt[concave_maxima_minima]]).T    
    combinations_list = list(combinations(maxima_minima, 2))
    
    rotation_matrix = np.array([[0, -1], [1, 0]])
    outward_normals = np.dot(rotation_matrix, tangents.T).T
    outward_normals /= np.linalg.norm(outward_normals, axis=-1, keepdims=True)

    grasps = []
    for combination in combinations_list:
        idx1, idx2 = combination
        print(np.dot(outward_normals[idx1], outward_normals[idx2])/(np.linalg.norm(outward_normals[idx1])*np.linalg.norm(outward_normals[idx2])))
        # if np.dot(outward_normals[idx1], outward_normals[idx2]) < 0:
        center = np.array([(xt[idx1] + xt[idx2])/2, (yt[idx1] + yt[idx2])/2])
        pt1 = np.array([xt[idx1] - center[0], yt[idx1] - center[1]])
        pt2 = np.array([xt[idx2] - center[0], yt[idx2] - center[1]])
        
        tau1 = outward_normals[idx1]
        tau2 = outward_normals[idx2]
        
        grasps.append([[idx1, idx2], np.dot(outward_normals[idx1], outward_normals[idx2]), np.linalg.norm(pt1 - pt2)])      
        
    plt.plot(xt, yt)
    plt.plot(largest_contour[:, 0], largest_contour[:, 1], "c--", linewidth=2)
    plt.plot(candidate_points[:, 0], candidate_points[:, 1], "ro", markersize=10)
    
    sorted_grasp = sorted(grasps, key=lambda x: (x[1], x[2]))

    best_grasp = sorted_grasp[4]
    x1 = xt[best_grasp[0][0]]
    y1 = yt[best_grasp[0][0]]
    x2 = xt[best_grasp[0][1]]
    y2 = yt[best_grasp[0][1]]
    plt.plot(x1, y1, "bo", markersize=10)
    plt.plot(x2, y2, "bo", markersize=10)

    random_indices = np.random.choice(len(xt), size=300, replace=False)
    # plot_random_lines(xt, yt, tangents, random_indices, 'red')
    # plot_random_lines(xt, yt, normals, random_indices, 'green')
    plot_random_lines(xt, yt, outward_normals, random_indices, 'green')

    plt.show()
