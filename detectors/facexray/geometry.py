"""Native five-point affine and triangulated first48-point face warps."""

import cv2
import numpy as np

from detectors.fwa.geometry import similarity


def five_points(points):
    return np.concatenate(
        [
            np.array((points[36] + points[39]) // 2, dtype=int).reshape(-1, 2),
            np.array((points[45] + points[42]) // 2, dtype=int).reshape(-1, 2),
            np.array(points[30], dtype=int).reshape(-1, 2),
            np.array(points[48], dtype=int).reshape(-1, 2),
            np.array(points[54], dtype=int).reshape(-1, 2),
        ],
        axis=0,
    )


def bilinear(image, coordinates):
    integer = np.int32(coordinates)
    x, y = integer
    # Native interpolation changes integer coordinates before computing dx/dy,
    # clips only the upper bound, and assumes the256 image recipe.
    x[x > 254] = 254
    y[y > 254] = 254
    dx, dy = coordinates - integer
    q11, q21 = image[y, x], image[y, x + 1]
    q12, q22 = image[y + 1, x], image[y + 1, x + 1]
    bottom = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    return (top * dy + bottom * (1 - dy)).T


def triangle_matrices(vertices, source, destination):
    for indices in vertices:
        source_triangle = np.vstack((source[indices, :].T, [1, 1, 1]))
        destination_triangle = np.vstack((destination[indices, :].T, [1, 1, 1]))
        yield np.dot(source_triangle, np.linalg.inv(destination_triangle))[:2, :]


def triangular_warp(image, source, destination, shape):
    from scipy.spatial import Delaunay

    height, width = shape[:2]
    result = np.zeros((height, width, 3), dtype=np.uint8)
    triangulation = Delaunay(destination)
    matrices = np.asarray(
        list(triangle_matrices(triangulation.simplices, source, destination))
    )
    minimum, maximum = np.min(destination, axis=0), np.max(destination, axis=0) + 1
    grid = np.asarray(
        [
            (x, y)
            for y in range(minimum[1], maximum[1])
            for x in range(minimum[0], maximum[0])
        ],
        dtype=np.uint32,
    )
    assignments = triangulation.find_simplex(grid)
    for index in range(len(triangulation.simplices)):
        coordinates = grid[assignments == index]
        transformed = np.dot(
            matrices[index],
            np.vstack((coordinates.T, np.ones(len(coordinates)))),
        )
        x, y = coordinates.T
        x[x > 255] = 255
        y[y > 255] = 255
        result[y, x] = bilinear(image, transformed)
    return result


def warp_face(image, source_points, destination_points, foreground_mask):
    height, width, _ = image.shape
    triangular = np.random.rand() < 0.5
    if triangular:
        warped = triangular_warp(
            image,
            np.array(source_points[:48]),
            np.array(destination_points[:48]),
            (height, width),
        )
        mask = np.mean(warped, axis=2) > 0
        receipt = {"warp": "triangular_first48"}
    else:
        matrix = np.array(
            similarity(five_points(source_points), five_points(destination_points))[:2]
        ).reshape(2, 3)
        warped = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        mask = (
            cv2.warpAffine(
                foreground_mask,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT,
            )
            > 0
        )
        receipt = {"warp": "five_point_affine", "matrix": matrix.tolist()}
    return warped, mask, receipt
