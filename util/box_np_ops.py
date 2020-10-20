import numba
import numpy as np


def camera_to_lidar(points, r_rect, velo2cam):
    """Converts points in camera coordinate to lidar coordinate.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.

    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def projection_matrix_to_CRT_kitti(proj):
    """Split projection matrix of kitti.

    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.

    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.

    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def remove_outside_points(points, rect, Trv2c, P2, image_shape):
    """Remove points which are outside of image.

    Args:
        points (np.ndarray, shape=[N, 3+dims]): Total points.
        rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.
        image_shape (list[int]): Shape of image.

    Returns:
        np.ndarray, shape=[N, 3+dims]: Filtered points.
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """Get frustum corners in camera coordinates.

    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float): Nearest distance of frustum.
        far_clip (float): Farthest distance of frustum.

    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], aixs=1)
    return ret_xyz