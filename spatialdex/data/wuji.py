import numpy as np
from scipy.spatial.transform import Rotation

MAPPING_RIGHT = [
    (0, 36, 1), (1, 37, -1), (2, 39, 1), (3, 42, 1), (4, 2, 1), (5, 1, 1), (6, 5, 1), (7, 8, 1), (8, 11, 1),
    (9, 10, 1), (10, 14, 1), (11, 17, 1), (12, 29, 1), (13, 28, 1), (14, 32, 1), (15, 35, 1), (16, 20, 1),
    (17, 19, 1), (18, 23, 1), (19, 26, 1),
]
MAPPING_LEFT = [
    (0, 36, 1), (1, 37, 1), (2, 39, 1), (3, 42, 1), (4, 2, -1), (5, 1, 1), (6, 5, -1), (7, 8, -1), (8, 11, -1),
    (9, 10, 1), (10, 14, -1), (11, 17, -1), (12, 29, -1), (13, 28, 1), (14, 32, -1), (15, 35, -1), (16, 20, -1),
    (17, 19, 1), (18, 23, -1), (19, 26, -1),
]
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def pack_wuji_pose(pose):
    squeeze = pose.ndim == 1
    if squeeze:
        pose = pose[None, :]
    out = np.zeros((pose.shape[0], 102), dtype=pose.dtype)
    out[:, 0:6] = pose[:, 0:6]
    for w, m, sign in MAPPING_LEFT:
        out[:, 6 + m] = sign * pose[:, 6 + w]
    out[:, 51:57] = pose[:, 26:32]
    for w, m, sign in MAPPING_RIGHT:
        out[:, 57 + m] = sign * pose[:, 32 + w]
    return out[0] if squeeze else out


def unpack_wuji_pose(layout):
    squeeze = layout.ndim == 1
    if squeeze:
        layout = layout[None, :]
    out = np.zeros((layout.shape[0], 52), dtype=layout.dtype)
    out[:, 0:6] = layout[:, 0:6]
    left = np.zeros((layout.shape[0], 20), dtype=layout.dtype)
    for w, m, sign in MAPPING_LEFT:
        left[:, w] = sign * layout[:, 6 + m]
    out[:, 6:26] = left
    out[:, 26:32] = layout[:, 51:57]
    right = np.zeros((layout.shape[0], 20), dtype=layout.dtype)
    for w, m, sign in MAPPING_RIGHT:
        right[:, w] = sign * layout[:, 57 + m]
    out[:, 32:52] = right
    return out[0] if squeeze else out


def hand_dims(side):
    table = MAPPING_LEFT if side == "left" else MAPPING_RIGHT
    return np.array(sorted({m for _, m, _ in table}), dtype=np.int64)


def camera_basis(camera_to_world_mujoco):
    return np.asarray(camera_to_world_mujoco, dtype=np.float64) @ OPENGL_TO_OPENCV


def transform_pose(pose, camera_to_world, to_camera, active):
    transform = np.linalg.inv(camera_to_world) if to_camera else camera_to_world
    rotation, translation = transform[:3, :3], transform[:3, 3]
    result = np.asarray(pose).astype(np.float64, copy=True)
    for arm, is_active in enumerate(active):
        if not is_active:
            continue
        start = arm * 26
        position = result[..., start : start + 3]
        rotvec = result[..., start + 3 : start + 6]
        result[..., start : start + 3] = np.einsum("ij,...j->...i", rotation, position) + translation
        shape = rotvec.shape
        rotated = np.einsum("ij,njk->nik", rotation, Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix())
        result[..., start + 3 : start + 6] = Rotation.from_matrix(rotated).as_rotvec().reshape(shape)
    return result.astype(np.asarray(pose).dtype, copy=False)


def anchor_delta_to_absolute(delta, anchor, active):
    target = delta.astype(np.float64, copy=False)
    state = anchor.astype(np.float64, copy=False)
    result = target.copy()
    for arm, is_active in enumerate(active):
        if not is_active:
            continue
        start = arm * 26
        position = state[..., start : start + 3].copy()
        rotation = Rotation.from_rotvec(state[..., start + 3 : start + 6].reshape(-1, 3)).as_matrix()
        rotation = rotation.reshape(state.shape[:-1] + (3, 3))
        delta_rotation = Rotation.from_rotvec(target[..., :, start + 3 : start + 6].reshape(-1, 3)).as_matrix()
        delta_rotation = delta_rotation.reshape(target.shape[:-2] + (target.shape[-2], 3, 3))
        for step in range(target.shape[-2]):
            next_position = position + target[..., step, start : start + 3]
            next_rotation = np.matmul(delta_rotation[..., step, :, :], rotation)
            result[..., step, start : start + 3] = next_position
            result[..., step, start + 3 : start + 6] = (
                Rotation.from_matrix(next_rotation.reshape(-1, 3, 3)).as_rotvec().reshape(next_position.shape)
            )
    return result.astype(delta.dtype, copy=False)
