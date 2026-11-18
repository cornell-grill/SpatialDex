import numpy as np

ALLEGRO_HUMAN_MAPPING_RIGHT = [
    (0, 0, -1),
    (1, 2, +1),
    (2, 5, +1),
    (3, 8, +1),
    (4, 9, +1),
    (5, 11, +1),
    (6, 14, +1),
    (7, 17, +1),
    (8, 27, +1),
    (9, 29, +1),
    (10, 32, +1),
    (11, 35, +1),
    (12, 36, +1),
    (13, 41, +1),
    (14, 37, -1),
    (15, 43, -1),
]
ALLEGRO_HUMAN_MAPPING_LEFT = [
    (0, 27, -1),
    (1, 29, -1),
    (2, 32, -1),
    (3, 35, -1),
    (4, 9, -1),
    (5, 11, -1),
    (6, 14, -1),
    (7, 17, -1),
    (8, 0, +1),
    (9, 2, -1),
    (10, 5, -1),
    (11, 8, -1),
    (12, 36, +1),
    (13, 41, -1),
    (14, 37, +1),
    (15, 43, +1),
]


def pack_allegro_pose_into_mano_layout(action_pre_pack):
    squeeze = action_pre_pack.ndim == 1
    if squeeze:
        action_pre_pack = action_pre_pack[None, :]
    W = action_pre_pack.shape[0]
    out = np.zeros((W, 102), dtype=action_pre_pack.dtype)
    out[:, 0:6] = action_pre_pack[:, 0:6]
    lh = action_pre_pack[:, 6:22]
    for a_idx, m_idx, sign in ALLEGRO_HUMAN_MAPPING_LEFT:
        out[:, 6 + m_idx] = sign * lh[:, a_idx]
    out[:, 51:57] = action_pre_pack[:, 22:28]
    rh = action_pre_pack[:, 28:44]
    for a_idx, m_idx, sign in ALLEGRO_HUMAN_MAPPING_RIGHT:
        out[:, 57 + m_idx] = sign * rh[:, a_idx]
    return out[0] if squeeze else out


def unpack_mano_layout_to_allegro_pose(action_mano_layout):
    squeeze = action_mano_layout.ndim == 1
    if squeeze:
        action_mano_layout = action_mano_layout[None, :]
    W = action_mano_layout.shape[0]
    out = np.zeros((W, 44), dtype=action_mano_layout.dtype)
    out[:, 0:6] = action_mano_layout[:, 0:6]
    lh = np.zeros((W, 16), dtype=action_mano_layout.dtype)
    for a_idx, m_idx, sign in ALLEGRO_HUMAN_MAPPING_LEFT:
        lh[:, a_idx] = sign * action_mano_layout[:, 6 + m_idx]
    out[:, 6:22] = lh
    out[:, 22:28] = action_mano_layout[:, 51:57]
    rh = np.zeros((W, 16), dtype=action_mano_layout.dtype)
    for a_idx, m_idx, sign in ALLEGRO_HUMAN_MAPPING_RIGHT:
        rh[:, a_idx] = sign * action_mano_layout[:, 57 + m_idx]
    out[:, 28:44] = rh
    return out[0] if squeeze else out


def hand_supervised_dim_indices(side):
    table = ALLEGRO_HUMAN_MAPPING_LEFT if side == "left" else ALLEGRO_HUMAN_MAPPING_RIGHT
    return np.array(sorted({m_idx for (_, m_idx, _) in table}), dtype=np.int64)
