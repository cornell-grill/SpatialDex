import numpy as np
from scipy.spatial.transform import Rotation


def decode(action, state):
    a = np.asarray(action, dtype=np.float64).reshape(-1, 2, 51)
    s = np.asarray(state, dtype=np.float64).reshape(2, 61)
    wrist = np.repeat(np.eye(4)[None, None], len(a), axis=0)
    wrist = np.repeat(wrist, 2, axis=1)
    wrist[..., :3, 3] = s[None, :, :3] + a[..., :3]
    anchor_rotation = Rotation.from_rotvec(s[:, 3:6]).as_matrix()
    wrist[..., :3, :3] = (
        Rotation.from_rotvec(a[..., 3:6].reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3) @ anchor_rotation[None]
    )
    return (wrist, s[None, :, 6:51] + a[..., 6:51])
