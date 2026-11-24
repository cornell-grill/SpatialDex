import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "wuji_tianji"
WIDTH, HEIGHT = 640, 480
CAMERA = "head_cam"
SIDES = ("left", "right")
ARM_DEFAULT_Q = {
    "left": np.array([-0.014, -0.551, 0.556, -0.932, 1.098, 0.0, 0.0]),
    "right": np.array([-0.172, -0.559, -0.350, -1.153, -0.863, 0.0, 0.0]),
}


def load_scene_model(xml_path):
    root = ET.parse(xml_path).getroot()
    for elem in root.iter():
        if elem.tag in ("mesh", "texture", "hfield") and "file" in elem.attrib:
            parts = [part for part in elem.attrib["file"].split("/") if part not in ("", ".", "..")]
            elem.set("file", str(ASSETS.joinpath(*parts)))
        elif elem.tag == "compiler":
            elem.set("meshdir", "")
            elem.set("texturedir", "")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def layout_bodies(model):
    """Body ids of the task object and of its target (the bin, or the dustpan when sweeping)."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    sweepable = [i for i, n in enumerate(names) if n.startswith("sweepable_") and n.endswith("_object")]
    if sweepable:
        return sweepable[-1], [i for i, n in enumerate(names) if n.startswith("dustpan_") and n.endswith("_object")][-1]
    return next(i for i, n in enumerate(names) if n in ("ball", "task_object_object")), names.index("bin")


class ArmIK:
    def __init__(self, model, data, side, posture, elbow=None):
        joints = [f"Joint{i}_{side[0].upper()}" for i in range(1, 8)]
        self.qpos_addrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
        self.configuration = mink.Configuration(model)
        self.configuration.update(data.qpos)
        self.ee_task = mink.FrameTask(
            frame_name=f"{side}_wrist", frame_type="site", position_cost=1.0, orientation_cost=0.8, lm_damping=1.0
        )
        self.tasks = [self.ee_task]
        if elbow is not None:
            elbow_task = mink.FrameTask(
                frame_name=f"Link4_{side[0].upper()}",
                frame_type="body",
                position_cost=(0.03, 0.08, 0.12),
                orientation_cost=0.1,
                lm_damping=1.0,
            )
            elbow_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=elbow[3:]), elbow[:3]))
            self.tasks.append(elbow_task)
        posture_task = mink.PostureTask(model=model, cost=0.01)
        target = self.configuration.q.copy()
        target[self.qpos_addrs] = posture
        posture_task.set_target(target)
        self.tasks.append(posture_task)
        self.limits = [mink.VelocityLimit(model, {j: 3.0 for j in joints}), mink.ConfigurationLimit(model)]

    def solve(self, qpos, target_pos, target_wxyz, dt=1 / 50, max_iters=5, threshold=1e-4):
        self.configuration.update(qpos)
        self.ee_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=target_wxyz), target_pos))
        for _ in range(max_iters):
            try:
                velocity = mink.solve_ik(
                    self.configuration, self.tasks, dt, solver="daqp", damping=1e-3, limits=self.limits
                )
            except mink.exceptions.NoSolutionFound:
                break
            self.configuration.integrate_inplace(velocity, dt)
            error = self.ee_task.compute_error(self.configuration)
            if np.linalg.norm(error[:3]) <= threshold and np.linalg.norm(error[3:]) <= threshold:
                break
        return self.configuration.q[self.qpos_addrs]


def rotmat_to_wxyz(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    s = 0.5 / np.sqrt(t + 1.0)
    return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])


class Scene:
    def __init__(self, xml_path, scene_path, recorder_ik):
        model = load_scene_model(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.model, self.data = model, data

        def index(kind, name):
            return mujoco.mj_name2id(model, kind, name)

        joint, site, actuator = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_ACTUATOR
        self.arm_q = {
            s: [model.jnt_qposadr[index(joint, f"Joint{i}_{s[0].upper()}")] for i in range(1, 8)] for s in SIDES
        }
        self.hand_q = {
            s: [model.jnt_qposadr[index(joint, f"{s}_finger{f}_joint{j}")] for f in range(1, 6) for j in range(1, 5)]
            for s in SIDES
        }
        self.arm_a = {s: [index(actuator, f"{s}_arm_joint{i}_actuator") for i in range(1, 8)] for s in SIDES}
        self.hand_a = {
            s: [index(actuator, f"{s}_finger{f}_joint{j}_actuator") for f in range(1, 6) for j in range(1, 5)]
            for s in SIDES
        }
        self.palm = {s: index(site, f"{s}_wrist") for s in SIDES}
        self.camera = index(mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)

        with h5py.File(scene_path, "r") as f:
            E = np.asarray(f["camera/extrinsic_matrix"][:])
            K = np.asarray(f["camera/intrinsic_matrix"][:])
            model.cam_pos[self.camera] = E[:3, 3]
            model.cam_quat[self.camera] = rotmat_to_wxyz(E[:3, :3])
            model.cam_fovy[self.camera] = float(
                np.degrees(2.0 * np.arctan2(int(f.attrs["image_height"]) / 2.0, K[1, 1]))
            )
            data.qpos[:] = f["initial_qpos"][:]
            data.qvel[:] = f["initial_qvel"][:]
            data.ctrl[:] = f["initial_ctrl"][:]
            if "initial_mocap_pos" in f:
                data.mocap_pos[:] = f["initial_mocap_pos"][:].reshape(data.mocap_pos.shape)
                data.mocap_quat[:] = f["initial_mocap_quat"][:].reshape(data.mocap_quat.shape)
            if recorder_ik:
                targets = {s: (f[f"ik_posture_target_{s}"][:], f[f"ik_elbow_target_{s}"][:]) for s in SIDES}
            else:
                targets = {s: (ARM_DEFAULT_Q[s], None) for s in SIDES}
            self.instruction = str(f.attrs["instruction"])
            self.active = (bool(f.attrs["active_left"]), bool(f.attrs["active_right"]))
            self.recorder_frame_dt = float(f.attrs["replay_nstep"]) * float(f.attrs["replay_timestep"])
        mujoco.mj_forward(model, data)

        self.ik = {s: ArmIK(model, data, s, np.asarray(targets[s][0], dtype=np.float64), targets[s][1]) for s in SIDES}

        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self.depth_renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self.depth_renderer.enable_depth_rendering()
        self.view = mujoco.MjvCamera()
        self.view.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.view.fixedcamid = self.camera

    def open_active_hands(self):
        qpos_to_joint = {int(adr): j for j, adr in enumerate(self.model.jnt_qposadr)}
        for side, active in zip(SIDES, self.active):
            if not active:
                continue
            ctrlrange = np.asarray(self.model.actuator_ctrlrange[self.hand_a[side]], dtype=np.float64)
            q_open = np.clip(np.zeros(len(self.hand_a[side])), ctrlrange[:, 0], ctrlrange[:, 1])
            for i, adr in enumerate(self.hand_q[side]):
                self.data.qpos[adr] = q_open[i]
                self.data.qvel[int(self.model.jnt_dofadr[qpos_to_joint[int(adr)]])] = 0.0
            for i, a in enumerate(self.hand_a[side]):
                self.data.ctrl[a] = q_open[i]
        mujoco.mj_forward(self.model, self.data)

    def jitter(self, rng, amount):
        """Shift the object, and the bin when there is one, by up to `amount` metres in x and y."""
        obj, target = layout_bodies(self.model)
        adr = int(self.model.jnt_qposadr[self.model.body_jntadr[obj]])
        self.data.qpos[adr : adr + 2] += rng.uniform(-amount, amount, 2)
        mocap = int(self.model.body_mocapid[target])
        if mocap >= 0:  # the dustpan is a tool the robot holds, so it stays where it was recorded
            self.data.mocap_pos[mocap, :2] += rng.uniform(-amount, amount, 2)
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.view)
        return self.renderer.render().copy()

    def render_depth(self):
        self.depth_renderer.update_scene(self.data, self.view)
        depth = self.depth_renderer.render().copy().astype(np.float32)
        depth = np.where(depth > 20.0, 0.0, depth)
        return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    def camera_pose(self):
        extr = np.eye(4)
        extr[:3, :3] = self.data.cam_xmat[self.camera].reshape(3, 3)
        extr[:3, 3] = self.data.cam_xpos[self.camera]
        fy = 0.5 * HEIGHT / np.tan(0.5 * np.deg2rad(self.model.cam_fovy[self.camera]))
        K = np.array([[fy, 0, 0.5 * WIDTH], [0, fy, 0.5 * HEIGHT], [0, 0, 1.0]], dtype=np.float64)
        return K, extr

    def hand_state(self, side):
        R = self.data.site_xmat[self.palm[side]].reshape(3, 3)
        position = self.data.site_xpos[self.palm[side]].astype(np.float32)
        rotvec = Rotation.from_matrix(R).as_rotvec().astype(np.float32)
        hand = np.array([self.data.qpos[a] for a in self.hand_q[side]], dtype=np.float32)
        return np.concatenate([position, rotvec, hand])

    def apply(self, side, target):
        q = Rotation.from_rotvec(target[3:6].astype(np.float64)).as_quat()
        q_arm = self.ik[side].solve(self.data.qpos, target[:3].astype(np.float64), np.array([q[3], q[0], q[1], q[2]]))
        for i, a in enumerate(self.arm_a[side]):
            self.data.ctrl[a] = q_arm[i]
        for i, a in enumerate(self.hand_a[side]):
            self.data.ctrl[a] = target[6 + i]

    def step(self, substeps):
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

    def body_position(self, name):
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if body < 0 else self.data.xpos[body].copy()

    def success(self):
        names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.model.nbody)
        ]
        sweepable = [n for n in names if n.startswith("sweepable_") and n.endswith("_object")]
        if sweepable:
            dustpan = [n for n in names if n.startswith("dustpan_") and n.endswith("_object")]
            obj, pan = self.body_position(sweepable[-1]), self.body_position(dustpan[-1])
            return bool(abs(obj[0] - pan[0]) < 0.09 and abs(obj[1] - pan[1]) < 0.09 and abs(obj[2] - pan[2]) < 0.05)
        obj = next(
            p for p in (self.body_position(n) for n in ("ball", "task_object_object")) if p is not None
        )
        bin_pos = self.body_position("bin")
        half, height, radius = 0.06, 0.04, 0.025
        in_xy = abs(obj[0] - bin_pos[0]) < half + radius and abs(obj[1] - bin_pos[1]) < half + radius
        return bool(in_xy and bin_pos[2] - 0.005 <= obj[2] <= bin_pos[2] + 2.0 * height + radius)

    def close(self):
        self.renderer.close()
        self.depth_renderer.close()
