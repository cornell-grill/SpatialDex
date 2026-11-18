import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from spatialdex.model import sonata

SONATA_CONFIG = dict(
    in_channels=9,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(3, 3, 3, 12, 3),
    enc_channels=(48, 96, 192, 384, 512),
    enc_num_head=(3, 6, 12, 24, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    traceable=True,
    enc_mode=True,
    mask_token=True,
)
SONATA_FEATURE_DIM = 1088


def estimate_normals(coord, k=16):
    _, index = cKDTree(coord).query(coord, k=k + 1)
    neighbors = coord[index[:, 1:]]
    delta = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", delta, delta) / float(k)
    normal = np.linalg.eigh(cov)[1][:, :, 0]
    normal = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-06, None)
    return normal.astype(np.float32, copy=False)


class SonataPointTransformer(nn.Module):
    def __init__(self, out_dim, normal_knn_k=16):
        super().__init__()
        self.normal_knn_k = normal_knn_k
        self.train_transform = self.build_transform(random_select=True)
        self.transform = self.build_transform(random_select=False)
        config = dict(SONATA_CONFIG)
        if getattr(sonata.model, "flash_attn", None) is None:
            config["enable_flash"] = False
        self.backbone = sonata.model.PointTransformerV3(**config)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.output_proj = nn.Linear(SONATA_FEATURE_DIM, out_dim)

    @staticmethod
    def build_transform(random_select):
        return sonata.transform.Compose(
            [
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                    return_inverse=True,
                    random_select=random_select,
                ),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "color", "inverse"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ]
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def point_dict(self, coord, color):
        coord = coord.detach().float().cpu().numpy()
        color = color.detach().float().cpu().numpy() * 255.0
        return {"coord": coord, "color": color, "normal": estimate_normals(coord, k=self.normal_knn_k)}

    @staticmethod
    def recover_features(point):
        for _ in range(2):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys() and "pooling_inverse" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point

    def forward(self, coord, color, mask):
        mask = mask.bool()
        device = coord.device
        batch, num_points, _ = coord.shape
        points, indices, shift = [], [], 0
        transform = self.train_transform if self.training else self.transform
        for b in range(batch):
            index = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            point = transform(self.point_dict(coord[b, index], color[b, index]))
            point["inverse"] = point["inverse"] + shift
            shift += int(point["coord"].shape[0])
            points.append(point)
            indices.append(index)
        point = sonata.data.collate_fn(points)
        for key in list(point.keys()):
            if isinstance(point[key], torch.Tensor):
                if torch.is_floating_point(point[key]):
                    point[key] = point[key].to(device=device, dtype=torch.float32, non_blocking=True)
                else:
                    point[key] = point[key].to(device=device, non_blocking=True)
        with torch.autocast("cuda", enabled=False), torch.no_grad():
            point = self.backbone(point)
        point = self.recover_features(point)
        features = point.feat[point.inverse]
        full = coord.new_zeros((batch, num_points, features.shape[-1]), dtype=features.dtype)
        start = 0
        for b, index in enumerate(indices):
            full[b, index] = features[start : start + index.numel()]
            start += index.numel()
        full = self.output_proj(full)
        return full * mask.unsqueeze(-1).to(full.dtype)
