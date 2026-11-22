<h2 align="center">SpatialDex: Reconstruction-Augmented Visuomotor Pre-Training of 3D-Grounded Bimanual Dexterity from RGB-Only Human Videos</h2>

<p align="center">
  Xingyi He*,
  Zhiwen Qiu*,
  Adhitya Polavaram*,
  Yunhao Cao*,
  Cory Fan,
  Julius Samwer,
  Saleh Kalantari,
  Kuan Fang
  <br>
  <sub>* Equal contribution</sub>
</p>

<p align="center">
  Cornell University
</p>

<p align="center">
  <a href="https://spatial-dex.github.io/">[Project Page]</a>
  <a href="https://arxiv.org/abs/2610.29716">[Paper]</a>
  <a href="https://spatial-dex.github.io/#video">[Video]</a>
  <a href="#checkpoints-and-datasets">[Downloads]</a>
  <a href="#citation">[Citation]</a>
</p>

SpatialDex is a 3D-grounded visuomotor policy for bimanual dexterous manipulation. It is pre-trained on 1M human-video clips whose scene geometry and hand motion are reconstructed from RGB alone, then adapted to a robot with a small number of teleoperated demonstrations.

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/media/real_unzip_bag.webp" width="100%" alt="Unzip Bag rollout"><br>
      <b>Unzip Bag</b>
    </td>
    <td align="center" width="33%">
      <img src="assets/media/real_sweep_trash.webp" width="100%" alt="Sweep Trash rollout"><br>
      <b>Sweep Trash</b>
    </td>
    <td align="center" width="33%">
      <img src="assets/media/real_pass_flower.webp" width="100%" alt="Pass Flower rollout"><br>
      <b>Pass Flower</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/media/real_spray_plant.webp" width="100%" alt="Spray Plant rollout"><br>
      <b>Spray Plant</b>
    </td>
    <td align="center">
      <img src="assets/media/real_pack_suitcase.webp" width="100%" alt="Pack Suitcase rollout"><br>
      <b>Pack Suitcase</b>
    </td>
    <td align="center">
      <img src="assets/media/real_pour_water.webp" width="100%" alt="Pour Water rollout"><br>
      <b>Pour Water</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <img src="assets/media/sim_relocate_ball.webp" width="100%" alt="Relocate Ball in simulation"><br>
      <b>Relocate Ball</b>
    </td>
    <td align="center" width="33%">
      <img src="assets/media/sim_pass_toy.webp" width="100%" alt="Pass Toy in simulation"><br>
      <b>Pass Toy</b>
    </td>
    <td align="center" width="33%">
      <img src="assets/media/sim_sweep_trash.webp" width="100%" alt="Sweep Trash in simulation"><br>
      <b>Sweep Trash</b>
    </td>
  </tr>
</table>

## Highlights

- **Real-robot dexterity:** 81.7% success across six bimanual dexterous tasks, compared with 43.3% for the strongest baseline (π<sub>0.5</sub>).
- **Two simulated embodiments:** 76.3% mean success across nine tasks, compared with 59.8% for the strongest baseline.
- **The value of geometry:** with the same videos, demonstrations, and training budget, pre-training with reconstructed geometry achieves 73.1% success, compared with 58.9% for pre-training on hand motion alone.
- **Reusable simulation resources:** the release includes a MuJoCo model of our Wuji + Tianji robot, three tasks, demonstrations, and post-training and evaluation code for experiments without the physical hardware.

## Contents

- [Method Overview](#method-overview)
- [Installation](#installation)
- [Checkpoints and Datasets](#checkpoints-and-datasets)
- [Post-Training](#post-training)
- [Simulation](#simulation)
  - [Wuji + Tianji](#wuji--tianji)
  - [DexJoCo](#dexjoco)
- [Citation](#citation)

## Method Overview

**Aligned scene and hand reconstruction.** From each RGB-only human video we reconstruct scene geometry and estimate hand motion, then align the two at one shared scale per video using the hand-size prior of the hand estimates. The result supplies depth observations, wrist and finger action targets, and future-geometry targets.

![Data pipeline](assets/media/pipeline.webp)

**3D-grounded policy.** Depth is back-projected to 3D points, encoded with a point-cloud encoder, and fused into the visual token grid of a pre-trained VLM through gated residual fusion. A training-only head predicts future scene geometry. A wrist expert predicts both wrist trajectories, and a finger expert predicts finger actions conditioned on them.

![Policy architecture](assets/media/policy.webp)

## Installation

Run the commands below from the SpatialDex repository root.

Create the environment and install dependencies:

```bash
conda create -n spatialdex python=3.10 -y
conda activate spatialdex
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install spconv-cu124 torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install flash-attn --no-build-isolation
```

Download the PaliGemma 2 processor files after accepting the Gemma license on Hugging Face:

```bash
huggingface-cli download google/paligemma2-3b-mix-224 --include "*.json" --local-dir weights/paligemma2-3b-mix-224
```

## Checkpoints and Datasets

Download a task-specific checkpoint to [run simulation rollouts](#simulation), or use the human-video pre-trained checkpoint to [post-train on robot demonstrations](#post-training).

| Stage / environment | Checkpoints | Dataset |
|---|---|---|
| Human-video pre-training | [Pre-trained checkpoint](https://drive.google.com/drive/folders/1RNyjRYEt5H-uJ_gn5VN2EJCad6eFv8Fb?usp=sharing) | Coming soon |
| [Wuji + Tianji](#wuji--tianji) | [Task-specific checkpoints](https://drive.google.com/drive/folders/1RNyjRYEt5H-uJ_gn5VN2EJCad6eFv8Fb?usp=sharing) | [Demonstrations](https://drive.google.com/drive/folders/1JIuJSCrXiCDoeL8Y6iPGrPxHZdh2-kq8?usp=sharing) |
| [DexJoCo](#dexjoco) | [Task-specific checkpoints](https://drive.google.com/drive/folders/1RNyjRYEt5H-uJ_gn5VN2EJCad6eFv8Fb?usp=sharing) | [Raw](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-Raw) / [LeRobot](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot) |

The DexJoCo dataset links point to the benchmark's upstream Raw and LeRobot releases.

Place `pretrain.safetensors` under `weights/`, Wuji + Tianji task checkpoints under `weights/wuji_tianji/`, and DexJoCo task checkpoints under `weights/dexjoco/`.

For each Wuji + Tianji checkpoint, also download the matching `<task>.json` file. It contains the normalization statistics and is loaded automatically from the same directory as the checkpoint.

<details>
<summary>Checkpoint directory layout</summary>

```
weights/
├── pretrain.safetensors
├── wuji_tianji/
│   ├── relocate_ball.safetensors
│   ├── relocate_ball.json
│   ├── pass_toy.safetensors
│   ├── pass_toy.json
│   ├── sweep_trash.safetensors
│   └── sweep_trash.json
└── dexjoco/
    ├── microwave.safetensors
    ├── assembly.safetensors
    ├── fold_glasses.safetensors
    ├── water_plant.safetensors
    ├── pick_bucket.safetensors
    └── hammer_nail.safetensors
```

</details>

Place the Wuji + Tianji dataset's task folders under `data/wuji_tianji/`. Evaluation uses the scenes in `eval/`; post-training uses the demonstrations in `train/`.

```
data/wuji_tianji/
├── relocate_ball/
│   ├── train/
│   └── eval/
├── pass_toy/
└── sweep_trash/
```

## Post-Training

Post-training adapts SpatialDex's human-video pre-trained checkpoint to a target dataset and robot embodiment, for tasks in simulation or the real world.

### Example: Wuji + Tianji

The example below uses the released Wuji + Tianji simulation dataset.

Download the pre-trained checkpoint and task dataset from [Checkpoints and Datasets](#checkpoints-and-datasets), placing the checkpoint at `weights/pretrain.safetensors`. This example uses **8 GPUs**, with a batch size of **16 per GPU**:

```bash
torchrun --nproc_per_node 8 -m spatialdex.train --task sweep_trash --output outputs/sweep_trash
```

The provided Wuji + Tianji configurations cover `relocate_ball`, `pass_toy`, and `sweep_trash`. Hyperparameters are in `configs/wuji_tianji/<task>.json`.

This configuration trains for 40,000 steps and saves `step_XXXXXX.safetensors` every 2,500 steps. To evaluate a trained checkpoint, pass its path to `--checkpoint` in the [Wuji + Tianji evaluation command](#wuji--tianji).

## Simulation

Evaluate the released checkpoints on three [Wuji + Tianji](#wuji--tianji) tasks and six [DexJoCo](#dexjoco) tasks. The table compares SpatialDex with π<sub>0.5</sub>, the strongest baseline in our experiments. Success rates (%) are averaged over three seeds, with 50 episodes per seed.

See the [project page](https://spatial-dex.github.io/) for real-robot results, all baselines, and ablations.

| Environment | Task | `--task` | π<sub>0.5</sub> | SpatialDex |
|---|---|---|---:|---:|
| [Wuji + Tianji](#wuji--tianji) | Relocate Ball | `relocate_ball` | 64.7 | **88.7** |
| | Pass Toy | `pass_toy` | 36.0 | **75.3** |
| | Sweep Trash | `sweep_trash` | 32.7 | **80.0** |
| [DexJoCo](#dexjoco) | Fold Glasses | `fold_glasses` | 72.0 | **77.3** |
| | Hammer Nail | `hammer_nail` | 84.7 | **94.0** |
| | Water Plant | `water_plant` | 88.7 | **91.3** |
| | Pick Bucket | `pick_bucket` | **84.0** | 79.3 |
| | Bimanual Assembly | `assembly` | 5.3 | **16.7** |
| | Bimanual Microwave | `microwave` | 70.0 | **84.0** |
| | **Mean over nine tasks** | | 59.8 | **76.3** |

### Wuji + Tianji

These three tasks use two 7-DoF Tianji arms with 20-DoF Wuji hands in MuJoCo, matching our real-robot setup. Observations come from one head RGB-D camera.

**Setup**

Download the task checkpoints, matching normalization statistics, and dataset from [Checkpoints and Datasets](#checkpoints-and-datasets). On a headless machine, set `MUJOCO_GL=egl` before evaluation.

**Evaluation**

```bash
python -m spatialdex.eval.wuji --task sweep_trash --checkpoint weights/wuji_tianji/sweep_trash.safetensors --output outputs/sweep_trash_eval
```

`--task` accepts `relocate_ball`, `pass_toy`, or `sweep_trash`. Evaluation runs every scene in `data/wuji_tianji/<task>/eval/` by default. It writes results to `results.json` and saves one rollout video per episode in the output directory.

**Options**

- `--episodes`: run selected episode numbers.
- `--no_video`: skip video recording.

<details>
<summary>Task instructions and success criteria</summary>

| Task | Active hands | Instruction (per hand) | Success criterion |
|---|---|---|---|
| `relocate_ball` | left, right, or both, set per episode | "Pick up the red ball and place it in the gray box-shaped container" | ball is inside the container |
| `pass_toy` | both | "Pick up the {object} and hand it off to the other hand" / "Receive the {object} from the other hand and place it in the gray box-shaped container" | object is inside the container |
| `sweep_trash` | both | "Use broom to sweep the {object} into the dustpan" / "Hold the dustpan to receive the {object}" | object is on the dustpan |

`{object}` is a yellow duck or a red Pringles can. Which hand takes which role varies between episodes.

</details>

### DexJoCo

These six tasks use Allegro hands on Franka arms in [DexJoCo](https://github.com/jellyho/dexjoco).

**Setup**

Download the task checkpoints from [Checkpoints and Datasets](#checkpoints-and-datasets) and install DexJoCo at commit `c592293`:

```bash
git clone https://github.com/jellyho/dexjoco
(
  cd dexjoco &&
  git checkout c592293 &&
  pip install -e dexjoco --no-deps
)
```

**Evaluation**

```bash
python -m spatialdex.eval.dexjoco --task microwave --output outputs/microwave
```

`--task` accepts `microwave`, `assembly`, `fold_glasses`, `water_plant`, `pick_bucket`, or `hammer_nail`. The checkpoint defaults to `weights/dexjoco/<task>.safetensors`. Evaluation runs 50 episodes by default. It writes results to `results.json` and saves one rollout video per episode in the output directory.

**Options**

- `--episodes`: set the number of evaluation episodes.
- `--seed`: set the starting seed.
- `--no-video`: skip video recording.

## Citation

If you use SpatialDex, our released checkpoints, or the Wuji + Tianji simulation resources in your research, please cite our paper.

```bibtex
@misc{he2026spatialdex,
      title={Reconstruction-Augmented Visuomotor Pre-Training of 3D-Grounded Bimanual Dexterity from RGB-Only Human Videos},
      author={Xingyi He and Zhiwen Qiu and Adhitya Polavaram and Yunhao Cao and Cory Fan and Julius Samwer and Saleh Kalantari and Kuan Fang},
      year={2026},
      eprint={2610.29716},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.29716},
}
```
