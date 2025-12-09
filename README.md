<p align="center">
  <img src="../render_example.png" width="70%" />
</p>

# Minimal NeRF (PyTorch)

A compact PyTorch implementation of Neural Radiance Fields (NeRF).  
The repository includes a clean training pipeline, rendering utilities, and a demo notebook for interactive testing.

This project reproduces a TinyNeRF-style baseline with practical improvements:

- hierarchical sampling (coarse + fine)
- positional encoding annealing
- direction normalization reuse
- precomputed ray banks for faster training

All experiments use the `nerf_synthetic/lego` dataset.

---

## 1. Environment Setup

Install dependencies:

```bash
pip install torch torchvision
pip install numpy matplotlib imageio pandas
```

---

## 2. Dataset

Download the Blender synthetic dataset (from the original NeRF repo):

https://github.com/bmild/nerf/tree/master/data/nerf_synthetic

Place the dataset as:

```bash
data/nerf_synthetic/lego/
```

Required files:

```
transforms_train.json
transforms_val.json
transforms_test.json
*.png
```

Folder tree:

```
data/
 └── nerf_synthetic/
      └── lego/
           ├── transforms_train.json
           ├── transforms_val.json
           ├── transforms_test.json
           ├── 000.png
           ├── 001.png
           └── ...
```

---

## 3. Training

Run training:

```bash
python train.py
```

This will generate:

- `nerf_lego_pe.pt` — model checkpoint  
- `training_metrics.csv` — logged losses and PSNR  
- `training_metrics_eval.csv` — test-set evaluation logs  
- plots
  - `train_vs_test_psnr_f.png`
  - `train_vs_test_loss_f.png`
- additional visualizations for positional encoding schedules  
- rendered example images (`render_example.png`)

---

## 4. Rendering

All rendering examples are demonstrated inside the notebook:

**`TinyNeRFdemo.ipynb`**

The notebook contains:
- utilities for generating rays and running the NeRF model  
- functions for rendering a fixed test view  
- functions for generating novel-camera renderings  
- comments explaining each step of the rendering pipeline

To reproduce the figures shown in the paper / report:
1. open the notebook  
2. run the “Rendering” section  
3. adjust the parameters (camera pose, pitch, yaw, sampling rates) as needed  

Rendered images (e.g., `render_example.png`, `novel_pitch_up_white.png`) will be saved to the repository directory.

---

## 5. Files

```
train.py            # main training script
render.py           # rendering utilities
dataset.py          # Blender dataset loader
nerf_model.py       # NeRF MLP + positional encodings
TinyNeRFdemo.ipynb  # interactive demo notebook
render_example.png  # sample rendering output
```

---

## 6. License

MIT License.
