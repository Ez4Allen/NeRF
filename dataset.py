# dataset.py (compact style)

import os, json
import numpy as np
import imageio.v2 as imageio
import torch
import torch.nn.functional as F

def load_blender_data_raw(basedir):
    splits = ['train', 'val', 'test']
    metas = {s: json.load(open(os.path.join(basedir, f'transforms_{s}.json'), 'r')) for s in splits}

    all_imgs, all_poses, counts = [], [], [0]
    for s in splits:
        meta = metas[s]
        imgs, poses = [], []
        for fr in meta['frames']:
            fname = os.path.join(basedir, fr['file_path'] + '.png')
            im = imageio.imread(fname).astype(np.float32) / 255.0
            imgs.append(im)
            poses.append(np.array(fr['transform_matrix'], dtype=np.float32))
        imgs = np.stack(imgs, 0); poses = np.stack(poses, 0)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs); all_poses.append(poses)

    imgs  = np.concatenate(all_imgs,  0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    return imgs, poses, [H, W, focal], i_split

def get_rays(H, W, focal, c2w):
    device = c2w.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy'
    )
    dirs = torch.stack([(i - W*0.5)/focal, -(j - H*0.5)/focal, -torch.ones_like(i)], -1)
    R, t = c2w[:3,:3], c2w[:3,3]
    rd = F.normalize(dirs @ R.T, dim=-1)
    ro = t.expand_as(rd)
    return ro, rd

@torch.no_grad()
def build_ray_bank(imgs, poses, H, W, focal):
    N = imgs.shape[0]
    ro_all, rd_all = [], []
    for n in range(N):
        ro, rd = get_rays(H, W, focal, poses[n])
        ro_all.append(ro); rd_all.append(rd)
    ro_all = torch.stack(ro_all, 0)  # [N,H,W,3]
    rd_all = torch.stack(rd_all, 0)
    rgb = imgs[..., :3]
    return ro_all.reshape(-1,3), rd_all.reshape(-1,3), rgb.reshape(-1,3)
