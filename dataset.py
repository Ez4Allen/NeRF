import os, json
import numpy as np
import imageio.v2 as imageio


def load_blender_data_raw(basedir):
    """
    Load the Blender (nerf_synthetic) dataset.
    
    Returns:
      imgs    : [N, H, W, C]  Raw images (RGB or RGBA depending on PNG)
      poses   : [N, 4, 4]     Camera-to-world matrices (c2w)
      hwf     : [H, W, focal] Intrinsics (pixel units)
      i_split : [train_ids, val_ids, test_ids]
    """

    splits = ['train', 'val', 'test']

    # Load JSON metadata for each split
    metas = {
        s: json.load(open(os.path.join(basedir, f"transforms_{s}.json"), "r"))
        for s in splits
    }

    all_imgs, all_poses, counts = [], [], [0]

    # Load every image and its corresponding c2w pose
    for s in splits:
        meta = metas[s]
        imgs, poses = [], []

        for frame in meta["frames"]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")

            # Load image (RGB or RGBA) and normalize to [0,1]
            im = imageio.imread(fname).astype(np.float32) / 255.0
            imgs.append(im)

            # Load camera-to-world transform matrix
            poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

        imgs  = np.stack(imgs,  axis=0)   # [Ns, H, W, C]
        poses = np.stack(poses, axis=0)   # [Ns, 4, 4]

        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # Concatenate train/val/test into one array each
    imgs  = np.concatenate(all_imgs,  axis=0)
    poses = np.concatenate(all_poses, axis=0)

    # Camera intrinsics (same for all frames)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas["train"]["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Split indices for train / val / test
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    return imgs, poses, [H, W, focal], i_split
