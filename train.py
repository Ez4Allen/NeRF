import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

from dataset import load_blender_data_raw, build_ray_bank
from model import TinyNeRF_PE
from render import sample_along_rays, volume_render, sample_pdf

def mse2psnr(x):  # x: scalar tensor
    return -10.0 * torch.log10(x)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--basedir', type=str, default='../data/nerf_synthetic/lego')
    p.add_argument('--iters', type=int, default=40000)
    p.add_argument('--B', type=int, default=2048)
    p.add_argument('--S_coarse', type=int, default=128)
    p.add_argument('--S_fine', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--down', type=int, default=2)
    p.add_argument('--pe_pos_end', type=int, default=8000)
    p.add_argument('--pe_dir_end', type=int, default=4000)
    p.add_argument('--near', type=float, default=2.0)
    p.add_argument('--far', type=float, default=6.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imgs_np, poses_np, hwf, i_split = load_blender_data_raw(args.basedir)
    H, W, focal = hwf
    imgs = torch.from_numpy(imgs_np).float().to(device)
    poses = torch.from_numpy(poses_np).float().to(device)

    if imgs.shape[-1] == 4:
        rgb, a = imgs[..., :3], imgs[..., 3:4]
        imgs = rgb * a + (1.0 - a)

    if args.down > 1:
        chw = imgs.permute(0,3,1,2)
        chw = F.interpolate(chw, size=(H//args.down, W//args.down), mode='area')
        imgs = chw.permute(0,2,3,1).contiguous()
        H //= args.down; W //= args.down; focal /= args.down

    i_train = i_split[0]
    imgs_tr = imgs[i_train]
    poses_tr = poses[i_train]

    with torch.no_grad():
        rays_o_all, rays_d_all, rgb_all = build_ray_bank(imgs_tr, poses_tr, H, W, focal)
        rays_o_all = rays_o_all.to(device)
        rays_d_all = rays_d_all.to(device)
        rgb_all    = rgb_all.to(device)

    model = TinyNeRF_PE(Lx=10, Ld=4, hidden=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_rays = rays_o_all.shape[0]
    t0 = time.time()

    for it in range(1, args.iters + 1):
        model.pe_xyz.set_window_by_iter(it, args.pe_pos_end)
        model.pe_dir.set_window_by_iter(it, args.pe_dir_end)

        idx = torch.randint(0, total_rays, (args.B,), device=device)
        rays_o = rays_o_all[idx]; rays_d = rays_d_all[idx]; target = rgb_all[idx]

        z_c, pts_c = sample_along_rays(rays_o, rays_d, args.near, args.far, S=args.S_coarse, stratified=True)
        dirs_c = F.normalize(rays_d, dim=-1)[:, None, :].expand(-1, args.S_coarse, -1)
        rgb_c, sigma_c = model(pts_c, dirs_c)
        rgb_map_c, acc_c, w_c = volume_render(rgb_c, sigma_c, z_c, rays_d, white_bkgd=True)

        with torch.no_grad():
            bins = 0.5 * (z_c[:, 1:] + z_c[:, :-1])
            w_mid = w_c[:, 1:-1, 0]
            L = min(bins.shape[-1], w_mid.shape[-1])
            z_f = sample_pdf(bins[:, :L], w_mid[:, :L], N_importance=args.S_fine, deterministic=False)

        z_all, _ = torch.sort(torch.cat([z_c, z_f], -1), -1)
        pts_all  = rays_o[:, None, :] + rays_d[:, None, :] * z_all[..., None]
        dirs_all = F.normalize(rays_d, dim=-1)[:, None, :].expand(-1, z_all.shape[1], -1)
        rgb_f, sigma_f = model(pts_all, dirs_all)
        rgb_map_f, acc_f, _ = volume_render(rgb_f, sigma_f, z_all, rays_d, white_bkgd=True)

        loss_main = F.mse_loss(rgb_map_f, target)
        loss_aux  = F.mse_loss(rgb_map_c, target)
        loss = loss_main + 0.1 * loss_aux

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it <= 10 or it % 50 == 0:
            psnr_f = mse2psnr(loss_main).item()
            psnr_c = mse2psnr(loss_aux).item()
            dt = time.time() - t0
            pos_on = int(model.pe_xyz.freq_weights.gt(0.999).sum().item())
            dir_on = int(model.pe_dir.freq_weights.gt(0.999).sum().item())
            print(f'Iter {it:5d} L={loss.item():.6f} PSNR_f={psnr_f:.2f} PSNR_c={psnr_c:.2f} '
                  f'[pos={pos_on}/{model.pe_xyz.L}, dir={dir_on}/{model.pe_dir.L}] ({dt:.1f}s)')
            t0 = time.time()

    print('Done.')

if __name__ == '__main__':
    main()
