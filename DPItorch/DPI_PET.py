#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPI_PET.py  – Deep Probabilistic Imaging on PET (single-scan training)

运行示例:
python DPI_PET.py \
  --activity_path  /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_activity_map/brain128_tumor_FDG_K1_40min.mat \
  --sinogram_path  /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min.mat \
  --gmat_path      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/G_system_matrix.mat \
  --rmat_path      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/r_noise_map.mat \
  --save_dir       /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/DPItorch/checkpoint/pet \
  --model_form     realnvp
"""

# ---------- imports ----------
import os, argparse, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import scipy.io, h5py
from scipy.sparse import csc_matrix
from pathlib import Path
from tqdm import trange
import time

from generative_model import realnvpfc_model, glow_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Torch version:", torch.__version__)
print("Device selected:", device)
if device.type == "cuda":
    print("CUDA device name :", torch.cuda.get_device_name(0))
else:
    print("CUDA not available, using CPU -- slower training")

# ---------- utils ----------
# def load_mat(fp: str, key: str):
#     """读取 v7 / v7.3 .mat"""
#     try:
#         mat = scipy.io.loadmat(fp)[key]
#         if key not in mat:
#             raise KeyError(f"[Matlab Matrix Load Error]: Key '{key}' not found in {fp}. Keys founded: {list(mat.keys())}")
#         return mat[key]
#     except NotImplementedError:
#         with h5py.File(fp, 'r') as f:
#             return np.array(f[key]).T
def load_mat(fp: str, key: str):
    """读取 v7 / v7.3 .mat"""
    try:
        return scipy.io.loadmat(fp)[key]
    except NotImplementedError:
        with h5py.File(fp, 'r') as f:
            return np.array(f[key]).T

def save_img(arr, path, cmap='hot'):
    arr_n = (arr - arr.min()) / (np.ptp(arr) + 1e-8)
    plt.imshow(arr_n, cmap=cmap); plt.axis('off'); plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0); plt.close()

def poisson_nll(lam, y, eps=1e-8):
    lam = torch.clamp(lam, min=eps)
    return (lam - y * torch.log(lam)).mean()

def loss_tv(x):
    """isotropic TV for batch images"""
    dy = torch.abs(x[:, :-1, :] - x[:, 1:, :]).mean()
    dx = torch.abs(x[:, :, :-1] - x[:, :, 1:]).mean()
    return dx + dy

def loss_l1(x):
    return torch.mean(torch.abs(x))

class ImgLogScale(nn.Module):
    """learnable global scale   image = softplus(raw) * exp(log_scale)"""
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor([np.log(init_scale)], dtype=torch.float32))

    def forward(self):
        return self.log_scale

# ---------- CLI ----------
parser = argparse.ArgumentParser("Deep Probabilistic Imaging Trainer for PET")
parser.add_argument("--activity_path", default=r"/gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_activity_map/brain128_tumor_FDG_K1_40min.mat")
parser.add_argument("--sinogram_path", default=r"/gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min.mat")
parser.add_argument("--gmat_path",     default=r"/gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/G_system_matrix.mat", help="MAT file with G_sparse")
parser.add_argument("--rmat_path",     default=r"/gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/r_noise_map.mat", help="MAT file with ri")
parser.add_argument("--save_dir",      default=r"/gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/DPItorch/checkpoint/pet")

parser.add_argument("--frame_idx", type=int, default=10, help="time frame index")
parser.add_argument("--roi_idx",   type=int, default=30, help="slice/ROI index")

# Flow model
parser.add_argument("--model_form", choices=["realnvp", "glow"], default="realnvp")
parser.add_argument("--n_flow",    type=int, default=8)
parser.add_argument("--n_block",   type=int, default=4)  # for Glow
parser.add_argument("--latent_dim",type=int, default=128)

# Training hyper-params
parser.add_argument("--n_epoch", type=int, default=100000)
parser.add_argument("--n_batch", type=int, default=64)
parser.add_argument("--lr",      type=float, default=1e-4)
parser.add_argument("--logdet",  type=float, default=1.0)
parser.add_argument("--tv",      type=float, default=1e3)
parser.add_argument("--l1",      type=float, default=0.0)
parser.add_argument("--clip",    type=float, default=1e-2)
parser.add_argument("--pretrained", default="", help="(optional) pre-trained generator .pth")
args = parser.parse_args()

Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# ---------- 1. 读取 PET 数据 (activity & sinogram) ----------
act4d  = load_mat(args.activity_path,  'tracer')      # (H,W,S,T)
sino4d = load_mat(args.sinogram_path, 'sinogram')     # (nb,na,S,T)

print("✔ activity shape:", act4d.shape)   # 应为 (128,128,60,18)
print("✔ sinogram shape:", sino4d.shape)  # 应为 (128,160,60,18)

H, W, S, T = act4d.shape
frame = np.clip(args.frame_idx, 0, T-1)
roi   = np.clip(args.roi_idx,   0, S-1)

print("====== frame:", frame, ", roi:", roi,"======")

# ground-truth activity (仅用于可视化对比)
act_gt = act4d[:, :, roi, frame]                      # (H,W)

# 观测 sinogram → 展平向量 y
y_vec = torch.tensor(
    sino4d[:, :, roi, frame].T.flatten(),
    dtype=torch.float32, device=device)               # (M,)

# ---------- 2. 系统矩阵 G_sparse & 背景噪声 r ----------
Gsp = csc_matrix(load_mat(args.gmat_path, 'G_sparse'))

print("✔ G shape:", Gsp.shape, "| non-zeros:", Gsp.nnz,
      f"({Gsp.nnz / (Gsp.shape[0] * Gsp.shape[1]):.2%} non zeros)")

coo = Gsp.tocoo()
A_torch = torch.sparse_coo_tensor(
    np.vstack((coo.row, coo.col)), coo.data, Gsp.shape,
    dtype=torch.float32, device=device).coalesce()     # (M,N)

print("✔ torch sparse A:", A_torch.shape, "| dtype:", A_torch.dtype)

ri_slice = load_mat(args.rmat_path, 'ri')[:, :, roi, frame]  # (128,160)
r_vec = torch.tensor(ri_slice.T.flatten(), dtype=torch.float32, device=device)


print("✔ noise r shape:", r_vec.shape)
# print(r_vec)
N_pix = H * W          # 每幅图展平后维度

# ---------- 3. 构造 / 加载 Flow 生成器 ----------
start_time = time.time()

if args.model_form == 'realnvp':
    Gnet = realnvpfc_model.RealNVP(N_pix, args.n_flow, affine=True).to(device)
else:
    z_shapes = glow_model.calc_z_shapes(1, H, args.n_flow, args.n_block)
    Gnet = glow_model.Glow(1, args.n_flow, args.n_block, affine=True).to(device)

elapsed = time.time() - start_time
print(f"✔ Flow model form: {args.model_form} | Built in {elapsed:.3f} seconds")

if args.pretrained:
    Gnet.load_state_dict(torch.load(args.pretrained, map_location=device))
    print("Pre-trained weights loaded:", args.pretrained)

# ---------- 4. 可学习全局缩放因子 ----------
init_scale = np.mean(act_gt)
logscale = ImgLogScale(init_scale).to(device)

print("[Logscale] Init scale =", init_scale)
# print("[Logscale] Initial factor =", torch.exp(logscale()).item())

# ---------- 5. 优化器 ----------
optimizer = optim.Adam(list(Gnet.parameters()) + list(logscale.parameters()),
                       lr=args.lr)

# n_params = sum(p.numel() for p in Gnet.parameters()) + sum(p.numel() for p in logscale.parameters())
# print(f"[Optimizer] Total trainable parameters: {n_params}")
# print(f"[Optimizer] Learning rate: {args.lr}, clip: {args.clip}")

logdet_w = args.logdet / N_pix
tv_w     = args.tv / N_pix
l1_w     = args.l1 / N_pix

loss_rec = {'total': [], 'data': [], 'logdet': [], 'tv': [], 'l1': []}

# ---------- 6. 训练循环 ----------
tick = time.time()  # 起始时间

for epoch in trange(args.n_epoch, desc="Training Epoch"):
    # 1) 采 z
    if args.model_form == 'realnvp':
        z = torch.randn(args.n_batch, N_pix, device=device)
    else:
        z = [torch.randn(args.n_batch, *s, device=device) for s in z_shapes]

    # 2) 生成正值图像 & logdet
    img_raw, logdet = Gnet.reverse(z)                # img_raw (B,N)
    img_raw = img_raw.view(-1, H, W)
    img_pos = torch.nn.functional.softplus(img_raw) * torch.exp(logscale())

    # 3) 前向投影 λ = A·x + r   （逐 batch）
    img_vec = img_pos.view(args.n_batch, -1)
    lam_list = [torch.sparse.mm(A_torch, img_vec[b:b+1].t()).squeeze() + r_vec
                for b in range(args.n_batch)]
    lam = torch.stack(lam_list)                      # (B,M)

    # # 4) 损失
    # loss_data  = poisson_nll(lam, y_vec)
    # loss_logdet= -logdet_w * logdet.mean()
    # loss_tv    = tv_w  * loss_tv(img_pos) if tv_w>0 else 0
    # loss_l1    = l1_w  * loss_l1(img_pos) if l1_w>0 else 0
    # loss = loss_data + loss_logdet + loss_tv + loss_l1
    # 4) 损失
    data_term = poisson_nll(lam, y_vec)
    logdet_term = -logdet_w * logdet.mean()
    tv_term = tv_w * loss_tv(img_pos) if tv_w > 0 else 0
    l1_term = l1_w * loss_l1(img_pos) if l1_w > 0 else 0
    loss = data_term + logdet_term + tv_term + l1_term

    # 5) 反向
    optimizer.zero_grad()

    loss_rec['total'].append(loss.item())
    loss_rec['data'].append(data_term.item())
    loss_rec['logdet'].append(logdet_term.item())
    loss_rec['tv'].append(tv_term.item() if tv_w > 0 else 0)
    loss_rec['l1'].append(l1_term.item() if l1_w > 0 else 0)

    loss.backward()
    nn.utils.clip_grad_norm_(list(Gnet.parameters()) + list(logscale.parameters()),
                             args.clip)
    optimizer.step()

    if epoch % 100 == 0:
        elapsed = time.time() - tick
        print(f"Epoch {epoch:04d} | total {loss.item():.4e} | "
              f"data {data_term.item():.4e} | logdet {logdet_term.item():.4e} | "
              f"TV {tv_term.item() if tv_w > 0 else 0:.4e} | "
              f"L1 {l1_term.item() if l1_w > 0 else 0:.4e} | "
              f"[Time] 100 iters in {elapsed:.2f}s")
        tick = time.time()

    if epoch % 1000 == 0:  # 每1000个epoch保存一次
        checkpoint_path = f"{args.save_dir}/checkpoint_nflow{args.n_flow}_nbatch{args.n_batch}_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': Gnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logscale_state_dict': logscale.state_dict(),
            'loss': loss_rec
        }, checkpoint_path)
        print(f"✔ Checkpoint saved to {checkpoint_path}")

# try:
#     # ---------- 7. 采样 & 保存结果 ----------
#     with torch.no_grad():
#         if args.model_form == 'realnvp':
#             z_sample = torch.randn(1, N_pix, device=device)
#             x_gen, _  = Gnet.reverse(z_sample)
#         else:  # glow
#             z_sample = [torch.randn(1, *s, device=device) for s in z_shapes]
#             x_gen, _  = Gnet.reverse(z_sample)
#
#         x_sample = torch.nn.functional.softplus(x_gen.view(H, W)) * torch.exp(logscale())
#
#     x_np = x_sample.cpu().numpy()
#
#     save_img(act_gt, f"{args.save_dir}/gt_f{frame}_roi{roi}.png")
#     save_img(x_np,   f"{args.save_dir}/recon_f{frame}_roi{roi}.png")
#     print("✔ saved to", args.save_dir)
# except Exception as e:                          # ← 采样或 PNG 写入出错
#     print("[X] Sampling / PNG save failed:", e)
#
# finally:
# ---------- 8. 保存模型 ----------
checkpoint_path_final = f"{args.save_dir}/finalmodel_nflow{args.n_flow}_nbatch{args.n_batch}_epoch{args.n_epoch}.pth"
torch.save({
    'model_state_dict': Gnet.state_dict(),
    'logscale_state_dict': logscale.state_dict()
}, checkpoint_path_final)
print(f"✔ Final model saved to {checkpoint_path_final}")

np.save(os.path.join(args.save_dir, "loss_curve.npy"), loss_rec)
print("✔ Loss curve saved to", os.path.join(args.save_dir, "loss_curve.npy"))
