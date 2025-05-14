# import matplotlib
# matplotlib.use('TkAgg')  # or use 'QtAgg' if you have Qt installed
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os
# import h5py
# def load_mat_variable(filepath, candidates):
#     """自动处理 MATLAB v7.2 和 v7.3 的 .mat 文件读取，并返回匹配变量"""
#     ext = os.path.splitext(filepath)[1]
#     if ext != '.mat':
#         raise ValueError("Only .mat files supported")
#
#     mat = {}
#     try:
#         import scipy.io as sio
#         mat = sio.loadmat(filepath)
#         print(f"Loaded (v7.2 or earlier) file: {filepath}")
#         print("Available keys:", mat.keys())
#     except NotImplementedError:
#         print(f"Falling back to h5py for v7.3+ file: {filepath}")
#         import h5py
#         with h5py.File(filepath, 'r') as f:
#             print("Available keys:", list(f.keys()))
#             for key in f.keys():
#                 mat[key] = np.array(f[key]).T  # MATLAB 是列主序，需转置
#
#     for key in candidates:
#         if key in mat:
#             print(f"Found variable: {key}")
#             return mat[key]
#
#     raise ValueError(f"None of {candidates} found in {filepath}")
#
# def visualize_pet_data(activity_path, sinogram_path=None, param_path=None):
#     # Load and process activity map
#     # activity = load_mat_variable(activity_path, ['x_true', 'image', 'frame_img', 'dynamic_img'])
#     activity = load_mat_variable(activity_path, ['tracer', 'x_true', 'image', 'frame_img', 'dynamic_img'])
#     # Handle up to 4D activity data
#     if activity.ndim == 4:
#         print(f"Activity shape {activity.shape}, slicing to (H, W) by [:,:,0,0]")
#         activity = activity[:, :, 0, 0]  # 取第一个时间帧和第一个ROI
#     elif activity.ndim == 3:
#         activity = activity[:, :, activity.shape[2] // 2]
#     activity_display = activity / np.max(activity)
#
#     # Load sinogram if provided
#     sinogram = load_mat_variable(sinogram_path, ['sinogram', 'sino', 'y']) if sinogram_path else None
#
#     # Load parameter map if provided
#     # param_map = load_mat_variable(param_path, ['K', 'param_map', 'true_K', 'K_true']) if param_path else None
#     param_map = load_mat_variable(param_path, ['KP', 'K', 'param_map', 'true_K', 'K_true']) if param_path else None
#     # Plotting
#     fig_count = 1 + int(sinogram is not None) + int(param_map is not None)
#     plt.figure(figsize=(5 * fig_count, 5))
#
#     plot_id = 1
#     plt.subplot(1, fig_count, plot_id)
#     plt.imshow(activity_display, cmap='hot')
#     plt.title("PET Activity Map")
#     plt.colorbar()
#     plot_id += 1
#
#     # if sinogram is not None:
#     #     plt.subplot(1, fig_count, plot_id)
#     #     plt.imshow(sinogram, cmap='gray', aspect='auto')
#     #     plt.title("PET Sinogram")
#     #     plt.colorbar()
#     #     plot_id += 1
#     if sinogram is not None:
#         if sinogram.ndim == 4:
#             print(f"Sinogram shape {sinogram.shape}, slicing to [:,:,0,0]")
#             sinogram = sinogram[:, :, 0, 0]  # 取第一个时间帧和第一个 ROI / 样本
#         elif sinogram.ndim == 3:
#             sinogram = sinogram[:, :, sinogram.shape[2] // 2]  # 取中间帧
#
#         plt.subplot(1, fig_count, plot_id)
#         plt.imshow(sinogram, cmap='gray', aspect='auto')
#         plt.title("PET Sinogram")
#         plt.colorbar()
#         plot_id += 1
#
#     if param_map is not None and param_map.ndim == 3:
#         for i, name in zip(range(min(3, param_map.shape[2])), ['K1', 'k2', 'k3']):
#             plt.figure()
#             plt.imshow(param_map[:, :, i], cmap='viridis')
#             plt.title(f"{name} Parameter Map")
#             plt.colorbar()
#
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Visualize PET Activity, Sinogram and Parameter Map")
#
#     parser.add_argument('--activity_path', type=str,
#                         default="/Users/linyuxuan/workSpace/DPI-main/dataset/brain128_tumor_activity_map/brain128_tumor_FDG_K1_40min.mat",
#                         help='Path to activity map .mat file')
#     parser.add_argument('--sinogram_path', type=str,
#                         default="/Users/linyuxuan/workSpace/DPI-main/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min.mat",
#                         help='Path to sinogram .mat file')
#     parser.add_argument('--param_path', type=str,
#                         default="/Users/linyuxuan/workSpace/DPI-main/dataset/brain128_tumor_true_parameter_map/brain128_tumor_true_parameter_map_KK1.mat",
#                         help='Path to true parameter map .mat file')
#
#     args = parser.parse_args()
#
#     visualize_pet_data(args.activity_path, args.sinogram_path, args.param_path)

# ---------- imports ----------
import os, argparse, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import scipy.io, h5py
from scipy.sparse import csc_matrix
from pathlib import Path

def load_mat(fp: str, key: str):
    """读取 v7 / v7.3 .mat"""
    try:
        return scipy.io.loadmat(fp)[key]
    except NotImplementedError:
        with h5py.File(fp, 'r') as f:
            return np.array(f[key]).T

mat = scipy.io.loadmat(r"E:\Code\fessler模拟sinogram\zubal\G_system_matrix.mat")
mat_2 = load_mat(r"E:\Code\fessler模拟sinogram\zubal\G_system_matrix.mat", "G_sparse")
print(mat['G_sparse'])
print(mat_2)
