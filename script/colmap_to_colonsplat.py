#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colmap_to_colonsplat.py
=======================
把「COLMAP SfM 位姿 + 单目深度」形式的内镜数据转换成 ColonSplat 可直接训练的
`transforms.json` 数据集格式。

------------------------------------------------------------------------------
输入 (--src)
------------------------------------------------------------------------------
主路径 (本仓库 preprocess 流程的产物，推荐)：
    poses_meta.json            由 COLMAP SfM 生成的位姿元数据
                               (intrinsics / poses{R_c2w,t_c2w,qvec,tvec} / align.alpha /
                                final.frames 已注册帧列表)
    images/*.png               每帧 RGB 图像 (本例 1000x1000)
    points3d.ply               (可选) 稠密初始化点云，坐标系 = COLMAP 世界系 × align.alpha
    sparse/points3D.ply        (可选) 稀疏 COLMAP 点云，坐标系 = COLMAP 原始世界系
    sparse/{images,cameras}.bin(可选) 备用位姿来源

备用路径：标准 COLMAP 目录 (sparse/0|sparse 下含 images.bin / cameras.bin / points3D.bin)，
          未提供 poses_meta.json 时自动使用。

深度 (--depth-dir，默认 `<src>_C3VD/depth`)：
    <stem>.npy                每帧 float32 深度，形状 (H,W) 或 (1,H,W)
    若缺失，可先运行仓库自带的 `python pre_dam_dep.py --dataset_root <dir> --rgb_paths images`

------------------------------------------------------------------------------
输出 (--dst)  —— ColonSplat_Dataset / readColonSplatInfo 需要的布局
------------------------------------------------------------------------------
    <dst>/transforms.json
    <dst>/images/frame_XXXX.png     (软链接或拷贝；文件名必须是 "<前缀>_<数字>"，
                                     因为 get_gt_plys 会用 image_name.split('_')[1])
    <dst>/depth_dam/XXXX.npy        (H,W) float32
    <dst>/ply/pc_0001.ply           归一化坐标系下的初始化点云（名字被硬编码）

------------------------------------------------------------------------------
坐标系约定（已在本数据集上逐项数值验证，勿随意改动）
------------------------------------------------------------------------------
* `poses_meta.json` 中 `R_c2w == qvec2rotmat(qvec).T`，`t_c2w == -R_c2w @ tvec`
  （误差 <1e-15）→ 即 3DGS 的 `R`（R_w2c 的转置）与 `T`（相机中心 C），
  世界系就是 COLMAP 世界系。
* ColonSplat_Dataset.readCamerasFromTransforms 读取 transform_matrix 后会执行
  `c2w[:3, 1:3] *= -1`（OpenGL→COLMAP），再 `w2c = inv(c2w)`、`R = w2c[:3,:3].T`、
  `T = w2c[:3,3]`。
  因此写入 `transform_matrix` 时必须 **预先反向翻转第 1、2 列**：
        M[:3,:3] = R_c2w @ diag(1,-1,-1)      M[:3,3] = C
  这样 loader 里的 `*= -1` 恰好把它还原成标准的 `[R_c2w | C]`。
  脚本内的 `self_check()` 会模拟 loader 复算一遍并断言，防止写反。

------------------------------------------------------------------------------
用法
------------------------------------------------------------------------------
    /root/miniconda3/envs/env/bin/python script/colmap_to_colonsplat.py \
        --src /mnt/t-1/ERCP_Video/seq_3 \
        --dst data_custom/seq_3

    # 只生成 transforms.json / depth 的清单，不落盘大文件
    ... --dry-run

    # 用稀疏 SfM 点云（几何最可靠，但较稀）
    ... --init sparse
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMG_PREFIX = "frame"


def log(*a):
    print("[convert]", *a, flush=True)


def warn(*a):
    print("[convert][WARN]", *a, flush=True)


# ==============================================================================
# 基础 IO
# ==============================================================================
def read_ply(path, need_rgb=True):
    """返回 (xyz Nx3, rgb Nx3 float 0-255 或 None, normals Nx3 或 None)。"""
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"{path} 中没有 vertex 元素")
    v = ply["vertex"].data
    names = set(v.dtype.names)
    for ax in ("x", "y", "z"):
        if ax not in names:
            raise ValueError(f"{path} 缺少字段 {ax}")
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    rgb, nrm = None, None
    if need_rgb and {"red", "green", "blue"} <= names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64)
        if rgb.max() <= 1.0 + 1e-6:      # 少数 ply 里颜色是 0-1 浮点
            rgb *= 255.0
    if {"nx", "ny", "nz"} <= names:
        nrm = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64)
    return xyz, rgb, nrm


def write_ply(path, xyz, rgb=None, normals=None):
    """按 dataset_readers.fetchPly / storePly 的字段约定写 ply。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, np.float64)
    n = len(xyz)
    if normals is None:
        normals = np.zeros((n, 3), np.float64)
    if rgb is None:
        rgb = np.full((n, 3), 128.0)
    rgb = np.clip(np.round(np.asarray(rgb, np.float64)), 0, 255).astype(np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = np.ascontiguousarray(xyz).T
    arr["nx"], arr["ny"], arr["nz"] = np.ascontiguousarray(normals, np.float64).T
    arr["red"], arr["green"], arr["blue"] = rgb.T
    with open(path, "wb") as f:
        PlyData([PlyElement.describe(arr, "vertex")], text=False).write(f)
    log(f"  写入 {path}  ({n} 点)")


def qvec2rotmat(q):
    w, x, y, z = np.asarray(q, np.float64)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


# ==============================================================================
# 位姿读取
# ==============================================================================
def load_poses_from_meta(src):
    """主路径：读 poses_meta.json。"""
    meta = json.loads((src / "poses_meta.json").read_text())
    frames = list(meta["final"]["frames"])
    poses = meta["poses"]
    K = meta["intrinsics"]
    alpha = float(meta.get("align", {}).get("alpha")
                  or meta.get("scale_alpha") or 1.0)

    R_list, C_list, err = [], [], 0.0
    for f in frames:
        if f not in poses:
            raise KeyError(f"poses_meta.json 的 poses 里找不到 {f}")
        p = poses[f]
        R = np.array(p["R_c2w"], np.float64).reshape(3, 3)
        C = np.array(p["t_c2w"], np.float64).reshape(3)
        if "qvec" in p and "tvec" in p:      # 交叉校验 R = qvec2rotmat^T, C = -R@tvec
            Rw2c = qvec2rotmat(p["qvec"])
            err = max(err,
                      float(np.abs(Rw2c.T - R).max()),
                      float(np.abs(-R @ np.array(p["tvec"], np.float64) - C).max()))
        R_list.append(R)
        C_list.append(C)

    log(f"位姿来源 : poses_meta.json  ({len(frames)} 帧已注册, alpha={alpha:.8f})")
    log(f"  与 COLMAP qvec/tvec 的最大互查误差 = {err:.3e}")
    if err > 1e-6:
        warn("R_c2w/t_c2w 与 qvec/tvec 不一致！请检查 poses_meta.json 的生成方式")

    return dict(frames=frames,
                R=np.stack(R_list), C=np.stack(C_list),
                fx=float(K["fx"]), fy=float(K["fy"]),
                cx=float(K["cx"]), cy=float(K["cy"]),
                W=int(K["width"]), H=int(K["height"]),
                alpha=alpha, src_kind="poses_meta.json")


def load_poses_from_colmap(src):
    """备用路径：直接读标准 COLMAP 二进制模型。"""
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

    sp = src / "sparse"
    if (sp / "0").is_dir() and (sp / "0" / "images.bin").exists():
        sp = sp / "0"
    extr = read_extrinsics_binary(str(sp / "images.bin"))
    intr = read_intrinsics_binary(str(sp / "cameras.bin"))
    cam = next(iter(intr.values()))
    m, par = cam.model, list(cam.params)

    if m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = par[0]; cx, cy = par[1], par[2]
        dist = par[3:]
    elif m in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = par[:4]
        dist = par[4:]
    else:
        raise ValueError(f"未支持的 COLMAP 相机模型 {m}")
    if any(abs(d) > 1e-6 for d in dist):
        warn(f"相机模型 {m} 带畸变 {dist}，本脚本不做去畸变，"
             f"请使用 undistorted 模型或让 COLMAP 输出 SIMPLE_RADIAL(dist=0)")
    if abs(fx - fy) > 1e-3:
        warn(f"fx({fx:.4f}) != fy({fy:.4f})，将使用 fx 作为 fovx 依据")

    items = sorted(extr.values(), key=lambda e: e.name)
    frames, R_list, C_list = [], [], []
    for e in items:
        Rc2w = qvec2rotmat(e.qvec).T
        C = -Rc2w @ np.array(e.tvec, np.float64)
        frames.append(e.name)
        R_list.append(Rc2w)
        C_list.append(C)
    log(f"位姿来源 : COLMAP bin  ({len(frames)} 帧, model={m})")
    return dict(frames=frames, R=np.stack(R_list), C=np.stack(C_list),
                fx=fx, fy=fy, cx=cx, cy=cy,
                W=int(cam.width), H=int(cam.height),
                alpha=1.0, src_kind="colmap_bin")


def load_poses(src):
    if (src / "poses_meta.json").exists():
        return load_poses_from_meta(src)
    for cand in (src / "sparse" / "0" / "images.bin", src / "sparse" / "images.bin"):
        if cand.exists():
            return load_poses_from_colmap(src)
    raise FileNotFoundError(
        f"{src} 下既没有 poses_meta.json，也没有 sparse[/0]/images.bin")


# ==============================================================================
# 几何工具
# ==============================================================================
def project(pts, R, C, intr):
    """COLMAP/3DGS 约定投影: X_cam = R^T (X - C)；返回 (u, v, z)。"""
    X = (R.T @ (pts.T - C.reshape(3, 1))).T
    z = X[:, 2]
    zc = np.clip(z, 1e-6, None)
    u = intr["fx"] * X[:, 0] / zc + intr["cx"]
    v = intr["fy"] * X[:, 1] / zc + intr["cy"]
    return u, v, z


def visible_mask(u, v, z, intr, margin=1.0):
    return ((z > 1e-3)
            & (u >= margin) & (u < intr["W"] - margin)
            & (v >= margin) & (v < intr["H"] - margin))


def count_views(pts, R, C, intr, frame_ids, margin=1.0):
    vis = np.zeros(len(pts), np.int32)
    for i in frame_ids:
        u, v, z = project(pts, R[i], C[i], intr)
        vis += visible_mask(u, v, z, intr, margin).astype(np.int32)
    return vis


def sample_colors(pts, rgb_src, R, C, intr, img_dir, frames, frame_ids, vis):
    """把点云投影到各帧取像素颜色，取中位数作为初始颜色（无源颜色时使用）。"""
    if not img_dir.exists():
        return None
    acc = np.zeros((len(pts), 3), np.float64)
    cnt = np.zeros(len(pts), np.float64)
    for i in frame_ids:
        u, v, z = project(pts, R[i], C[i], intr)
        m = visible_mask(u, v, z, intr)
        if m.sum() == 0:
            continue
        img = np.asarray(Image.open(img_dir / frames[i]).convert("RGB"), np.float64)
        ui = np.clip(u[m].astype(int), 0, intr["W"] - 1)
        vi = np.clip(v[m].astype(int), 0, intr["H"] - 1)
        acc[m] += img[vi, ui]
        cnt[m] += 1
    ok = cnt > 0
    out = np.full((len(pts), 3), 128.0)
    out[ok] = acc[ok] / cnt[ok, None]
    if rgb_src is not None:             # 有源颜色时只填补采样到的位置
        out = rgb_src.copy()
    return out


# ==============================================================================
# loader 行为模拟 / 自检
# ==============================================================================
def transform_matrix_from(R, C):
    """按 ColonSplat_Dataset 的约定生成 transform_matrix（OpenGL 侧，预翻转）。"""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R @ np.diag([1.0, -1.0, -1.0])
    M[:3, 3] = C
    return M


def simulate_loader(M):
    """复刻 ColonSplat_Dataset.readCamerasFromTransforms 的数学运算。"""
    c2w = np.array(M, np.float64)
    c2w[:3, 1:3] *= -1                       # OpenGL -> COLMAP
    w2c = np.linalg.inv(c2w)
    return np.transpose(w2c[:3, :3]), w2c[:3, 3]


def self_check(R, C, n=5):
    """验证写入的 transform_matrix 经 loader 解析后能还原 (R, tvec)。"""
    worst = 0.0
    for i in np.linspace(0, len(R) - 1, min(n, len(R))).astype(int):
        R_hat, T_hat = simulate_loader(transform_matrix_from(R[i], C[i]))
        tvec = -R[i].T @ C[i]
        worst = max(worst, float(np.abs(R_hat - R[i]).max()),
                    float(np.abs(T_hat - tvec).max()))
    if worst > 1e-9:
        raise AssertionError(f"transform_matrix 自检失败, 最大误差 {worst:.3e}")
    log(f"  transform_matrix 自检通过 (最大误差 {worst:.3e})")


# ==============================================================================
# 主流程
# ==============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="COLMAP 风格内镜数据 -> ColonSplat (transforms.json) 数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="源数据目录")
    p.add_argument("--dst", type=Path, default=None,
                   help="输出目录，默认 <repo>/data_custom/<src 名>")
    p.add_argument("--init", choices=["auto", "dense", "sparse"], default="auto",
                   help="初始化点云来源：dense=points3d.ply(alpha 坐标系)，"
                        "sparse=sparse/points3D.ply(原始 COLMAP 坐标系)")
    p.add_argument("--min-views", type=int, default=3,
                   help="点云必须被至少这么多帧看到，否则丢弃（去离群点）")
    p.add_argument("--vis-stride", type=int, default=2,
                   help="可见性统计时每隔几帧采样一次")
    p.add_argument("--depth-dir", type=Path, default=None,
                   help="深度 npy 目录，默认 <src>_C3VD/depth")
    p.add_argument("--no-depth", action="store_true", help="不生成 depth_dam（训练会缺深度监督）")
    p.add_argument("--no-normalize", action="store_true",
                   help="不做尺度归一化（默认归一化到 nerf radius=1.1）")
    p.add_argument("--scale", type=float, default=None,
                   help="手动指定归一化世界缩放（覆盖自动计算）")
    p.add_argument("--test-every", type=int, default=5, help="仅用于文档说明，不写入文件")
    p.add_argument("--copy-images", action="store_true",
                   help="拷贝而不是软链接图像")
    p.add_argument("--link-gt-ply", action="store_true",
                   help="额外生成 ply/pc_<stem>.ply 软链接指向 pc_0001.ply。"
                        "只有 test_geometry.py 需要逐帧 gt_plys；"
                        "本数据没有逐帧 GT mesh，该指标本身无意义，仅在不想让它报错时使用")
    p.add_argument("--relative-paths", action="store_true",
                   help="transforms.json 里用相对路径（注意 loader 会二次拼接，慎用）")
    p.add_argument("--dry-run", action="store_true", help="只统计与校验，不写文件")
    return p


def main():
    args = build_parser().parse_args()
    src = args.src.resolve()
    dst = (args.dst or (REPO_ROOT / "data_custom" / src.name)).resolve()

    if not src.is_dir():
        raise SystemExit(f"源目录不存在: {src}")

    log(f"源  : {src}")
    log(f"目标: {dst}")

    # ---------------- 1. 位姿 ----------------
    P = load_poses(src)
    frames, R, C = P["frames"], P["R"], P["C"]
    intr = dict(fx=P["fx"], fy=P["fy"], cx=P["cx"], cy=P["cy"], W=P["W"], H=P["H"])
    fovx = 2.0 * np.arctan(intr["W"] / (2.0 * intr["fx"]))
    log(f"内参 : fx={intr['fx']:.4f} fy={intr['fy']:.4f} cx={intr['cx']:.3f} "
        f"cy={intr['cy']:.3f} {intr['W']}x{intr['H']}  ->  camera_angle_x={fovx:.8f} rad")

    img_dir = src / "images"
    have_imgs = [f for f in frames if (img_dir / f).exists()]
    if len(have_imgs) != len(frames):
        missing = [f for f in frames if f not in set(have_imgs)][:5]
        raise FileNotFoundError(f"{img_dir} 缺少 {len(frames) - len(have_imgs)} 张图像，"
                                f"例如 {missing}")

    # ---------------- 2. 初始化点云 ----------------
    log(f"初始化点云来源: {args.init}")
    init_kind, pts, cols, nrms = None, None, None, None

    def _try_dense():
        p = src / "points3d.ply"
        if not p.exists():
            return None
        xyz, rgb, nrm = read_ply(p)
        return xyz / P["alpha"], rgb, nrm      # alpha 坐标系 -> COLMAP 原始坐标系

    def _try_sparse():
        for p in (src / "sparse" / "points3D.ply", src / "poses_work" / "sparse_points.npy"):
            if p.exists():
                if p.suffix == ".npy":
                    return np.load(p).astype(np.float64), None, None
                xyz, rgb, nrm = read_ply(p)
                return xyz, rgb, nrm
        return None

    wants = {"auto": [_try_dense, _try_sparse],
             "dense": [_try_dense],
             "sparse": [_try_sparse]}[args.init]
    for fn in wants:
        got = fn()
        if got is not None:
            pts, cols, nrms = got
            init_kind = fn.__name__.replace("_try_", "")
            break
    if pts is None:
        raise FileNotFoundError("找不到可用的初始化点云 (points3d.ply / sparse/points3D.ply)")

    # 稠密点云建在 alpha 缩放坐标系上，若 alpha != 1 需换算；换错会整体投影失败
    if init_kind == "dense" and abs(P["alpha"] - 1.0) > 1e-9:
        log(f"  {src.name}/points3d.ply 位于 alpha 缩放坐标系, "
            f"已除以 alpha={P['alpha']:.8f} 换回 COLMAP 世界系")

    log(f"  原始点数 {len(pts)}")

    # ---------------- 3. 可见性过滤 ----------------
    samp = list(range(0, len(frames), max(1, args.vis_stride)))
    vis = count_views(pts, R, C, intr, samp)
    keep = vis >= max(1, args.min_views)
    log(f"  可见性过滤 (>= {args.min_views} / {len(samp)} 采样帧): "
        f"{keep.sum()}/{len(pts)} 保留 ({100.0 * keep.mean():.1f}%)")
    if keep.sum() < 500:
        warn("过滤后点数过少，放宽 --min-views 到 1")
        keep = vis >= 1
    pts = pts[keep]
    cols = cols[keep] if cols is not None else None
    nrms = nrms[keep] if nrms is not None else None

    # ---------------- 4. 场景归一化 ----------------
    if args.no_normalize:
        centroid = np.zeros(3)
        sfac = 1.0
        log("  未做归一化 (--no-normalize)")
    else:
        centroid = C.mean(axis=0)
        maxd = float(np.linalg.norm(C - centroid, axis=1).max())
        sfac = args.scale if args.scale is not None else 1.0 / max(maxd, 1e-9)
        log(f"  归一化: 相机中心质心 {np.round(centroid, 4)}  "
            f"最大半径 {maxd:.4f}  缩放 {sfac:.6f}  ->  nerf radius = {1.1 * maxd * sfac:.3f}")
    C = (C - centroid) * sfac
    pts = (pts - centroid) * sfac

    # ---------------- 5. 自检 + 统计 ----------------
    self_check(R, C)
    u, v, z = project(pts, R[0], C[0], intr)
    log(f"  第 0 帧投影: 可见 {visible_mask(u, v, z, intr).mean() * 100:.1f}%")
    log(f"  归一化后点云 bbox: {np.round(pts.min(0), 3)} .. {np.round(pts.max(0), 3)}")
    log(f"  归一化后相机中心到原点距离: "
        f"中位 {np.median(np.linalg.norm(C, axis=1)):.3f}  最大 {np.linalg.norm(C, axis=1).max():.3f}")

    # ---------------- 6. 深度 ----------------
    depth_dir = (args.depth_dir or (src.parent / f"{src.name}_C3VD" / "depth")).resolve()
    has_depth = (not args.no_depth) and depth_dir.is_dir()
    if not args.no_depth:
        if has_depth:
            log(f"深度来源: {depth_dir}")
        else:
            warn(f"深度目录不存在: {depth_dir} -> 将不生成 depth_dam/，"
                 f"训练前请先跑 pre_dam_dep.py 生成单目深度")

    # ---------------- 7. 颜色 ----------------
    if cols is None:
        log("源点云无颜色，改为投影采样图像颜色 ...")
        cols = sample_colors(pts, None, R, C, intr, img_dir, frames, samp, vis)

    # ---------------- 8. 落盘 ----------------
    names = [f"{IMG_PREFIX}_{Path(f).stem}" for f in frames]

    if args.dry_run:
        log("--dry-run: 跳过所有写盘操作")
    else:
        dst.mkdir(parents=True, exist_ok=True)

        # 8.1 images/
        out_img = dst / "images"
        out_img.mkdir(exist_ok=True)
        for f, nm in zip(frames, names):
            target = out_img / f"{nm}.png"
            if target.exists() or target.is_symlink():
                target.unlink()
            if args.copy_images:
                shutil.copy2(img_dir / f, target)
            else:
                os.symlink((img_dir / f).resolve(), target)
        log(f"  写入 {out_img}  ({len(frames)} 张, "
            f"{'拷贝' if args.copy_images else '软链接'})")

        # 8.2 depth_dam/
        if has_depth:
            out_dep = dst / "depth_dam"
            out_dep.mkdir(exist_ok=True)
            n_bad = 0
            for f, nm in zip(frames, names):
                cands = [depth_dir / f"{Path(f).stem}.npy", depth_dir / f]
                src_d = next((c for c in cands if c.exists()), None)
                if src_d is None:
                    warn(f"缺深度: {f}")
                    n_bad += 1
                    continue
                d = np.load(src_d)
                d = np.squeeze(d).astype(np.float32)
                if d.ndim != 2:
                    raise ValueError(f"{src_d} squeeze 后维度为 {d.shape}")
                if d.shape != (intr["H"], intr["W"]):
                    warn(f"{src_d} 尺寸 {d.shape} != ({intr['H']},{intr['W']})，已最近邻缩放")
                    d = np.asarray(Image.fromarray(d).resize(
                        (intr["W"], intr["H"]), Image.NEAREST), np.float32)
                np.save(out_dep / f"{nm}.npy", d)
            log(f"  写入 {out_dep}  ({len(frames) - n_bad} 个, 形状 (H,W) float32)")

        # 8.3 ply/pc_0001.ply
        write_ply(dst / "ply" / "pc_0001.ply", pts, cols, nrms)

        # 8.3b 可选：逐帧 gt ply 软链接（仅 test_geometry.py 会用到）
        if args.link_gt_ply:
            n_link = 0
            for nm in names:
                lp = dst / "ply" / f"pc_{nm.split('_')[1]}.ply"
                if lp.exists() or lp.is_symlink():
                    lp.unlink()
                os.symlink("pc_0001.ply", lp)   # 相对软链接
                n_link += 1
            log(f"  写入 {dst / 'ply'} 逐帧软链接 x{n_link} -> pc_0001.ply "
                f"(注意: 并非真实逐帧 GT mesh, 几何指标无参考意义)")

        # 8.4 transforms.json
        def fmt(p):
            # 注意：不能用 Path.resolve()，那会跟随软链接回到源目录，
            #      导致文件名丢掉 "frame_" 前缀 -> get_gt_plys 里 split('_')[1] 越界
            p = os.path.abspath(str(p))
            if args.relative_paths:
                rel = os.path.relpath(p, str(dst))
                return rel.replace(os.sep, "/")
            return p

        tframes = []
        for i, (f, nm) in enumerate(zip(frames, names)):
            fr = {
                "file_path": fmt(out_img / f"{nm}.png"),
                "transform_matrix": transform_matrix_from(R[i], C[i]).tolist(),
                "time": (i / (len(frames) - 1)) if len(frames) > 1 else 0.0,
            }
            if has_depth:
                fr["depth_path"] = fmt(dst / "depth_dam" / f"{nm}.npy")
            tframes.append(fr)

        tj = {
            "camera_angle_x": float(fovx),
            "camera_angle_y": float(2.0 * np.arctan(intr["H"] / (2.0 * intr["fy"]))),
            "fl_x": intr["fx"], "fl_y": intr["fy"],
            "cx": intr["cx"], "cy": intr["cy"],
            "w": intr["W"], "h": intr["H"],
            "scale": float(sfac), "offset": (centroid * sfac).tolist(),
            "source": str(src), "source_poses": P["src_kind"],
            "init_cloud": init_kind, "num_init_points": int(len(pts)),
            "frames": tframes,
        }
        with open(dst / "transforms.json", "w") as fp:
            json.dump(tj, fp, indent=1)
        log(f"  写入 {dst / 'transforms.json'}")

    # ---------------- 9. 深度方向抽查 ----------------
    if has_depth:
        cors, noc = [], 0
        for i in np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int):
            u, v, z = project(pts, R[i], C[i], intr)
            m = visible_mask(u, v, z, intr)
            if m.sum() < 50:
                continue
            dp = np.load(depth_dir / f"{Path(frames[i]).stem}.npy")
            dp = np.squeeze(dp)
            ui = np.clip(u[m].astype(int), 0, intr["W"] - 1)
            vi = np.clip(v[m].astype(int), 0, intr["H"] - 1)
            d = dp[vi, ui].astype(np.float64)
            cors.append(float(np.corrcoef(z[m], d)[0, 1]))
            noc += 1
        if cors:
            cors = np.array(cors)
            log(f"深度一致性抽查 ({noc} 帧): corr(几何 z, 深度) 中位 {np.median(cors):+.3f} "
                f"范围 [{cors.min():+.3f}, {cors.max():+.3f}]")
            if np.median(cors) < 0:
                warn("深度与几何呈负相关 -> 深度可能是视差(disparity)，训练前需取倒数！")

    # ---------------- 10. 收尾 ----------------
    if args.dry_run:
        log("dry-run 完成。")
        return

    log("")
    log("=" * 74)
    log(f"转换完成: {dst}")
    log(f"  帧数            : {len(frames)}  (train/test = "
        f"{sum(1 for i in range(len(frames)) if (i - 1) % args.test_every != 0)}/"
        f"{sum(1 for i in range(len(frames)) if (i - 1) % args.test_every == 0)})")
    log(f"  图像            : {dst / 'images'}  -> image_name = {names[0]} ...")
    log(f"  深度            : {(dst / 'depth_dam') if has_depth else '缺失'}")
    log(f"  初始化点云      : {dst / 'ply' / 'pc_0001.ply'}  ({len(pts)} 点, 来自 {init_kind})")
    log(f"  nerf radius     : 约 {1.1 if not args.no_normalize else 'N/A'} "
        f"(决定 spatial_lr_scale / 各类尺度阈值)")
    log("")
    log("下一步（注意：默认开启了稠密化，不要照抄 run_custom_ds.sh 里")
    log("        --densify_from_iter 500000 / --pruning_from_iter 500000，")
    log("        那套参数是为有 GT mesh 密初始化的数据关掉稠密化用的）：")
    log("")
    log(f"  python train.py -s {dst} --port 6019 --expname custom_ds/{dst.name} \\")
    log("      --depth_weight 0.25 --dcol_weight 0.005 --knn_weight 0.01 \\")
    log("      --col_smooth_weight 0.0001 --iterations 30000")
    log("")
    log("  渲染 / 评测：")
    log(f"  python render.py -s {dst} --model_path output/custom_ds/{dst.name} --skip_video")
    log(f"  python metrics.py --model_path output/custom_ds/{dst.name}")
    log("=" * 74)
    log("提示：scene/__init__.py 里 `args.extra_mark=\"c3vd2\"` 是硬编码的，")
    log("      但因为本数据集含 transforms.json，会正确命中 colonsplat 分支，")
    log("      无需修改该文件。")
    if not args.link_gt_ply and not args.dry_run:
        log("提示：test_geometry.py 需要逐帧 ply/pc_<stem>.ply（本数据集只有 "
            "pc_0001.ply），")
        log("      若要让它不报错可加 --link-gt-ply；但本数据没有逐帧 GT mesh，")
        log("      其几何指标本身没有参考意义。")


if __name__ == "__main__":
    main()
