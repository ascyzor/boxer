#!/usr/bin/env python3

# pyre-unsafe
"""
Single-image BoxerNet inference on COLMAP-reconstructed scenes.

Assumptions:
  - COLMAP sparse reconstruction is in <colmap_dir>/sparse/0/
  - Original images are in <colmap_dir>/images/
  - World coordinate system is z-up (gravity = [0, 0, -1])

Usage:
  python boxer_colmap.py --colmap_dir ~/data/mipnerf360/room --image DSCF4667.JPG
  python boxer_colmap.py --colmap_dir ~/data/mipnerf360/room --image 0
"""

import argparse
import os
import struct
import sys

import cv2
import numpy as np
import torch

from boxernet.boxernet import BoxerNet
from loaders.base_loader import BaseLoader
from owl.owl_wrapper import OwlWrapper
from utils.demo_utils import CKPT_PATH
from utils.file_io import ObbCsvWriter2
from utils.image import draw_bb3s, put_text, render_bb2
from utils.taxonomy import load_text_labels
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import pad_string, string2tensor

# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

# Number of intrinsic parameters per camera model
_COLMAP_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE: fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,   # RADIAL: f, cx, cy, k1, k2
    4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 12,  # OPENCV_FISHEYE
    6: 8,   # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 9,  # THIN_PRISM_FISHEYE
}


def _rb(f, num_bytes, fmt, endian="<"):
    return struct.unpack(endian + fmt, f.read(num_bytes))


def read_cameras_bin(path):
    """Returns {camera_id: {w, h, model_id, params}}."""
    cameras = {}
    with open(path, "rb") as f:
        (num,) = _rb(f, 8, "Q")
        for _ in range(num):
            (cam_id, model_id) = _rb(f, 8, "ii")
            (w, h) = _rb(f, 16, "QQ")
            n = _COLMAP_NUM_PARAMS.get(model_id, 0)
            params = _rb(f, 8 * n, "d" * n)
            cameras[cam_id] = {
                "w": int(w),
                "h": int(h),
                "model_id": model_id,
                "params": params,
            }
    return cameras


def read_images_bin(path):
    """Returns {image_id: {name, qvec (4,), tvec (3,), camera_id}}.

    COLMAP pose convention: T_cam_world, i.e. p_cam = R @ p_world + t.
    qvec is (qw, qx, qy, qz).
    """
    images = {}
    with open(path, "rb") as f:
        (num,) = _rb(f, 8, "Q")
        for _ in range(num):
            (image_id,) = _rb(f, 4, "i")
            qvec = np.array(_rb(f, 32, "dddd"), dtype=np.float64)   # qw qx qy qz
            tvec = np.array(_rb(f, 24, "ddd"), dtype=np.float64)    # tx ty tz
            (camera_id,) = _rb(f, 4, "i")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            # Skip 2D point observations: each is x(f64)+y(f64)+point3D_id(i64)=24 bytes
            (num_pts2d,) = _rb(f, 8, "Q")
            f.read(num_pts2d * 24)
            images[image_id] = {
                "name": name.decode("utf-8"),
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
            }
    return images


def read_points3d_bin(path):
    """Returns (N, 3) float32 array of world-frame XYZ from COLMAP points3D.bin."""
    pts = []
    with open(path, "rb") as f:
        (num,) = _rb(f, 8, "Q")
        for _ in range(num):
            _rb(f, 8, "Q")              # point3D_id
            xyz = _rb(f, 24, "ddd")     # x y z
            f.read(3)                   # r g b (uint8 × 3)
            _rb(f, 8, "d")              # reprojection error
            (track_len,) = _rb(f, 8, "Q")
            f.read(track_len * 8)       # track: image_id(i32) + point2D_idx(i32) each
            pts.append(xyz)
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def estimate_world_up(images_data):
    """Estimate the world 'up' direction from all registered COLMAP cameras.

    In OpenCV/COLMAP camera convention, Y points down, so the camera 'up'
    direction in world frame is -R_wc[:,1] (negative second column of R_world_cam).
    Averaging this over all cameras gives a robust estimate of world up.

    Returns:
        (3,) float32 numpy array, unit vector pointing 'up' in world frame.
    """
    up_dirs = []
    for img in images_data.values():
        T_wc = colmap_pose_to_T_world_cam(img["qvec"], img["tvec"])
        up_dirs.append(-T_wc.R.numpy()[:, 1])
    avg_up = np.array(up_dirs).mean(axis=0)
    return (avg_up / np.linalg.norm(avg_up)).astype(np.float32)


def rotation_to_z_up(up_world):
    """Compute R_align (3×3 float32) such that R_align @ up_world ≈ [0, 0, 1].

    Uses Rodrigues' rotation formula.
    """
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = up_world / np.linalg.norm(up_world)
    v = np.cross(up, target)
    c = float(np.dot(up, target))
    s = float(np.linalg.norm(v))
    if s < 1e-6:
        return np.eye(3, dtype=np.float32)
    v_skew = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float32,
    )
    return np.eye(3, dtype=np.float32) + v_skew + v_skew @ v_skew * (1.0 - c) / (s ** 2)


def apply_world_rotation(T_wc, sdp_w, R_align):
    """Rotate the world coordinate system by R_align.

    Transforms T_world_cam and sdp_w so that the new world frame is z-up.

    Args:
        T_wc:    PoseTW for one camera (unbatched)
        sdp_w:   (N, 3) float32 tensor of world-frame sparse points
        R_align: (3, 3) float32 numpy array

    Returns:
        (T_wc_aligned, sdp_w_aligned)
    """
    R = torch.from_numpy(R_align)
    new_T_wc = PoseTW.from_Rt(R @ T_wc.R, R @ T_wc.t)

    if sdp_w.shape[0] > 0:
        valid = ~torch.isnan(sdp_w[:, 0])
        new_sdp = sdp_w.clone()
        new_sdp[valid] = new_sdp[valid] @ R.T  # row vectors: p_new = p @ R^T
        return new_T_wc, new_sdp
    return new_T_wc, sdp_w


def colmap_pose_to_T_world_cam(qvec, tvec):
    """Convert COLMAP T_cam_world (qvec+tvec) → BoxerNet T_world_cam (PoseTW).

    COLMAP stores world-to-camera: p_cam = R_cw @ p_world + t_cw.
    BoxerNet expects world-from-camera: T_world_cam = T_cam_world^{-1}.
    """
    q = torch.from_numpy(qvec.astype(np.float32))  # (qw, qx, qy, qz)
    t = torch.from_numpy(tvec.astype(np.float32))
    T_cam_world = PoseTW.from_qt(q, t)
    return T_cam_world.inverse()


def letterbox_resize(img_rgb, target=960):
    """Resize img_rgb into a target×target square canvas without distortion.

    Scales uniformly so the longer dimension fits target, then pads the
    shorter dimension symmetrically with zeros.

    Returns:
        canvas:    (target, target, 3) uint8
        scale:     uniform scale factor applied to the original image
        pad_top:   zero-rows added above the scaled image
        pad_left:  zero-cols added left of the scaled image
    """
    h0, w0 = img_rgb.shape[:2]
    scale = target / max(w0, h0)
    new_w = round(w0 * scale)
    new_h = round(h0 * scale)
    img_scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    pad_top = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_scaled
    return canvas, scale, pad_top, pad_left


def build_camera_tw(colmap_cam, target=960):
    """Build a pinhole CameraTW for a letterboxed target×target image.

    Uses a single uniform scale factor (preserving fx ≈ fy) and offsets
    the principal point by the letterbox padding.
    Supports SIMPLE_PINHOLE (model_id=0) and PINHOLE (model_id=1).
    """
    w0, h0 = colmap_cam["w"], colmap_cam["h"]
    model_id = colmap_cam["model_id"]
    params = colmap_cam["params"]

    if model_id == 0:       # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model_id == 1:     # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model_id={model_id}. "
            "Only SIMPLE_PINHOLE (0) and PINHOLE (1) are supported."
        )

    scale = target / max(w0, h0)
    pad_top = (target - round(h0 * scale)) // 2
    pad_left = (target - round(w0 * scale)) // 2
    return BaseLoader.pinhole_from_K(
        w=target,
        h=target,
        fx=fx * scale,
        fy=fy * scale,
        cx=cx * scale + pad_left,
        cy=cy * scale + pad_top,
    )


def build_sdp(points_xyz, num_samples=10000):
    """Sample COLMAP sparse points as BoxerNet sdp_w input.

    Returns (num_samples, 3) float32 tensor, NaN-padded when fewer than num_samples points.
    """
    n = len(points_xyz)
    if n == 0:
        return torch.full((num_samples, 3), float("nan"), dtype=torch.float32)

    if n > num_samples:
        idx = np.random.choice(n, size=num_samples, replace=False)
        pts = points_xyz[idx]
    else:
        pts = points_xyz

    sdp = torch.from_numpy(pts).float()
    if sdp.shape[0] < num_samples:
        pad = torch.full(
            (num_samples - sdp.shape[0], 3), float("nan"), dtype=torch.float32
        )
        sdp = torch.cat([sdp, pad], dim=0)
    return sdp


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _jet_bgr(scores):
    if len(scores) == 0:
        return []
    vals = np.clip(np.array(scores, dtype=np.float32), 0.0, 1.0)
    u8 = (vals * 255).astype(np.uint8).reshape(1, -1)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)[0]
    return [tuple(int(c) for c in row) for row in bgr]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BoxerNet single-image 3D detection on COLMAP data"
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        required=True,
        help="COLMAP root dir (must contain sparse/0/ and images/)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="0",
        help="Image filename (e.g. DSCF4667.JPG) or 0-based integer index",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="lvisplus",
        help="Label taxonomy (e.g. lvisplus) or comma-separated custom labels",
    )
    parser.add_argument(
        "--thresh2d", type=float, default=0.5, help="OWLv2 confidence threshold"
    )
    parser.add_argument(
        "--thresh3d", type=float, default=0.2, help="BoxerNet confidence threshold"
    )
    parser.add_argument(
        "--detector_hw",
        type=int,
        default=960,
        help="Resolution OWLv2 resizes to for detection",
    )
    parser.add_argument(
        "--no_sdp", action="store_true", help="Disable COLMAP sparse point input"
    )
    parser.add_argument(
        "--no_align", action="store_true",
        help="Skip automatic world-up alignment (use if COLMAP world is already z-up)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(
            CKPT_PATH, "boxernet_hw960in4x6d768-wssxpf9p.ckpt"
        ),
        help="Path to BoxerNet checkpoint",
    )
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    # --- Device ---
    if torch.cuda.is_available() and not args.force_cpu:
        device = "cuda"
    elif torch.backends.mps.is_available() and not args.force_cpu:
        device = "mps"
    else:
        device = "cpu"
    print(f"==> Device: {device}")

    sparse_dir = os.path.join(args.colmap_dir, "sparse", "0")
    images_dir = os.path.join(args.colmap_dir, "images")

    for p in [sparse_dir, images_dir]:
        if not os.path.isdir(p):
            print(f"ERROR: directory not found: {p}")
            sys.exit(1)

    # --- Parse COLMAP data ---
    print("==> Reading COLMAP cameras...")
    cameras = read_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))

    print("==> Reading COLMAP images...")
    images_data = read_images_bin(os.path.join(sparse_dir, "images.bin"))
    sorted_images = sorted(images_data.values(), key=lambda x: x["name"])
    print(f"    {len(sorted_images)} registered images")

    # --- Select image ---
    try:
        idx = int(args.image)
        if not (0 <= idx < len(sorted_images)):
            print(f"ERROR: index {idx} out of range [0, {len(sorted_images) - 1}]")
            sys.exit(1)
        img_entry = sorted_images[idx]
    except ValueError:
        matches = [v for v in sorted_images if v["name"] == args.image]
        if not matches:
            print(f"ERROR: '{args.image}' not found in COLMAP reconstruction")
            sys.exit(1)
        img_entry = matches[0]

    img_name = img_entry["name"]
    img_stem = os.path.splitext(img_name)[0]
    print(f"==> Selected image: {img_name}")

    # --- Load and resize image to 960×960 ---
    img_path = os.path.join(images_dir, img_name)
    if not os.path.exists(img_path):
        print(f"ERROR: image file not found: {img_path}")
        sys.exit(1)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"ERROR: cv2 failed to load: {img_path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cam_entry = cameras[img_entry["camera_id"]]
    img_resized_rgb, _scale, _pad_top, _pad_left = letterbox_resize(img_rgb, target=960)
    img_tensor = BaseLoader.img_to_tensor(img_resized_rgb)  # (1, 3, 960, 960) float32
    print(f"==> Image letterboxed to 960×960 (scale={_scale:.4f}, pad_top={_pad_top}, pad_left={_pad_left})")

    # --- Camera (intrinsics with uniform scale + padding offset) ---
    cam = build_camera_tw(cam_entry, target=960)

    # --- Pose: COLMAP T_cam_world → T_world_cam ---
    T_world_cam = colmap_pose_to_T_world_cam(img_entry["qvec"], img_entry["tvec"])

    # --- Sparse depth points from COLMAP reconstruction ---
    if args.no_sdp:
        sdp_w = torch.zeros(0, 3, dtype=torch.float32)
        print("==> SDP disabled (--no_sdp)")
    else:
        print("==> Reading COLMAP sparse points...")
        points_xyz = read_points3d_bin(os.path.join(sparse_dir, "points3D.bin"))
        sdp_w = build_sdp(points_xyz, num_samples=10000)
        n_valid = int((~torch.isnan(sdp_w[:, 0])).sum())
        print(f"    {len(points_xyz)} total points, {n_valid} sampled")

    # --- World alignment: rotate COLMAP world frame to z-up for BoxerNet ---
    if not args.no_align:
        up_world = estimate_world_up(images_data)
        dot_z = float(np.dot(up_world, [0, 0, 1]))
        print(f"==> Estimated world up: [{up_world[0]:.3f}, {up_world[1]:.3f}, {up_world[2]:.3f}]  dot([0,0,1])={dot_z:.3f}")
        if abs(dot_z - 1.0) > 0.01:
            R_align = rotation_to_z_up(up_world)
            T_world_cam, sdp_w = apply_world_rotation(T_world_cam, sdp_w, R_align)
            print(f"==> Applied world rotation to align z-up (was off by {np.degrees(np.arccos(np.clip(dot_z,-1,1))):.1f}°)")
        else:
            print("==> World already z-up, no rotation needed")

    # --- Load BoxerNet ---
    print(f"==> Loading BoxerNet from {args.ckpt}")
    boxernet = BoxerNet.load_from_checkpoint(args.ckpt, device=device)
    boxernet.eval()
    print(f"    Input resolution: {boxernet.hw}×{boxernet.hw}")

    # --- Load text labels ---
    label_list = args.labels.split(",") if "," in args.labels else [args.labels]
    text_labels = load_text_labels(label_list)
    print(f"==> Text labels: {len(text_labels)} ({args.labels})")

    # --- Load OWLv2 ---
    print("==> Loading OWLv2...")
    # precision=None: auto-detect bfloat16 on CUDA, matching run_boxer.py behaviour.
    # Passing the default "float32" causes a dtype mismatch when the cached text
    # embeddings were previously saved in bfloat16.
    owl = OwlWrapper(
        device,
        text_prompts=text_labels,
        min_confidence=args.thresh2d,
        precision=None,
    )

    # --- Output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2D Detection ---
    print("==> Running OWLv2...")
    img_255 = img_tensor.clone() * 255.0
    bb2d, scores2d, label_ints, _ = owl.forward(
        img_255,
        rotated=False,
        resize_to_HW=(args.detector_hw, args.detector_hw),
    )
    labels2d = [text_labels[i] for i in label_ints]
    print(f"    {len(bb2d)} detections (thresh={args.thresh2d})")

    # --- Write 2D visualization ---
    img_np_bgr = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2BGR)
    viz_2d = img_np_bgr.copy()
    if len(bb2d) > 0:
        texts_2d = [f"{l[:12]} {s:.2f}" for l, s in zip(labels2d, scores2d.tolist())]
        viz_2d = render_bb2(
            viz_2d,
            bb2d,
            rotated=False,
            texts=texts_2d,
            clr=_jet_bgr(scores2d.tolist()),
        )
    put_text(viz_2d, f"2D: OWLv2 {args.detector_hw}x{args.detector_hw} | {len(bb2d)} dets", scale=0.6, line=0)
    put_text(viz_2d, img_name, scale=0.4, line=2)
    out_2d = os.path.join(args.output_dir, f"{img_stem}_2d.png")
    cv2.imwrite(out_2d, viz_2d)
    print(f"==> Saved 2D PNG: {out_2d}")

    if len(bb2d) == 0:
        print("==> No 2D detections — skipping BoxerNet.")
        return

    # --- BoxerNet 3D inference ---
    print("==> Running BoxerNet...")
    sem_name_to_id = {label: i for i, label in enumerate(text_labels)}

    datum = {
        "img0": img_tensor,
        "cam0": cam,
        "T_world_rig0": T_world_cam,
        "sdp_w": sdp_w,
        "bb2d": bb2d,
        "time_ns0": torch.tensor(0, dtype=torch.int64),
        "rotated0": torch.tensor([0]),
    }

    if device == "mps":
        outputs = boxernet.forward(datum)
    else:
        precision = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        with torch.autocast(device_type=device, dtype=precision):
            outputs = boxernet.forward(datum)

    obb_pr_w = outputs["obbs_pr_w"].cpu()[0]

    # Assign semantic IDs from OWLv2 label strings
    assert len(obb_pr_w) == len(labels2d), (
        f"OBB count {len(obb_pr_w)} != label count {len(labels2d)}"
    )
    sem_ids = torch.zeros(len(labels2d), dtype=torch.int32)
    for i, label in enumerate(labels2d):
        if label not in sem_name_to_id:
            sem_name_to_id[label] = len(sem_name_to_id)
        sem_ids[i] = sem_name_to_id[label]
    obb_pr_w.set_sem_id(sem_ids)
    sem_id_to_name = {v: k for k, v in sem_name_to_id.items()}

    # Filter by 3D confidence and combine with 2D scores
    scores3d = obb_pr_w.prob.squeeze(-1).clone()
    keepers = scores3d >= args.thresh3d
    obb_pr_w = obb_pr_w[keepers].clone()
    scores3d = scores3d[keepers].clone()
    labels3d = [labels2d[i] for i in range(len(labels2d)) if keepers[i]]
    obb_pr_w.set_prob((scores2d[keepers] + scores3d) / 2.0)
    print(f"    {obb_pr_w.shape[0]} detections after thresh3d={args.thresh3d}")

    # Embed text labels in OBBs
    if len(labels3d) > 0:
        text_data = torch.stack(
            [string2tensor(pad_string(lab, max_len=128)) for lab in labels3d]
        )
        obb_pr_w.set_text(text_data)

    # --- CSV output ---
    csv_path = os.path.join(args.output_dir, f"{img_stem}_3dbbs.csv")
    writer = ObbCsvWriter2(csv_path)
    writer.write(obb_pr_w, timestamps_ns=0, sem_id_to_name=sem_id_to_name)
    writer.close()
    print(f"==> Saved CSV: {csv_path}")

    # --- Write 3D visualization ---
    viz_3d = img_np_bgr.copy()
    if len(labels3d) > 0:
        texts_3d = [
            f"{l[:12]} {s:.2f}" for l, s in zip(labels3d, scores3d.tolist())
        ]
        viz_3d = draw_bb3s(
            viz=viz_3d,
            T_world_rig=T_world_cam.float(),
            cam=cam.float(),
            obbs=obb_pr_w,
            already_rotated=False,
            rotate_label=False,
            colors=_jet_bgr(scores3d.tolist()),
            texts=texts_3d,
        )
    put_text(viz_3d, f"3D: Boxer 960x960 | {obb_pr_w.shape[0]} dets", scale=0.6, line=0)
    put_text(viz_3d, img_name, scale=0.4, line=2)
    out_3d = os.path.join(args.output_dir, f"{img_stem}_3d.png")
    cv2.imwrite(out_3d, viz_3d)
    print(f"==> Saved 3D PNG: {out_3d}")


if __name__ == "__main__":
    main()