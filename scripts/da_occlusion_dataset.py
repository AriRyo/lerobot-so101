# segment_occlusion_dataset.py
import os, sys, hashlib
from copy import deepcopy
from typing import Optional

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DONE, REWARD

# =========================
# 設定（ここだけ触れば動きます）
# =========================
repo_id = "AriRyo/pickplace-v4_black"
new_repo_id_base = "AriRyo/pickplace-v4_black_da_colored"

# デフォルト: 10%, 25%, 40%
occlusion_levels = [5, 15, 30]

# 手法: "grid", "mean_shift", "graph", "createSuperpixelSLIC"
# ※ "mean_shit" と "overley" は入力した場合に自動で補正します
methods = ["grid", "mean_shift", "graph", "createSuperpixelSLIC"]

# 変形: "fill", "overlay"
augmentations = ["fill", "overlay"]

# grid
grid_rows, grid_cols = 28, 42

# overlay
overlay_alpha_min, overlay_alpha_max = 0.35, 0.75

# mean_shift（Mean shift フィルタ + kmeans + 連結成分）
mean_shift_sp, mean_shift_sr, mean_shift_k = 12, 18, 10

# graph
graph_sigma, graph_k, graph_min_size = 0.5, 300.0, 120

# createSuperpixelSLIC
slic_region_size, slic_ruler, slic_iterations = 25, 10.0, 10

# ラベル選択時に小さすぎる領域を除外（必要なら 0 に）
min_region_area_pixels = 200

base_seed = 1234567
push_to_hub = True


# =========================
# 共通ユーティリティ
# =========================
def tensor_to_hwc_numpy(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().numpy()
    return np.transpose(a, (1, 2, 0)) if a.ndim == 3 and a.shape[0] in (1, 3) else a

def to_u8(img: np.ndarray) -> np.ndarray:
    return (img * 255.0).clip(0, 255).astype(np.uint8) if np.issubdtype(img.dtype, np.floating) else np.clip(img, 0, 255).astype(np.uint8)

def convert_display_to_dtype(img_u8: np.ndarray, orig_dtype: torch.dtype) -> np.ndarray:
    if orig_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        target = {torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64, torch.bfloat16: np.float32}[orig_dtype]
        return (img_u8.astype(np.float32) / 255.0).astype(target)
    if orig_dtype == torch.uint8:
        return img_u8.astype(np.uint8)
    if orig_dtype in (torch.int16, torch.int32, torch.int64):
        target = {torch.int16: np.int16, torch.int32: np.int32, torch.int64: np.int64}[orig_dtype]
        return img_u8.astype(target)
    return img_u8.astype(np.uint8)

def compute_cell_edges(length: int, cells: int) -> list[int]:
    edges, base, rem = [0], length // cells, length % cells
    for i in range(cells):
        edges.append(edges[-1] + base + (1 if i < rem else 0))
    return edges

def make_seed(base: int, *tokens: object) -> int:
    s = "|".join(map(str, tokens)).encode("utf-8")
    d = hashlib.sha256(s).digest()
    return (int.from_bytes(d[:8], "little") ^ (base & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF

def norm_method(m: str) -> str:
    m = m.strip()
    if m == "mean_shit":
        m = "mean_shift"
    return m

def norm_aug(a: str) -> str:
    a = a.strip()
    if a == "overley":
        a = "overlay"
    return a


# =========================
# grid（長方形セル選択 + セル単位で色）
# =========================
def choose_grid_rectangle(rows: int, cols: int, target_cells: int, rng: np.random.Generator, tries: int = 80) -> tuple[int, int, int, int]:
    target_cells = max(1, min(rows * cols, int(target_cells)))
    best, best_score = None, 10**18
    for _ in range(tries):
        h = int(rng.integers(1, rows + 1))
        w = int(np.ceil(target_cells / float(h)))
        if 1 <= w <= cols:
            score = abs(h * w - target_cells)
            if score < best_score:
                r0 = int(rng.integers(0, rows - h + 1))
                c0 = int(rng.integers(0, cols - w + 1))
                best, best_score = (r0, c0, h, w), score
                if score == 0:
                    break
    if best is None:
        h = max(1, min(rows, int(round(np.sqrt(target_cells)))))
        w = max(1, min(cols, int(np.ceil(target_cells / float(h)))))
        best = (int(rng.integers(0, rows - h + 1)), int(rng.integers(0, cols - w + 1)), h, w)
    return best  # r0,c0,h,w

def apply_grid(img_u8: np.ndarray, ratio: float, aug: str, rng: np.random.Generator) -> np.ndarray:
    out = img_u8.copy()
    h, w = out.shape[:2]
    re, ce = compute_cell_edges(h, grid_rows), compute_cell_edges(w, grid_cols)
    r0, c0, rh, cw = choose_grid_rectangle(grid_rows, grid_cols, int(round(grid_rows * grid_cols * ratio)), rng)
    a = float(rng.uniform(min(overlay_alpha_min, overlay_alpha_max), max(overlay_alpha_min, overlay_alpha_max))) if aug == "overlay" else 1.0
    for r in range(r0, r0 + rh):
        y0, y1 = re[r], re[r + 1]
        for c in range(c0, c0 + cw):
            x0, x1 = ce[c], ce[c + 1]
            color = rng.integers(0, 256, size=(3,), dtype=np.uint8)
            if aug == "fill":
                out[y0:y1, x0:x1] = color
            else:
                patch = out[y0:y1, x0:x1].astype(np.float32)
                out[y0:y1, x0:x1] = ((1.0 - a) * patch + a * color.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return out


# =========================
# セグメンテーション（labels）
# =========================
def labels_mean_shift(img_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img_u8.shape[:2]
    filtered = cv2.pyrMeanShiftFiltering(img_u8, sp=mean_shift_sp, sr=mean_shift_sr)
    Z = filtered.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cv2.setRNGSeed(int(rng.integers(0, 2**31 - 1)))
    _, lab, _ = cv2.kmeans(Z, mean_shift_k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    lab2 = lab.reshape(h, w).astype(np.int32)

    out = np.full((h, w), -1, np.int32)
    nid = 0
    for c in range(mean_shift_k):
        m = (lab2 == c).astype(np.uint8)
        n, cc = cv2.connectedComponents(m, connectivity=4)
        for i in range(1, n):
            out[cc == i] = nid
            nid += 1
    out[out < 0] = 0
    return out

def labels_graph(img_u8: np.ndarray) -> np.ndarray:
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
        raise RuntimeError("graph を使うには cv2.ximgproc.segmentation が必要です（opencv-contrib-python 相当）。")
    seg = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=graph_sigma, k=graph_k, min_size=graph_min_size)
    return seg.processImage(img_u8).astype(np.int32)

def labels_createSuperpixelSLIC(img_u8: np.ndarray) -> np.ndarray:
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("createSuperpixelSLIC を使うには cv2.ximgproc が必要です（opencv-contrib-python 相当）。")
    sp = cv2.ximgproc.createSuperpixelSLIC(img_u8, algorithm=cv2.ximgproc.SLICO, region_size=slic_region_size, ruler=slic_ruler)
    sp.iterate(slic_iterations)
    return sp.getLabels().astype(np.int32)

def choose_region_ids(labels: np.ndarray, ratio: float, rng: np.random.Generator) -> list[int]:
    total = labels.size
    target = int(round(total * ratio))
    ids, cnt = np.unique(labels, return_counts=True)
    cand = [(int(i), int(c)) for i, c in zip(ids, cnt) if int(c) >= int(min_region_area_pixels)]
    if not cand:
        cand = [(int(i), int(c)) for i, c in zip(ids, cnt)]
    rng.shuffle(cand)
    chosen, covered = [], 0
    for i, c in cand:
        chosen.append(i); covered += c
        if covered >= target:
            break
    return chosen

def apply_recolor_by_labels(img_u8: np.ndarray, labels: np.ndarray, ratio: float, aug: str, rng: np.random.Generator) -> np.ndarray:
    out = img_u8.copy()
    for rid in choose_region_ids(labels, ratio, rng):
        m = (labels == rid)
        color = rng.integers(0, 256, size=(3,), dtype=np.uint8)
        if aug == "fill":
            out[m] = color
        else:
            a = float(rng.uniform(min(overlay_alpha_min, overlay_alpha_max), max(overlay_alpha_min, overlay_alpha_max)))
            out_f = out[m].astype(np.float32)
            out[m] = ((1.0 - a) * out_f + a * color.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return out


# =========================
# データセット作成
# =========================
def load_source_dataset() -> LeRobotDataset:
    try:
        return LeRobotDataset(repo_id, video_backend="pyav")
    except (RuntimeError, ImportError) as err:
        print(f"PyAV backend failed ({err}), falling back to torchvision video_reader.")
        return LeRobotDataset(repo_id, video_backend="video_reader")

def initialize_target_datasets(ds: LeRobotDataset, modes: list[str]) -> dict[str, LeRobotDataset]:
    root_parent = ds.root.parent.parent
    out: dict[str, LeRobotDataset] = {}
    for mode in modes:
        new_repo_id = f"{new_repo_id_base}_{mode}"
        name = new_repo_id.split("/")[-1]
        root = root_parent / name
        if root.exists():
            raise FileExistsError(f"既に {root} が存在します。削除するか new_repo_id_base を変えてください。")
        root.parent.mkdir(parents=True, exist_ok=True)
        out[mode] = LeRobotDataset.create(
            repo_id=new_repo_id,
            fps=int(ds.fps),
            features=deepcopy(ds.meta.info["features"]),
            root=root,
            robot_type=ds.meta.robot_type,
            use_videos=len(ds.meta.video_keys) > 0,
        )
    return out


def main() -> None:
    ms = [norm_method(m) for m in methods]
    ag = [norm_aug(a) for a in augmentations]
    for m in ms:
        if m not in ("grid", "mean_shift", "graph", "createSuperpixelSLIC"):
            raise ValueError(f"未知の手法: {m}")
    for a in ag:
        if a not in ("fill", "overlay"):
            raise ValueError(f"未知の変形: {a}")

    ds = load_source_dataset()
    sample = ds[0]
    image_keys = [k for k in sample.keys() if k.startswith("observation.images.")]
    if not image_keys:
        raise RuntimeError("observation.images.* が見つかりません。")

    modes, mode_info = [], {}
    for m in ms:
        for a in ag:
            for p in occlusion_levels:
                mode = f"{m}_{a}_{p}"
                modes.append(mode)
                mode_info[mode] = (m, a, float(p) / 100.0)

    datasets = initialize_target_datasets(ds, modes)

    skip_keys = {"task_index", "timestamp", "episode_index", "frame_index", "index"}
    total_frames = ds.meta.total_frames
    prev_episode_idx: Optional[int] = None

    need_graph = any(m == "graph" for m in ms)
    need_slic = any(m == "createSuperpixelSLIC" for m in ms)
    if (need_graph or need_slic) and not hasattr(cv2, "ximgproc"):
        raise RuntimeError("graph / createSuperpixelSLIC を使うには opencv-contrib-python 相当が必要です（cv2.ximgproc がありません）。")
    if need_graph and not hasattr(cv2.ximgproc, "segmentation"):
        raise RuntimeError("graph を使うには cv2.ximgproc.segmentation が必要です。")

    for global_idx in range(total_frames):
        print(f"Writing frame {global_idx + 1} / {total_frames}", end="\r")
        frame = ds[global_idx]
        episode_idx = int(frame["episode_index"].item())

        if prev_episode_idx is None:
            prev_episode_idx = episode_idx
        elif episode_idx != prev_episode_idx:
            for mode in modes:
                if datasets[mode].episode_buffer["size"] > 0:
                    datasets[mode].save_episode()
            prev_episode_idx = episode_idx

        base_payload: dict[str, object] = {}
        for k, v in frame.items():
            if k == "task" or k in skip_keys or k.startswith("observation.images."):
                continue
            if isinstance(v, torch.Tensor):
                c = v.detach().cpu()
                if k in (DONE, REWARD) and c.dim() == 0:
                    c = c.unsqueeze(0)
                if k.startswith("complementary_info") and c.dim() == 0:
                    c = c.unsqueeze(0)
                base_payload[k] = c
            else:
                base_payload[k] = v

        task_value = frame["task"]
        if isinstance(task_value, torch.Tensor):
            task_value = task_value.item() if task_value.dim() == 0 else task_value.tolist()
        if not isinstance(task_value, str):
            task_value = str(task_value)

        image_tensors = {k: frame[k].detach().cpu() for k in image_keys}
        images_u8 = {k: to_u8(tensor_to_hwc_numpy(t)) for k, t in image_tensors.items()}

        # 1フレーム・1カメラあたり、labels は手法ごとに一度だけ作る
        labels_cache: dict[str, dict[str, np.ndarray]] = {k: {} for k in image_keys}
        for image_key in image_keys:
            img_u8 = images_u8[image_key]
            for m in ms:
                if m == "grid":
                    continue
                seed = make_seed(base_seed, repo_id, "labels", m, image_key, episode_idx, global_idx)
                rng = np.random.default_rng(seed)
                if m == "mean_shift":
                    labels_cache[image_key][m] = labels_mean_shift(img_u8, rng)
                elif m == "graph":
                    labels_cache[image_key][m] = labels_graph(img_u8)
                elif m == "createSuperpixelSLIC":
                    labels_cache[image_key][m] = labels_createSuperpixelSLIC(img_u8)

        for mode in modes:
            method, aug, ratio = mode_info[mode]
            payload = dict(base_payload)
            for image_key in image_keys:
                img_u8 = images_u8[image_key]
                seed = make_seed(base_seed, repo_id, new_repo_id_base, mode, image_key, episode_idx, global_idx)
                rng = np.random.default_rng(seed)

                if method == "grid":
                    edited = apply_grid(img_u8, ratio, aug, rng)
                else:
                    labels = labels_cache[image_key][method]
                    edited = apply_recolor_by_labels(img_u8, labels, ratio, aug, rng)

                payload[image_key] = convert_display_to_dtype(edited, image_tensors[image_key].dtype)

            payload["task"] = task_value
            datasets[mode].add_frame(payload)

    print()

    for mode in modes:
        if datasets[mode].episode_buffer["size"] > 0:
            datasets[mode].save_episode()
        datasets[mode].finalize()
        print(f"編集済みデータセットを {datasets[mode].root} に保存しました。")
        if push_to_hub:
            new_repo_id = f"{new_repo_id_base}_{mode}"
            print(f"{new_repo_id} を Hugging Face Hub へアップロードします...")
            datasets[mode].push_to_hub()
        else:
            print("push_to_hub=False のため、アップロードはスキップしました。")


if __name__ == "__main__":
    main()
