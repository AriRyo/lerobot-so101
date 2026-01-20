import os
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DONE, REWARD


# =========================
# 設定
# =========================
repo_id = "AriRyo/pickplace-v4_black"

# 出力先のベース（各モード名が後ろに付きます）
# 例: AriRyo/pickplace-v4_black_grid_fill_10 など
new_repo_id_base = "AriRyo/pickplace-v4_black"

# 10%, 25%, 40% を作る
occlusion_levels = [10, 25, 40]

# グリッド分割
grid_rows = 28
grid_cols = 42

# 半透明オーバーレイのアルファ範囲
overlay_alpha_min = 0.35
overlay_alpha_max = 0.75

# 再現性のための固定シード（これを変えると生成結果も変わります）
base_seed = 1234567

# Hugging Face Hub へアップロードするか
push_to_hub = True


# =========================
# 共通ユーティリティ
# =========================
def tensor_to_hwc_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def to_display_image(np_hwc: np.ndarray) -> np.ndarray:
    # float(0..1想定) を uint8(0..255) に寄せる
    if np.issubdtype(np_hwc.dtype, np.floating):
        return (np_hwc * 255.0).clip(0, 255).astype(np.uint8)
    return np.clip(np_hwc, 0, 255).astype(np.uint8)


def convert_display_to_dtype(display_img: np.ndarray, orig_dtype: torch.dtype) -> np.ndarray:
    if orig_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        target_dtype = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.bfloat16: np.float32,
        }[orig_dtype]
        converted = display_img.astype(np.float32) / 255.0
        return converted.astype(target_dtype)
    if orig_dtype == torch.uint8:
        return display_img.astype(np.uint8)
    if orig_dtype in (torch.int16, torch.int32, torch.int64):
        target_dtype = {
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }[orig_dtype]
        return display_img.astype(target_dtype)
    return display_img.astype(np.uint8)


def compute_cell_edges(length: int, cells: int) -> list[int]:
    edges = [0]
    base = length // cells
    remainder = length % cells
    for idx in range(cells):
        step = base + (1 if idx < remainder else 0)
        edges.append(edges[-1] + step)
    return edges


def parse_mode_name(mode: str) -> tuple[str, int]:
    parts = mode.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid mode name: {mode}")
    level = int(parts[-1])
    base = "_".join(parts[:-1])
    return base, level


def make_deterministic_seed(base_seed_value: int, *tokens: object) -> int:
    """
    再現性のために、入力トークン列から 64bit シードを作る。
    """
    text = "|".join(str(t) for t in tokens)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed64 = int.from_bytes(digest[:8], "little", signed=False)
    seed64 ^= (base_seed_value & 0xFFFFFFFFFFFFFFFF)
    return seed64


@dataclass(frozen=True)
class GridRectangle:
    r0: int
    c0: int
    h: int
    w: int


def choose_grid_rectangle(
    rows: int,
    cols: int,
    target_cells: int,
    rng: np.random.Generator,
    tries: int = 80,
) -> GridRectangle:
    """
    target_cells に近い面積の、連結したグリッド長方形を選ぶ（近似）。
    """
    target_cells = max(1, min(rows * cols, int(target_cells)))

    best_rect: Optional[GridRectangle] = None
    best_score = 10**18

    # 複数候補を試して、target に近いものを採用
    for _ in range(tries):
        # 高さを先にランダムに決めて幅を決める（幅が cols を超えたらやり直し）
        h = int(rng.integers(1, rows + 1))
        w = int(np.ceil(target_cells / float(h)))
        if w < 1 or w > cols:
            continue

        area = h * w
        score = abs(area - target_cells)
        if score < best_score:
            r0 = int(rng.integers(0, rows - h + 1))
            c0 = int(rng.integers(0, cols - w + 1))
            best_rect = GridRectangle(r0=r0, c0=c0, h=h, w=w)
            best_score = score
            if best_score == 0:
                break

    # どうしても見つからない場合のフォールバック
    if best_rect is None:
        h = max(1, min(rows, int(round(np.sqrt(target_cells)))))
        w = max(1, min(cols, int(np.ceil(target_cells / float(h)))))
        r0 = int(rng.integers(0, rows - h + 1))
        c0 = int(rng.integers(0, cols - w + 1))
        best_rect = GridRectangle(r0=r0, c0=c0, h=h, w=w)

    return best_rect


# =========================
# データ拡張本体
# =========================
def apply_grid_fill_per_cell_color(
    image_bgr_u8: np.ndarray,
    rows: int,
    cols: int,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    画像全体を rows x cols に区切り、面積が ratio 程度になるグリッド長方形を選び、
    その長方形内を「セルごとにランダム色」で塗りつぶす（不透明）。
    """
    out = image_bgr_u8.copy()
    h_img, w_img = out.shape[:2]
    row_edges = compute_cell_edges(h_img, rows)
    col_edges = compute_cell_edges(w_img, cols)

    target_cells = int(round(rows * cols * float(ratio)))
    rect = choose_grid_rectangle(rows, cols, target_cells, rng)

    for r in range(rect.r0, rect.r0 + rect.h):
        y0 = row_edges[r]
        y1 = row_edges[r + 1]
        for c in range(rect.c0, rect.c0 + rect.w):
            x0 = col_edges[c]
            x1 = col_edges[c + 1]
            color = rng.integers(0, 256, size=(3,), dtype=np.uint8)  # BGR
            out[y0:y1, x0:x1] = color

    return out


def apply_grid_overlay_per_cell_color(
    image_bgr_u8: np.ndarray,
    rows: int,
    cols: int,
    ratio: float,
    rng: np.random.Generator,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    """
    画像全体を rows x cols に区切り、面積が ratio 程度になるグリッド長方形を選び、
    その長方形内に「セルごとにランダム色」の半透明オーバーレイをかける。
    """
    out = image_bgr_u8.copy()
    h_img, w_img = out.shape[:2]
    row_edges = compute_cell_edges(h_img, rows)
    col_edges = compute_cell_edges(w_img, cols)

    target_cells = int(round(rows * cols * float(ratio)))
    rect = choose_grid_rectangle(rows, cols, target_cells, rng)

    alpha_min = float(np.clip(alpha_min, 0.0, 1.0))
    alpha_max = float(np.clip(alpha_max, 0.0, 1.0))
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min
    alpha = float(rng.uniform(alpha_min, alpha_max))

    for r in range(rect.r0, rect.r0 + rect.h):
        y0 = row_edges[r]
        y1 = row_edges[r + 1]
        for c in range(rect.c0, rect.c0 + rect.w):
            x0 = col_edges[c]
            x1 = col_edges[c + 1]
            color = rng.integers(0, 256, size=(3,), dtype=np.uint8)  # BGR

            patch = out[y0:y1, x0:x1].astype(np.float32)
            color_f = color.astype(np.float32).reshape(1, 1, 3)
            blended = (1.0 - alpha) * patch + alpha * color_f
            out[y0:y1, x0:x1] = blended.clip(0, 255).astype(np.uint8)

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
    datasets: dict[str, LeRobotDataset] = {}

    for mode in modes:
        new_repo_id = f"{new_repo_id_base}_{mode}"
        new_dataset_name = new_repo_id.split("/")[-1]
        new_dataset_root = root_parent / new_dataset_name

        if new_dataset_root.exists():
            raise FileExistsError(
                f"既に {new_dataset_root} が存在します。別の new_repo_id_base を指定するか、ディレクトリを削除してください。"
            )

        new_dataset_root.parent.mkdir(parents=True, exist_ok=True)

        datasets[mode] = LeRobotDataset.create(
            repo_id=new_repo_id,
            fps=int(ds.fps),
            features=deepcopy(ds.meta.info["features"]),
            root=new_dataset_root,
            robot_type=ds.meta.robot_type,
            use_videos=len(ds.meta.video_keys) > 0,
        )

    return datasets


def main() -> None:
    ds = load_source_dataset()

    # モード一覧
    modes: list[str] = []
    for p in occlusion_levels:
        modes.append(f"grid_fill_{p}")
        modes.append(f"grid_overlay_{p}")

    datasets = initialize_target_datasets(ds, modes)

    skip_keys = {"task_index", "timestamp", "episode_index", "frame_index", "index"}
    total_frames = ds.meta.total_frames

    sample_frame = ds[0]
    image_keys = [k for k in sample_frame.keys() if k.startswith("observation.images.")]

    prev_episode_idx: Optional[int] = None

    for global_idx in range(total_frames):
        print(f"Writing frame {global_idx + 1} / {total_frames}", end="\r")

        frame = ds[global_idx]
        episode_idx = int(frame["episode_index"].item())

        # エピソード境界で save_episode
        if prev_episode_idx is None:
            prev_episode_idx = episode_idx
        elif episode_idx != prev_episode_idx:
            for mode in modes:
                if datasets[mode].episode_buffer["size"] > 0:
                    datasets[mode].save_episode()
            prev_episode_idx = episode_idx

        # 画像以外のフィールド
        base_payload: dict[str, object] = {}
        for key_inner, value in frame.items():
            if key_inner == "task" or key_inner in skip_keys:
                continue
            if key_inner.startswith("observation.images."):
                continue
            if isinstance(value, torch.Tensor):
                copied = value.detach().cpu()
                if key_inner in (DONE, REWARD) and copied.dim() == 0:
                    copied = copied.unsqueeze(0)
                if key_inner.startswith("complementary_info") and copied.dim() == 0:
                    copied = copied.unsqueeze(0)
                base_payload[key_inner] = copied
            else:
                base_payload[key_inner] = value

        # task
        task_value = frame["task"]
        if isinstance(task_value, torch.Tensor):
            task_value = task_value.item() if task_value.dim() == 0 else task_value.tolist()
        if not isinstance(task_value, str):
            task_value = str(task_value)

        # 画像を一度だけ numpy 化
        image_tensors: dict[str, torch.Tensor] = {k: frame[k].detach().cpu() for k in image_keys}
        np_images: dict[str, np.ndarray] = {k: tensor_to_hwc_numpy(t) for k, t in image_tensors.items()}

        for mode in modes:
            payload = dict(base_payload)
            base_mode, level = parse_mode_name(mode)
            ratio = float(level) / 100.0

            for image_key in image_keys:
                img_tensor = image_tensors[image_key]
                np_img_hwc = np_images[image_key]
                display_img = to_display_image(np_img_hwc)

                # モード・フレーム・カメラごとに seed を固定
                seed = make_deterministic_seed(
                    base_seed,
                    repo_id,
                    new_repo_id_base,
                    mode,
                    image_key,
                    episode_idx,
                    global_idx,
                )
                rng = np.random.default_rng(seed)

                if base_mode == "grid_fill":
                    edited = apply_grid_fill_per_cell_color(
                        display_img,
                        grid_rows,
                        grid_cols,
                        ratio,
                        rng,
                    )
                elif base_mode == "grid_overlay":
                    edited = apply_grid_overlay_per_cell_color(
                        display_img,
                        grid_rows,
                        grid_cols,
                        ratio,
                        rng,
                        overlay_alpha_min,
                        overlay_alpha_max,
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                payload[image_key] = convert_display_to_dtype(edited, img_tensor.dtype)

            payload["task"] = task_value
            datasets[mode].add_frame(payload)

    print()

    # flush
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

