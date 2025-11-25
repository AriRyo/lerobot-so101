import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import tkinter as tk  # preview 用
import torch
import pickle  # 進捗保存用

from copy import deepcopy
from PIL import Image, ImageTk
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DONE, REWARD


def show_preview(images: list[np.ndarray], titles: list[str], window_title: str) -> None:
    if not enable_preview:
        return

    scaled_images = []
    for image in images:
        display_img = image
        if preview_scale != 1.0:
            h, w = image.shape[:2]
            display_img = cv2.resize(
                image,
                (int(w * preview_scale), int(h * preview_scale)),
                interpolation=cv2.INTER_NEAREST,
            )
        scaled_images.append(display_img)

    root = tk.Tk()
    root.title(window_title)
    root.resizable(False, False)

    frame = tk.Frame(root)
    frame.pack()

    photos = []
    for img, title in zip(scaled_images, titles):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)
        photos.append(photo)

        sub_frame = tk.Frame(frame)
        sub_frame.pack(side=tk.LEFT, padx=5, pady=5)

        label = tk.Label(sub_frame, image=photo)
        label.pack()
        tk.Label(sub_frame, text=title, pady=2).pack()

    tk.Label(root, text="Enter/Space/Esc で閉じます。", pady=8).pack()

    def close_window(_event=None) -> None:
        root.destroy()

    root.bind("<Return>", close_window)
    root.bind("<space>", close_window)
    root.bind("<Escape>", close_window)
    root.protocol("WM_DELETE_WINDOW", close_window)

    root.mainloop()


repo_id = "AriRyo/redblack-pickplace-v3_black"
new_repo_id = "AriRyo/black-pickplace-v3_da"
episode_indices: list[int] = []

augmentation_levels = [20, 50, 80, 100]
mask_modes: list[str] = []
for p in augmentation_levels:
    mask_modes.append(f"mask_{p}")
    mask_modes.append(f"grid_swap_{p}")
    mask_modes.append(f"mosaic_{p}")

grid_rows = 28
grid_cols = 42
selector_scale = 2
preview_scale = 2
enable_preview = False
push_to_hub = True

mosaic_block_size = 24

start_frame_index = 0

auto_select_enabled = True
auto_search_radius = 1  # 追従の探索範囲を広めに
contour_blur_ksize = 5
contour_canny_low = 60
contour_canny_high = 180
contour_dilate_iterations = 1
contour_min_area = 150

motion_diff_threshold = 25          # 差分画素とみなすしきい値
motion_cell_ratio_threshold = 0.05   # セル内で何割以上差分があれば「動いたセル」とするか
motion_neighbor_radius = 1       # 動作セルの周囲何セルまで含めるか

# 進捗保存ファイル
PROGRESS_PATH = "annotation_progress.pkl"


def tensor_to_hwc_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def to_display_image(np_hwc: np.ndarray) -> np.ndarray:
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


def build_mask_from_cells(
    image_shape: tuple[int, int, int] | tuple[int, int],
    cells: list[tuple[int, int]],
    row_edges: list[int],
    col_edges: list[int],
    rows: int,
    cols: int,
) -> np.ndarray:
    """選択セル + その周囲 3×3 をマスクに含める。"""
    h = image_shape[0]
    w = image_shape[1]
    mask = np.zeros((h, w), dtype=np.uint8)
    for row, col in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = row + dr
                cc = col + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                y0 = row_edges[rr]
                y1 = row_edges[rr + 1]
                x0 = col_edges[cc]
                x1 = col_edges[cc + 1]
                mask[y0:y1, x0:x1] = 255
    return mask


def get_neighborhood_cells(
    center: tuple[int, int],
    rows: int,
    cols: int,
    radius: int = 1,
) -> list[tuple[int, int]]:
    row_c, col_c = center
    neighbors: list[tuple[int, int]] = []
    for d_row in range(-radius, radius + 1):
        for d_col in range(-radius, radius + 1):
            row = row_c + d_row
            col = col_c + d_col
            if row < 0 or row >= rows or col < 0 or col >= cols:
                continue
            neighbors.append((row, col))
    return neighbors


def extract_cell_bounds(
    row_edges: list[int],
    col_edges: list[int],
    cell: tuple[int, int],
) -> tuple[int, int, int, int]:
    row, col = cell
    y0 = row_edges[row]
    y1 = row_edges[row + 1]
    x0 = col_edges[col]
    x1 = col_edges[col + 1]
    return y0, y1, x0, x1


def cell_centroid(
    row_edges: list[int],
    col_edges: list[int],
    cell: tuple[int, int],
) -> tuple[float, float]:
    y0, y1, x0, x1 = extract_cell_bounds(row_edges, col_edges, cell)
    return (y0 + y1) * 0.5, (x0 + x1) * 0.5


def point_to_cell(
    row_edges: list[int],
    col_edges: list[int],
    y: float,
    x: float,
    rows: int,
    cols: int,
) -> tuple[int, int]:
    row_idx = int(np.clip(np.searchsorted(row_edges, y, side="right") - 1, 0, rows - 1))
    col_idx = int(np.clip(np.searchsorted(col_edges, x, side="right") - 1, 0, cols - 1))
    return row_idx, col_idx


def cells_from_bbox(
    bbox: tuple[float, float, float, float],
    row_edges: list[int],
    col_edges: list[int],
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    y0, y1, x0, x1 = bbox
    y0 = max(0.0, y0)
    y1 = max(y0 + 1.0, y1)
    x0 = max(0.0, x0)
    x1 = max(x0 + 1.0, x1)
    row_start = int(np.clip(np.searchsorted(row_edges, y0, side="right") - 1, 0, rows - 1))
    row_end = int(np.clip(np.searchsorted(row_edges, y1 - 1e-3, side="right") - 1, 0, rows - 1))
    col_start = int(np.clip(np.searchsorted(col_edges, x0, side="right") - 1, 0, cols - 1))
    col_end = int(np.clip(np.searchsorted(col_edges, x1 - 1e-3, side="right") - 1, 0, cols - 1))

    cells: list[tuple[int, int]] = []
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            cells.append((row, col))
    return cells


def clamp_anchor_for_shape(
    anchor_row: int,
    anchor_col: int,
    rows: int,
    cols: int,
    shape_size: tuple[int, int],
) -> tuple[int, int]:
    shape_rows, shape_cols = shape_size
    max_row_anchor = max(0, rows - shape_rows)
    max_col_anchor = max(0, cols - shape_cols)
    clamped_row = int(np.clip(anchor_row, 0, max_row_anchor))
    clamped_col = int(np.clip(anchor_col, 0, max_col_anchor))
    return clamped_row, clamped_col


def cell_set_centroid(
    row_edges: list[int],
    col_edges: list[int],
    cells: set[tuple[int, int]],
) -> tuple[float, float]:
    if not cells:
        return 0.0, 0.0
    sum_y = 0.0
    sum_x = 0.0
    for cell in cells:
        cy, cx = cell_centroid(row_edges, col_edges, cell)
        sum_y += cy
        sum_x += cx
    count = float(len(cells))
    return sum_y / count, sum_x / count


def detect_primary_contour(patch: np.ndarray) -> Optional[tuple[tuple[float, float], tuple[int, int, int, int]]]:
    if patch.size == 0:
        return None

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if contour_blur_ksize >= 3 and contour_blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (contour_blur_ksize, contour_blur_ksize), 0)

    edges = cv2.Canny(gray, contour_canny_low, contour_canny_high)
    if contour_dilate_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=contour_dilate_iterations)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_contour)
    if area < contour_min_area:
        return None

    moments = cv2.moments(best_contour)
    if moments["m00"] != 0.0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        x, y, w, h = cv2.boundingRect(best_contour)
        cx = x + w * 0.5
        cy = y + h * 0.5

    bbox = cv2.boundingRect(best_contour)
    return (cx, cy), bbox


def draw_detection_overlays(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    centroids: list[tuple[float, float]],
) -> np.ndarray:
    overlay = image.copy()
    for (x0, y0, x1, y1) in bboxes:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    for (cx, cy) in centroids:
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1)
    return overlay


@dataclass
class ContourReference:
    reference_cells: set[tuple[int, int]]
    search_radius: int = auto_search_radius
    last_anchor_cell: Optional[tuple[int, int]] = None
    last_matched_cells: Optional[set[tuple[int, int]]] = None
    last_centroid: Optional[tuple[float, float]] = None
    last_bbox: Optional[tuple[int, int, int, int]] = None
    shape_offsets: tuple[tuple[int, int], ...] = field(init=False)
    reference_anchor: tuple[int, int] = field(init=False)
    shape_size: tuple[int, int] = field(init=False)
    template_patch: Optional[np.ndarray] = None
    template_gray: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.set_reference_cells(self.reference_cells)

    def set_reference_cells(
        self,
        cells: set[tuple[int, int]],
        image: Optional[np.ndarray] = None,
        row_edges: Optional[list[int]] = None,
        col_edges: Optional[list[int]] = None,
        force_update_template: bool = False,
    ) -> None:
        if not cells:
            raise ValueError("ContourReference requires at least one cell")
        cells_set = set(cells)
        group_changed = (not hasattr(self, "reference_cells")) or (self.reference_cells != cells_set)
        self.reference_cells = cells_set
        min_row = min(row for row, _ in cells)
        min_col = min(col for _, col in cells)
        self.reference_anchor = (min_row, min_col)
        offsets = [(row - min_row, col - min_col) for row, col in cells]
        offsets.sort()
        self.shape_offsets = tuple(offsets)
        max_row_offset = max(offset[0] for offset in offsets)
        max_col_offset = max(offset[1] for offset in offsets)
        self.shape_size = (max_row_offset + 1, max_col_offset + 1)
        self.last_anchor_cell = self.reference_anchor
        self.last_matched_cells = set(cells)
        if (group_changed or force_update_template) and image is not None and row_edges is not None and col_edges is not None:
            self.template_patch, self.template_gray = self.extract_template(image, row_edges, col_edges)

    def extract_template(self, image: np.ndarray, row_edges: list[int], col_edges: list[int]) -> tuple[np.ndarray, np.ndarray]:
        ref_rows = [row for row, _ in self.reference_cells]
        ref_cols = [col for _, col in self.reference_cells]
        min_row_ref = min(ref_rows)
        max_row_ref = max(ref_rows)
        min_col_ref = min(ref_cols)
        max_col_ref = max(ref_cols)
        y0 = row_edges[min_row_ref]
        y1 = row_edges[max_row_ref + 1]
        x0 = col_edges[min_col_ref]
        x1 = col_edges[max_col_ref + 1]
        patch = image[y0:y1, x0:x1].copy()
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return patch, gray


auto_reference_entries: dict[str, list[ContourReference]] = {}


def parse_mode_name(mode: str) -> tuple[str, int]:
    parts = mode.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid mode name: {mode}")
    level = int(parts[-1])
    base = "_".join(parts[:-1])
    return base, level


def auto_select_cells(
    image: np.ndarray,
    reference_entries: list[ContourReference],
    rows: int,
    cols: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]], list[tuple[float, float]]]:
    if len(reference_entries) == 0:
        return [], [], []

    row_edges = compute_cell_edges(image.shape[0], rows)
    col_edges = compute_cell_edges(image.shape[1], cols)

    suggested_cells: list[tuple[int, int]] = []
    used_cells: set[tuple[int, int]] = set()
    bboxes: list[tuple[int, int, int, int]] = []
    centroids: list[tuple[float, float]] = []

    for entry in reference_entries:
        current_cells = entry.last_matched_cells or entry.reference_cells
        if not current_cells:
            continue

        min_row = min(row for row, _ in current_cells)
        max_row = max(row for row, _ in current_cells)
        min_col = min(col for _, col in current_cells)
        max_col = max(col for _, col in current_cells)

        row_min = max(0, min_row - entry.search_radius)
        row_max = min(rows - 1, max_row + entry.search_radius)
        col_min = max(0, min_col - entry.search_radius)
        col_max = min(cols - 1, max_col + entry.search_radius)

        y0 = row_edges[row_min]
        y1 = row_edges[row_max + 1]
        x0 = col_edges[col_min]
        x1 = col_edges[col_max + 1]
        patch = image[y0:y1, x0:x1]

        candidate_cells: list[tuple[int, int]] = []
        found_match = False

        if entry.template_gray is not None:
            search_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            th, tw = entry.template_gray.shape[:2]
            ph, pw = search_gray.shape[:2]
            if ph >= th and pw >= tw:
                res = cv2.matchTemplate(search_gray, entry.template_gray, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc

                anchor_y = y0 + top_left[1]
                anchor_x = x0 + top_left[0]
                anchor_row = int(np.clip(np.searchsorted(row_edges, anchor_y, side="right") - 1, 0, rows - 1))
                anchor_col = int(np.clip(np.searchsorted(col_edges, anchor_x, side="right") - 1, 0, cols - 1))
                anchor_row, anchor_col = clamp_anchor_for_shape(anchor_row, anchor_col, rows, cols, entry.shape_size)
                translated_cells: set[tuple[int, int]] = set()
                for dr, dc in entry.shape_offsets:
                    rr = anchor_row + dr
                    cc = anchor_col + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        translated_cells.add((rr, cc))
                if len(translated_cells) == len(entry.shape_offsets):
                    entry.last_anchor_cell = (anchor_row, anchor_col)
                    entry.last_matched_cells = set(translated_cells)
                    found_match = True
                    candidate_cells.extend(sorted(translated_cells))

                    min_row_t = min(rr for rr, _ in translated_cells)
                    max_row_t = max(rr for rr, _ in translated_cells)
                    min_col_t = min(cc for _, cc in translated_cells)
                    max_col_t = max(cc for _, cc in translated_cells)
                    bbox = (
                        int(col_edges[min_col_t]),
                        int(row_edges[min_row_t]),
                        int(col_edges[max_col_t + 1]),
                        int(row_edges[max_row_t + 1]),
                    )
                    entry.last_bbox = bbox
                    bboxes.append(bbox)
                    centroid_y, centroid_x = cell_set_centroid(row_edges, col_edges, translated_cells)
                    entry.last_centroid = (centroid_y, centroid_x)
                    centroids.append((centroid_x, centroid_y))

        if not found_match:
            if entry.last_centroid is None:
                centroid_y, centroid_x = cell_set_centroid(row_edges, col_edges, set(current_cells))
                entry.last_centroid = (centroid_y, centroid_x)
            centroid_y, centroid_x = entry.last_centroid
            centroids.append((centroid_x, centroid_y))
            if entry.last_bbox is not None:
                bboxes.append(entry.last_bbox)
            candidate_cells.extend(sorted(current_cells))

        for cell in candidate_cells:
            if cell in used_cells:
                continue
            used_cells.add(cell)
            suggested_cells.append(cell)

    return suggested_cells, bboxes, centroids
def detect_motion_cells(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    rows: int,
    cols: int,
    diff_threshold: int = motion_diff_threshold,
    cell_ratio_threshold: float = motion_cell_ratio_threshold,
) -> list[tuple[int, int]]:
    """
    前フレーム(prev_img)と現在フレーム(curr_img)の画素差分から、
    「よく動いているグリッドセル」を返す。
    """
    if prev_img is None or prev_img.shape != curr_img.shape:
        return []

    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, diff_bin = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # ノイズ除去用にモルフォロジー演算を少し入れる
    kernel = np.ones((3, 3), np.uint8)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    h, w = curr_img.shape[:2]
    row_edges = compute_cell_edges(h, rows)
    col_edges = compute_cell_edges(w, cols)

    moving_cells: list[tuple[int, int]] = []

    for r in range(rows):
        y0 = row_edges[r]
        y1 = row_edges[r + 1]
        for c in range(cols):
            x0 = col_edges[c]
            x1 = col_edges[c + 1]

            cell_mask = diff_bin[y0:y1, x0:x1]
            if cell_mask.size == 0:
                continue

            ratio = float(np.count_nonzero(cell_mask)) / float(cell_mask.size)
            if ratio >= cell_ratio_threshold:
                moving_cells.append((r, c))

    return moving_cells


def update_reference_entries(
    image: np.ndarray,
    manual_cells: list[tuple[int, int]],
    selected_cells: list[tuple[int, int]],
    manual_regions: list[list[tuple[int, int]]],
    existing_entries: list[ContourReference],
    rows: int,
    cols: int,
) -> list[ContourReference]:
    """
    ドラッグした各領域をそれぞれひとつのテンプレートとして扱う。
    manual_regions が空の場合は既存の参照を維持する。
    """
    # このフレームで新しいドラッグが無ければ、参照はそのまま
    if not manual_cells and not manual_regions:
        return existing_entries

    # manual_regions があるフレームでは、
    # 「今回ドラッグした領域だけ」を自動追従の参照として作り直す
    row_edges = compute_cell_edges(image.shape[0], rows)
    col_edges = compute_cell_edges(image.shape[1], cols)

    new_entries: list[ContourReference] = []

    # manual_regions: 1ドラッグ = 1領域
    regions = manual_regions
    for region in regions:
        if not region:
            continue
        ref_cells = set(region)
        entry = ContourReference(reference_cells=ref_cells)
        entry.set_reference_cells(
            ref_cells,
            image,
            row_edges,
            col_edges,
            force_update_template=True,
        )
        new_entries.append(entry)

    return new_entries


def apply_mosaic_outside_mask(
    image: np.ndarray,
    mask: np.ndarray,
    block_size: int,
) -> np.ndarray:
    if block_size <= 1:
        return image

    height, width = image.shape[:2]
    small_w = max(1, width // block_size)
    small_h = max(1, height // block_size)
    pixelated = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)

    result = image.copy()
    result[mask == 0] = pixelated[mask == 0]
    return result


def apply_random_grid_swap(
    image: np.ndarray,
    selected_cells: list[tuple[int, int]],
    rows: int,
    cols: int,
    ratio: float = 1.0,
) -> np.ndarray:
    """
    グリッドをランダムに入れ替える。
    ratio: 0.2, 0.5, 0.8, 1.0 のように、保護領域以外のセルのうち何割を対象にするか。
    保護領域は「選択セル + 周囲(3x3)」。
    """
    if len(selected_cells) == 0:
        return image

    height, width = image.shape[:2]
    row_edges = compute_cell_edges(height, rows)
    col_edges = compute_cell_edges(width, cols)

    protected_cells: set[tuple[int, int]] = set()
    for row, col in selected_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = row + dr
                cc = col + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    protected_cells.add((rr, cc))

    swap_targets = [
        (r, c) for r in range(rows) for c in range(cols) if (r, c) not in protected_cells
    ]
    if len(swap_targets) <= 1 or ratio <= 0.0:
        return image

    if ratio >= 1.0:
        active_targets = swap_targets
    else:
        num_targets = max(1, int(len(swap_targets) * ratio))
        num_targets = min(num_targets, len(swap_targets))
        if num_targets <= 1:
            return image
        active_targets = random.sample(swap_targets, num_targets)

    shuffled_targets = active_targets.copy()
    random.shuffle(shuffled_targets)
    if active_targets == shuffled_targets:
        shuffled_targets = shuffled_targets[1:] + shuffled_targets[:1]

    source = image.copy()
    result = image.copy()

    def cell_bounds(cell: tuple[int, int]) -> tuple[int, int, int, int]:
        r, c = cell
        y0 = row_edges[r]
        y1 = row_edges[r + 1]
        x0 = col_edges[c]
        x1 = col_edges[c + 1]
        return y0, y1, x0, x1

    for src_cell, dst_cell in zip(active_targets, shuffled_targets):
        y0s, y1s, x0s, x1s = cell_bounds(src_cell)
        y0d, y1d, x0d, x1d = cell_bounds(dst_cell)

        donor_patch = source[y0d:y1d, x0d:x1d]

        dst_h = y1s - y0s
        dst_w = x1s - x0s
        if donor_patch.shape[0] == dst_h and donor_patch.shape[1] == dst_w:
            result[y0s:y1s, x0s:x1s] = donor_patch
            continue

        resized_patch = cv2.resize(donor_patch, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
        result[y0s:y1s, x0s:x1s] = resized_patch

    return result


def ensure_display() -> Optional[object]:
    if os.environ.get("DISPLAY"):
        return None
    try:
        from pyvirtualdisplay import Display  # type: ignore
    except ImportError:
        sys.stderr.write(
            "DISPLAY が設定されていない環境です。GUI を表示するには `pip install pyvirtualdisplay` と `sudo apt install xvfb` を実行するか、\n"
            "`xvfb-run python scripts/object_mask_dataset_tool.py` のように仮想ディスプレイを用意してください。\n"
        )
        sys.exit(1)

    virtual_display = Display(visible=0, size=(1920, 1080))
    virtual_display.start()
    os.environ["DISPLAY"] = virtual_display.new_display_var
    print("Started virtual display for headless environment.")
    return virtual_display


def load_source_dataset() -> LeRobotDataset:
    try:
        return LeRobotDataset(repo_id, video_backend="pyav")
    except (RuntimeError, ImportError) as err:
        print(f"PyAV backend failed ({err}), falling back to torchvision video_reader.")
        return LeRobotDataset(repo_id, video_backend="video_reader")


def initialize_target_datasets(ds: LeRobotDataset) -> dict[str, LeRobotDataset]:
    root_parent = ds.root.parent.parent
    datasets: dict[str, LeRobotDataset] = {}
    for mode in mask_modes:
        new_repo_id_mode = f"{new_repo_id}_{mode}"
        new_dataset_name = new_repo_id_mode.split("/")[-1]
        new_dataset_root = root_parent / new_dataset_name

        if new_dataset_root.exists():
            raise FileExistsError(
                f"既に {new_dataset_root} が存在します。別の new_repo_id を指定するか、ディレクトリを削除してください。"
            )

        new_dataset_root.parent.mkdir(parents=True, exist_ok=True)

        datasets[mode] = LeRobotDataset.create(
            repo_id=new_repo_id_mode,
            fps=int(ds.fps),
            features=deepcopy(ds.meta.info["features"]),
            root=new_dataset_root,
            robot_type=ds.meta.robot_type,
            use_videos=len(ds.meta.video_keys) > 0,
        )

    return datasets


class GridSelector:
    """
    OpenCV ベースの UI。
    左ドラッグで矩形範囲を追加（複数ドラッグで追加可能）。
    Backspace / Delete で直前のドラッグ範囲削除。
    """

    def __init__(
        self,
        image: np.ndarray,
        window_title: str,
        rows: int,
        cols: int,
        display_scale: float = 1.0,
        initial_cells: Optional[list[tuple[int, int]]] = None,
        initial_regions: Optional[list[list[tuple[int, int]]]] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.window_name = "GridSelector"
        self.display_scale = display_scale

        self.image = image.copy()
        self.orig_height, self.orig_width = self.image.shape[:2]

        if self.display_scale != 1.0:
            self.display_width = int(self.orig_width * self.display_scale)
            self.display_height = int(self.orig_height * self.display_scale)
            self.display_image_base = cv2.resize(
                self.image,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            self.display_image_base = self.image.copy()
            self.display_height, self.display_width = self.display_image_base.shape[:2]

        self.display_cell_w = self.display_width / self.cols
        self.display_cell_h = self.display_height / self.rows
        self.row_edges = compute_cell_edges(self.orig_height, self.rows)
        self.col_edges = compute_cell_edges(self.orig_width, self.cols)

        # 自動追従セル / 手動ドラッグセル / 描画用セル
        self.auto_cells: list[tuple[int, int]] = []
        self.manual_cells: list[tuple[int, int]] = []
        self.selected_cells: list[tuple[int, int]] = []

        self._manual_cell_set: set[tuple[int, int]] = set()
        self.drag_regions: list[list[tuple[int, int]]] = []

        # 既存フレームに戻ってきたときのために drag_regions（=手動領域）を再現
        if initial_regions:
            for region in initial_regions:
                region_cells: list[tuple[int, int]] = []
                for cell in region:
                    cell = tuple(cell)
                    if cell in self._manual_cell_set:
                        continue
                    self._manual_cell_set.add(cell)
                    self.manual_cells.append(cell)
                    region_cells.append(cell)
                if region_cells:
                    self.drag_regions.append(region_cells)
        elif initial_cells:
            # 自動追従などからの初期候補は auto_cells としてだけ扱う
            for cell in initial_cells:
                cell = tuple(cell)
                if cell not in self.auto_cells:
                    self.auto_cells.append(cell)

        # auto + manual の和集合を描画用セルとして反映
        self._update_selected_cells()

        self.drag_start_x: Optional[int] = None
        self.drag_start_y: Optional[int] = None
        self.drag_cur_x: Optional[int] = None
        self.drag_cur_y: Optional[int] = None
        self.dragging: bool = False

        self.action: Optional[str] = None
        self.window_title = window_title

    def _update_selected_cells(self) -> None:
        merged: set[tuple[int, int]] = set()
        merged.update(tuple(c) for c in self.auto_cells)
        merged.update(tuple(c) for c in self.manual_cells)
        # 表示・マスク用にセルをソートして保持
        self.selected_cells = sorted(merged)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start_x = x
            self.drag_start_y = y
            self.drag_cur_x = x
            self.drag_cur_y = y
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_cur_x = x
            self.drag_cur_y = y
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_cur_x = x
            self.drag_cur_y = y
            self._finalize_drag_region()

    def _finalize_drag_region(self):
        if (
            self.drag_start_x is None
            or self.drag_start_y is None
            or self.drag_cur_x is None
            or self.drag_cur_y is None
        ):
            return

        if self.auto_cells:
            self.auto_cells.clear()

        x0 = max(0, min(self.display_width - 1, min(self.drag_start_x, self.drag_cur_x)))
        x1 = max(0, min(self.display_width - 1, max(self.drag_start_x, self.drag_cur_x)))
        y0 = max(0, min(self.display_height - 1, min(self.drag_start_y, self.drag_cur_y)))
        y1 = max(0, min(self.display_height - 1, max(self.drag_start_y, self.drag_cur_y)))

        col_start = int(x0 // self.display_cell_w)
        col_end = int(x1 // self.display_cell_w)
        row_start = int(y0 // self.display_cell_h)
        row_end = int(y1 // self.display_cell_h)

        col_start = max(0, min(self.cols - 1, col_start))
        col_end = max(0, min(self.cols - 1, col_end))
        row_start = max(0, min(self.rows - 1, row_start))
        row_end = max(0, min(self.rows - 1, row_end))

        if col_end < col_start or row_end < row_start:
            return

        new_cells: list[tuple[int, int]] = []
        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                cell = (r, c)
                if cell not in self._manual_cell_set:
                    self._manual_cell_set.add(cell)
                    self.manual_cells.append(cell)
                    new_cells.append(cell)

        if new_cells:
            self.drag_regions.append(new_cells)

        # 手動追加分を反映して描画セルを更新
        self._update_selected_cells()

        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_cur_x = None
        self.drag_cur_y = None

    def _draw_grid(self, img: np.ndarray) -> None:
        for c in range(1, self.cols):
            x = int(round(c * self.display_cell_w))
            cv2.line(img, (x, 0), (x, self.display_height), (255, 255, 255), 1)
        for r in range(1, self.rows):
            y = int(round(r * self.display_cell_h))
            cv2.line(img, (0, y), (self.display_width, y), (255, 255, 255), 1)

    def _draw_selection(self, img: np.ndarray) -> None:
        if not self.selected_cells:
            return
        for row, col in self.selected_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = row + dr
                    cc = col + dc
                    if rr < 0 or rr >= self.rows or cc < 0 or cc >= self.cols:
                        continue
                    x0 = int(round(cc * self.display_cell_w))
                    y0 = int(round(rr * self.display_cell_h))
                    x1 = int(round((cc + 1) * self.display_cell_w))
                    y1 = int(round((rr + 1) * self.display_cell_h))
                    is_center = (rr, cc) == (row, col)
                    if is_center:
                        fill_color = (0, 0, 255)
                        border_color = (0, 0, 180)
                    else:
                        fill_color = (0, 165, 255)
                        border_color = (0, 140, 220)
                    cv2.rectangle(img, (x0, y0), (x1, y1), fill_color, thickness=-1)
                    cv2.rectangle(img, (x0, y0), (x1, y1), border_color, thickness=2)
                    if is_center:
                        cv2.line(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
                        cv2.line(img, (x0, y1), (x1, y0), (255, 255, 255), 1)

    def _draw_drag_rect(self, img: np.ndarray) -> None:
        if (
            not self.dragging
            or self.drag_start_x is None
            or self.drag_start_y is None
            or self.drag_cur_x is None
            or self.drag_cur_y is None
        ):
            return
        x0 = max(0, min(self.display_width - 1, self.drag_start_x))
        y0 = max(0, min(self.display_height - 1, self.drag_start_y))
        x1 = max(0, min(self.display_width - 1, self.drag_cur_x))
        y1 = max(0, min(self.display_height - 1, self.drag_cur_y))
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    def _compose_display_image(self) -> np.ndarray:
        disp = self.display_image_base.copy()
        overlay = disp.copy()
        self._draw_selection(overlay)
        alpha = 0.4  # 半透明
        cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0, disp)
        self._draw_grid(disp)
        self._draw_drag_rect(disp)
        return disp

    def on_undo(self):
        if not self.drag_regions:
            return
        last_region = self.drag_regions.pop()
        for cell in last_region:
            cell = tuple(cell)
            if cell in self._manual_cell_set:
                self._manual_cell_set.remove(cell)
            try:
                self.manual_cells.remove(cell)
            except ValueError:
                pass
        # 手動領域を削ったので表示セルを更新
        self._update_selected_cells()

    def finish(self):
        self.action = "confirm"

    def skip(self):
        self.action = "skip"
        self.auto_cells.clear()
        self.manual_cells.clear()
        self.selected_cells = []
        self._manual_cell_set.clear()
        self.drag_regions = []

    def quit_episode(self):
        self.action = "quit"
        self.auto_cells.clear()
        self.manual_cells.clear()
        self.selected_cells = []
        self._manual_cell_set.clear()
        self.drag_regions = []

    def go_back(self):
        self.action = "back"

    def stop_all(self):
        self.action = "stop"
        self.auto_cells.clear()
        self.manual_cells.clear()
        self.selected_cells = []
        self._manual_cell_set.clear()
        self.drag_regions = []

    def run(self) -> str:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(self.window_name, self.window_title)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)

        info_text = (
            "Drag: 範囲追加 / Backspace: 直前の範囲削除 / "
            "Enter,Space: 確定 / s: スキップ / b: 戻る / "
            "q: エピソード残りスキップ / e,Esc: 中断保存"
        )
        print(info_text, end="\r")

        while True:
            disp = self._compose_display_image()
            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(10) & 0xFF

            if key != 255:
                if key in (13, 10, 32):
                    self.finish()
                elif key in (8, 127):  # Backspace / Delete の両方を許可
                    self.on_undo()
                elif key in (ord("s"), ord("S"), ord("n"), ord("N")):
                    self.skip()
                elif key in (ord("q"), ord("Q")):
                    self.quit_episode()
                elif key in (ord("b"), ord("B")):
                    self.go_back()
                elif key in (27, ord("e"), ord("E")):
                    self.stop_all()

            if self.action is not None:
                break

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.stop_all()
                break

        return self.action or "skip"

    def get_action(self) -> str:
        return self.action or "skip"

    def get_mask(self) -> Optional[np.ndarray]:
        if self.action != "confirm" or len(self.selected_cells) == 0:
            return None
        return build_mask_from_cells(
            self.image.shape,
            self.selected_cells,
            self.row_edges,
            self.col_edges,
            self.rows,
            self.cols,
        )

    def get_manual_cells(self) -> list[tuple[int, int]]:
        return list(self.manual_cells)

    def get_drag_regions(self) -> list[list[tuple[int, int]]]:
        return [list(region) for region in self.drag_regions]


def run_annotation(ds: LeRobotDataset, datasets: dict[str, LeRobotDataset]) -> None:
    global auto_reference_entries

    target_episode_set = set(episode_indices)
    skip_keys = {"task_index", "timestamp", "episode_index", "frame_index", "index"}
    total_frames = ds.meta.total_frames

    max_episode_index = len(ds.meta.episodes) - 1
    invalid_episodes = sorted(idx for idx in target_episode_set if idx > max_episode_index or idx < 0)
    if invalid_episodes:
        print(f"警告: 指定されたエピソード {invalid_episodes} は存在しません。無視します。")
        target_episode_set -= set(invalid_episodes)

    if not target_episode_set:
        print("編集対象のエピソードが存在しないため、全フレームを対象として実施します。")
        target_episode_set = set(range(max_episode_index + 1))

    print("編集と新データセットへの書き込みを開始します...")

    sample_frame = ds[0]
    image_keys = [k for k in sample_frame.keys() if k.startswith("observation.images.")]
    prev_frames: dict[str, Optional[np.ndarray]] = {k: None for k in image_keys}
    prev_episode_for_cam: dict[str, Optional[int]] = {k: None for k in image_keys}

    # safe_start_index は進捗ファイルから上書きされる可能性があるので変数にしておく
    safe_start_index = max(0, min(start_frame_index, total_frames - 1)) if total_frames > 0 else 0

    # 進捗読み込み
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "rb") as f:
            saved = pickle.load(f)
        frame_selection: dict[str, dict[int, list[tuple[int, int]]]] = saved.get("frame_selection", {})
        frame_regions: dict[str, dict[int, list[list[tuple[int, int]]]]] = saved.get("frame_regions", {})
        camera_progress: dict[str, int] = saved.get("camera_progress", {})
        saved_target_episode = saved.get("target_episode_set")
        saved_safe_start_index = saved.get("safe_start_index")
        if saved_target_episode is not None:
            target_episode_set = set(saved_target_episode)
        if saved_safe_start_index is not None:
            safe_start_index = int(saved_safe_start_index)
        print("既存のアノテーション進捗を読み込みました。")
    else:
        frame_selection = {}
        frame_regions = {}
        camera_progress = {}

    # キーが足りない場合の初期化
    for key in image_keys:
        frame_selection.setdefault(key, {})
        frame_regions.setdefault(key, {})
        camera_progress.setdefault(key, 0)

    # 進捗保存関数
    def save_progress():
        data = {
            "frame_selection": frame_selection,
            "frame_regions": frame_regions,
            "camera_progress": camera_progress,
            "target_episode_set": list(target_episode_set),
            "safe_start_index": safe_start_index,
        }
        with open(PROGRESS_PATH, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 画像そのものは保持せず、「どのセルを選んだか」だけ保持する
    auto_reference_entries = {}

    # 前回実行の最後に選択したセルを history として復元
    last_selection_history: dict[str, list[tuple[int, int]]] = {}
    for key in image_keys:
        if frame_selection[key]:
            last_idx = max(frame_selection[key].keys())
            last_selection_history[key] = frame_selection[key][last_idx]
        else:
            last_selection_history[key] = []

    safe_start_index = max(0, min(safe_start_index, total_frames - 1)) if total_frames > 0 else 0
    early_stop = False

    # -------- 第1パス：UI でアノテーション（選択セルだけ記録） --------
    for key in image_keys:
        if early_stop:
            break

        cam_name = key.split(".")[-1]
        print(f"Processing camera {cam_name}")

        prev_episode: Optional[int] = None
        skip_episode_rest = False

        # このカメラで前回どこまで進んだか
        global_idx = camera_progress.get(key, 0)
        if global_idx >= total_frames:
            print(f"Camera {cam_name} は既に全フレーム処理済みのためスキップします。")
            continue

        while global_idx < total_frames:
            frame = ds[global_idx]
            episode_idx = int(frame["episode_index"].item())
            frame_index = int(frame["frame_index"].item())

            if prev_episode is None:
                prev_episode = episode_idx
            elif episode_idx != prev_episode:
                # エピソードが変わったらスキップ状態と履歴をリセット
                skip_episode_rest = False
                last_selection_history[key] = []
                prev_episode = episode_idx
                prev_frames[key] = None
                prev_episode_for_cam[key] = None

            # アノテーション対象外条件
            if (
                global_idx < safe_start_index
                or episode_idx not in target_episode_set
                or skip_episode_rest
            ):
                global_idx += 1
                camera_progress[key] = global_idx
                save_progress()
                continue

            img_tensor = frame[key].detach().cpu()
            np_img_hwc = tensor_to_hwc_numpy(img_tensor)
            display_img = to_display_image(np_img_hwc)
            motion_cells: list[tuple[int, int]] = []
            prev_img = prev_frames.get(key)
            if prev_img is not None and prev_episode_for_cam.get(key) == episode_idx:
                # まず全セルからの差分セルを検出
                motion_cells_raw = detect_motion_cells(
                    prev_img,
                    display_img,
                    grid_rows,
                    grid_cols,
                )

                # 前回の選択セル
                prev_cells = last_selection_history.get(key, [])

                if motion_cells_raw and prev_cells:
                    # 前回選択セルの近傍セル集合
                    neighbor_set: set[tuple[int, int]] = set()
                    for cell in prev_cells:
                        neighbors = get_neighborhood_cells(
                            cell,
                            grid_rows,
                            grid_cols,
                            radius=motion_neighbor_radius,  # 近傍半径
                        )
                        neighbor_set.update(neighbors)

                    # 差分セルのうち、近傍セルに含まれるものだけを残す
                    motion_cells = [c for c in motion_cells_raw if c in neighbor_set]
                else:
                    # 前回選択が無い場合は、動きセルを使わない（完全に無視）
                    motion_cells = []
            else:
                # 前フレーム情報が無い場合は、動きセルを使わない
                motion_cells = []

            existing_selection = frame_selection[key].get(global_idx)
            existing_regions = frame_regions[key].get(global_idx)
            auto_cells: list[tuple[int, int]] = []
            auto_bboxes: list[tuple[int, int, int, int]] = []
            auto_centroids: list[tuple[float, float]] = []
            initial_cells_list: list[tuple[int, int]] = []
            if existing_selection is not None:
                initial_cells_list = existing_selection
            else:
                # 自動追従（テンプレート）
                if auto_select_enabled and key in auto_reference_entries:
                    auto_cells, auto_bboxes, auto_centroids = auto_select_cells(
                        display_img,
                        auto_reference_entries[key],
                        grid_rows,
                        grid_cols,
                    )
                history = last_selection_history[key]

                candidate_set: set[tuple[int, int]] = set()
                if auto_cells:
                    candidate_set.update(auto_cells)
                elif history:
                    candidate_set.update(history)

                # ここで「近傍だけに絞った motion_cells」を統合
                candidate_set.update(motion_cells)

                initial_cells_list = sorted(candidate_set)

            title = f"Ep {episode_idx} Cam {cam_name} Frame {frame_index}"

            display_img_for_ui = display_img
            if auto_bboxes or auto_centroids:
                display_img_for_ui = draw_detection_overlays(
                    display_img,
                    auto_bboxes,
                    auto_centroids,
                )

            selector = GridSelector(
                display_img_for_ui.copy(),
                title,
                grid_rows,
                grid_cols,
                display_scale=selector_scale,
                initial_cells=initial_cells_list if initial_cells_list else None,
                initial_regions=existing_regions,
            )

            selector.run()
            action = selector.get_action()

            if action == "back":
                if global_idx > safe_start_index:
                    global_idx -= 1
                else:
                    print("これ以上前のフレームには戻れません。")
                # ここでは progress は更新しない
                continue

            if action == "stop":
                # ここまでの進捗を保存して終了（第2パスには進まない）
                early_stop = True
                save_progress()
                print("\nアノテーションを中断しました。次回実行時に続きから再開します。")
                break

            if action == "quit":
                skip_episode_rest = True
                print(f"エピソード {episode_idx}: q が押されたため残りフレームをスキップします。")
                prev_frames[key] = display_img.copy()
                prev_episode_for_cam[key] = episode_idx
                global_idx += 1
                camera_progress[key] = global_idx
                save_progress()
                continue

            if action == "skip":
                # 選択なし（このフレームは元画像を使う）
                prev_frames[key] = display_img.copy()
                prev_episode_for_cam[key] = episode_idx
                global_idx += 1
                camera_progress[key] = global_idx
                save_progress()
                continue

            mask = selector.get_mask()
            selected_cells_copy = [tuple(cell) for cell in selector.selected_cells]

            if mask is None or not selected_cells_copy:
                prev_frames[key] = display_img.copy()
                prev_episode_for_cam[key] = episode_idx
                global_idx += 1
                camera_progress[key] = global_idx
                save_progress()
                continue

            manual_cells = selector.get_manual_cells()
            drag_regions = selector.get_drag_regions()
            row_edges = selector.row_edges
            col_edges = selector.col_edges

            selection_mask = build_mask_from_cells(
                display_img.shape,
                selected_cells_copy,
                row_edges,
                col_edges,
                grid_rows,
                grid_cols,
            )
            background_mask = (selection_mask == 0)

            # プレビュー用（ここで画像は使うが保存はしない）
            if enable_preview:
                edited_displays = []
                mode_titles = []
                for mode in mask_modes:
                    base_mode, level = parse_mode_name(mode)
                    level_ratio = max(0.0, min(1.0, level / 100.0))
                    edited_display = display_img.copy()

                    if base_mode == "mask":
                        edited_display = display_img.copy()

                        if level_ratio > 0.0 and np.any(background_mask):
                            avg_color = display_img.reshape(-1, display_img.shape[2]).mean(axis=0)
                            fill_color = np.clip(np.round(avg_color), 0, 255).astype(display_img.dtype)

                            if level_ratio >= 1.0:
                                edited_display[background_mask] = fill_color
                            else:
                                ys, xs = np.where(background_mask)
                                num_pixels = len(ys)
                                num_to_mask = max(1, int(num_pixels * level_ratio))
                                num_to_mask = min(num_to_mask, num_pixels)
                                if num_to_mask > 0:
                                    idx = np.random.choice(num_pixels, size=num_to_mask, replace=False)
                                    edited_display[ys[idx], xs[idx]] = fill_color

                    elif base_mode == "grid_swap":
                        edited_display = apply_random_grid_swap(
                            edited_display,
                            selected_cells_copy,
                            grid_rows,
                            grid_cols,
                            level_ratio,
                        )

                    elif base_mode == "mosaic":
                        edited_display = display_img.copy()

                        if level_ratio > 0.0 and np.any(background_mask):
                            block_size = max(1, int(mosaic_block_size * level_ratio))

                            height, width = display_img.shape[:2]
                            small_w = max(1, width // block_size)
                            small_h = max(1, height // block_size)
                            pixelated = cv2.resize(display_img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                            pixelated = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)

                            if level_ratio >= 1.0:
                                edited_display[background_mask] = pixelated[background_mask]
                            else:
                                ys, xs = np.where(background_mask)
                                num_pixels = len(ys)
                                num_to_mosaic = max(1, int(num_pixels * level_ratio))
                                num_to_mosaic = min(num_to_mosaic, len(ys))
                                if num_to_mosaic > 0:
                                    idx = np.random.choice(num_pixels, size=num_to_mosaic, replace=False)
                                    edited_display[ys[idx], xs[idx]] = pixelated[ys[idx], xs[idx]]

                    else:
                        raise ValueError(f"Unknown mask_mode: {mode}")

                    edited_displays.append(edited_display)
                    mode_titles.append(mode)

                show_preview(edited_displays, mode_titles, f"Preview Ep{episode_idx} Cam {cam_name}")

            # 自動追従用テンプレート更新（今回ドラッグした領域のみを参照にする）
            if auto_select_enabled:
                existing_entries = auto_reference_entries.get(key, [])
                auto_reference_entries[key] = update_reference_entries(
                    display_img,
                    manual_cells,
                    selected_cells_copy,
                    drag_regions,
                    existing_entries,
                    grid_rows,
                    grid_cols,
                )

            # このフレーム・このカメラで選んだセルだけを記録
            frame_selection[key][global_idx] = selected_cells_copy
            frame_regions[key][global_idx] = drag_regions
            last_selection_history[key] = selected_cells_copy

            prev_frames[key] = display_img.copy()
            prev_episode_for_cam[key] = episode_idx
            global_idx += 1
            camera_progress[key] = global_idx
            # save_progress()

        if early_stop:
            break

    # ここまでで early_stop なら、第2パスには進まない
    if early_stop:
        return
    
    save_progress()

    # すべてのカメラで total_frames まで到達しているか確認
    annotation_done = all(camera_progress[k] >= total_frames for k in image_keys)
    if not annotation_done:
        print("アノテーションがまだ完了していません。次回実行時に続きから再開します。")
        save_progress()
        return

    print("全カメラのアノテーションが完了しました。データセットを書き出します。")

    # -------- 第2パス：元データから読み直して、その場で画像加工して書き込み --------
    prev_episode_idx: Optional[int] = None

    mode_level: dict[str, tuple[str, float]] = {}
    for mode in mask_modes:
        base_mode, level = parse_mode_name(mode)
        level_ratio = max(0.0, min(1.0, level / 100.0))
        mode_level[mode] = (base_mode, level_ratio)

    for global_idx in range(total_frames):
        print(f"Writing frame {global_idx + 1} / {total_frames}", end="\r")

        frame_cache: dict[str, dict[str, object]] = {}

        frame = ds[global_idx]
        episode_idx = int(frame["episode_index"].item())

        # エピソード境界で save_episode
        if prev_episode_idx is None:
            prev_episode_idx = episode_idx
        elif episode_idx != prev_episode_idx:
            for mode in mask_modes:
                if datasets[mode].episode_buffer["size"] > 0:
                    datasets[mode].save_episode()
            prev_episode_idx = episode_idx

        # 画像以外のフィールドを組み立て
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

        # 各カメラ画像を一度だけ numpy に変換
        image_tensors: dict[str, torch.Tensor] = {
            k: frame[k].detach().cpu() for k in image_keys
        }
        np_images: dict[str, np.ndarray] = {
            k: tensor_to_hwc_numpy(t) for k, t in image_tensors.items()
        }

        # このフレームで「どのカメラにも選択が一切無いか」を先にチェック（必要なら利用）
        any_selection_this_frame = any(
            bool(frame_selection.get(image_key, {}).get(global_idx))
            for image_key in image_keys
        )

        # 各モードごとにフレームを書き込む
        for mode in mask_modes:
            payload = dict(base_payload)
            base_mode, level_ratio = mode_level[mode]

            for image_key in image_keys:
                img_tensor = image_tensors[image_key]
                np_img_hwc = np_images[image_key]
                display_img = to_display_image(np_img_hwc)

                selected_cells = frame_selection.get(image_key, {}).get(global_idx)

                # このカメラに選択セルが無い場合は元画像をそのまま使う
                if not selected_cells:
                    payload[image_key] = np_img_hwc
                    continue

                if image_key not in frame_cache:
                    # 初回: このフレーム・このカメラ用のマスク等を計算
                    row_edges = compute_cell_edges(display_img.shape[0], grid_rows)
                    col_edges = compute_cell_edges(display_img.shape[1], grid_cols)

                    selection_mask = build_mask_from_cells(
                        display_img.shape,
                        selected_cells,
                        row_edges,
                        col_edges,
                        grid_rows,
                        grid_cols,
                    )
                    background_mask = (selection_mask == 0)

                    ys_bg, xs_bg = np.where(background_mask)
                    num_bg_pixels = len(ys_bg)

                    avg_color = display_img.reshape(-1, display_img.shape[2]).mean(axis=0)
                    avg_color = np.clip(np.round(avg_color), 0, 255).astype(display_img.dtype)

                    mosaic_ratios = sorted({
                        r for (bm, r) in mode_level.values() if bm == "mosaic"
                    })

                    mosaic_images: dict[float, np.ndarray] = {}
                    if num_bg_pixels > 0 and mosaic_ratios:
                        height, width = display_img.shape[:2]
                        for r in mosaic_ratios:
                            if r <= 0.0:
                                continue
                            block_size = max(1, int(mosaic_block_size * r))
                            small_w = max(1, width // block_size)
                            small_h = max(1, height // block_size)
                            pixelated = cv2.resize(display_img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                            pixelated = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)
                            mosaic_images[r] = pixelated

                    frame_cache[image_key] = {
                        "background_mask": background_mask,
                        "ys_bg": ys_bg,
                        "xs_bg": xs_bg,
                        "num_bg_pixels": num_bg_pixels,
                        "avg_color": avg_color,
                        "mosaic_images": mosaic_images,
                    }

                cache = frame_cache[image_key]
                background_mask = cache["background_mask"]
                ys_bg = cache["ys_bg"]
                xs_bg = cache["xs_bg"]
                num_bg_pixels = cache["num_bg_pixels"]
                avg_color = cache["avg_color"]
                mosaic_images = cache["mosaic_images"]

                edited_display = display_img.copy()


                if base_mode == "mask":
                    if level_ratio > 0.0 and num_bg_pixels > 0:
                        if level_ratio >= 1.0:
                            edited_display[background_mask] = avg_color
                        else:
                            num_to_mask = max(1, int(num_bg_pixels * level_ratio))
                            num_to_mask = min(num_to_mask, num_bg_pixels)
                            idx = np.random.choice(num_bg_pixels, size=num_to_mask, replace=False)
                            edited_display[ys_bg[idx], xs_bg[idx]] = avg_color

                elif base_mode == "grid_swap":
                    edited_display = apply_random_grid_swap(
                        edited_display,
                        selected_cells,
                        grid_rows,
                        grid_cols,
                        level_ratio,
                    )

                elif base_mode == "mosaic":
                    if level_ratio > 0.0 and num_bg_pixels > 0:
                        pixelated = mosaic_images.get(level_ratio, None)
                        if pixelated is None:
                            block_size = max(1, int(mosaic_block_size * level_ratio))
                            height, width = display_img.shape[:2]
                            small_w = max(1, width // block_size)
                            small_h = max(1, height // block_size)
                            pixelated = cv2.resize(display_img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                            pixelated = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)

                        if level_ratio >= 1.0:
                            edited_display[background_mask] = pixelated[background_mask]
                        else:
                            num_to_mosaic = max(1, int(num_bg_pixels * level_ratio))
                            num_to_mosaic = min(num_to_mosaic, num_bg_pixels)
                            idx = np.random.choice(num_bg_pixels, size=num_to_mosaic, replace=False)
                            edited_display[ys_bg[idx], xs_bg[idx]] = pixelated[ys_bg[idx], xs_bg[idx]]

                else:
                    raise ValueError(f"Unknown mask_mode: {mode}")

                payload[image_key] = convert_display_to_dtype(
                    edited_display,
                    img_tensor.dtype,
                )

            payload["task"] = task_value
            datasets[mode].add_frame(payload)

    print()  # 改行

    # 残りのエピソードを flush
    for mode in mask_modes:
        if datasets[mode].episode_buffer["size"] > 0:
            datasets[mode].save_episode()
        datasets[mode].finalize()

        print(f"編集済みデータセットを {datasets[mode].root} に保存しました。")

        if push_to_hub:
            new_repo_id_mode = f"{new_repo_id}_{mode}"
            print(f"{new_repo_id_mode} を Hugging Face Hub へアップロードします...")
            datasets[mode].push_to_hub()
        else:
            print("push_to_hub=False のため、アップロードはスキップしました。")

    # 完了したので進捗ファイルは削除
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
        print(f"{PROGRESS_PATH} を削除しました。")


def main() -> None:
    virtual_display = ensure_display()
    try:
        ds = load_source_dataset()
        datasets = initialize_target_datasets(ds)
        run_annotation(ds, datasets)
    finally:
        if virtual_display is not None:
            virtual_display.stop()


if __name__ == "__main__":
    main()
