import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
import torch

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


repo_id = "AriRyo/test-pickplace-v3"
new_repo_id = "AriRyo/test-pickplace-v3"
episode_indices: list[int] = []
mask_modes = ["mask", "grid_swap", "mosaic"]
grid_rows = 16
grid_cols = 24
selector_scale = 2
preview_scale = 2
enable_preview = False
push_to_hub = False
mosaic_block_size = 24

auto_select_enabled = True
auto_search_radius = 1
contour_blur_ksize = 5
contour_canny_low = 60
contour_canny_high = 180
contour_dilate_iterations = 1
contour_min_area = 150


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


def group_connected_cells_4(cells: list[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining: set[tuple[int, int]] = set(cells)
    groups: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        group: set[tuple[int, int]] = {start}
        while stack:
            row, col = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + dr, col + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    group.add(neighbor)
        groups.append(group)
    return groups


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

    def set_reference_cells(self, cells: set[tuple[int, int]], image: Optional[np.ndarray] = None, row_edges: Optional[list[int]] = None, col_edges: Optional[list[int]] = None, force_update_template: bool = False) -> None:
        if not cells:
            raise ValueError("ContourReference requires at least one cell")
        cells_set = set(cells)
        group_changed = (not hasattr(self, 'reference_cells')) or (self.reference_cells != cells_set)
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
        # Only update template if group changed or forced
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
        # Template matching if template is available
        if entry.template_gray is not None:
            search_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            th, tw = entry.template_gray.shape[:2]
            ph, pw = search_gray.shape[:2]
            if ph >= th and pw >= tw:
                res = cv2.matchTemplate(search_gray, entry.template_gray, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                # Compute anchor in grid
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
                    # Bounding box and centroid
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
            # fallback: previous logic (no template or failed)
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


def update_reference_entries(
    image: np.ndarray,
    manual_cells: list[tuple[int, int]],
    selected_cells: list[tuple[int, int]],
    existing_entries: list[ContourReference],
    rows: int,
    cols: int,
) -> list[ContourReference]:
    if not manual_cells and not existing_entries:
        return []

    row_edges = compute_cell_edges(image.shape[0], rows)
    col_edges = compute_cell_edges(image.shape[1], cols)

    unique_manual: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in manual_cells:
        if cell not in seen:
            unique_manual.append(cell)
            seen.add(cell)

    manual_groups = group_connected_cells_4(unique_manual) if unique_manual else []

    remaining = list(existing_entries)
    updated: list[ContourReference] = []
    current_selected = set(selected_cells)
    max_distance = max(1, auto_search_radius * 3)

    def pop_closest_group(anchor: tuple[int, int]) -> tuple[Optional[ContourReference], int]:
        best_idx = -1
        best_dist = sys.maxsize
        for idx, entry in enumerate(remaining):
            entry_anchor = entry.last_anchor_cell or entry.reference_anchor
            dist = abs(entry_anchor[0] - anchor[0]) + abs(entry_anchor[1] - anchor[1])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx == -1:
            return None, sys.maxsize
        return remaining.pop(best_idx), best_dist

    for group in manual_groups:
        min_row = min(row for row, _ in group)
        min_col = min(col for _, col in group)
        anchor = (min_row, min_col)
        entry, distance = pop_closest_group(anchor)
        if entry is None or distance > max_distance:
            entry = ContourReference(reference_cells=set(group))
            entry.set_reference_cells(set(group), image, row_edges, col_edges, force_update_template=True)
        else:
            entry.set_reference_cells(set(group), image, row_edges, col_edges, force_update_template=True)

        centroid_y, centroid_x = cell_set_centroid(row_edges, col_edges, entry.reference_cells)
        entry.last_centroid = (centroid_y, centroid_x)

        ref_rows = [row for row, _ in entry.reference_cells]
        ref_cols = [col for _, col in entry.reference_cells]
        min_row_ref = min(ref_rows)
        max_row_ref = max(ref_rows)
        min_col_ref = min(ref_cols)
        max_col_ref = max(ref_cols)
        bbox = (
            int(col_edges[min_col_ref]),
            int(row_edges[min_row_ref]),
            int(col_edges[max_col_ref + 1]),
            int(row_edges[max_row_ref + 1]),
        )
        entry.last_bbox = bbox
        updated.append(entry)

    for entry in remaining:
        cells = entry.last_matched_cells or entry.reference_cells
        if not cells:
            continue
        if not current_selected:
            updated.append(entry)
            continue
        if any(cell in current_selected for cell in cells):
            updated.append(entry)

    updated.sort(key=lambda item: item.reference_anchor)
    return updated


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
) -> np.ndarray:
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
    if len(swap_targets) <= 1:
        return image

    shuffled_targets = swap_targets.copy()
    random.shuffle(shuffled_targets)
    if swap_targets == shuffled_targets:
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

    for src_cell, dst_cell in zip(swap_targets, shuffled_targets):
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

    virtual_display = Display(visible=0, size=(1280, 720))
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
    def __init__(
        self,
        image: np.ndarray,
        window_title: str,
        rows: int,
        cols: int,
        display_scale: float = 1.0,
        initial_cells: Optional[list[tuple[int, int]]] = None,
    ):
        self.image = image.copy()
        self.rows = rows
        self.cols = cols
        self.display_scale = display_scale
        self.selected_cells: list[tuple[int, int]] = []
        self.manual_cells: list[tuple[int, int]] = []
        self._manual_cell_set: set[tuple[int, int]] = set()
        self._confirm_keys_down: set[str] = set()
        self._keyboard_sequences = (
            "<BackSpace>",
            "<KeyPress-Return>",
            "<KeyRelease-Return>",
            "<KeyPress-space>",
            "<KeyRelease-space>",
            "<Escape>",
            "s",
            "S",
            "n",
            "N",
            "q",
            "Q",
        )
        self.action: Optional[str] = None
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.resizable(False, False)

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        if self.display_scale != 1.0:
            new_size = (
                int(pil_img.width * self.display_scale),
                int(pil_img.height * self.display_scale),
            )
            pil_img = pil_img.resize(new_size, Image.NEAREST)

        self.photo = ImageTk.PhotoImage(pil_img)

        self.orig_height, self.orig_width = rgb.shape[0], rgb.shape[1]
        self.display_width, self.display_height = pil_img.size
        self.canvas = tk.Canvas(self.root, width=self.display_width, height=self.display_height)
        self.canvas.pack()
        self.canvas_image = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.display_cell_w = self.display_width / self.cols
        self.display_cell_h = self.display_height / self.rows
        self.row_edges = compute_cell_edges(self.orig_height, self.rows)
        self.col_edges = compute_cell_edges(self.orig_width, self.cols)

        if initial_cells:
            self.selected_cells = [tuple(cell) for cell in initial_cells]
        self.manual_cells.clear()
        self._manual_cell_set.clear()
        self._confirm_keys_down.clear()

        self._draw_grid()
        self._draw_selection()

        info = (
            "クリックでセルを追加/解除 / Backspaceで直前のセル削除 /"
            " Enter・Spaceで確定 / s・n・Escでスキップ / qでエピソード終了"
        )
        tk.Label(self.root, text=info).pack(pady=4)

        self._enable_user_input()
        self.root.protocol("WM_DELETE_WINDOW", self.skip)

    def update_image_and_title(
        self,
        new_image: np.ndarray,
        new_title: str,
        initial_cells: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        self._disable_user_input()

        self.image = new_image.copy()
        self.root.title(new_title)

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        if self.display_scale != 1.0:
            new_size = (
                int(pil_img.width * self.display_scale),
                int(pil_img.height * self.display_scale),
            )
            pil_img = pil_img.resize(new_size, Image.NEAREST)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)

        self.orig_height, self.orig_width = rgb.shape[0], rgb.shape[1]
        self.display_width, self.display_height = pil_img.size
        self.display_cell_w = self.display_width / self.cols
        self.display_cell_h = self.display_height / self.rows
        self.row_edges = compute_cell_edges(self.orig_height, self.rows)
        self.col_edges = compute_cell_edges(self.orig_width, self.cols)

        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.delete("grid")
        self.selected_cells = [tuple(cell) for cell in initial_cells] if initial_cells else []
        self.manual_cells.clear()
        self._manual_cell_set.clear()
        self._confirm_keys_down.clear()
        self._draw_grid()
        self._draw_selection()

        self.root.update_idletasks()
        self._enable_user_input()

    def run(self) -> str:
        self.action = None
        self.root.mainloop()
        return self.action or "skip"

    def _draw_grid(self) -> None:
        for c in range(1, self.cols):
            x = int(round(c * self.display_cell_w))
            self.canvas.create_line(x, 0, x, self.display_height, fill="white", width=1, tags="grid")
        for r in range(1, self.rows):
            y = int(round(r * self.display_cell_h))
            self.canvas.create_line(0, y, self.display_width, y, fill="white", width=1, tags="grid")

    def _draw_selection(self) -> None:
        self.canvas.delete("highlight")
        if len(self.selected_cells) == 0:
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
                    outline = "#ff3333" if is_center else "#ffa500"
                    fill = "#ff6666" if is_center else "#ffd27f"
                    stipple = "gray25" if is_center else "gray50"
                    width = 3 if is_center else 2
                    self.canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        y1,
                        outline=outline,
                        fill=fill,
                        stipple=stipple,
                        width=width,
                        tags="highlight",
                    )
                    if is_center:
                        self.canvas.create_line(
                            x0,
                            y0,
                            x1,
                            y1,
                            fill="white",
                            width=1,
                            tags="highlight",
                        )
                        self.canvas.create_line(
                            x0,
                            y1,
                            x1,
                            y0,
                            fill="white",
                            width=1,
                            tags="highlight",
                        )

    def _enable_user_input(self) -> None:
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<BackSpace>", self.on_undo)
        self.root.bind("<KeyPress-Return>", self.on_confirm_key_press)
        self.root.bind("<KeyRelease-Return>", self.on_confirm_key_release)
        self.root.bind("<KeyPress-space>", self.on_confirm_key_press)
        self.root.bind("<KeyRelease-space>", self.on_confirm_key_release)
        self.root.bind("<Escape>", self.skip)
        self.root.bind("s", self.skip)
        self.root.bind("S", self.skip)
        self.root.bind("n", self.skip)
        self.root.bind("N", self.skip)
        self.root.bind("q", self.quit_episode)
        self.root.bind("Q", self.quit_episode)

    def _disable_user_input(self) -> None:
        self.canvas.unbind("<Button-1>")
        for sequence in self._keyboard_sequences:
            self.root.unbind(sequence)
        self._confirm_keys_down.clear()

    def on_confirm_key_press(self, event) -> None:
        if event.keysym not in {"Return", "space"}:
            return
        if event.keysym in self._confirm_keys_down:
            return
        self._confirm_keys_down.add(event.keysym)

    def on_confirm_key_release(self, event) -> None:
        if event.keysym not in {"Return", "space"}:
            return
        if event.keysym not in self._confirm_keys_down:
            return
        self._confirm_keys_down.discard(event.keysym)
        self.finish()

    def on_click(self, event):
        col = min(int(event.x // self.display_cell_w), self.cols - 1)
        row = min(int(event.y // self.display_cell_h), self.rows - 1)
        cell = (row, col)
        if cell in self.selected_cells:
            self.selected_cells.remove(cell)
            if cell in self._manual_cell_set:
                self._manual_cell_set.remove(cell)
                self.manual_cells = [c for c in self.manual_cells if c != cell]
        else:
            self.selected_cells.append(cell)
            if cell not in self._manual_cell_set:
                self.manual_cells.append(cell)
                self._manual_cell_set.add(cell)
        self._draw_selection()

    def on_undo(self, event=None):
        if self.selected_cells:
            cell = self.selected_cells.pop()
            if cell in self._manual_cell_set:
                self._manual_cell_set.remove(cell)
                self.manual_cells = [c for c in self.manual_cells if c != cell]
        self._draw_selection()

    def finish(self, event=None):
        self._confirm_keys_down.clear()
        self.action = "confirm"
        self.root.quit()

    def skip(self, event=None):
        self._confirm_keys_down.clear()
        self.action = "skip"
        self.selected_cells = []
        self.manual_cells.clear()
        self._manual_cell_set.clear()
        self.root.quit()

    def quit_episode(self, event=None):
        self._confirm_keys_down.clear()
        self.action = "quit"
        self.selected_cells = []
        self.manual_cells.clear()
        self._manual_cell_set.clear()
        self.root.quit()

    def get_action(self) -> str:
        return self.action or "skip"

    def get_mask(self):
        if self.action != "confirm" or len(self.selected_cells) == 0:
            return None
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for row, col in self.selected_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = row + dr
                    cc = col + dc
                    if rr < 0 or rr >= self.rows or cc < 0 or cc >= self.cols:
                        continue
                    y0 = self.row_edges[rr]
                    y1 = self.row_edges[rr + 1]
                    x0 = self.col_edges[cc]
                    x1 = self.col_edges[cc + 1]
                    mask[y0:y1, x0:x1] = 255
        return mask

    def get_manual_cells(self) -> list[tuple[int, int]]:
        return list(self.manual_cells)


def run_annotation(ds: LeRobotDataset, datasets: dict[str, LeRobotDataset]) -> None:
    global auto_reference_entries

    target_episode_set = set(episode_indices)
    skip_keys = {"task_index", "timestamp", "episode_index", "frame_index", "index"}
    total_frames = ds.meta.total_frames
    prev_episode = None
    skip_episode_rest = False
    selection_history: dict[str, list[tuple[int, int]]] = {}
    edited_frames = {
        mode: [dict() for _ in range(total_frames)]
        for mode in mask_modes
    }
    auto_reference_entries = {}
    base_frames: list[dict[str, object] | None] = [None] * total_frames
    task_values: list[str] = [""] * total_frames
    episode_by_frame: list[int] = [-1] * total_frames

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

    for key in image_keys:
        cam_name = key.split(".")[-1]
        print(f"Processing camera {cam_name}")
        selection_history[key] = []
        prev_episode = None
        skip_episode_rest = False
        selector = None

        for global_idx in range(total_frames):
            frame = ds[global_idx]
            episode_idx = int(frame["episode_index"].item())
            frame_index = int(frame["frame_index"].item())

            stored_episode = episode_by_frame[global_idx]
            if stored_episode == -1:
                episode_by_frame[global_idx] = episode_idx
            elif stored_episode != episode_idx:
                episode_idx = stored_episode

            if base_frames[global_idx] is None:
                base_data: dict[str, object] = {}
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
                        base_data[key_inner] = copied
                    else:
                        base_data[key_inner] = value
                base_frames[global_idx] = base_data

                task_value = frame["task"]
                if isinstance(task_value, torch.Tensor):
                    task_value = task_value.item() if task_value.dim() == 0 else task_value.tolist()
                if not isinstance(task_value, str):
                    task_value = str(task_value)
                task_values[global_idx] = task_value
            elif task_values[global_idx] == "":
                task_value = frame["task"]
                if isinstance(task_value, torch.Tensor):
                    task_value = task_value.item() if task_value.dim() == 0 else task_value.tolist()
                if not isinstance(task_value, str):
                    task_value = str(task_value)
                task_values[global_idx] = task_value

            if prev_episode is None:
                prev_episode = episode_idx
            elif episode_idx != prev_episode:
                skip_episode_rest = False
                selection_history[key] = []
                prev_episode = episode_idx

            for other_key in image_keys:
                if other_key == key:
                    continue
                for mode in mask_modes:
                    if other_key not in edited_frames[mode][global_idx]:
                        edited_frames[mode][global_idx][other_key] = tensor_to_hwc_numpy(
                            frame[other_key].detach().cpu()
                        )

            if episode_idx in target_episode_set and not skip_episode_rest:
                img_tensor = frame[key].detach().cpu()
                np_img_hwc = tensor_to_hwc_numpy(img_tensor)
                display_img = to_display_image(np_img_hwc)

                history = selection_history[key]
                auto_cells: list[tuple[int, int]] = []
                if auto_select_enabled and key in auto_reference_entries:
                    auto_cells, auto_bboxes, auto_centroids = auto_select_cells(
                        display_img,
                        auto_reference_entries[key],
                        grid_rows,
                        grid_cols,
                    )
                else:
                    auto_bboxes = []
                    auto_centroids = []

                initial_cells_list: list[tuple[int, int]] = []
                if auto_cells:
                    initial_cells_list = auto_cells
                elif history:
                    initial_cells_list = history

                title = f"Ep {episode_idx} Cam {cam_name} Frame {frame_index}"

                display_img_for_ui = display_img
                if auto_bboxes or auto_centroids:
                    display_img_for_ui = draw_detection_overlays(
                        display_img,
                        auto_bboxes,
                        auto_centroids,
                    )

                if selector is None:
                    selector = GridSelector(
                        display_img_for_ui.copy(),
                        title,
                        grid_rows,
                        grid_cols,
                        display_scale=selector_scale,
                        initial_cells=initial_cells_list if initial_cells_list else None,
                    )
                else:
                    selector.update_image_and_title(
                        display_img_for_ui.copy(),
                        title,
                        initial_cells=initial_cells_list if initial_cells_list else None,
                    )

                action = selector.run()
                mask = selector.get_mask()

                if action == "quit":
                    skip_episode_rest = True
                    print(f"エピソード {episode_idx}: q が押されたため残りフレームをスキップします。")
                    for mode in mask_modes:
                        edited_frames[mode][global_idx][key] = np_img_hwc
                    continue

                if action == "skip" or mask is None:
                    for mode in mask_modes:
                        edited_frames[mode][global_idx][key] = np_img_hwc
                    continue

                selected_cells_copy = [tuple(cell) for cell in selector.selected_cells]
                manual_cells = selector.get_manual_cells()
                edited_displays = []
                mode_titles = []
                for mode in mask_modes:
                    edited_display = display_img.copy()
                    if mode == "mask":
                        avg_color = display_img.reshape(-1, display_img.shape[2]).mean(axis=0)
                        fill_color = np.clip(np.round(avg_color), 0, 255).astype(display_img.dtype)
                        edited_display[mask == 0] = fill_color
                    elif mode == "grid_swap":
                        edited_display = apply_random_grid_swap(
                            edited_display,
                            selected_cells_copy,
                            grid_rows,
                            grid_cols,
                        )
                    elif mode == "mosaic":
                        edited_display = apply_mosaic_outside_mask(
                            edited_display,
                            mask,
                            mosaic_block_size,
                        )
                    else:
                        raise ValueError(f"Unknown mask_mode: {mode}")

                    edited_displays.append(edited_display)
                    mode_titles.append(mode)
                    edited_frames[mode][global_idx][key] = convert_display_to_dtype(
                        edited_display,
                        img_tensor.dtype,
                    )
                show_preview(edited_displays, mode_titles, f"Preview Ep{episode_idx} Cam {cam_name}")
                if auto_select_enabled:
                    existing = auto_reference_entries.get(key, [])
                    auto_reference_entries[key] = update_reference_entries(
                        display_img,
                        manual_cells,
                        selected_cells_copy,
                        existing,
                        grid_rows,
                        grid_cols,
                    )
                selection_history[key] = selected_cells_copy
            else:
                original_np = tensor_to_hwc_numpy(frame[key].detach().cpu())
                for mode in mask_modes:
                    edited_frames[mode][global_idx][key] = original_np

        if selector is not None:
            selector.root.destroy()

    prev_episode_combined: Optional[int] = None
    for global_idx in range(total_frames):
        if (
            episode_by_frame[global_idx] == -1
            or base_frames[global_idx] is None
            or task_values[global_idx] == ""
        ):
            frame = ds[global_idx]
            if episode_by_frame[global_idx] == -1:
                episode_by_frame[global_idx] = int(frame["episode_index"].item())
            if base_frames[global_idx] is None:
                base_data: dict[str, object] = {}
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
                        base_data[key_inner] = copied
                    else:
                        base_data[key_inner] = value
                base_frames[global_idx] = base_data
            if task_values[global_idx] == "":
                task_value = frame["task"]
                if isinstance(task_value, torch.Tensor):
                    task_value = task_value.item() if task_value.dim() == 0 else task_value.tolist()
                if not isinstance(task_value, str):
                    task_value = str(task_value)
                task_values[global_idx] = task_value

        episode_idx = episode_by_frame[global_idx]
        if prev_episode_combined is None:
            prev_episode_combined = episode_idx
        elif episode_idx != prev_episode_combined:
            for mode in mask_modes:
                if datasets[mode].episode_buffer["size"] > 0:
                    datasets[mode].save_episode()
            prev_episode_combined = episode_idx

        base_payload = base_frames[global_idx] or {}
        fallback_cache: dict[str, np.ndarray] = {}
        frame_cache: Optional[dict[str, object]] = None
        for mode in mask_modes:
            payload = dict(base_payload)
            for image_key in image_keys:
                if image_key not in edited_frames[mode][global_idx]:
                    if image_key not in fallback_cache:
                        if frame_cache is None:
                            frame_cache = ds[global_idx]
                        fallback_cache[image_key] = tensor_to_hwc_numpy(
                            frame_cache[image_key].detach().cpu()
                        )
                    payload[image_key] = fallback_cache[image_key]
                else:
                    payload[image_key] = edited_frames[mode][global_idx][image_key]
            payload["task"] = task_values[global_idx]
            datasets[mode].add_frame(payload)

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
