#%%
import os
import random
import sys
from pathlib import Path
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
    for i, (img, title) in enumerate(zip(scaled_images, titles)):
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

#%%
# ハードコードされたパラメータ
repo_id = "AriRyo/test-pickplace-v3"  # 元のデータセットID
new_repo_id = "AriRyo/test-pickplace-v3"  # 新しいデータセットID
episode_indices = []  # 編集するエピソードのインデックス（例: 最初の3つ）
mask_modes = ["mask", "grid_swap", "mosaic"]  # 適用するモードのリスト
grid_rows = 12  # グリッドの行数
grid_cols = 16  # グリッドの列数
selector_scale = 2  # 選択用GUIの拡大率
preview_scale = 2  # プレビュー表示の拡大率
enable_preview = False  # プレビュー表示を有効化するか
push_to_hub = True  # Hugging Face Hub へアップロードするか
mosaic_block_size = 24  # モザイクのブロックサイズ（ピクセル）


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

#%%
# ヘッドレス環境対応（Xvfb）
virtual_display = None
if not os.environ.get("DISPLAY"):
    try:
        from pyvirtualdisplay import Display  # type: ignore

        virtual_display = Display(visible=0, size=(1280, 720))
        virtual_display.start()
        os.environ["DISPLAY"] = virtual_display.new_display_var
        print("Started virtual display for headless environment.")
    except ImportError:
        sys.stderr.write(
            "DISPLAY が設定されていない環境です。GUI を表示するには `pip install pyvirtualdisplay` と `sudo apt install xvfb` を実行するか、\n"
            "`xvfb-run python scripts/object_mask_dataset_tool.py` のように仮想ディスプレイを用意してください。\n"
        )
        sys.exit(1)

#%%
# データセットをロード
try:
    ds = LeRobotDataset(repo_id, video_backend="pyav")
except (RuntimeError, ImportError) as err:
    print(f"PyAV backend failed ({err}), falling back to torchvision video_reader.")
    ds = LeRobotDataset(repo_id, video_backend="video_reader")

#%%
# グリッドで領域を選択するクラス（Tkinter版）
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
        # 一時的にユーザー入力を無効化してレンダリング完了まで待機
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
        self._confirm_keys_down.clear()
        self._draw_grid()
        self._draw_selection()

        self.root.update_idletasks()

        # 入力を再有効化
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
        else:
            self.selected_cells.append(cell)
        self._draw_selection()

    def on_undo(self, event=None):
        if self.selected_cells:
            self.selected_cells.pop()
        self._draw_selection()

    def finish(self, event=None):
        self._confirm_keys_down.clear()
        self.action = "confirm"
        self.root.quit()

    def skip(self, event=None):
        self._confirm_keys_down.clear()
        self.action = "skip"
        self.selected_cells = []
        self.root.quit()

    def quit_episode(self, event=None):
        self._confirm_keys_down.clear()
        self.action = "quit"
        self.selected_cells = []
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

#%%
# 新しいデータセットを作成
root_parent = ds.root.parent.parent
datasets = {}
for mode in mask_modes:
    new_repo_id_mode = f"{new_repo_id}_{mode}"
    new_dataset_name = new_repo_id_mode.split("/")[-1]
    new_dataset_root = root_parent / new_dataset_name

    if new_dataset_root.exists():
        raise FileExistsError(f"既に {new_dataset_root} が存在します。別の new_repo_id を指定するか、ディレクトリを削除してください。")

    new_dataset_root.parent.mkdir(parents=True, exist_ok=True)

    datasets[mode] = LeRobotDataset.create(
        repo_id=new_repo_id_mode,
        fps=int(ds.fps),
        features=deepcopy(ds.meta.info["features"]),
        root=new_dataset_root,
        robot_type=ds.meta.robot_type,
        use_videos=len(ds.meta.video_keys) > 0,
    )

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
# ベースとなる非画像データやタスク情報をフレーム単位で保持
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

# image_keysを取得（最初のフレームから）
sample_frame = ds[0]
image_keys = [k for k in sample_frame.keys() if k.startswith("observation.images.")]

selection_history: dict[str, list[tuple[int, int]]] = {}

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
            title = f"Ep {episode_idx} Cam {cam_name} Frame {frame_index}"

            if selector is None:
                selector = GridSelector(
                    display_img.copy(),
                    title,
                    grid_rows,
                    grid_cols,
                    display_scale=selector_scale,
                    initial_cells=history if history else None,
                )
            else:
                selector.update_image_and_title(
                    display_img.copy(),
                    title,
                    initial_cells=history if history else None,
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

# 最終エピソードを保存してクローズ
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

if virtual_display is not None:
    virtual_display.stop()