# segmentation_preview.py
import os
import sys
import argparse
import numpy as np
import cv2
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def tensor_to_hwc_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def to_uint8_image(np_hwc: np.ndarray) -> np.ndarray:
    if np.issubdtype(np_hwc.dtype, np.floating):
        return (np_hwc * 255.0).clip(0, 255).astype(np.uint8)
    return np.clip(np_hwc, 0, 255).astype(np.uint8)


def boundary_mask_from_labels(labels: np.ndarray) -> np.ndarray:
    # ラベル差分で境界抽出
    b = np.zeros(labels.shape, dtype=np.uint8)
    b[:, 1:] |= (labels[:, 1:] != labels[:, :-1])
    b[1:, :] |= (labels[1:, :] != labels[:-1, :])
    b = (b * 255).astype(np.uint8)
    b = cv2.dilate(b, np.ones((3, 3), np.uint8), iterations=1)
    return b


def overlay_boundaries(bgr: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    out[boundary_mask > 0] = (0, 0, 255)  # 赤
    return out


def colorize_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    h, w = labels.shape
    lab = labels.astype(np.int32)
    max_label = int(lab.max()) if lab.size else 0
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 256, size=(max_label + 1, 3), dtype=np.uint8)
    return colors[lab.reshape(-1)].reshape(h, w, 3)


def meanshift_kmeans_segmentation(
    bgr: np.ndarray,
    sp: int = 12,
    sr: int = 18,
    k: int = 10,
    kmeans_iter: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    # Mean shift filtering
    ms = cv2.pyrMeanShiftFiltering(bgr, sp=sp, sr=sr)

    # k-means on color (after mean shift) -> labels
    Z = ms.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, kmeans_iter, 1.0)
    rng = np.random.default_rng(seed)
    attempts = 1
    # OpenCV kmeans は乱数を内部使用するので、入力を軽くシャッフルせずそのまま実行
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    labels_2d = labels.reshape(bgr.shape[:2]).astype(np.int32)

    colored = colorize_labels(labels_2d, seed=seed)
    boundary = boundary_mask_from_labels(labels_2d)
    overlay = overlay_boundaries(bgr, boundary)
    return colored, overlay


def graph_based_segmentation(
    bgr: np.ndarray,
    sigma: float = 0.5,
    k: float = 300.0,
    min_size: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
        raise RuntimeError("cv2.ximgproc.segmentation が見つかりません。OpenCV contrib が必要です。")

    segmenter = cv2.ximgproc.segmentation.createGraphSegmentation(
        sigma=sigma, k=k, min_size=min_size
    )
    labels = segmenter.processImage(bgr)  # (H,W) int32
    labels = labels.astype(np.int32)

    colored = colorize_labels(labels, seed=seed)
    boundary = boundary_mask_from_labels(labels)
    overlay = overlay_boundaries(bgr, boundary)
    return colored, overlay


def slic_segmentation(
    bgr: np.ndarray,
    region_size: int = 25,
    ruler: float = 10.0,
    iterations: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("cv2.ximgproc が見つかりません。OpenCV contrib が必要です。")

    # SLICO はパラメータ調整が比較的楽
    slic = cv2.ximgproc.createSuperpixelSLIC(
        bgr, algorithm=cv2.ximgproc.SLICO, region_size=region_size, ruler=ruler
    )
    slic.iterate(iterations)
    labels = slic.getLabels().astype(np.int32)  # (H,W)
    contour = slic.getLabelContourMask(thick_line=True)  # 255 on boundary

    colored = colorize_labels(labels, seed=seed)
    overlay = overlay_boundaries(bgr, contour)
    return colored, overlay


def load_dataset(repo_id: str) -> LeRobotDataset:
    try:
        return LeRobotDataset(repo_id, video_backend="pyav")
    except Exception as err:
        print(f"PyAV backend failed ({err}), falling back to torchvision video_reader.", file=sys.stderr)
        return LeRobotDataset(repo_id, video_backend="video_reader")


def tile_and_label(images_bgr: list[np.ndarray], titles: list[str]) -> np.ndarray:
    # 画像サイズを揃えて横連結、上にタイトルを描画
    h = min(img.shape[0] for img in images_bgr)
    w = min(img.shape[1] for img in images_bgr)
    resized = [cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST) for img in images_bgr]

    labeled = []
    for img, t in zip(resized, titles):
        out = img.copy()
        cv2.putText(out, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        labeled.append(out)

    return cv2.hconcat(labeled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="AriRyo/pickplace-v4_black")
    parser.add_argument("--camera_key", type=str, default="", help="例: observation.images.top（空なら自動で先頭を選択）")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="seg_vis")
    parser.add_argument("--show", action="store_true", help="cv2.imshow で表示（ヘッドレスでは非推奨）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.repo_id)
    sample = ds[0]
    image_keys = [k for k in sample.keys() if k.startswith("observation.images.")]
    if not image_keys:
        raise RuntimeError("observation.images.* が見つかりません。データセットの features を確認してください。")

    cam_key = args.camera_key if args.camera_key else image_keys[0]
    if cam_key not in image_keys:
        raise RuntimeError(f"camera_key={cam_key} が見つかりません。候補: {image_keys}")

    total = ds.meta.total_frames
    indices = [i for i in range(args.start, min(total, args.start + args.step * args.num_frames), args.step)]
    indices = indices[: args.num_frames]

    for n, idx in enumerate(indices):
        frame = ds[idx]
        np_hwc = tensor_to_hwc_numpy(frame[cam_key])
        img_rgb = to_uint8_image(np_hwc)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 1) Mean shift（フィルタ）+ k-means（色クラスタ）
        ms_col, ms_ov = meanshift_kmeans_segmentation(
            img_bgr, sp=12, sr=18, k=10, seed=args.seed + idx
        )

        # 2) グラフベース領域分割
        gb_col, gb_ov = graph_based_segmentation(
            img_bgr, sigma=0.5, k=300.0, min_size=120, seed=args.seed + idx
        )

        # 3) SLIC
        sl_col, sl_ov = slic_segmentation(
            img_bgr, region_size=25, ruler=10.0, iterations=10, seed=args.seed + idx
        )

        panel = tile_and_label(
            [
                img_bgr,
                ms_ov, ms_col,
                gb_ov, gb_col,
                sl_ov, sl_col,
            ],
            [
                "original",
                "Mean shift overlay", "Mean shift labels",
                "Graph overlay", "Graph labels",
                "SLIC overlay", "SLIC labels",
            ],
        )

        out_path = os.path.join(args.out_dir, f"frame_{idx:06d}.png")
        cv2.imwrite(out_path, panel)
        print(f"saved: {out_path}")

        if args.show:
            cv2.imshow("segmentation compare", panel)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
