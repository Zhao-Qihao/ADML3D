import argparse
import os
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from tqdm import tqdm

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BEVFusion predictions by projecting 3D bboxes onto multi-view camera images and saving a video."
    )
    parser.add_argument("config", help="Path to BEVFusion config file")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--threshold", type=float, default=0.01, help="Score threshold for predictions")
    parser.add_argument("--step", type=int, default=40, help="Number of steps to visualize (-1 for full)")
    parser.add_argument("--cam_order", type=int, nargs="*", default=None, help="Custom camera order, e.g., 0 1 2 3")
    parser.add_argument("--fps", type=float, default=2.0, help="FPS of output video")
    parser.add_argument("--out", type=str, default=None, help="Output video path (default: work_dir/visualization/visualization_bevfusion.mp4)")
    parser.add_argument("--show-pred", action="store_true", help="Show predictions instead of ground truth")
    return parser.parse_args()


def project_to_image(points, lidar2cam, cam2img):
    """Project 3D points to image plane, supporting 3x3 or 4x4 cam2img."""
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = np.dot(lidar2cam, points_hom.T).T
    valid_mask = points_cam[:, 2] > 0

    if cam2img.shape == (3, 3):
        proj = np.dot(cam2img, points_cam[:, :3].T).T
        proj = proj / proj[:, 2:3]
        return proj[:, :2], valid_mask
    elif cam2img.shape == (4, 4):
        points_cam_h = np.hstack((points_cam[:, :3], np.ones((points_cam.shape[0], 1))))
        proj_h = np.dot(cam2img, points_cam_h.T).T
        proj = proj_h[:, :3] / proj_h[:, 2:3]
        return proj[:, :2], valid_mask
    else:
        raise ValueError(f"Unsupported cam2img shape: {cam2img.shape}")


def draw_projected_3d_bboxes(ax, bboxes, labels, lidar2cam, cam2img, img_shape, classes, linewidth=1):
    """
    Projects 3D bounding boxes to 2D and draws them on the given axis with class-specific colors.
    """
    H, W = img_shape[:2]

    def clip_line(pt1, pt2, W, H):
        """Clip a line segment to the image bounds."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Both points are inside
        if (0 <= x1 < W and 0 <= y1 < H) and (0 <= x2 < W and 0 <= y2 < H):
            return [pt1, pt2]

        # If both are out of bounds, skip
        if (x1 < 0 and x2 < 0) or (x1 >= W and x2 >= W) or (y1 < 0 and y2 < 0) or (y1 >= H and y2 >= H):
            return None

        # Simple clipping
        pt1_clipped = [np.clip(x1, 0, W - 1), np.clip(y1, 0, H - 1)]
        pt2_clipped = [np.clip(x2, 0, W - 1), np.clip(y2, 0, H - 1)]
        return [pt1_clipped, pt2_clipped]

    if bboxes is None or len(bboxes) == 0:
        return

    # only take the first 7 dims for corners
    lidar_boxes = LiDARInstance3DBoxes(bboxes[:, :7])
    corners_3d = lidar_boxes.corners.cpu().numpy()  # (N, 8, 3)

    # Edges of a 3D bbox
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
    ]

    for idx, corners in enumerate(corners_3d):
        name = classes[labels[idx]]
        color = np.array(OBJECT_PALETTE[name]) / 255
        
        projected, valid_mask = project_to_image(corners, lidar2cam, cam2img)
        if not valid_mask.all():
            continue
        for j, k in lines:
            pt1, pt2 = projected[j], projected[k]
            clipped = clip_line(pt1, pt2, W, H)
            if clipped:
                ax.plot([clipped[0][0], clipped[1][0]],
                        [clipped[0][1], clipped[1][1]],
                        color=color, linewidth=linewidth)


def infer_grid(n_views: int):
    # Prefer 2x3 for 6, 2x2 for 4; otherwise near-square grid
    if n_views == 6:
        return 2, 3
    if n_views == 4:
        return 2, 2
    rows = int(math.floor(math.sqrt(n_views)))
    cols = int(math.ceil(n_views / rows))
    return rows, cols


def main():
    args = parse_args()
    init_default_scope("mmdet3d")

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1

    classes = cfg.class_names

    # dataloader
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # model
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(get_device())
    model.eval()

    os.makedirs(os.path.join(cfg.get("work_dir", "./work_dirs/default"), "visualization"), exist_ok=True)
    video_path = args.out or os.path.join(cfg.get("work_dir", "./work_dirs/default"),
                                          "visualization",
                                          "visualization_bevfusion.mp4")

    video_writer = None
    steps = args.step if args.step != -1 else len(dataloader)

    pbar = tqdm(range(steps))
    data_iter = iter(dataloader)
    for _ in pbar:
        data = next(data_iter)  # each iteration yields one batch
        with torch.no_grad():
            outputs = model.test_step(data)
        
        if args.show_pred:
            # visualize predictions
            pred = outputs[0].pred_instances_3d
            bboxes = pred["bboxes_3d"].tensor.detach().cpu()
            labels = pred["labels_3d"].detach().cpu() if "labels_3d" in pred else None
            scores = pred["scores_3d"].detach().cpu() if "scores_3d" in pred else None
            if args.threshold is not None and scores is not None:
                keep = scores >= args.threshold
                bboxes = bboxes[keep]
                if labels is not None:
                    labels = labels[keep]
        else:
            # visualize ground truth
            gt_data = data['data_samples'][0].eval_ann_info
            bboxes = gt_data['gt_bboxes_3d'].tensor.detach().cpu()
            labels = gt_data['gt_labels_3d']

        # sample meta
        sample = data["data_samples"][0]
        # img_path: list[str]; lidar2cam/cam2img: list[np.ndarray]
        img_paths = getattr(sample, "img_path", None)
        lidar2cam_list = getattr(sample, "lidar2cam", None)
        cam2img_list = getattr(sample, "cam2img", None)

        if img_paths is None or lidar2cam_list is None or cam2img_list is None:
            print("Missing camera metadata (img_path/lidar2cam/cam2img). Skipping frame.")
            continue

        num_views = len(img_paths)
        cam_order = args.cam_order if args.cam_order is not None else list(range(num_views))
        rows, cols = infer_grid(len(cam_order))

        # build figure
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = np.array(axs).reshape(-1)
        for i, cam_id in enumerate(cam_order):
            if i >= len(axs):
                break
            img_path = img_paths[cam_id]
            img = cv2.imread(img_path)
            if img is None:
                axs[i].axis("off")
                axs[i].set_title(f"Camera {cam_id} (missing)")
                continue
            # BGR -> RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            lidar2cam = np.array(lidar2cam_list[cam_id], dtype=np.float64)
            cam2img = np.array(cam2img_list[cam_id], dtype=np.float64)

            ax = axs[i]
            ax.imshow(img_rgb)
            draw_projected_3d_bboxes(ax, bboxes, labels, lidar2cam, cam2img, img_rgb.shape, classes)
            ax.axis("off")
            ax.set_title(f"Camera {cam_id}")

        # hide unused axes
        for j in range(len(cam_order), len(axs)):
            axs[j].axis("off")

        plt.tight_layout()

        # figure -> frame
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        # init writer after first frame (match exact size)
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frame.shape[:2]
            video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))

        video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()
    print(f"Saved video to: {video_path}")


if __name__ == "__main__":
    main()