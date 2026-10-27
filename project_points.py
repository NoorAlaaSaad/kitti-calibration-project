"""Project KITTI Velodyne points into rectified camera 03 images."""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Keep Matplotlib's cache inside the project and render without a desktop.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import colormaps, colors, pyplot as plt

DATE = "2011_09_26"
DRIVE = f"{DATE}_drive_0005_sync"
DEFAULT_SCENE = ROOT / "data" / "raw-data" / DRIVE / DATE / DRIVE
DEFAULT_CALIBRATION = ROOT / "data" / "raw-data" / f"{DATE}_calib" / DATE


@dataclass
class Calibration:
    """Matrices map column vectors from the source to the destination frame."""

    velo_to_cam0: np.ndarray
    cam0_to_cam3: np.ndarray
    rectify_cam0: np.ndarray
    rectify_cam3: np.ndarray
    projection_cam3: np.ndarray
    image_size: tuple[int, int]  # width, height


@dataclass
class Projection:
    uv: np.ndarray
    depth: np.ndarray  # forward distance (z) in rectified camera 03, in meters
    point_indices: np.ndarray


def read_calibration(path: Path) -> dict[str, np.ndarray]:
    """Read KITTI's row-major numeric fields; ignore the calibration timestamp."""
    fields = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split(":", 1)
        if key != "calib_time":
            fields[key] = np.array(value.split(), dtype=np.float64)
    return fields


def rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build the homogeneous rigid transform [[R, t], [0, 1]]."""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation).reshape(3)
    return transform


def load_calibration(directory: Path) -> Calibration:
    lidar = read_calibration(directory / "calib_velo_to_cam.txt")
    cameras = read_calibration(directory / "calib_cam_to_cam.txt")
    return Calibration(
        velo_to_cam0=rigid_transform(lidar["R"], lidar["T"]),
        cam0_to_cam3=rigid_transform(cameras["R_03"], cameras["T_03"]),
        rectify_cam0=rigid_transform(cameras["R_rect_00"], np.zeros(3)),
        rectify_cam3=rigid_transform(cameras["R_rect_03"], np.zeros(3)),
        projection_cam3=cameras["P_rect_03"].reshape(3, 4),
        image_size=tuple(int(value) for value in cameras["S_rect_03"]),
    )


def load_lidar(path: Path) -> np.ndarray:
    """Each binary record is float32 [x, y, z, reflectance], not [x, y, z, w]."""
    if path.stat().st_size == 0 or path.stat().st_size % 16 != 0:
        raise ValueError(f"Expected nonempty float32 (x, y, z, reflectance) records: {path}")
    return np.fromfile(path, dtype="<f4").reshape(-1, 4)[:, :3] # <f4: little-endian float32


def project_points(
    points: np.ndarray, calibration: Calibration, image_shape: tuple[int, int]
) -> Projection:
    """Apply the requested rigid transforms, rectify camera 03, then project."""
    height, width = image_shape
    # Replace reflectance with homogeneous w=1 so translations apply correctly.
    points_h = np.column_stack((points, np.ones(len(points))))
    transform = (
        calibration.rectify_cam3
        @ calibration.cam0_to_cam3
        @ calibration.velo_to_cam0
    ) # Its last row is [0, 0, 0, 1]

    # Nonfinite input points must not reach the perspective division.
    finite = np.isfinite(points_h).all(axis=1)
    indices = np.flatnonzero(finite)
    rectified = (transform @ points_h[finite].T).T[:, :3] # shape (N, 3)
    depth = rectified[:, 2]

    # P_rect_03's fourth column already contains the camera baseline.
    # We already moved into camera 03, so use only its intrinsic 3x3 block.
    intrinsics = calibration.projection_cam3[:, :3]
    image_h = (intrinsics @ rectified.T).T # shape (N, 3)
    in_front = (depth > 0) & (image_h[:, 2] > 0) & np.isfinite(image_h).all(axis=1)
    uv = image_h[in_front, :2] / image_h[in_front, 2:3] # shape (N, 2)
    inside = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return Projection(uv[inside], depth[in_front][inside], indices[in_front][inside])


def reference_projection_error(
    points: np.ndarray, calibration: Calibration, projection: Projection
) -> float:
    """Compare with KITTI's P_rect_03 @ R_rect_00 @ T_cam0_velo formula."""
    if len(projection.uv) == 0:
        return 0.0
    points_h = np.column_stack((points[projection.point_indices], np.ones(len(projection.uv))))
    reference_h = (
        calibration.projection_cam3
        @ calibration.rectify_cam0
        @ calibration.velo_to_cam0
        @ points_h.T
    ).T
    reference_uv = reference_h[:, :2] / reference_h[:, 2:3]
    return float(np.linalg.norm(projection.uv - reference_uv, axis=1).max())


def rasterize_depth(
    uv: np.ndarray, depths: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    """Use the nearest point per pixel. Zero means no LiDAR measurement."""
    height, width = image_shape
    depth_image = np.full((height, width), np.inf, dtype=np.float32)
    # Projection has already removed invalid and out-of-bounds coordinates.
    pixels = uv.astype(int) # round down
    flat_indices = pixels[:, 1] * width + pixels[:, 0]
    np.minimum.at(depth_image.ravel(), flat_indices, depths)
    depth_image[~np.isfinite(depth_image)] = 0
    return depth_image


def render_images(
    image: np.ndarray, depth_image: np.ndarray, max_depth: float, radius: int
) -> dict[str, np.ndarray]:
    """Create readable point displays; retain exact depths separately in .npy."""
    black = np.zeros(image.shape[:2], dtype=np.uint8)
    grayscale = black.copy()
    overlay = image.copy()
    rows, columns = np.nonzero(depth_image)
    depths = depth_image[rows, columns] # advanced indexing
    normalized = np.clip(depths / max_depth, 0, 1)

    # Matplotlib returns RGB; OpenCV drawing and image writing require BGR.
    cmap = colormaps["twilight"].resampled(256)
    rgb = (cmap(normalized)[:, :3] * 255).astype(np.uint8)
    bgr = rgb[:, ::-1]
    gray = (1 + 254 * (1 - normalized)).astype(np.uint8)

    # Draw distant points first so nearby points remain visible on overlap.
    for i in np.argsort(depths)[::-1]:
        center = (int(columns[i]), int(rows[i]))
        cv2.circle(black, center, radius, 255, thickness=-1)
        cv2.circle(grayscale, center, radius, int(gray[i]), thickness=-1)
        cv2.circle(overlay, center, radius, tuple(int(c) for c in bgr[i]), thickness=-1)
    return {
        "points_on_black": black,
        "depth_grayscale": grayscale,
        "overlay_twilight": overlay,
    }


def save_overview(
    image: np.ndarray, images: dict[str, np.ndarray], max_depth: float,
    frame: int, output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(18, 7), layout="constrained")
    panels = [
        (image[:, :, ::-1], "Camera 03 — original", None),
        (images["points_on_black"], "Projected LiDAR on black", "gray"),
        (images["depth_grayscale"], "Depth — near points are brighter", "gray"),
        (images["overlay_twilight"][:, :, ::-1], "Camera 03 + twilight depth", None),
    ]
    for axis, (panel, title, cmap) in zip(axes.flat, panels):
        axis.imshow(panel, cmap=cmap, vmin=0, vmax=255)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    for axis, cmap in [(axes[1, 0], "gray_r"), (axes[1, 1], "twilight")]:
        scale = plt.cm.ScalarMappable(norm=colors.Normalize(0, max_depth), cmap=cmap)
        figure.colorbar(scale, ax=axis, orientation="horizontal", shrink=0.8,
                        pad=0.02, label="Camera depth z (m); colors saturate at the limit")
    figure.suptitle(f"KITTI scene 0005 | Camera 03 | Frame {frame:010d}", fontsize=16)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--frame", type=int, default=0, help="Synchronized frame index (default: 0)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Color scale limit in meters")
    parser.add_argument("--radius", type=int, default=2, help="Display circle radius in pixels")
    args = parser.parse_args()
    if args.frame < 0 or not np.isfinite(args.max_depth) or args.max_depth <= 0 or args.radius < 0:
        parser.error("frame and radius must be nonnegative; max-depth must be finite and positive")

    stem = f"{args.frame:010d}"
    try:
        # Same frame index pairs the synced point cloud and camera 03 image.
        image_path = args.scene_dir / "image_03" / "data" / f"{stem}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read camera 03 image: {image_path}")
        calibration = load_calibration(args.calib_dir)
        if (image.shape[1], image.shape[0]) != calibration.image_size:
            raise ValueError("Image dimensions do not match S_rect_03; use the synced, rectified scene")
        points = load_lidar(args.scene_dir / "velodyne_points" / "data" / f"{stem}.bin")
        projection = project_points(points, calibration, image.shape[:2])
        if len(projection.uv) == 0:
            raise ValueError("No points project inside camera 03")
        depth_image = rasterize_depth(projection.uv, projection.depth, image.shape[:2])
        images = render_images(image, depth_image, args.max_depth, args.radius)
        output = args.output_dir / f"frame_{stem}"
        output.mkdir(parents=True, exist_ok=True)
        for name, rendered in images.items():
            if not cv2.imwrite(str(output / f"{name}.png"), rendered):
                raise OSError(f"Could not save {name}.png")
        np.save(output / "depth_meters.npy", depth_image)
        save_overview(image, images, args.max_depth, args.frame, output / "overview.png")

        error = reference_projection_error(points, calibration, projection)
        summary = {
            "scene": args.scene_dir.name,
            "camera": "03",
            "frame": args.frame,
            "image_size": list(calibration.image_size),
            "input_points": len(points),
            "visible_points": len(projection.uv),
            "occupied_pixels": int(np.count_nonzero(depth_image)),
            "depth_min_m": float(projection.depth.min()),
            "depth_max_m": float(projection.depth.max()),
            "color_limit_m": args.max_depth,
            "display_radius_px": args.radius,
            "reference_max_error_px": error,
            "matrices": {
                "T_cam0_from_velo": calibration.velo_to_cam0.tolist(),
                "T_cam3_from_cam0": calibration.cam0_to_cam3.tolist(),
                "R_rect_00": calibration.rectify_cam0[:3, :3].tolist(),
                "R_rect_03": calibration.rectify_cam3[:3, :3].tolist(),
                "P_rect_03": calibration.projection_cam3.tolist(),
                "T_rect_cam3_from_velo": (
                    calibration.rectify_cam3 @ calibration.cam0_to_cam3 @ calibration.velo_to_cam0
                ).tolist(),
            },
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Camera 03, frame {stem}: {len(points):,} input points")
    print(f"{len(projection.uv):,} visible points; {summary['occupied_pixels']:,} depth pixels")
    print(f"Maximum difference from KITTI reference projection: {error:.6f} px")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
