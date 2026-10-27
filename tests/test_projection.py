"""Small numerical examples that exercise the projection's geometry and visibility."""

from pathlib import Path

import numpy as np
import pytest

from project_points import (
    Calibration,
    load_calibration,
    load_lidar,
    project_points,
    rasterize_depth,
    read_calibration,
    reference_projection_error,
    rigid_transform,
)


def identity_calibration() -> Calibration:
    """A unit-focal-length camera makes expected pixel coordinates immediate."""
    return Calibration(
        velo_to_cam0=np.eye(4),
        cam0_to_cam3=np.eye(4),
        rectify_cam0=np.eye(4),
        rectify_cam3=np.eye(4),
        projection_cam3=np.column_stack((np.eye(3), np.zeros(3))),
        image_size=(6, 4),
    )


def test_rigid_transform_rotates_then_translates() -> None:
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform = rigid_transform(rotation, np.array([2, 3, 4]))

    np.testing.assert_allclose(transform @ [1, 2, 3, 1], [0, 4, 7, 1])
    np.testing.assert_array_equal(transform[3], [0, 0, 0, 1])


def test_projection_applies_each_coordinate_transform_in_order() -> None:
    """For the first point: (1,2,8) -> (9,4,2) -> (-2,8,3) -> (-2,-3,8)."""
    rotation_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rotation_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotation_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    calibration = Calibration(
        velo_to_cam0=rigid_transform(rotation_y, np.array([1, 2, 3])),
        cam0_to_cam3=rigid_transform(rotation_z, np.array([2, -1, 1])),
        rectify_cam0=np.eye(4),
        rectify_cam3=rigid_transform(rotation_x, np.zeros(3)),
        # The final column must not be applied to points already in camera 03.
        projection_cam3=np.array([[4, 0, 10, 777], [0, 6, 10, -555], [0, 0, 1, 8]]),
        image_size=(20, 20),
    )

    result = project_points(np.array([[1, 2, 8], [2, 1, 6]]), calibration, (20, 20))

    np.testing.assert_allclose(result.uv, [[9, 7.75], [28 / 3, 8]])
    np.testing.assert_allclose(result.depth, [8, 6])
    np.testing.assert_array_equal(result.point_indices, [0, 1])


def test_visibility_checks_float_coordinates_and_positive_camera_depth() -> None:
    points = np.array(
        [
            [1, 1, 1],
            [0, 0, 2],
            [5.9, 3.9, 1],
            [-0.001, 1, 1],
            [1, -0.001, 1],
            [6, 1, 1],
            [1, 4, 1],
            [1, 1, 0],
            [-1, -1, -1],
            [np.nan, 1, 1],
            [1, np.inf, 1],
            [1, 1, np.inf],
            [2, 2, 2],
        ]
    )

    result = project_points(points, identity_calibration(), (4, 6))

    np.testing.assert_array_equal(result.point_indices, [0, 1, 2, 12])
    np.testing.assert_allclose(result.uv, [[1, 1], [0, 0], [5.9, 3.9], [1, 1]])
    np.testing.assert_allclose(result.depth, [1, 2, 1, 2])


def test_projection_accepts_an_empty_point_cloud() -> None:
    result = project_points(np.empty((0, 3)), identity_calibration(), (4, 6))

    assert result.uv.shape == (0, 2)
    assert result.depth.shape == (0,)
    assert result.point_indices.shape == (0,)


def test_rasterization_keeps_nearest_depth_and_is_independent_of_input_order() -> None:
    uv = np.array([[1.1, 2.2], [1.9, 2.9], [1.1, 2.2], [5.9, 0.1]])
    depths = np.array([10, 2, 7, 4])
    expected = np.zeros((4, 6), dtype=np.float32)
    expected[2, 1] = 2
    expected[0, 5] = 4

    result = rasterize_depth(uv, depths, (4, 6))
    shuffled = rasterize_depth(uv[::-1], depths[::-1], (4, 6))

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(shuffled, expected)


def test_empty_rasterization_has_a_black_background() -> None:
    result = rasterize_depth(np.empty((0, 2)), np.empty(0), (4, 6))

    np.testing.assert_array_equal(result, np.zeros((4, 6), dtype=np.float32))


def test_lidar_reader_excludes_reflectance(tmp_path: Path) -> None:
    scan = tmp_path / "scan.bin"
    np.array([[1, 2, 3, 0.25], [-4, 5, 6, 0.75]], dtype="<f4").tofile(scan)

    np.testing.assert_array_equal(load_lidar(scan), [[1, 2, 3], [-4, 5, 6]])


def test_calibration_reader_skips_timestamp_and_preserves_numeric_fields(tmp_path: Path) -> None:
    path = tmp_path / "calibration.txt"
    path.write_text("calib_time: 09-Jan-2012 13:57:47\n\nT: 1 -2 3e-3\n", encoding="utf-8")

    result = read_calibration(path)

    assert "calib_time" not in result
    np.testing.assert_allclose(result["T"], [1, -2, 0.003])


def test_camera03_projection_agrees_with_kitti_reference_on_available_data() -> None:
    """Optional integration check; the large KITTI download is not a test dependency."""
    data = Path(__file__).resolve().parents[1] / "data" / "raw-data"
    calibration_dir = data / "2011_09_26_calib" / "2011_09_26"
    scan = (
        data
        / "2011_09_26_drive_0005_sync"
        / "2011_09_26"
        / "2011_09_26_drive_0005_sync"
        / "velodyne_points"
        / "data"
        / "0000000000.bin"
    )
    if not scan.is_file() or not (calibration_dir / "calib_cam_to_cam.txt").is_file():
        pytest.skip("The optional local KITTI scene and calibration are unavailable.")

    calibration = load_calibration(calibration_dir)
    points = load_lidar(scan)
    width, height = calibration.image_size
    result = project_points(points, calibration, (height, width))

    assert points.shape == (123397, 3)
    # One point lies across an image boundary in the official reference chain;
    # the explicit camera-03 route retains 20,501 points before rasterization.
    assert result.uv.shape == (20501, 2)
    # The raw-camera and official rectified-reference calibrations have a small
    # residual difference; demanding exact equality would reject valid data.
    assert reference_projection_error(points, calibration, result) < 0.06
