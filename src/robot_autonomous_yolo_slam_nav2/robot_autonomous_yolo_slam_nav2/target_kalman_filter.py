"""Kalman Filter sensor fusion of Camera (YOLO bbox/footpoint) and 2D LiDAR in base_link.

State vector in ``base_link``:
    x = [px, py, vx, vy]^T

Why Camera + LiDAR fusion works better than raw single-frame LiDAR:
1. Camera provides high-rate, low-noise azimuth/bearing ``theta_cam`` from the
   bounding-box center ray and a coarse monocular range estimate ``r_cam``
   from apparent torso width, body height, and ground-contact footpoint.
   Its measurement covariance ``R_cam`` is anisotropic: tight perpendicular
   to the viewing ray (bearing) and wide along the viewing ray (depth).
2. LiDAR provides accurate metric range ``r_lidar`` from either 3D-to-2D bbox
   projection or an angular bearing-window search around the camera ray, with
   tight radial variance in ``R_lidar``.
3. Sequential Kalman updates fuse both sensors: LiDAR locks metric depth while
   Camera pins down lateral bearing and bridges 4 Hz LiDAR scan gaps or brief
   leg-occlusion dropouts without dropping target validity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .metric_association import (
    BaseToCameraTransform,
    BoundingBox,
    CameraCalibration,
    ImageSize,
    LidarPoint,
    MetricPersonObservation,
    Point3D,
    associate_metric_observations,
)
from .metric_target_scan_helpers import (
    LaserScanFields,
    LaserToBaseTransform,
    bearing_window_point,
)
from .metric_target_selection_helpers import metric_obstacles_to_xyz
from .target_state import TargetSelectionResult, TrackId

if TYPE_CHECKING:
    from .metric_target_branches import BranchDecision, LidarInputs


Cov2x2 = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class KalmanFusionConfig:
    """Tunable parameters for Camera + LiDAR Kalman Filter fusion."""

    process_pos_noise: float = 0.15
    process_vel_noise: float = 0.60
    velocity_damping: float = 0.85
    max_speed_m_s: float = 2.2
    max_coast_s: float = 1.2
    lidar_coast_s: float = 1.5
    lidar_bbox_range_std_m: float = 0.04
    lidar_bbox_lateral_rel_std: float = 0.025
    lidar_window_range_std_m: float = 0.07
    lidar_window_lateral_rel_std: float = 0.04
    camera_bearing_std_rad: float = 0.035
    camera_range_rel_std: float = 0.25
    camera_min_range_std_m: float = 0.25
    expected_person_width_m: float = 0.45
    expected_person_height_m: float = 1.68
    min_bbox_height_px: float = 20.0
    min_bbox_width_px: float = 12.0
    min_range_m: float = 0.25
    max_range_m: float = 6.0
    bearing_window_half_angle_rad: float = 0.22
    mahalanobis_gate_sq: float = 16.0
    allow_camera_only: bool = True


@dataclass(frozen=True, slots=True)
class CameraFusionInputs:
    """Validated camera target geometry and calibration for one cycle."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    calibration: CameraCalibration
    camera_transform: BaseToCameraTransform
    image_size: ImageSize
    stamp_ns: int
    footpoint_valid: bool = False
    footpoint_x: float = 0.0
    footpoint_y: float = 0.0
    footpoint_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class CameraMeasurement:
    """2D base_link measurement and anisotropic covariance derived from camera."""

    point_base_link: Point3D
    range_m: float
    bearing_rad: float
    covariance_2x2: Cov2x2
    confidence: float
    provenance: str
    stamp_ns: int


@dataclass(frozen=True, slots=True)
class LidarMeasurement:
    """2D base_link measurement and anisotropic covariance derived from LiDAR."""

    point_base_link: Point3D
    range_m: float
    bearing_rad: float
    covariance_2x2: Cov2x2
    confidence: float
    provenance: str
    stamp_ns: int


def polar_to_cartesian_covariance(
    bearing_rad: float,
    radial_std_m: float,
    lateral_std_m: float,
) -> Cov2x2:
    """Build a 2x2 Cartesian covariance rotated to align with ``bearing_rad``."""
    cos_b = math.cos(bearing_rad)
    sin_b = math.sin(bearing_rad)
    var_r = max(1e-4, radial_std_m * radial_std_m)
    var_lat = max(1e-4, lateral_std_m * lateral_std_m)
    r00 = cos_b * cos_b * var_r + sin_b * sin_b * var_lat
    r01 = cos_b * sin_b * (var_r - var_lat)
    r11 = sin_b * sin_b * var_r + cos_b * cos_b * var_lat
    return ((r00, r01), (r01, r11))


def camera_origin_in_base(base_to_camera: BaseToCameraTransform) -> Point3D:
    """Compute camera optical center in base_link from ``p_cam = R * p_base + t``."""
    rows = base_to_camera.rotation
    tx, ty, tz = base_to_camera.translation
    return (
        -(rows[0][0] * tx + rows[1][0] * ty + rows[2][0] * tz),
        -(rows[0][1] * tx + rows[1][1] * ty + rows[2][1] * tz),
        -(rows[0][2] * tx + rows[1][2] * ty + rows[2][2] * tz),
    )


def camera_pixel_ray_in_base(
    pixel_x: float,
    pixel_y: float,
    calibration: CameraCalibration,
    base_to_camera: BaseToCameraTransform,
) -> Point3D | None:
    """Unproject an image pixel into a direction vector in ``base_link``."""
    if calibration.fx <= 0.0 or calibration.fy <= 0.0:
        return None
    ray_cam = (
        (pixel_x - calibration.cx) / calibration.fx,
        (pixel_y - calibration.cy) / calibration.fy,
        1.0,
    )
    rows = base_to_camera.rotation
    ray_base = (
        rows[0][0] * ray_cam[0] + rows[1][0] * ray_cam[1] + rows[2][0] * ray_cam[2],
        rows[0][1] * ray_cam[0] + rows[1][1] * ray_cam[1] + rows[2][1] * ray_cam[2],
        rows[0][2] * ray_cam[0] + rows[1][2] * ray_cam[1] + rows[2][2] * ray_cam[2],
    )
    if not all(math.isfinite(v) for v in ray_base):
        return None
    if math.hypot(ray_base[0], ray_base[1]) <= 1e-6:
        return None
    return ray_base


def estimate_camera_measurement(
    camera: CameraFusionInputs,
    config: KalmanFusionConfig = KalmanFusionConfig(),
    prior_range_m: float | None = None,
) -> CameraMeasurement | None:
    """Estimate target bearing and monocular depth from a confirmed YOLO bbox."""
    left, top, right, bottom = camera.bbox
    width_img, height_img = camera.image_size
    if (
        width_img <= 0
        or height_img <= 0
        or not all(math.isfinite(v) for v in (left, top, right, bottom))
    ):
        return None
    bbox_w = right - left
    bbox_h = bottom - top
    if bbox_w < config.min_bbox_width_px or bbox_h < config.min_bbox_height_px:
        return None
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    if not (0.0 <= center_x < width_img and 0.0 <= center_y < height_img):
        return None

    ray_base = camera_pixel_ray_in_base(
        center_x, center_y, camera.calibration, camera.camera_transform
    )
    if ray_base is None:
        return None
    horiz_norm = math.hypot(ray_base[0], ray_base[1])
    unit_x = ray_base[0] / horiz_norm
    unit_y = ray_base[1] / horiz_norm

    border_margin = 4.0
    horiz_clipped = left <= border_margin or right >= (width_img - border_margin)
    vert_clipped = top <= border_margin or bottom >= (height_img - border_margin)

    depth_w = camera.calibration.fx * config.expected_person_width_m / bbox_w
    depth_h = camera.calibration.fy * config.expected_person_height_m / bbox_h

    # Combine width and height monocular cues according to border clipping.
    if not vert_clipped and not horiz_clipped:
        est_depth = 0.55 * depth_h + 0.45 * depth_w
        rel_std = config.camera_range_rel_std
        provenance = "camera_bbox_wh"
    elif vert_clipped and not horiz_clipped:
        # At close range (0.8m - 1.5m), vertical bbox often clips the 240px height,
        # while torso/shoulder width remains unclipped.
        est_depth = max(depth_w, 0.7 * depth_h)
        rel_std = config.camera_range_rel_std * 1.15
        provenance = "camera_bbox_width"
    elif horiz_clipped and not vert_clipped:
        est_depth = depth_h
        rel_std = config.camera_range_rel_std * 1.15
        provenance = "camera_bbox_height"
    else:
        est_depth = min(depth_w, depth_h)
        rel_std = config.camera_range_rel_std * 1.5
        provenance = "camera_bbox_clipped"

    origin = camera_origin_in_base(camera.camera_transform)

    # Fuse ground-plane intersection when validated footpoint is available.
    if (
        camera.footpoint_valid
        and math.isfinite(camera.footpoint_x)
        and math.isfinite(camera.footpoint_y)
        and camera.footpoint_confidence >= 0.3
        and origin[2] > 0.05
    ):
        foot_ray = camera_pixel_ray_in_base(
            camera.footpoint_x,
            camera.footpoint_y,
            camera.calibration,
            camera.camera_transform,
        )
        if foot_ray is not None and foot_ray[2] < -0.03:
            foot_horiz = math.hypot(foot_ray[0], foot_ray[1])
            ground_range = (origin[2] / (-foot_ray[2])) * foot_horiz
            if config.min_range_m <= ground_range <= config.max_range_m:
                w_foot = min(0.5, 0.5 * float(camera.footpoint_confidence))
                est_depth = (1.0 - w_foot) * est_depth + w_foot * ground_range
                rel_std *= 0.85
                provenance = "camera_footpoint_bbox"

    if not math.isfinite(est_depth) or est_depth <= 0.0:
        return None
    if est_depth > config.max_range_m * 1.5:
        return None
    est_depth = min(max(est_depth, config.min_range_m), config.max_range_m)

    # Blend softly with prior Kalman/LiDAR range when available so monocular scale
    # shifts (e.g. person turning sideways) do not cause radial jumps.
    if prior_range_m is not None and math.isfinite(prior_range_m) and prior_range_m > 0.0:
        radial_anchor = 0.35 * est_depth + 0.65 * prior_range_m
    else:
        radial_anchor = est_depth

    px = origin[0] + radial_anchor * unit_x
    py = origin[1] + radial_anchor * unit_y
    pz = 0.0
    range_m = math.hypot(px, py)
    bearing_rad = math.atan2(py, px)
    if range_m <= 0.0 or not math.isfinite(range_m) or not math.isfinite(bearing_rad):
        return None

    radial_std = max(config.camera_min_range_std_m, est_depth * rel_std)
    lateral_std = max(0.03, range_m * config.camera_bearing_std_rad)
    cov = polar_to_cartesian_covariance(bearing_rad, radial_std, lateral_std)
    conf = min(0.85, max(0.35, float(camera.confidence) * 0.75))
    return CameraMeasurement(
        point_base_link=(px, py, pz),
        range_m=range_m,
        bearing_rad=bearing_rad,
        covariance_2x2=cov,
        confidence=conf,
        provenance=provenance,
        stamp_ns=camera.stamp_ns,
    )


def extract_lidar_measurement(
    selection: TargetSelectionResult,
    lidar: LidarInputs,
    config: KalmanFusionConfig = KalmanFusionConfig(),
    predicted_range_m: float | None = None,
) -> tuple[LidarMeasurement | None, tuple[Point3D, ...]]:
    """Extract target LiDAR measurement (bbox or bearing-window fallback) and obstacles."""
    assoc = associate_metric_observations(
        selection,
        lidar.points,
        lidar.calibration,
        lidar.camera_transform,
        lidar.image_size,
        lidar.stamp_ns,
    )
    obstacles = metric_obstacles_to_xyz(assoc.obstacles)
    target_obs = selection.target_observation
    if target_obs is None:
        return None, obstacles

    if assoc.target_observation is not None:
        pt = assoc.target_observation.point_base_link
        r_m = assoc.target_observation.range_m
        b_rad = assoc.target_observation.bearing_rad
        cov = polar_to_cartesian_covariance(
            b_rad,
            config.lidar_bbox_range_std_m,
            max(0.04, r_m * config.lidar_bbox_lateral_rel_std),
        )
        return (
            LidarMeasurement(
                point_base_link=pt,
                range_m=r_m,
                bearing_rad=b_rad,
                covariance_2x2=cov,
                confidence=lidar.position_confidence,
                provenance="lidar_bbox",
                stamp_ns=lidar.stamp_ns,
            ),
            obstacles,
        )

    # Fallback: bearing-window search when 2D LiDAR scan plane misses vertical bbox
    # (e.g. close-range vertical crop, camera pitch, or leg gap).
    scan = getattr(lidar, "scan", None)
    laser_transform = getattr(lidar, "laser_transform", None)
    if scan is not None and laser_transform is not None:
        left, top, right, bottom = target_obs.bbox
        bbox_w = right - left
        bbox_h = bottom - top
        if bbox_w >= config.min_bbox_width_px and bbox_h >= config.min_bbox_height_px:
            half_angle = max(
                0.08,
                min(
                    config.bearing_window_half_angle_rad,
                    0.6 * math.atan2(bbox_w, 2.0 * max(1.0, lidar.calibration.fx)),
                ),
            )
            window_pt = bearing_window_point(
                target_obs.bbox,
                lidar.calibration,
                lidar.camera_transform,
                lidar.image_size,
                scan,
                laser_transform,
                half_angle,
            )
            if window_pt is not None:
                r_m = math.hypot(window_pt[0], window_pt[1])
                b_rad = math.atan2(window_pt[1], window_pt[0])
                expected_d = (
                    lidar.calibration.fx * config.expected_person_width_m / max(1.0, bbox_w)
                )
                depth_lo = max(config.min_range_m, expected_d * 0.35)
                depth_hi = min(config.max_range_m, expected_d * 3.0)
                if predicted_range_m is not None and predicted_range_m > 0.0:
                    depth_lo = max(config.min_range_m, min(depth_lo, predicted_range_m - 0.8))
                    depth_hi = min(config.max_range_m, max(depth_hi, predicted_range_m + 0.9))
                # Ensure point is not an already-claimed non-target obstacle.
                claimed_by_obstacle = any(
                    math.hypot(window_pt[0] - ox, window_pt[1] - oy) < 0.25
                    for ox, oy, _oz in obstacles
                )
                if not claimed_by_obstacle and depth_lo <= r_m <= depth_hi:
                    cov = polar_to_cartesian_covariance(
                        b_rad,
                        config.lidar_window_range_std_m,
                        max(0.05, r_m * config.lidar_window_lateral_rel_std),
                    )
                    return (
                        LidarMeasurement(
                            point_base_link=window_pt,
                            range_m=r_m,
                            bearing_rad=b_rad,
                            covariance_2x2=cov,
                            confidence=max(0.75, lidar.position_confidence - 0.08),
                            provenance="lidar_bearing_window",
                            stamp_ns=lidar.stamp_ns,
                        ),
                        obstacles,
                    )

    return None, obstacles


class TargetKalmanFilter:
    """4-state constant-velocity Kalman Filter fusing Camera and LiDAR in base_link."""

    def __init__(self, config: KalmanFusionConfig | None = None) -> None:
        self._config = config or KalmanFusionConfig()
        self._track_id: int | None = None
        self._x = np.zeros(4, dtype=float)  # [px, py, vx, vy]
        self._z_height = 0.0
        self._p = np.eye(4, dtype=float)
        self._last_stamp_ns: int = 0
        self._last_lidar_stamp_ns: int = 0
        self._has_lidar_lock: bool = False
        self._initialized: bool = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def track_id(self) -> int | None:
        return self._track_id

    @property
    def has_lidar_lock(self) -> bool:
        return self._has_lidar_lock

    @property
    def position_xy(self) -> tuple[float, float] | None:
        if not self._initialized:
            return None
        return (float(self._x[0]), float(self._x[1]))

    @property
    def velocity_xy(self) -> tuple[float, float] | None:
        if not self._initialized:
            return None
        return (float(self._x[2]), float(self._x[3]))

    @property
    def range_m(self) -> float | None:
        if not self._initialized:
            return None
        return math.hypot(float(self._x[0]), float(self._x[1]))

    def reset(self) -> None:
        """Clear filter state on target loss or identity switch."""
        self._track_id = None
        self._x = np.zeros(4, dtype=float)
        self._z_height = 0.0
        self._p = np.eye(4, dtype=float)
        self._last_stamp_ns = 0
        self._last_lidar_stamp_ns = 0
        self._has_lidar_lock = False
        self._initialized = False

    def _initialize_state(
        self,
        track_id: int,
        point: Point3D,
        cov_2x2: Cov2x2,
        stamp_ns: int,
        is_lidar: bool,
    ) -> None:
        self._track_id = track_id
        self._x = np.array([point[0], point[1], 0.0, 0.0], dtype=float)
        self._z_height = float(point[2])
        self._p = np.zeros((4, 4), dtype=float)
        self._p[0, 0] = cov_2x2[0][0]
        self._p[0, 1] = cov_2x2[0][1]
        self._p[1, 0] = cov_2x2[1][0]
        self._p[1, 1] = cov_2x2[1][1]
        self._p[2, 2] = 0.5
        self._p[3, 3] = 0.5
        self._last_stamp_ns = stamp_ns
        if is_lidar:
            self._has_lidar_lock = True
            self._last_lidar_stamp_ns = stamp_ns
        else:
            self._has_lidar_lock = False
            self._last_lidar_stamp_ns = 0
        self._initialized = True

    def predict(self, stamp_ns: int) -> bool:
        """Advance state and covariance to ``stamp_ns``."""
        if not self._initialized or stamp_ns <= 0:
            return False
        dt_s = (stamp_ns - self._last_stamp_ns) / 1_000_000_000.0
        if dt_s < -0.5 or dt_s > self._config.max_coast_s:
            self.reset()
            return False
        if dt_s <= 0.0:
            return True

        damping = math.pow(self._config.velocity_damping, dt_s * 10.0)
        f_mat = np.array(
            [
                [1.0, 0.0, dt_s, 0.0],
                [0.0, 1.0, 0.0, dt_s],
                [0.0, 0.0, damping, 0.0],
                [0.0, 0.0, 0.0, damping],
            ],
            dtype=float,
        )
        self._x = f_mat @ self._x
        speed = math.hypot(float(self._x[2]), float(self._x[3]))
        if speed > self._config.max_speed_m_s and speed > 0.0:
            scale = self._config.max_speed_m_s / speed
            self._x[2] *= scale
            self._x[3] *= scale

        qp = self._config.process_pos_noise
        qv = self._config.process_vel_noise
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        q_mat = np.array(
            [
                [0.333 * dt3 * qv + qp * dt_s, 0.0, 0.5 * dt2 * qv, 0.0],
                [0.0, 0.333 * dt3 * qv + qp * dt_s, 0.0, 0.5 * dt2 * qv],
                [0.5 * dt2 * qv, 0.0, dt_s * qv, 0.0],
                [0.0, 0.5 * dt2 * qv, 0.0, dt_s * qv],
            ],
            dtype=float,
        )
        self._p = f_mat @ self._p @ f_mat.T + q_mat
        self._last_stamp_ns = stamp_ns
        return True

    def update_position(
        self,
        point: Point3D,
        cov_2x2: Cov2x2,
        *,
        is_lidar: bool,
        stamp_ns: int,
    ) -> bool:
        """Apply a 2D Cartesian measurement update with Mahalanobis gating."""
        if not all(math.isfinite(v) for v in point):
            return False
        z = np.array([point[0], point[1]], dtype=float)
        r_mat = np.array(
            [
                [cov_2x2[0][0], cov_2x2[0][1]],
                [cov_2x2[1][0], cov_2x2[1][1]],
            ],
            dtype=float,
        )
        h_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        innovation = z - (h_mat @ self._x)
        s_mat = h_mat @ self._p @ h_mat.T + r_mat
        det = float(s_mat[0, 0] * s_mat[1, 1] - s_mat[0, 1] * s_mat[1, 0])
        if not math.isfinite(det) or det <= 1e-12:
            return False
        s_inv = np.array(
            [
                [s_mat[1, 1] / det, -s_mat[0, 1] / det],
                [-s_mat[1, 0] / det, s_mat[0, 0] / det],
            ],
            dtype=float,
        )
        mahalanobis_sq = float(innovation.T @ s_inv @ innovation)
        if not math.isfinite(mahalanobis_sq):
            return False

        # Outlier handling:
        # - If we only had a weak monocular camera lock and LiDAR arrives for the
        #   first time, snap depth to LiDAR immediately.
        # - If already LiDAR-locked and a measurement jumps beyond the gate
        #   (e.g. a single ray hitting a background wall), inflate R so the
        #   estimate resists teleporting while still converging if persistent.
        if is_lidar and not self._has_lidar_lock:
            self._x[0] = point[0]
            self._x[1] = point[1]
            self._z_height = float(point[2])
            self._p[0, 0] = cov_2x2[0][0]
            self._p[0, 1] = cov_2x2[0][1]
            self._p[1, 0] = cov_2x2[1][0]
            self._p[1, 1] = cov_2x2[1][1]
            self._has_lidar_lock = True
            self._last_lidar_stamp_ns = stamp_ns
            return True

        if mahalanobis_sq > self._config.mahalanobis_gate_sq:
            inflate = mahalanobis_sq / self._config.mahalanobis_gate_sq
            weight = max(4.0, inflate * 4.0) if (not is_lidar and self._has_lidar_lock) else max(3.0, inflate * 2.5)
            r_mat = r_mat + weight * s_mat
            s_mat = h_mat @ self._p @ h_mat.T + r_mat
            det = float(s_mat[0, 0] * s_mat[1, 1] - s_mat[0, 1] * s_mat[1, 0])
            if not math.isfinite(det) or det <= 1e-12:
                return False
            s_inv = np.array(
                [
                    [s_mat[1, 1] / det, -s_mat[0, 1] / det],
                    [-s_mat[1, 0] / det, s_mat[0, 0] / det],
                ],
                dtype=float,
            )

        k_gain = self._p @ h_mat.T @ s_inv
        self._x = self._x + (k_gain @ innovation)
        speed = math.hypot(float(self._x[2]), float(self._x[3]))
        if speed > self._config.max_speed_m_s and speed > 0.0:
            scale = self._config.max_speed_m_s / speed
            self._x[2] *= scale
            self._x[3] *= scale

        ident = np.eye(4, dtype=float)
        ikh = ident - (k_gain @ h_mat)
        self._p = (ikh @ self._p @ ikh.T) + (k_gain @ r_mat @ k_gain.T)

        if is_lidar:
            self._z_height = float(point[2])
            self._has_lidar_lock = True
            self._last_lidar_stamp_ns = stamp_ns
        if stamp_ns > self._last_stamp_ns:
            self._last_stamp_ns = stamp_ns
        return True

    def posterior_confidence(self, base_confidence: float) -> float:
        """Compute geometry confidence (0..1) from posterior position variance."""
        if not self._initialized:
            return 0.0
        pos_var = max(0.0, float(self._p[0, 0] + self._p[1, 1]))
        pos_std = math.sqrt(pos_var)
        uncertainty_penalty = 1.0 / (1.0 + 1.8 * pos_std)
        conf = base_confidence * (0.4 + 0.6 * uncertainty_penalty)
        return min(0.99, max(0.1, conf))

    def fuse(
        self,
        selection: TargetSelectionResult,
        lidar: LidarInputs | None,
        camera: CameraFusionInputs | None,
    ) -> BranchDecision:
        """Fuse Camera and LiDAR measurements for the locked target."""
        from .metric_target_branches import BranchDecision

        target_obs = selection.target_observation
        if target_obs is None:
            self.reset()
            return BranchDecision("none", None, 0.0, "no_geometry", ())

        track_id = int(target_obs.track_id)
        if self._track_id is not None and self._track_id != track_id:
            self.reset()

        lidar_meas: LidarMeasurement | None = None
        obstacles: tuple[Point3D, ...] = ()
        if lidar is not None:
            lidar_meas, obstacles = extract_lidar_measurement(
                selection,
                lidar,
                self._config,
                predicted_range_m=self.range_m if self._has_lidar_lock else None,
            )

        cam_meas: CameraMeasurement | None = None
        if camera is not None:
            prior_r = (
                self.range_m
                if self._has_lidar_lock
                else (lidar_meas.range_m if lidar_meas is not None else None)
            )
            cam_meas = estimate_camera_measurement(
                camera, self._config, prior_range_m=prior_r
            )

        if lidar_meas is None and cam_meas is None:
            return BranchDecision("none", None, 0.0, "no_geometry", obstacles)

        # Determine effective cycle timestamp.
        stamps = [
            m.stamp_ns for m in (lidar_meas, cam_meas) if m is not None and m.stamp_ns > 0
        ]
        if not stamps:
            return BranchDecision("none", None, 0.0, "no_geometry", obstacles)
        cycle_stamp_ns = max(stamps)

        if not self._initialized:
            if lidar_meas is not None:
                self._initialize_state(
                    track_id,
                    lidar_meas.point_base_link,
                    lidar_meas.covariance_2x2,
                    cycle_stamp_ns,
                    is_lidar=True,
                )
                if cam_meas is not None:
                    self.update_position(
                        cam_meas.point_base_link,
                        cam_meas.covariance_2x2,
                        is_lidar=False,
                        stamp_ns=cycle_stamp_ns,
                    )
            elif cam_meas is not None and self._config.allow_camera_only:
                self._initialize_state(
                    track_id,
                    cam_meas.point_base_link,
                    cam_meas.covariance_2x2,
                    cycle_stamp_ns,
                    is_lidar=False,
                )
            else:
                return BranchDecision("none", None, 0.0, "no_geometry", obstacles)
        else:
            if not self.predict(cycle_stamp_ns):
                # Filter expired over a long gap; re-initialize from current sensors.
                if lidar_meas is not None:
                    self._initialize_state(
                        track_id,
                        lidar_meas.point_base_link,
                        lidar_meas.covariance_2x2,
                        cycle_stamp_ns,
                        is_lidar=True,
                    )
                elif cam_meas is not None and self._config.allow_camera_only:
                    self._initialize_state(
                        track_id,
                        cam_meas.point_base_link,
                        cam_meas.covariance_2x2,
                        cycle_stamp_ns,
                        is_lidar=False,
                    )
                else:
                    return BranchDecision("none", None, 0.0, "no_geometry", obstacles)
            else:
                if cam_meas is not None:
                    self.update_position(
                        cam_meas.point_base_link,
                        cam_meas.covariance_2x2,
                        is_lidar=False,
                        stamp_ns=cycle_stamp_ns,
                    )
                if lidar_meas is not None:
                    self.update_position(
                        lidar_meas.point_base_link,
                        lidar_meas.covariance_2x2,
                        is_lidar=True,
                        stamp_ns=cycle_stamp_ns,
                    )

        fx = float(self._x[0])
        fy = float(self._x[1])
        fz = float(self._z_height)
        f_range = math.hypot(fx, fy)
        if not (math.isfinite(fx) and math.isfinite(fy) and f_range > 0.0):
            self.reset()
            return BranchDecision("none", None, 0.0, "no_geometry", obstacles)

        # Determine source, stamp, confidence, and provenance.
        if lidar_meas is not None and cam_meas is not None:
            source = "fused"
            provenance = f"kalman_fused_{lidar_meas.provenance}+{cam_meas.provenance}"
            base_conf = max(lidar_meas.confidence, cam_meas.confidence)
            out_stamp_ns = cycle_stamp_ns
        elif lidar_meas is not None:
            source = "lidar"
            provenance = f"kalman_{lidar_meas.provenance}"
            base_conf = lidar_meas.confidence
            out_stamp_ns = lidar_meas.stamp_ns
        else:
            assert cam_meas is not None
            lidar_age_s = (
                (cycle_stamp_ns - self._last_lidar_stamp_ns) / 1_000_000_000.0
                if self._has_lidar_lock and self._last_lidar_stamp_ns > 0
                else math.inf
            )
            if self._has_lidar_lock and 0.0 <= lidar_age_s <= self._config.lidar_coast_s:
                source = "fused"
                provenance = f"kalman_coast_lidar+{cam_meas.provenance}"
                base_conf = max(cam_meas.confidence, 0.85 * (1.0 - 0.2 * lidar_age_s))
            elif self._config.allow_camera_only:
                source = "camera"
                provenance = f"kalman_{cam_meas.provenance}"
                base_conf = cam_meas.confidence
            else:
                return BranchDecision("none", None, 0.0, "no_geometry", obstacles)
            out_stamp_ns = cam_meas.stamp_ns

        confidence = self.posterior_confidence(base_conf)
        observation = MetricPersonObservation(
            track_id=TrackId(track_id),
            point_base_link=(fx, fy, fz),
            source_stamp_ns=out_stamp_ns,
        )
        return BranchDecision(
            source=source,
            observation=observation,
            confidence=confidence,
            provenance=provenance,
            obstacles=obstacles,
        )
