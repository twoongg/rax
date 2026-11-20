"""Point clouds from a RoboDojo observation: whole objects from depth, not only one point.

Uses only what `robodojo_observe` or a motion tool returns: RGB images, metric depth
images and camera calibration. Every 3-D value is in the environment frame (metres)
that robot commands and `robodojo_pixel_to_position` use.

    from point_cloud import observation, cloud, segment, project, overlay

    obs = observation(ctx.call("robodojo_observe"))
    xyz, rgb, cam = cloud(obs)                      # every depth pixel as a point
    part = segment(obs, "cam_high", 412, 233)       # one object grown from a pixel
    print(part["centroid"], part["size"], part["top"], part["obb"])

Needs numpy, scipy and Pillow (the agent's Python has them). Copy this file into your
project folder so registered bundles carry it. The region growing and its colour test
follow the `cloud` and `segment` calls of AMP's robot SDK.
"""
import base64
import io
import json
import math

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
VOXEL_M = 0.004            # Segmentation grid; a segment's points are the original ones.
INTENSITY_WEIGHT = 0.1     # Brightness share of the colour feature that color_tol compares.
MIN_SEGMENT_POINTS = 20


class Observation:
    """One decoded observation.

    rgb[camera]: (h, w, 3) uint8; depth[camera]: (h, w) float32 metres, 0 = no depth;
    K[camera]: 3x3 intrinsics in RGB pixels; T[camera]: 4x4 camera-to-world pose (USD
    camera axes: +X right, +Y up, -Z forward); origin: the environment origin in world
    coordinates; cameras: the cameras with both depth and calibration.
    """

    def __init__(self, reply):
        blocks = reply.get("content") if isinstance(reply, dict) else reply
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("Pass the reply of a robot tool: a dict with a 'content' list")
        text = next((b.get("text") for b in blocks if b.get("type") == "text"), None)
        if text is None:
            raise ValueError("The reply has no JSON text block")
        packet = json.loads(text)
        self.step_id = packet.get("step_id")
        self.origin = np.asarray(packet.get("environment_origin_world_m") or (0.0, 0.0, 0.0),
                                 dtype=np.float64)
        self.rgb, self.depth, self.K, self.T = {}, {}, {}, {}
        depth_info = packet.get("depth") or {}
        for item in packet.get("attachments") or []:
            camera, block = item["camera"], blocks[item["content_index"]]
            image = Image.open(io.BytesIO(base64.b64decode(block["data"])))
            if item["kind"] == "rgb":
                self.rgb[camera] = np.asarray(image.convert("RGB"))
            elif item["kind"] == "depth":
                scale = float((depth_info.get(camera) or {}).get("meters_per_unit", 0.001))
                self.depth[camera] = np.asarray(image, dtype=np.float32) * scale
        for camera, parameters in (packet.get("camera_parameters") or {}).items():
            if parameters and "intrinsic_matrix" in parameters and "camera_to_world_usd" in parameters:
                self.K[camera] = np.asarray(parameters["intrinsic_matrix"], dtype=np.float64)
                self.T[camera] = np.asarray(parameters["camera_to_world_usd"], dtype=np.float64)
        self.cameras = tuple(c for c in CAMERAS if c in self.depth and c in self.K)
        if not self.cameras:
            raise ValueError("This observation has no depth images with calibration; "
                             "point clouds need the rgbd observation profile")

    def require(self, camera):
        if camera not in self.cameras:
            raise ValueError(f"No depth and calibration for {camera!r}; available: {list(self.cameras)}")


def observation(reply):
    """Decode the reply of `robodojo_observe` (or of a motion tool, which carries its final
    frame) into an Observation."""
    return reply if isinstance(reply, Observation) else Observation(reply)


def _unproject(obs, camera, u, v, z):
    """Environment-frame points of depth-image pixels (u, v) at optical depth z."""
    depth = obs.depth[camera]
    height, width = obs.rgb[camera].shape[:2] if camera in obs.rgb else depth.shape
    # Calibration is in RGB pixels; a smaller depth image maps onto them.
    su, sv = width / depth.shape[1], height / depth.shape[0]
    pixels = np.stack([(u + 0.5) * su - 0.5, (v + 0.5) * sv - 0.5, np.ones_like(z)])
    ray = np.linalg.solve(obs.K[camera], pixels)
    ray = ray / ray[2]
    cv = ray * z
    usd = np.stack([cv[0], -cv[1], -cv[2]])  # CV optical axes to USD camera axes.
    T = obs.T[camera]
    world = (T[:3, :3] @ usd).T + T[:3, 3]
    return world - obs.origin


def _camera_list(obs, cameras):
    """The requested cameras (a name or a list; None = all with depth), in obs.cameras order."""
    if cameras is None:
        return list(obs.cameras)
    cameras = [cameras] if isinstance(cameras, str) else list(cameras)
    for camera in cameras:
        obs.require(camera)
    return [c for c in obs.cameras if c in cameras]


def _in_box(xyz, box):
    keep = np.ones(len(xyz), dtype=bool)
    for axis, (low, high) in enumerate(box):
        keep &= (xyz[:, axis] >= low) & (xyz[:, axis] <= high)
    return keep


def cloud(obs, cameras=None, *, stride=1, max_depth=2.0, box=None):
    """Every valid depth pixel of `cameras` (default: all with depth) as a 3-D point.

    Returns (xyz (N, 3) float32 metres, rgb (N, 3) uint8, cam (N,) the point's camera as an
    index into obs.cameras). `box` = ((x0, x1), (y0, y1), (z0, z1)) keeps only the
    points inside it. Only visible surfaces appear; nothing is removed but what `box` and
    `max_depth` cut.
    """
    obs = observation(obs)
    cameras = _camera_list(obs, cameras)
    xyzs, rgbs, cams = [], [], []
    for camera in cameras:
        depth = obs.depth[camera]
        v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[v, u]
        keep = np.isfinite(z) & (z > 0) & (z <= max_depth)
        u, v, z = u[keep], v[keep], z[keep].astype(np.float64)
        xyz = _unproject(obs, camera, u, v, z)
        if camera in obs.rgb:
            image = obs.rgb[camera]
            ru = np.clip(np.rint((u + 0.5) * image.shape[1] / depth.shape[1] - 0.5), 0, image.shape[1] - 1)
            rv = np.clip(np.rint((v + 0.5) * image.shape[0] / depth.shape[0] - 0.5), 0, image.shape[0] - 1)
            rgb = image[rv.astype(int), ru.astype(int)]
        else:
            rgb = np.full((len(z), 3), 180, dtype=np.uint8)
        if box is not None:
            inside = _in_box(xyz, box)
            xyz, rgb = xyz[inside], rgb[inside]
        xyzs.append(xyz.astype(np.float32))
        rgbs.append(np.asarray(rgb, dtype=np.uint8))
        cams.append(np.full(len(xyz), obs.cameras.index(camera), dtype=np.int16))
    if not xyzs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros(0, np.int16)
    return np.concatenate(xyzs), np.concatenate(rgbs), np.concatenate(cams)


def pixel_point(obs, camera, u, v, *, radius=2):
    """The 3-D point under RGB pixel (u, v): the median depth of a small window."""
    obs = observation(obs)
    obs.require(camera)
    depth = obs.depth[camera]
    height, width = obs.rgb[camera].shape[:2] if camera in obs.rgb else depth.shape
    du = int(round((u + 0.5) * depth.shape[1] / width - 0.5))
    dv = int(round((v + 0.5) * depth.shape[0] / height - 0.5))
    if not (0 <= du < depth.shape[1] and 0 <= dv < depth.shape[0]):
        raise ValueError(f"Pixel ({u}, {v}) is outside {camera}'s {width}x{height} image")
    window = depth[max(0, dv - radius):dv + radius + 1, max(0, du - radius):du + radius + 1]
    valid = window[np.isfinite(window) & (window > 0)]
    if not valid.size:
        raise ValueError(f"No depth at ({u}, {v}) in {camera}; pick a pixel on a visible surface")
    return _unproject(obs, camera, np.array([du]), np.array([dv]),
                      np.array([float(np.median(valid))]))[0]


def table_height(xyz, *, bin_m=0.001):
    """(z, count): the height of the largest horizontal surface in a cloud, the table in
    a cam_high view. Check it: a large flat object or the floor can win instead."""
    z = np.asarray(xyz, dtype=np.float64)[:, 2]
    z = z[np.isfinite(z)]
    if not len(z):
        raise ValueError("Empty cloud")
    bins = np.arange(z.min(), z.max() + 2 * bin_m, bin_m)
    counts, edges = np.histogram(z, bins=bins)
    peak = int(np.argmax(np.convolve(counts, np.ones(3), mode="same")))
    centre = 0.5 * (edges[peak] + edges[peak + 1])
    near = z[np.abs(z - centre) <= 3 * bin_m]
    return float(np.median(near)), int(len(near))


def _color_feature(rgb):
    """Shading-tolerant colour: chroma over intensity, plus a little intensity."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    i = c.mean(axis=1, keepdims=True)
    return np.hstack([(c - i) / (i + 0.03), INTENSITY_WEIGHT * i])


def _voxelize(xyz, rgb, size):
    keys = np.floor(xyz / size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n = len(counts)
    centres = np.stack([np.bincount(inverse, xyz[:, a], n) for a in range(3)], axis=1) / counts[:, None]
    colours = np.stack([np.bincount(inverse, rgb[:, a], n) for a in range(3)], axis=1) / counts[:, None]
    return centres, colours, inverse


def _obb(xyz):
    """The smallest upright box around the points: the minimum-area rectangle of their
    footprint (its x side the longer one), from the table up to their top."""
    xy = xyz[:, :2]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except (QhullError, ValueError):  # Fewer than three points, or all on one line.
        hull = xy
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), math.pi / 2))
    best = None
    for angle in angles if len(angles) else [0.0]:
        c, s = math.cos(angle), math.sin(angle)
        local = hull @ np.array([[c, -s], [s, c]])  # Coordinates along (c, s) and (-s, c).
        low, high = local.min(axis=0), local.max(axis=0)
        area = float(np.prod(high - low))
        if best is None or area < best[0]:
            best = (area, angle, low, high)
    _, yaw, low, high = best
    c, s = math.cos(yaw), math.sin(yaw)
    centre_xy = np.array([[c, -s], [s, c]]) @ (0.5 * (low + high))
    length, width = high - low
    if length < width:
        yaw, length, width = yaw + math.pi / 2, width, length
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2  # A box axis has no sign: [-90, 90).
    z0, z1 = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    return {"center": [float(centre_xy[0]), float(centre_xy[1]), 0.5 * (z0 + z1)],
            "size": [float(length), float(width), z1 - z0],
            "yaw_deg": math.degrees(yaw)}


def segment(obs, camera, u, v, *, radius_mm=11, color_tol=None, cameras=None, box=None,
            table_z=None, table_margin_m=0.005, reach_m=0.3, max_depth=2.0):
    """One object grown from the point under RGB pixel (u, v) of `camera`.

    The cloud of `cameras` (default: all with depth) loses the table (everything below
    table_z + table_margin_m; table_z defaults to table_height of cam_high's cloud) and
    what lies outside `box`. Points then join while neighbours are within radius_mm, and, with
    color_tol (0.08 works), while their colours stay close. The robot is not removed: an
    object touching the gripper grows into it; cut it off with `box` or color_tol.

    Returns a dict: points (N, 3) float32 and rgb (N, 3) uint8 of the object; n_points;
    seed; centroid; aabb [[lo], [hi]]; size (aabb extent); top (centre of the aabb's top
    face); top_z; bottom_z; obb {center, size, yaw_deg}: the smallest upright box, its size
    [longer side, shorter side, height] and yaw (degrees about z, of the longer side, in
    [-90, 90)); table_z; cameras. All of it describes the visible surface only.
    """
    obs = observation(obs)
    obs.require(camera)
    radius = float(radius_mm) / 1000.0
    if not 0.002 <= radius <= 0.05:
        raise ValueError("radius_mm must be in [2, 50]")
    if color_tol is not None and not color_tol > 0:
        raise ValueError("color_tol must be a positive number or None")
    cameras = _camera_list(obs, cameras)
    seed = pixel_point(obs, camera, u, v)
    xyz, rgb, _ = cloud(obs, cameras, max_depth=max_depth, box=box)
    if table_z is None:
        # The overview camera sees mostly table; a close wrist view may see mostly object.
        overview = ["cam_high"] if "cam_high" in obs.cameras else cameras
        table_z, _ = table_height(cloud(obs, overview, stride=2, max_depth=max_depth)[0])
    if seed[2] <= table_z + table_margin_m:
        raise ValueError(f"The pixel hits the table (z {seed[2]:.4f} m, table {table_z:.4f} m); "
                         "pick a pixel inside the object")
    near = np.all(np.abs(xyz - seed) <= reach_m, axis=1) & (xyz[:, 2] > table_z + table_margin_m)
    index = np.flatnonzero(near)
    points = xyz[index].astype(np.float64)
    if not len(points):
        raise ValueError("No points above the table near that pixel")
    distance = np.linalg.norm(points - seed, axis=1)
    start = int(np.argmin(distance))
    if distance[start] > max(radius, 0.01):
        raise ValueError("The point under that pixel is outside `box` or `cameras`; pick another pixel")
    centres, colours, inverse = _voxelize(points, rgb[index].astype(np.float64), min(VOXEL_M, 0.5 * radius))
    pairs = cKDTree(centres).query_pairs(radius, output_type="ndarray")
    if color_tol is not None and len(pairs):
        feature = _color_feature(colours)
        pairs = pairs[np.linalg.norm(feature[pairs[:, 0]] - feature[pairs[:, 1]], axis=1) < color_tol]
    n = len(centres)
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    labels = connected_components(graph, directed=False)[1]
    member = labels[inverse] == labels[inverse[start]]
    part, part_rgb = xyz[index[member]], rgb[index[member]]
    if len(part) < MIN_SEGMENT_POINTS:
        raise ValueError(f"Only {len(part)} points grew from that pixel; pick a pixel inside the object")
    low, high = part.min(axis=0).astype(np.float64), part.max(axis=0).astype(np.float64)
    return {
        "points": part, "rgb": part_rgb, "n_points": int(len(part)),
        "seed": seed.tolist(), "centroid": part.mean(axis=0, dtype=np.float64).tolist(),
        "aabb": [low.tolist(), high.tolist()], "size": (high - low).tolist(),
        "top": [float(0.5 * (low[0] + high[0])), float(0.5 * (low[1] + high[1])), float(high[2])],
        "top_z": float(high[2]), "bottom_z": float(low[2]),
        "obb": _obb(part.astype(np.float64)), "table_z": float(table_z),
        "cameras": cameras,
    }


def project(obs, camera, xyz):
    """(uv (N, 2) RGB pixels, z (N,) optical depth in metres) of environment-frame points;
    points behind the camera have z <= 0."""
    obs = observation(obs)
    obs.require(camera)
    T = obs.T[camera]
    world = np.asarray(xyz, dtype=np.float64).reshape(-1, 3) + obs.origin
    usd = (world - T[:3, 3]) @ T[:3, :3]
    cv = np.stack([usd[:, 0], -usd[:, 1], -usd[:, 2]], axis=1)
    z = cv[:, 2]
    pixels = cv @ obs.K[camera].T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels[:, :2] / pixels[:, 2:3]
    return uv, z


def overlay(obs, camera, xyz, *, color=(255, 0, 255)):
    """A copy of `camera`'s RGB image with the visible-side projection of `xyz` painted in
    `color`, to check a segment or a cloud against what the camera sees."""
    obs = observation(obs)
    image = obs.rgb[camera].copy()
    uv, z = project(obs, camera, xyz)
    u, v = np.rint(uv[:, 0]), np.rint(uv[:, 1])
    keep = (z > 0) & np.isfinite(u) & np.isfinite(v)
    keep &= (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
    image[v[keep].astype(int), u[keep].astype(int)] = color
    return image
