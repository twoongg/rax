"""Point clouds from a RoboDojo observation: whole objects from depth, not only one point.

Uses only what `robodojo_observe` or a motion tool returns: RGB images, metric depth
images, camera calibration and the measured link6 positions. Every 3-D value is in the
environment frame (metres) that robot commands and `robodojo_pixel_to_position` use.

    from point_cloud import observation, objects, segment, local, world, overlay

    obs = observation(ctx.call("robodojo_observe"))
    for item in objects(obs):                       # every object on the table, no pixel needed
        print(item["index"], item["label"], item["obb"])
    part = segment(obs, "cam_high", 412, 233)       # one object grown from a chosen pixel

Needs numpy, scipy and Pillow (the agent's Python has them). Copy this file into your
project folder so registered bundles carry it. The region growing, its colour test and
the object list follow the `cloud`, `segment` and `objects` calls of AMP's robot SDK.
"""
import base64
import colorsys
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
ROBOT_TOUCH_M = 0.05       # A part reaching this close to a link6 position is the robot's.
FLOAT_M = 0.04             # A part whose lowest point is this far above the table...
SUPPORT_GAP_M = 0.015      # ...and has nothing this close beneath it floats.
TINY_VOXELS = 20           # A colour region this small joins the neighbour it touches most.
ABSORB_VOXELS = 60         # A part this small joins a much larger, similar-coloured part...
ABSORB_M = 0.022           # ...this close: a thin edge or tab seen apart from its body.
COLOR_MERGE = 0.18


class Observation:
    """One decoded observation.

    rgb[camera]: (h, w, 3) uint8; depth[camera]: (h, w) float32 metres, 0 = no depth;
    K[camera]: 3x3 intrinsics in RGB pixels; T[camera]: 4x4 camera-to-world pose (USD
    camera axes: +X right, +Y up, -Z forward); origin: the environment origin in world
    coordinates; eef: (arms, 3) measured link6 positions; cameras: the cameras with both
    depth and calibration.
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
        try:
            self.eef = np.asarray(packet.get("eef_positions"), dtype=np.float64).reshape(-1, 3)
        except (TypeError, ValueError):
            self.eef = np.zeros((0, 3))
        self.eef = self.eef[np.isfinite(self.eef).all(axis=1)]
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


def _table_z(obs, cameras, max_depth):
    # The overview camera sees mostly table; a close wrist view may see mostly object.
    overview = ["cam_high"] if "cam_high" in obs.cameras else cameras
    return table_height(cloud(obs, overview, stride=2, max_depth=max_depth)[0])[0]


def _color_feature(rgb):
    """Shading-tolerant colour: chroma over intensity, plus a little intensity."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    i = c.mean(axis=1, keepdims=True)
    return np.hstack([(c - i) / (i + 0.03), INTENSITY_WEIGHT * i])


def _color_name(rgb):
    """A coarse English name of an RGB colour."""
    r, g, b = (float(v) / 255.0 for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18:
        return "black"
    if s < 0.18 or (s < 0.3 and v < 0.35):
        return "white" if v > 0.75 else "grey"
    names = ((15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (200, "teal"),
             (255, "blue"), (290, "purple"), (340, "pink"), (361, "red"))
    name = next(n for limit, n in names if h * 360.0 < limit)
    return "brown" if name in ("red", "orange", "yellow") and v < 0.55 and s > 0.3 else name


def _voxelize(xyz, rgb, size):
    keys = np.floor(xyz / size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n = len(counts)
    centres = np.stack([np.bincount(inverse, xyz[:, a], n) for a in range(3)], axis=1) / counts[:, None]
    colours = np.stack([np.bincount(inverse, rgb[:, a], n) for a in range(3)], axis=1) / counts[:, None]
    return centres, colours, inverse


def _components(n, edges):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _fold_tiny(labels, pairs):
    """Colour regions under TINY_VOXELS (the mixed colours along an edge) join the
    neighbouring region they share the most links with."""
    n = int(labels.max()) + 1
    size = np.bincount(labels, minlength=n)
    a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
    cross = a != b
    src, dst = np.concatenate([a[cross], b[cross]]), np.concatenate([b[cross], a[cross]])
    keep = size[src] < TINY_VOXELS
    if not keep.any():
        return labels
    links, count = np.unique(src[keep] * n + dst[keep], return_counts=True)
    s, d = links // n, links % n
    order = np.lexsort((-size[d], -count, s))  # Per tiny region: most links, then the larger.
    s, d = s[order], d[order]
    first = np.r_[True, s[1:] != s[:-1]]
    return _components(n, np.stack([s[first], d[first]], axis=1))[labels]


def _groups(xyz, rgb, radius, color_tol):
    """Labels per voxel and the voxel of each point: points join while voxel neighbours
    are within radius (and, with color_tol, while their colours stay close). Returns
    (voxel labels, point-to-voxel index, voxel centres, voxel colours, neighbour pairs)."""
    centres, colours, inverse = _voxelize(xyz, rgb.astype(np.float64), min(VOXEL_M, 0.5 * radius))
    pairs = cKDTree(centres).query_pairs(radius, output_type="ndarray")
    linked = pairs
    if color_tol is not None and len(pairs):
        feature = _color_feature(colours)
        linked = pairs[np.linalg.norm(feature[pairs[:, 0]] - feature[pairs[:, 1]], axis=1) < color_tol]
    labels = _components(len(centres), linked)
    if color_tol is not None and len(pairs):
        labels = _fold_tiny(labels, pairs)
    return labels, inverse, centres, colours, pairs


def _absorb(labels, centres, colours, pairs):
    """Parts under ABSORB_VOXELS join a part at least four times larger, within ABSORB_M
    and of similar colour, unless the two touch (then colour split them on purpose)."""
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.ravel()
    n = int(labels.max()) + 1
    size = np.bincount(labels, minlength=n)
    small, big = np.flatnonzero(size < ABSORB_VOXELS), np.flatnonzero(size >= ABSORB_VOXELS)
    if not len(small) or not len(big):
        return labels
    feature = _color_feature(colours)
    mean = np.stack([np.bincount(labels, feature[:, k], n) for k in range(feature.shape[1])], axis=1)
    mean /= np.maximum(size, 1)[:, None]
    a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
    touching = set(zip(np.minimum(a, b)[a != b].tolist(), np.maximum(a, b)[a != b].tolist()))
    in_big = np.isin(labels, big)
    tree, big_labels = cKDTree(centres[in_big]), labels[in_big]
    edges = []
    for r in small:
        distance, nearest = tree.query(centres[labels == r], distance_upper_bound=ABSORB_M)
        hit = np.isfinite(distance)
        if not hit.any():
            continue
        candidates, count = np.unique(big_labels[nearest[hit]], return_counts=True)
        t = int(candidates[np.argmax(count)])
        if (size[t] >= 4 * size[r] and (min(r, t), max(r, t)) not in touching
                and np.linalg.norm(mean[t] - mean[r]) < COLOR_MERGE):
            edges.append((r, t))
    return _components(n, edges)[labels] if edges else labels


def _obb(xyz):
    """The smallest upright box around the points: the minimum-area rectangle of their
    footprint (its x side the longer one), from their bottom up to their top."""
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
        local_xy = hull @ np.array([[c, -s], [s, c]])  # Coordinates along (c, s) and (-s, c).
        low, high = local_xy.min(axis=0), local_xy.max(axis=0)
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


def describe(points, rgb=None, *, table_z=None):
    """The summary that segment and objects return, for any points you selected yourself
    (for example a `cloud(..., box=...)` around a held object).

    A dict: points; rgb; n_points; centroid; aabb [[lo], [hi]]; size (aabb extent); top
    (centre of the aabb's top face); top_z; bottom_z; obb {center, size, yaw_deg}, the
    smallest upright box: size [longer side, shorter side, height], yaw of the longer side
    in degrees about z, in [-90, 90); color (mean RGB); label ("<colour> · LxWxH cm", the
    height measured from table_z when given); table_z.
    """
    part = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(part) < 3:
        raise ValueError("describe needs at least three points")
    part_rgb = (np.full((len(part), 3), 180, np.uint8) if rgb is None
                else np.asarray(rgb, dtype=np.uint8).reshape(-1, 3))
    low, high = part.min(axis=0).astype(np.float64), part.max(axis=0).astype(np.float64)
    box = _obb(part.astype(np.float64))
    color = [int(round(float(c))) for c in part_rgb.astype(np.float64).mean(axis=0)]
    height = high[2] - (low[2] if table_z is None else float(table_z))
    dims = "x".join(str(max(1, round(100 * d))) for d in (box["size"][0], box["size"][1], height))
    return {
        "points": part, "rgb": part_rgb, "n_points": int(len(part)),
        "centroid": part.mean(axis=0, dtype=np.float64).tolist(),
        "aabb": [low.tolist(), high.tolist()], "size": (high - low).tolist(),
        "top": [float(0.5 * (low[0] + high[0])), float(0.5 * (low[1] + high[1])), float(high[2])],
        "top_z": float(high[2]), "bottom_z": float(low[2]), "obb": box,
        "color": color, "label": f"{_color_name(color)} · {dims} cm",
        "table_z": None if table_z is None else float(table_z),
    }


def _check(radius_mm, color_tol):
    radius = float(radius_mm) / 1000.0
    if not 0.002 <= radius <= 0.05:
        raise ValueError("radius_mm must be in [2, 50]")
    if color_tol is not None and not color_tol > 0:
        raise ValueError("color_tol must be a positive number or None")
    return radius


def segment(obs, camera, u, v, *, radius_mm=11, color_tol=None, cameras=None, box=None,
            table_z=None, table_margin_m=0.005, reach_m=0.3, max_depth=2.0):
    """One object grown from the point under RGB pixel (u, v) of `camera`.

    The cloud of `cameras` (default: all with depth) loses the table (everything below
    table_z + table_margin_m; table_z defaults to table_height of cam_high's cloud) and
    what lies outside `box`. Points then join while neighbours are within radius_mm, and,
    with color_tol (0.08 works), while their colours stay close. The robot is not removed:
    an object touching the gripper grows into it; cut it off with `box` or color_tol.

    Returns describe()'s dict plus seed and cameras. All of it describes the visible
    surface only.
    """
    obs = observation(obs)
    obs.require(camera)
    radius = _check(radius_mm, color_tol)
    cameras = _camera_list(obs, cameras)
    seed = pixel_point(obs, camera, u, v)
    xyz, rgb, _ = cloud(obs, cameras, max_depth=max_depth, box=box)
    table_z = _table_z(obs, cameras, max_depth) if table_z is None else float(table_z)
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
    labels, inverse, _, _, _ = _groups(points, rgb[index], radius, color_tol)
    labels = labels[inverse]
    member = index[labels == labels[start]]
    if len(member) < MIN_SEGMENT_POINTS:
        raise ValueError(f"Only {len(member)} points grew from that pixel; pick a pixel inside the object")
    return {**describe(xyz[member], rgb[member], table_z=table_z), "seed": seed.tolist(), "cameras": cameras}


def objects(obs, cameras=None, *, radius_mm=11, color_tol=None, box=None, table_z=None,
            table_margin_m=0.005, min_points=50, include_robot=False, max_depth=2.0):
    """Every object on the table, with no pixel to choose: what a controller uses to find
    objects again in each new layout.

    The cloud of `cameras` (default: all with depth) loses the table and what lies outside
    `box`; the rest splits into parts that are not connected within radius_mm (with
    color_tol, also at colour edges). A small fragment within 22 mm of a part at least four
    times larger and of similar colour joins it (a thin edge or tab seen apart from its
    body). Each part gets describe()'s dict plus:
    robot: it reaches within 5 cm of a measured link6 position (an arm, or an object held
    or touched by the gripper); supported: it rests on the table or on another part
    (False: it floats, like a piece of an arm). Unless include_robot, robot and floating
    parts and parts under min_points are left out. The list is sorted by footprint (obb
    length x width), largest first, and `index` is the position in it; indices are not
    kept from one call to the next, so choose objects by size, colour and position.
    """
    obs = observation(obs)
    radius = _check(radius_mm, color_tol)
    cameras = _camera_list(obs, cameras)
    xyz, rgb, _ = cloud(obs, cameras, max_depth=max_depth, box=box)
    table_z = _table_z(obs, cameras, max_depth) if table_z is None else float(table_z)
    above = np.flatnonzero(xyz[:, 2] > table_z + table_margin_m)
    if not len(above):
        return []
    labels, inverse, centres, colours, pairs = _groups(xyz[above].astype(np.float64), rgb[above],
                                                       radius, color_tol)
    labels = _absorb(labels, centres, colours, pairs)[inverse]
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    parts = [above[group] for group in np.split(order, bounds) if len(group) >= min_points]
    found = []
    for member in parts:
        item = describe(xyz[member], rgb[member], table_z=table_z)
        points = item["points"].astype(np.float64)
        item["robot"] = bool(len(obs.eef)) and float(
            np.min(np.linalg.norm(points[:, None, :] - obs.eef[None], axis=2))) < ROBOT_TOUCH_M
        found.append(item)
    for item in found:
        bottom = item["bottom_z"]
        item["supported"] = bottom <= table_z + FLOAT_M
        if not item["supported"]:
            # Something just beneath its lowest points holds it up (a stack, not an arm).
            under = np.vstack([o["points"] for o in found if o is not item] or [np.zeros((0, 3))])
            under = under[(under[:, 2] < bottom + 0.002) & (under[:, 2] > bottom - SUPPORT_GAP_M)]
            rim = item["points"][item["points"][:, 2] < bottom + 0.01]
            if len(under) and len(rim):
                gap = cKDTree(under[:, :2]).query(rim[:, :2], distance_upper_bound=SUPPORT_GAP_M)[0]
                item["supported"] = bool(np.isfinite(gap).any())
    if not include_robot:
        found = [item for item in found if item["supported"] and not item["robot"]]
    found.sort(key=lambda item: -item["obb"]["size"][0] * item["obb"]["size"][1])
    for index, item in enumerate(found):
        item["index"] = index
        item["cameras"] = cameras
    return found


def _box(item):
    return item["obb"] if "obb" in item else item


def flip(item):
    """The same upright box turned by 180 degrees: its +x end becomes the other end.
    Use it once you know which end of an object is which (see profile)."""
    box = dict(_box(item))
    box["yaw_deg"] = (box["yaw_deg"] + 360.0) % 360.0 - 180.0
    return box


def local(item, xyz):
    """Points in an object's own frame: origin at its obb centre, x along the obb's yaw, y
    to the left of x, z up. `item` is an obb or a dict that has one."""
    box = _box(item)
    a = math.radians(box["yaw_deg"])
    d = np.asarray(xyz, dtype=np.float64).reshape(-1, 3) - np.asarray(box["center"], dtype=np.float64)
    return np.stack([d[:, 0] * math.cos(a) + d[:, 1] * math.sin(a),
                     -d[:, 0] * math.sin(a) + d[:, 1] * math.cos(a), d[:, 2]], axis=1)


def world(item, offsets):
    """Environment-frame points of offsets given in an object's own frame (see local)."""
    box = _box(item)
    a = math.radians(box["yaw_deg"])
    o = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
    return np.stack([o[:, 0] * math.cos(a) - o[:, 1] * math.sin(a),
                     o[:, 0] * math.sin(a) + o[:, 1] * math.cos(a), o[:, 2]], axis=1) + \
        np.asarray(box["center"], dtype=np.float64)


def profile(item, bin_m=0.005):
    """An object's shape along its obb's long axis, from the -x end to the +x end, in
    slices about bin_m long: a list of {s (metres from the centre along +x), n (points),
    width (extent across, metres), top_z}. Ends differ in width or height when the object
    is not symmetric; slices longer than the feature at an end hide it."""
    points = np.asarray(item["points"], dtype=np.float64)
    coords = local(item, points)
    half = item["obb"]["size"][0] / 2
    edges = np.linspace(-half, half, max(2, int(round(2 * half / bin_m))) + 1)
    slot = np.clip(np.searchsorted(edges, coords[:, 0], side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        inside = slot == k
        rows.append({"s": float(0.5 * (edges[k] + edges[k + 1])), "n": int(inside.sum()),
                     "width": float(np.ptp(coords[inside, 1])) if inside.any() else 0.0,
                     "top_z": float(points[inside, 2].max()) if inside.any() else None})
    return rows


def project(obs, camera, xyz):
    """(uv (N, 2) RGB pixels, z (N,) optical depth in metres) of environment-frame points;
    points behind the camera have z <= 0."""
    obs = observation(obs)
    obs.require(camera)
    T = obs.T[camera]
    world_xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3) + obs.origin
    usd = (world_xyz - T[:3, 3]) @ T[:3, :3]
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
