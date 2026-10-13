"""obstacle_cloud: /nav/cloud_level -> /costmap_cloud (obstacle points only, voxel-downsampled, range-limited).

Ground points are dropped by the leveler's flags; points above the obstacle band (ceilings, light fixtures) never
reach this node; overhanging obstacles below the band's upper limit are kept. A voxel needs `min_points`
returns to survive, which removes isolated range outliers that would otherwise paint phantom obstacles.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from .cloud_leveler import OBSTACLE
from .pointcloud import cloud_to_struct, make_cloud


class ObstacleCloud(Node):
    def __init__(self):
        super().__init__("obstacle_cloud")
        d = self.declare_parameter
        d("input_topic", "/nav/cloud_level")
        d("output_topic", "/costmap_cloud")
        d("max_range", 8.0)
        d("min_range", 0.10)
        d("voxel_size", 0.05)
        d("min_points", 2)             # per voxel_size * speckle_factor voxel
        d("speckle_factor", 2.0)
        g = self.get_parameter
        self.max_range = float(g("max_range").value)
        self.min_range = float(g("min_range").value)
        self.voxel = float(g("voxel_size").value)
        self.min_points = int(g("min_points").value)
        self.speckle_voxel = self.voxel * float(g("speckle_factor").value)
        self.pub = self.create_publisher(PointCloud2, g("output_topic").value, 5)
        self.create_subscription(PointCloud2, g("input_topic").value, self.on_cloud, 5)

    @staticmethod
    def _keys(xyz: np.ndarray, voxel: float) -> np.ndarray:
        idx = np.floor(xyz / voxel).astype(np.int64)
        idx -= idx.min(axis=0)
        dims = idx.max(axis=0) + 1
        return (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]

    def on_cloud(self, msg: PointCloud2) -> None:
        pts = cloud_to_struct(msg)
        obs = pts[pts["flags"] == OBSTACLE]
        xyz = np.column_stack((obs["x"], obs["y"], obs["z"]))
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        xyz = xyz[(r >= self.min_range) & (r <= self.max_range)]
        if len(xyz):
            if self.min_points > 1:
                _, inv, counts = np.unique(self._keys(xyz, self.speckle_voxel), return_inverse=True, return_counts=True)
                xyz = xyz[counts[inv] >= self.min_points]
            if len(xyz):
                _, first = np.unique(self._keys(xyz, self.voxel), return_index=True)
                xyz = xyz[first]
        self.pub.publish(make_cloud(msg.header, {
            "x": xyz[:, 0].astype(np.float32), "y": xyz[:, 1].astype(np.float32), "z": xyz[:, 2].astype(np.float32)}))


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
