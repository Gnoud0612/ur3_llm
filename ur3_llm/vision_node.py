#!/usr/bin/env python3
"""Detect the colored blocks in RGB-D frames and publish base-frame state."""

import json
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_geometry_msgs  # noqa: F401 - registers geometry message transforms.
import tf2_ros


class CameraPerception(Node):
    """Estimate object centers from camera pixels, depth, intrinsics and TF."""

    COLOR_RANGES = {
        'red_cube': [((0, 90, 55), (10, 255, 255)),
                     ((170, 90, 55), (179, 255, 255))],
        # The table is warm brown and can fall in the yellow hue band. Require
        # a more saturated yellow so the surface is not reported as a block.
        'yellow_cube': [((15, 130, 70), (42, 255, 255))],
        'green_cube': [((38, 65, 45), (88, 255, 255))],
        'blue_cube': [((95, 75, 45), (132, 255, 255))],
        'purple_cube': [((130, 65, 45), (165, 255, 255))],
    }

    def __init__(self):
        super().__init__('camera_perception')
        self.declare_parameter('rgb_topic', '/overhead_camera/image_raw')
        self.declare_parameter('depth_topic', '/overhead_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/overhead_camera/camera_info')
        self.declare_parameter('objects_topic', '/vision/objects')
        self.declare_parameter('debug_image_topic', '/vision/debug')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('zone_names', ['zone_a', 'zone_b', 'zone_c'])
        self.declare_parameter('zone_centers_xy', [0.27, 0.14, 0.27, -0.14, 0.43, 0.14])
        self.declare_parameter('zone_radius', 0.06)
        self.declare_parameter('min_area', 70.0)
        self.declare_parameter('depth_patch', 7)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('show_window', False)
        self.declare_parameter('window_name', 'UR3 RGB-D Camera (Top View)')

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.zone_radius = float(self.get_parameter('zone_radius').value)
        self.min_area = float(self.get_parameter('min_area').value)
        self.depth_patch = max(1, int(self.get_parameter('depth_patch').value))
        self.min_depth = float(self.get_parameter('min_depth_m').value)
        self.max_depth = float(self.get_parameter('max_depth_m').value)
        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        zone_names = list(self.get_parameter('zone_names').value)
        centers = list(self.get_parameter('zone_centers_xy').value)
        if len(centers) != 2 * len(zone_names):
            raise ValueError('zone_centers_xy requires one x,y pair per zone name')
        self.zone_centers = {
            name: (float(centers[2 * i]), float(centers[2 * i + 1]))
            for i, name in enumerate(zone_names)
        }

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_depth = None
        self.camera_info = None
        self.last_tf_warning = 0.0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state_pub = self.create_publisher(
            String, self.get_parameter('objects_topic').value, 10)
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_image_topic').value, 1)
        self.create_subscription(Image, self.depth_topic, self._depth_callback, 10)
        self.create_subscription(CameraInfo, self.info_topic, self._info_callback, 10)
        self.create_subscription(Image, self.rgb_topic, self._rgb_callback, 10)
        self.get_logger().info(
            f'RGB-D perception listening to {self.rgb_topic} and {self.depth_topic}')
        if self.show_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                waiting = np.full((480, 640, 3), 32, dtype=np.uint8)
                cv2.putText(
                    waiting, 'Waiting for Gazebo RGB-D frames...', (28, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    1, cv2.LINE_AA)
                cv2.imshow(self.window_name, waiting)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.get_logger().error(
                    f'Could not open the camera preview window: {exc}. '
                    'Check that this process has access to a desktop display.')
                self.show_window = False

    def _info_callback(self, msg):
        with self.lock:
            self.camera_info = msg

    def _depth_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = np.asarray(image, dtype=np.float32)
            if msg.encoding in ('16UC1', 'mono16'):
                depth *= 0.001
            with self.lock:
                self.latest_depth = depth
        except Exception as exc:  # A malformed frame should not stop perception.
            self.get_logger().warning(f'Could not decode depth image: {exc}')

    def _rgb_callback(self, msg):
        with self.lock:
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            camera_info = self.camera_info
        if depth is None or camera_info is None:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'Could not decode RGB image: {exc}')
            return
        if depth.shape[:2] != bgr.shape[:2]:
            depth = cv2.resize(
                depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        objects = {}
        for object_name, color_ranges in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in color_ranges:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8)),
                )
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            detection = self._largest_block(mask)
            if detection is None:
                continue
            u, v, area, contour = detection
            camera_depth = self._sample_depth(depth, u, v)
            if camera_depth is None:
                continue
            base_point = self._pixel_to_base(u, v, camera_depth, msg, camera_info)
            if base_point is None:
                continue
            objects[object_name] = {
                'x': round(float(base_point[0]), 4),
                'y': round(float(base_point[1]), 4),
                'z': round(float(base_point[2]), 4),
                'pixel': [int(u), int(v)],
                'depth': round(float(camera_depth), 4),
                'area': round(float(area), 1),
            }
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)

        zones = {}
        for zone_name, (zone_x, zone_y) in self.zone_centers.items():
            occupants = [
                (math.hypot(obj['x'] - zone_x, obj['y'] - zone_y), name)
                for name, obj in objects.items()
                if math.hypot(obj['x'] - zone_x, obj['y'] - zone_y) <= self.zone_radius
            ]
            occupant = min(occupants)[1] if occupants else None
            zones[zone_name] = {
                'x': zone_x,
                'y': zone_y,
                'occupied_by': occupant,
                'free': occupant is None,
            }

        # Keep the ROS debug topic and a desktop preview visually identical.
        # Coordinates are shown in base_link, the frame used by the planner.
        for object_name, observation in objects.items():
            u, v = observation['pixel']
            cv2.drawMarker(
                bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
            x, y = observation['x'], observation['y']
            cv2.putText(
                bgr, f'{object_name} ({x:.3f}, {y:.3f})',
                (u + 8, max(18, v - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                bgr, f'{object_name} ({x:.3f}, {y:.3f})',
                (u + 8, max(18, v - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (20, 20, 20), 1, cv2.LINE_AA)

        zone_summary = 'Zones: ' + ' | '.join(
            f'{name.replace("zone_", "").upper()}={info["occupied_by"] or "free"}'
            for name, info in zones.items())
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 28), (24, 24, 24), -1)
        cv2.putText(
            bgr, zone_summary, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1, cv2.LINE_AA)

        payload = {
            'timestamp': time.time(),
            'frame': self.base_frame,
            'objects': objects,
            'zones': zones,
        }
        state_msg = String()
        state_msg.data = json.dumps(payload, separators=(',', ':'))
        self.state_pub.publish(state_msg)
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().debug(f'Could not publish debug image: {exc}')
        if self.show_window:
            try:
                cv2.imshow(self.window_name, bgr)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.get_logger().error(
                    f'Could not open the camera preview window: {exc}. '
                    'Check that this process has access to a desktop display.')
                self.show_window = False

    def _largest_block(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            _, _, width, height = cv2.boundingRect(contour)
            if width < 8 or height < 8 or max(width, height) > 120:
                continue
            aspect = width / float(height)
            rectangularity = area / float(width * height)
            # Reject large background/robot color patches. A cube should have
            # a compact, roughly square image footprint, even when partly cut.
            if not 0.55 <= aspect <= 1.80 or rectangularity < 0.55:
                continue
            candidates.append((area, contour))
        if not candidates:
            return None
        _, contour = max(candidates, key=lambda item: item[0])
        moments = cv2.moments(contour)
        if moments['m00'] <= 0:
            return None
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        return center_x, center_y, cv2.contourArea(contour), contour

    def _sample_depth(self, depth, u, v):
        radius = max(1, self.depth_patch // 2)
        height, width = depth.shape[:2]
        patch = depth[
            max(0, v - radius):min(height, v + radius + 1),
            max(0, u - radius):min(width, u + radius + 1),
        ]
        valid = patch[np.isfinite(patch) & (patch >= self.min_depth) & (patch <= self.max_depth)]
        return float(np.median(valid)) if valid.size else None

    def _pixel_to_base(self, u, v, depth, rgb_msg, camera_info):
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx <= 0 or fy <= 0:
            return None
        source_frame = camera_info.header.frame_id or rgb_msg.header.frame_id
        if not source_frame:
            return None

        point = PointStamped()
        point.header = rgb_msg.header
        point.header.frame_id = source_frame
        point.point.x = (float(u) - cx) * depth / fx
        point.point.y = (float(v) - cy) * depth / fy
        point.point.z = depth
        try:
            stamp = Time.from_msg(rgb_msg.header.stamp)
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, source_frame, stamp, timeout=Duration(seconds=0.25))
            transformed = tf2_geometry_msgs.do_transform_point(point, transform)
            return transformed.point.x, transformed.point.y, transformed.point.z
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_tf_warning > 2.0:
                self.get_logger().warning(
                    f'Waiting for TF {source_frame} -> {self.base_frame}: {exc}')
                self.last_tf_warning = now
            return None


def main(args=None):
    rclpy.init(args=args)
    node = CameraPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
