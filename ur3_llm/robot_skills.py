"""MoveIt arm motion and trajectory-controlled physical gripper skills."""

import math
import threading
import time

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    PlanningScene,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


class RobotSkills:
    """Concrete pick/place/home skills; all arm paths are planned by MoveIt."""

    ARM_JOINTS = (
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    )

    def __init__(self, node, scene):
        self.node = node
        self.scene = scene
        self.motion = scene['motion']
        self._gripper_state_lock = threading.Lock()
        self._gripper_positions = {}
        self._gripper_velocities = {}
        self._active_gripper_hold = None
        self._joint_state_sub = node.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)
        self.move_group = ActionClient(node, MoveGroup, '/move_action')
        self.execute_trajectory = ActionClient(
            node, ExecuteTrajectory, '/execute_trajectory')
        self.cartesian_path = node.create_client(
            GetCartesianPath, '/compute_cartesian_path')
        self.gripper = ActionClient(
            node, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory')
        self.node.get_logger().info('Waiting for MoveIt move_action...')
        if not self.move_group.wait_for_server(timeout_sec=45.0):
            raise RuntimeError('MoveIt action /move_action is unavailable.')
        self.node.get_logger().info('Waiting for MoveIt Cartesian path service...')
        if not self.cartesian_path.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(
                'MoveIt service /compute_cartesian_path is unavailable.')
        if not self.execute_trajectory.wait_for_server(timeout_sec=30.0):
            raise RuntimeError(
                'MoveIt action /execute_trajectory is unavailable.')
        self.apply_scene_client = node.create_client(
            ApplyPlanningScene, '/apply_planning_scene')
        if not self.apply_scene_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError('MoveIt service /apply_planning_scene is unavailable.')
        self.planning_scene_ids = set()
        self.known_object_positions = {}
        self.node.get_logger().info('Waiting for physical gripper controller...')
        if not self.gripper.wait_for_server(timeout_sec=60.0):
            raise RuntimeError(
                'Action /gripper_controller/follow_joint_trajectory is unavailable. '
                'Keep simulation.launch.py running and check that gripper_controller '
                'reached the active state in its terminal.')

    def sync_planning_scene(self, state, excluded_objects=(), exact_object=None):
        """Update MoveIt collision geometry from camera state and table calibration."""
        excluded = set(excluded_objects)
        scene_diff = PlanningScene()
        scene_diff.is_diff = True
        scene_diff.robot_state.is_diff = True

        table = CollisionObject()
        table.header.frame_id = 'base_link'
        table.id = 'work_table_surface'
        table.operation = CollisionObject.ADD
        table_box = SolidPrimitive()
        table_box.type = SolidPrimitive.BOX
        table_box.dimensions = [0.90, 0.80, 0.02]
        table_pose = PoseStamped().pose
        table_pose.position.x = 0.25
        table_pose.position.z = -0.01
        table_pose.orientation.w = 1.0
        table.primitives.append(table_box)
        table.primitive_poses.append(table_pose)
        scene_diff.world.collision_objects.append(table)

        table_z = float(self.scene['workspace']['table_z'])
        for name, observation in state.get('objects', {}).items():
            if name in excluded:
                continue
            surface_z = float(observation['z'])
            if surface_z <= table_z + 0.10:
                self.known_object_positions[name] = (
                    float(observation['x']),
                    float(observation['y']),
                    max(table_z + 0.025, surface_z - 0.025),
                )

        current_ids = {'work_table_surface'}
        for name, xyz in self.known_object_positions.items():
            if name in excluded:
                continue
            collision = CollisionObject()
            collision.header.frame_id = 'base_link'
            collision.id = name
            collision.operation = CollisionObject.ADD
            cube = SolidPrimitive()
            cube.type = SolidPrimitive.BOX
            if name == exact_object:
                # Keep the grasp target as an obstacle during the open-finger
                # descent, using its actual dimensions for the narrow approach.
                cube.dimensions = [0.05, 0.05, 0.05]
            else:
                # Add clearance around other camera-estimated cubes so arm
                # tracking and small depth/TF errors do not cause a bump.
                cube.dimensions = [0.07, 0.07, 0.06]
            cube_pose = PoseStamped().pose
            cube_pose.position.x, cube_pose.position.y, cube_pose.position.z = xyz
            cube_pose.orientation.w = 1.0
            collision.primitives.append(cube)
            collision.primitive_poses.append(cube_pose)
            scene_diff.world.collision_objects.append(collision)
            current_ids.add(name)

        for stale_id in self.planning_scene_ids - current_ids:
            remove = CollisionObject()
            remove.header.frame_id = 'base_link'
            remove.id = stale_id
            remove.operation = CollisionObject.REMOVE
            scene_diff.world.collision_objects.append(remove)

        request = ApplyPlanningScene.Request()
        request.scene = scene_diff
        result = self._wait_future(self.apply_scene_client.call_async(request), timeout=8.0)
        if result is None or not result.success:
            self.node.get_logger().error('MoveIt rejected the camera collision-scene update.')
            return False
        self.planning_scene_ids = current_ids
        return True

    def pick(self, object_name, position, state_provider, retry_count=0):
        """Approach with the block as an obstacle, then grasp its latest camera pose."""
        x, y = float(position['x']), float(position['y'])
        approach_z = float(self.motion['approach_z'])
        grasp_z = float(self.motion['grasp_z'])
        lift_z = float(self.motion['lift_z'])
        self.node.get_logger().info(f'pick({object_name}) at camera ({x:.3f}, {y:.3f})')
        if not self.open_gripper():
            return False

        # Read the block from a clear side view immediately before approach.
        # Going straight to its center can hide it under the wrist, leaving a
        # stale camera coordinate that makes the open fingers clip its side.
        workspace = self.scene['workspace']
        view_z = approach_z + 0.10
        offset = 0.14
        view_candidates = (
            (x + offset, y), (x - offset, y),
            (x, y + offset), (x, y - offset),
        )
        visible_state = None
        for view_x, view_y in view_candidates:
            if not (float(workspace['x_min']) <= view_x <= float(workspace['x_max'])
                    and float(workspace['y_min']) <= view_y <= float(workspace['y_max'])
                    and math.hypot(view_x, view_y) <= float(workspace['max_reach'])):
                continue
            if not self.move_to_pose(view_x, view_y, view_z):
                continue
            view_finished = time.monotonic()
            try:
                view_state = state_provider(timeout=2.0, after_time=view_finished)
            except TimeoutError:
                continue
            view_observation = view_state.get('objects', {}).get(object_name)
            if view_observation is None:
                continue
            if float(view_observation['z']) > float(workspace['table_z']) + 0.12:
                continue
            visible_state = view_state
            x, y = float(view_observation['x']), float(view_observation['y'])
            break
        if visible_state is None:
            self.node.get_logger().error(
                f'Could not get a fresh clear camera view of {object_name}; '
                'will not descend.')
            return False
        if not self.sync_planning_scene(visible_state):
            return False

        # A block can move if the arm clips it, or if the camera position was
        # stale. Keep it as an obstacle during pre-grasp motion, then observe
        # again before descending. Allow one camera-guided re-approach only.
        latest_state = visible_state
        for attempt in range(2):
            if attempt > 0 and not self.sync_planning_scene(latest_state):
                return False
            if not self.move_to_pose(x, y, approach_z):
                return False
            approach_finished = time.monotonic()
            try:
                latest_state = state_provider(
                    timeout=2.0, after_time=approach_finished)
            except TimeoutError:
                self.node.get_logger().error(
                    f'No fresh camera frame after approaching {object_name}; '
                    'will not descend.')
                return False

            observation = latest_state.get('objects', {}).get(object_name)
            if observation is None:
                # At the centered pre-grasp pose the overhead camera can be
                # fully blocked by the wrist/gripper. The approach was planned
                # with this padded block collision object present, so retain
                # the last camera-confirmed coordinates and continue vertically.
                self.node.get_logger().warning(
                    f'{object_name} is occluded by the arm at pre-grasp; '
                    'using its clear-view camera pose for the grasp.')
                break
            if float(observation['z']) > float(
                    self.scene['workspace']['table_z']) + 0.12:
                self.node.get_logger().error(
                    f'{object_name} is no longer on the table after the approach.')
                return False

            latest_x = float(observation['x'])
            latest_y = float(observation['y'])
            shift = math.hypot(latest_x - x, latest_y - y)
            if shift > 0.015:
                if attempt == 1:
                    self.sync_planning_scene(latest_state)
                    self.node.get_logger().error(
                        f'{object_name} moved {shift:.3f} m during re-approach; '
                        'stopping before grasp.')
                    return False
                self.node.get_logger().warning(
                    f'Camera found {object_name} shifted by {shift:.3f} m; '
                    'updating the collision scene and re-approaching safely.')
                x, y = latest_x, latest_y
                visible_state = latest_state
                continue
            x, y = latest_x, latest_y
            visible_state = latest_state
            break

        # Keep the exact-size cube in the scene while open fingers descend.
        # If the camera alignment is still off, MoveIt rejects the path before
        # the real robot can push the block. Remove it only after closing.
        grasp_scene = dict(latest_state)
        grasp_scene['objects'] = dict(latest_state.get('objects', {}))
        grasp_scene['objects'].setdefault(
            object_name, visible_state['objects'][object_name])
        if not self.sync_planning_scene(grasp_scene, exact_object=object_name):
            return False
        self.node.get_logger().info(
            f'Camera-confirmed {object_name} grasp pose ({x:.3f}, {y:.3f})')
        # OMPL often rejects the final low pose near the table even when the
        # straight vertical path is collision-free. Plan this short segment as
        # a Cartesian path and reject partial paths before moving the real arm.
        if not self.move_cartesian_z(x, y, approach_z, grasp_z):
            return False

        # Descending can still nudge a cube if the Gazebo pose and the last
        # overhead-camera estimate differ. Check a fresh frame before closing
        # the fingers so we do not squeeze at a stale coordinate.
        descent_finished = time.monotonic()
        try:
            contact_state = state_provider(
                timeout=2.0, after_time=descent_finished)
        except TimeoutError:
            self.node.get_logger().error(
                f'No fresh camera frame after lowering to {object_name}; '
                'stopping before the gripper closes.')
            return False

        contact_observation = contact_state.get('objects', {}).get(object_name)
        if contact_observation is not None:
            contact_x = float(contact_observation['x'])
            contact_y = float(contact_observation['y'])
            shift = math.hypot(contact_x - x, contact_y - y)
            self.node.get_logger().info(
                f'Post-descent camera sees {object_name} at '
                f'({contact_x:.3f}, {contact_y:.3f}), '
                f'shift={shift:.3f} m.')
            if shift > 0.010:
                if not self.sync_planning_scene(
                        contact_state, exact_object=object_name):
                    return False
                if not self.move_cartesian_z(
                        x, y, grasp_z, approach_z):
                    return False
                if retry_count >= 1:
                    self.node.get_logger().error(
                        f'{object_name} moved {shift:.3f} m while lowering; '
                        'stopping after one camera-guided retry.')
                    return False
                self.node.get_logger().warning(
                    f'Camera found {object_name} moved {shift:.3f} m '
                    'during descent; retreating and retrying at its new pose.')
                return self.pick(
                    object_name, contact_observation, state_provider,
                    retry_count=retry_count + 1)
            if not self.sync_planning_scene(
                    contact_state, exact_object=object_name):
                return False
        else:
            self.node.get_logger().warning(
                f'{object_name} is occluded at the grasp pose; '
                'retaining the last camera-confirmed position.')

        if not self.close_gripper():
            self.node.get_logger().error(
                f'Could not confirm {object_name} jaw closure; '
                'leaving the gripper in its current position.')
            return False
        if not self.sync_planning_scene(
                grasp_scene, excluded_objects={object_name}):
            self.node.get_logger().error(
                f'Could not clear the grasped {object_name} from the world collision scene.')
            return False
        if not self.move_cartesian_z(x, y, grasp_z, lift_z):
            self.node.get_logger().error(
                f'Could not lift {object_name}; keeping the gripper closed.')
            return False
        # Clear the source before judging the grasp. From directly above, the
        # gripper can hide a block that is still sitting on the table.
        y_min = float(self.scene['workspace']['y_min'])
        y_max = float(self.scene['workspace']['y_max'])
        retreat_y = [value for value in (y + 0.12, y - 0.12)
                     if y_min <= value <= y_max]
        for clear_y in retreat_y:
            if self.move_to_pose(x, clear_y, lift_z):
                return True
        self.node.get_logger().error(
            f'Could not move clear of {object_name}; keeping the gripper closed.')
        return False

    def place(self, object_name, x, y, fallback_positions=()):
        """Place physically, trying other camera-free positions before release."""
        candidates = [(float(x), float(y))]
        for candidate in fallback_positions:
            point = (float(candidate[0]), float(candidate[1]))
            if point not in candidates:
                candidates.append(point)

        approach_z = float(self.motion['approach_z'])
        grasp_z = float(self.motion['grasp_z'])
        selected = None
        for candidate_x, candidate_y in candidates:
            self.node.get_logger().info(
                f'place({object_name}) planning at '
                f'({candidate_x:.3f}, {candidate_y:.3f})')
            if not self.move_to_pose(candidate_x, candidate_y, approach_z):
                continue
            if not self.move_cartesian_z(
                    candidate_x, candidate_y, approach_z, grasp_z):
                continue
            selected = (candidate_x, candidate_y)
            break
        if selected is None:
            self.node.get_logger().error(
                f'No collision-free place pose found for {object_name}; '
                'keeping the gripper closed.')
            return None
        if not self.open_gripper():
            return None
        if not self.move_cartesian_z(
                selected[0], selected[1], grasp_z,
                float(self.motion['lift_z'])):
            self.node.get_logger().warning(
                'Block was released, but the post-place lift failed; '
                'camera verification will decide whether to continue.')
        return selected

    def home(self):
        self.node.get_logger().info('home()')
        if not self.open_gripper():
            return False
        return self.move_to_joints(self.motion['home_joints'])

    def open_gripper(self):
        positions, _ = self._gripper_position_snapshot()
        if self._active_gripper_hold is None and all(
                abs(positions.get(name, float('inf'))) < 0.002
                for name in ('left_finger_joint', 'right_finger_joint')):
            self.node.get_logger().info(
                'Gripper is already open; skipping a redundant trajectory.')
            return True
        return self._send_gripper_goal([0.0, 0.0])

    def close_gripper(self):
        # Keep a position trajectory active while the arm lifts and carries
        # the cube. Humble's JTC effort mode rejects trajectory effort fields;
        # its PID therefore needs the position goal to remain active to retain
        # pinch force. The next open_gripper goal preempts this hold.
        # 40 mm inward travel lets the opposite jaw reach cubes offset by up
        # to about 15 mm instead of letting the first jaw push them away.
        target = [0.015, -0.015]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'left_finger_joint', 'right_finger_joint']
        for duration in (2, 300):
            point = JointTrajectoryPoint()
            point.positions = target
            point.velocities = [0.0, 0.0]
            point.time_from_start = DurationMsg(sec=duration, nanosec=0)
            goal.trajectory.points.append(point)
        goal.path_tolerance = [
            JointTolerance(name='left_finger_joint', position=-1.0, velocity=-1.0),
            JointTolerance(name='right_finger_joint', position=-1.0, velocity=-1.0),
        ]
        goal.goal_tolerance = [
            JointTolerance(name='left_finger_joint', position=0.024, velocity=0.10),
            JointTolerance(name='right_finger_joint', position=0.024, velocity=0.10),
        ]
        goal.goal_time_tolerance = DurationMsg(sec=3, nanosec=0)

        accepted_event = threading.Event()
        response = {'handle': None, 'error': None}

        def on_goal_response(future):
            try:
                response['handle'] = future.result()
            except Exception as exc:
                response['error'] = exc
            finally:
                accepted_event.set()

        self.gripper.send_goal_async(goal).add_done_callback(on_goal_response)
        if not accepted_event.wait(timeout=5.0):
            self.node.get_logger().error(
                'Timed out waiting for the gripper hold trajectory response.')
            return False
        handle = response['handle']
        if response['error'] is not None or handle is None or not handle.accepted:
            self.node.get_logger().error(
                f'Gripper hold trajectory was rejected: {response["error"]!r}.')
            return False
        self._active_gripper_hold = handle

        # Wait for the two jaws to move inward before starting the arm lift.
        # Camera verification after the lift remains the actual grasp check.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            positions, velocities = self._gripper_position_snapshot()
            left = positions.get('left_finger_joint')
            right = positions.get('right_finger_joint')
            if left is not None and right is not None \
                    and left >= 0.004 and right <= -0.004:
                self.node.get_logger().info(
                    f'Gripper hold active at left={left:.4f}, right={right:.4f}; '
                    'keeping the pinch command active through lift.')
                return True
            time.sleep(0.05)

        self.node.get_logger().error(
            'Gripper hold was accepted, but both jaws did not move inward; '
            'aborting before lift.')
        handle.cancel_goal_async()
        self._active_gripper_hold = None
        return False

    def _on_joint_state(self, message):
        positions = dict(zip(message.name, message.position))
        velocities = dict(zip(message.name, message.velocity))
        with self._gripper_state_lock:
            for name in ('left_finger_joint', 'right_finger_joint'):
                if name in positions:
                    self._gripper_positions[name] = float(positions[name])
                if name in velocities:
                    self._gripper_velocities[name] = float(velocities[name])

    def _gripper_position_snapshot(self):
        with self._gripper_state_lock:
            return dict(self._gripper_positions), dict(self._gripper_velocities)

    def _send_gripper_goal(self, positions):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['left_finger_joint', 'right_finger_joint']
        # Contact is an expected terminal condition for a parallel gripper:
        # the jaws should keep applying effort after a cube blocks travel.
        # The camera lift check in executor.py is the grasp success criterion.
        goal.goal_tolerance = [
            JointTolerance(name='left_finger_joint', position=0.024, velocity=0.10),
            JointTolerance(name='right_finger_joint', position=0.024, velocity=0.10),
        ]
        goal.path_tolerance = [
            JointTolerance(name='left_finger_joint', position=-1.0, velocity=-1.0),
            JointTolerance(name='right_finger_joint', position=-1.0, velocity=-1.0),
        ]
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.velocities = [0.0, 0.0]
        point.time_from_start = DurationMsg(sec=2, nanosec=0)
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = DurationMsg(sec=3, nanosec=0)
        result = self._send_action(self.gripper, goal, timeout=12.0)
        if result is None:
            self.node.get_logger().error('Gripper trajectory timed out or was rejected.')
            return False
        code = result.result.error_code
        measured, velocities = self._gripper_position_snapshot()
        measured_text = ', '.join(
            (f'{name}={measured[name]:.4f} (v={velocities[name]:.4f})'
             if name in velocities else f'{name}={measured[name]:.4f}')
            for name in ('left_finger_joint', 'right_finger_joint')
            if name in measured)
        if measured_text:
            self.node.get_logger().info(
                f'Gripper final joint positions: {measured_text}; '
                f'target=({positions[0]:.4f}, {positions[1]:.4f}).')
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.node.get_logger().error(
                f'Gripper action failed with error_code={code}, '
                f'error_string={result.result.error_string!r}.')
            return False
        self._active_gripper_hold = None
        return True

    def move_to_pose(self, x, y, z):
        target = PoseStamped()
        target.header.frame_id = 'base_link'
        target.header.stamp = self.node.get_clock().now().to_msg()
        target.pose.position.x = float(x)
        target.pose.position.y = float(y)
        target.pose.position.z = float(z)
        # Tool0 +Z points down after a half-turn about its X axis.
        target.pose.orientation.x = 1.0
        target.pose.orientation.w = 0.0

        request = MotionPlanRequest()
        request.group_name = 'ur_manipulator'
        request.start_state.is_diff = True
        request.num_planning_attempts = 16
        request.allowed_planning_time = 15.0
        request.max_velocity_scaling_factor = float(self.motion['max_velocity_scaling'])
        request.max_acceleration_scaling_factor = float(self.motion['max_acceleration_scaling'])
        request.planner_id = 'RRTConnectkConfigDefault'

        constraints = Constraints()
        position = PositionConstraint()
        position.header.frame_id = 'base_link'
        position.link_name = 'tool0'
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        # This pose is the grasp approach; an 18 mm tolerance can put an open
        # finger into a 50 mm cube before the camera-guided vertical descent.
        sphere.dimensions = [0.018]
        position.constraint_region.primitives.append(sphere)
        position.constraint_region.primitive_poses.append(target.pose)
        position.weight = 1.0
        constraints.position_constraints.append(position)

        orientation = OrientationConstraint()
        orientation.header.frame_id = 'base_link'
        orientation.link_name = 'tool0'
        orientation.orientation = target.pose.orientation
        # OMPL uses a safe height to reach the target. The subsequent
        # Cartesian path aligns this exact orientation before descending.
        orientation.absolute_x_axis_tolerance = 0.12
        orientation.absolute_y_axis_tolerance = 0.12
        orientation.absolute_z_axis_tolerance = 3.14159
        orientation.weight = 1.0
        constraints.orientation_constraints.append(orientation)
        request.goal_constraints.append(constraints)

        goal = MoveGroup.Goal()
        goal.request = request
        action_result = self._send_action(self.move_group, goal, timeout=150.0)
        if action_result is None:
            self.node.get_logger().error(f'MoveIt failed to reach pose ({x:.3f}, {y:.3f}, {z:.3f}).')
            return False
        if action_result.result.error_code.val != 1:
            self.node.get_logger().error(
                f'MoveIt failed at tool0 target '
                f'({x:.3f}, {y:.3f}, {z:.3f}), '
                f'error_code={action_result.result.error_code.val}.')
            return False
        return True

    def move_cartesian_z(self, x, y, start_z, target_z):
        """Plan and execute a collision-checked vertical tool motion in MoveIt."""
        waypoints = []
        for z in (start_z, target_z):
            waypoint = PoseStamped()
            waypoint.header.frame_id = 'base_link'
            waypoint.header.stamp = self.node.get_clock().now().to_msg()
            waypoint.pose.position.x = float(x)
            waypoint.pose.position.y = float(y)
            waypoint.pose.position.z = float(z)
            waypoint.pose.orientation.x = 1.0
            waypoint.pose.orientation.w = 0.0
            waypoints.append(waypoint.pose)

        request = GetCartesianPath.Request()
        request.header.frame_id = 'base_link'
        request.header.stamp = self.node.get_clock().now().to_msg()
        request.start_state.is_diff = True
        request.group_name = 'ur_manipulator'
        request.link_name = 'tool0'
        request.waypoints = waypoints
        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.prismatic_jump_threshold = 0.0
        request.revolute_jump_threshold = 0.0
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = float(
            self.motion['max_velocity_scaling'])
        request.max_acceleration_scaling_factor = float(
            self.motion['max_acceleration_scaling'])

        self.node.get_logger().info(
            f'MoveIt Cartesian vertical path: z={float(start_z):.3f} '
            f'-> {float(target_z):.3f} at ({float(x):.3f}, {float(y):.3f})')
        response = self._wait_future(
            self.cartesian_path.call_async(request), timeout=30.0)
        if response is None:
            self.node.get_logger().error('MoveIt Cartesian path request failed.')
            return False
        if response.error_code.val != 1 or response.fraction < 0.999:
            self.node.get_logger().error(
                'MoveIt rejected the vertical path: '
                f'fraction={response.fraction:.3f}, '
                f'error_code={response.error_code.val}; arm was not moved.')
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        result = self._send_action(
            self.execute_trajectory, goal, timeout=60.0)
        if result is None:
            self.node.get_logger().error(
                'MoveIt failed to execute the Cartesian vertical path.')
            return False
        if result.result.error_code.val != 1:
            self.node.get_logger().error(
                'MoveIt Cartesian trajectory execution failed: '
                f'error_code={result.result.error_code.val}.')
            return False
        return True

    def move_to_joints(self, values):
        if len(values) != len(self.ARM_JOINTS):
            self.node.get_logger().error('Home target must contain six joint values.')
            return False
        request = MotionPlanRequest()
        request.group_name = 'ur_manipulator'
        request.start_state.is_diff = True
        request.num_planning_attempts = 8
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = float(self.motion['max_velocity_scaling'])
        request.max_acceleration_scaling_factor = float(self.motion['max_acceleration_scaling'])
        constraints = Constraints()
        for name, value in zip(self.ARM_JOINTS, values):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(value)
            joint.tolerance_above = 0.04
            joint.tolerance_below = 0.04
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        request.goal_constraints.append(constraints)
        goal = MoveGroup.Goal()
        goal.request = request
        action_result = self._send_action(self.move_group, goal, timeout=150.0)
        if action_result is None:
            self.node.get_logger().error('MoveIt failed to return home.')
            return False
        if action_result.result.error_code.val != 1:
            self.node.get_logger().error(
                f'MoveIt home planning/execution error: {action_result.result.error_code.val}')
            return False
        return True

    def _send_action(self, client, goal, timeout):
        done = threading.Event()
        response_received = threading.Event()
        cancel_finished = threading.Event()
        cancel_requested = threading.Event()
        cancel_lock = threading.Lock()
        holder = {
            'value': None,
            'exception': None,
            'goal_handle': None,
            'cancel_sent': False,
        }

        def cancel_goal(goal_handle):
            if goal_handle is None or not goal_handle.accepted:
                return
            with cancel_lock:
                if holder['cancel_sent']:
                    return
                holder['cancel_sent'] = True
            try:
                cancel_future = goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(lambda _: cancel_finished.set())
            except Exception as exc:
                self.node.get_logger().error(f'Could not cancel active motion: {exc}')
                cancel_finished.set()

        def on_goal_response(future):
            try:
                goal_handle = future.result()
                holder['goal_handle'] = goal_handle
                if goal_handle is None or not goal_handle.accepted:
                    holder['value'] = None
                    done.set()
                    return
                if cancel_requested.is_set():
                    cancel_goal(goal_handle)
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(on_result)
            except Exception as exc:  # Surface action transport errors to the caller.
                holder['exception'] = exc
                done.set()
            finally:
                response_received.set()

        def on_result(future):
            try:
                holder['value'] = future.result()
            except Exception as exc:
                holder['exception'] = exc
            finally:
                done.set()

        send_future = client.send_goal_async(goal)
        send_future.add_done_callback(on_goal_response)
        try:
            if not done.wait(timeout):
                self.node.get_logger().error('Action timed out; cancelling its motion goal.')
                cancel_requested.set()
                response_received.wait(timeout=1.0)
                cancel_goal(holder['goal_handle'])
                if not cancel_finished.wait(timeout=2.0) or not done.wait(timeout=2.0):
                    raise RuntimeError(
                        'Timed-out robot action did not confirm cancellation; '
                        'stopping the skill sequence.')
                return None
        except KeyboardInterrupt:
            self.node.get_logger().warning(
                'Interrupt received; cancelling the active robot motion before shutdown.')
            cancel_requested.set()
            response_received.wait(timeout=1.0)
            cancel_goal(holder['goal_handle'])
            cancel_finished.wait(timeout=1.0)
            done.wait(timeout=1.0)
            raise
        if holder['exception'] is not None:
            self.node.get_logger().error(f'Action transport error: {holder["exception"]}')
            return None
        return holder['value']

    def _wait_future(self, future, timeout):
        done = threading.Event()
        result = {'value': None, 'exception': None}

        def on_done(completed):
            try:
                result['value'] = completed.result()
            except Exception as exc:
                result['exception'] = exc
            finally:
                done.set()

        future.add_done_callback(on_done)
        if not done.wait(timeout):
            self.node.get_logger().error('MoveIt planning-scene service timed out.')
            return None
        if result['exception'] is not None:
            self.node.get_logger().error(f'Planning-scene service error: {result["exception"]}')
            return None
        return result['value']
