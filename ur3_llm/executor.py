#!/usr/bin/env python3
"""Interactive natural-language planner, validator and ROS skill executor."""

import json
import os
import sys
import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
import yaml

from ur3_llm.llm_planner import LLMPlanner
from ur3_llm.plan_validator import PlanValidator
from ur3_llm.robot_skills import RobotSkills


class SkillPlanningExecutor(Node):
    """Consume camera observations and execute only validated named skills."""

    def __init__(self):
        super().__init__('ur3_llm_executor')
        package_share = get_package_share_directory('ur3_llm')
        with open(os.path.join(package_share, 'config', 'scene.yaml'), encoding='utf-8') as f:
            self.scene = yaml.safe_load(f)

        self.state_condition = threading.Condition()
        self.latest_state = None
        self.state_received_at = None
        self.max_state_age = float(self.scene['vision']['max_state_age_s'])
        self.state_sub = self.create_subscription(
            String, '/vision/objects', self._vision_callback, 10)
        self.planner = LLMPlanner()
        self.validator = PlanValidator(self.scene)
        self.skills = RobotSkills(self, self.scene)
        self.get_logger().info('Executor ready. Camera state is required before planning.')

    def _vision_callback(self, msg):
        try:
            state = json.loads(msg.data)
            if not isinstance(state.get('objects'), dict) or not isinstance(state.get('zones'), dict):
                raise ValueError('objects and zones must be JSON objects')
            with self.state_condition:
                self.latest_state = state
                self.state_received_at = time.monotonic()
                self.state_condition.notify_all()
        except Exception as exc:
            self.get_logger().warning(f'Ignoring malformed camera state: {exc}')

    def _state_snapshot(self, timeout=2.0, after_time=None):
        deadline = time.monotonic() + timeout
        with self.state_condition:
            while True:
                age = (float('inf') if self.state_received_at is None
                       else time.monotonic() - self.state_received_at)
                is_new_enough = (
                    after_time is None
                    or (self.state_received_at is not None
                        and self.state_received_at > after_time)
                )
                if (self.latest_state is not None and age <= self.max_state_age
                        and is_new_enough):
                    state = json.loads(json.dumps(self.latest_state))
                    state['age_s'] = age
                    return state
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Không có camera state mới trong thời hạn cho phép.')
                self.state_condition.wait(min(remaining, 0.25))

    def _wait_for_state(self, predicate, timeout, description):
        deadline = time.monotonic() + timeout
        last_state = None
        while time.monotonic() < deadline:
            try:
                last_state = self._state_snapshot(timeout=0.25)
            except TimeoutError:
                continue
            if predicate(last_state):
                return last_state
            with self.state_condition:
                self.state_condition.wait(0.15)
        self.get_logger().error(f'Camera did not confirm {description}.')
        return None

    def run_command(self, command):
        print(f'\nUSER: {command}')
        try:
            state = self._state_snapshot(timeout=5.0)
            core_plan, source = self.planner.generate_core_plan(command, state)
            print(f'PLAN SOURCE: {source}')
            print('LLM CORE PLAN:')
            print(json.dumps(core_plan, indent=2, ensure_ascii=False))
            plan = self.validator.compile_plan(core_plan, state)
        except Exception as exc:
            self.get_logger().error(f'Could not plan command: {exc}')
            return False

        print('CAMERA-AWARE PLAN:')
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        valid, reason = self.validator.validate(plan, state)
        print(f'VALIDATION: {reason}')
        if not valid:
            self.get_logger().error('Plan rejected before execution.')
            return False
        if not self.skills.sync_planning_scene(state):
            self.get_logger().error('MoveIt collision scene could not be synchronized; task aborted.')
            return False

        return self._execute_plan(plan)

    def _execute_plan(self, plan):
        goal = plan['goal']
        for index, step in enumerate(plan['plan'], start=1):
            skill = step['skill']
            self.get_logger().info(f'STEP {index}/{len(plan["plan"])}: {skill}')
            try:
                if skill == 'check_zone':
                    state = self._state_snapshot()
                    actual = state['zones'][step['zone']].get('occupied_by')
                    if actual != step['observed_occupant']:
                        raise RuntimeError(
                            f'{step["zone"]} changed after planning '
                            f'({step["observed_occupant"]} -> {actual}); replan required.')
                    self.get_logger().info(f'{step["zone"]}: occupied_by={actual}')
                elif skill == 'find_free_position':
                    state = self._state_snapshot()
                    point = step['position']
                    free = self._is_temp_position_free(
                        point['x'], point['y'], state, excluded_object=step['object'])
                    if not free:
                        raise RuntimeError('Temporary position is no longer free; replan required.')
                    self.get_logger().info(
                        f'find_free_position({step["object"]}) -> '
                        f'({point["x"]:.3f}, {point["y"]:.3f})')
                elif skill == 'pick':
                    state = self._state_snapshot()
                    observed = state['objects'].get(step['object'])
                    if observed is None:
                        raise RuntimeError(f'Camera no longer sees {step["object"]}.')
                    if float(observed['z']) > self.scene['workspace']['table_z'] + 0.12:
                        raise RuntimeError(f'{step["object"]} does not appear to be on the table.')
                    # Keep the target cube in MoveIt's collision scene while
                    # planning the arm's route to the pre-grasp pose. The skill
                    # removes only that cube after reaching the approach pose.
                    if not self.skills.sync_planning_scene(state):
                        raise RuntimeError('Could not synchronize the approach collision scene.')
                    if not self.skills.pick(
                            step['object'], observed, self._state_snapshot):
                        try:
                            latest = self._state_snapshot()
                            self.skills.sync_planning_scene(latest)
                        except TimeoutError:
                            self.skills.sync_planning_scene(state)
                        raise RuntimeError(f'pick({step["object"]}) failed in MoveIt/gripper.')
                    raised = self._wait_for_state(
                        lambda latest: (
                            (
                                step['object'] in latest['objects']
                                and float(latest['objects'][step['object']]['z'])
                                > self.scene['workspace']['table_z'] + 0.075
                            )
                            # The overhead camera may see only the gripper after
                            # it closes over a block. Keep holding in that case;
                            # the release must still be confirmed by the camera.
                            or (
                                step['object'] not in latest['objects']
                                and all(
                                    zone.get('occupied_by') != step['object']
                                    for zone in latest['zones'].values()
                                )
                            )
                        ),
                        timeout=3.0,
                        description=f'{step["object"]} lifted or occluded at the gripper',
                    )
                    if raised is None:
                        self.skills.open_gripper()
                        latest = self._state_snapshot()
                        self.skills.sync_planning_scene(latest)
                        raise RuntimeError(
                            f'Camera did not confirm that {step["object"]} was grasped.')
                    if step['object'] in raised['objects']:
                        self.get_logger().info(
                            f'Camera confirmed lifted {step["object"]}.')
                    else:
                        self.get_logger().warning(
                            f'{step["object"]} is occluded by the gripper; '
                            'keep holding and require camera confirmation after release.')
                elif skill == 'place':
                    target = step['target']
                    fallback_positions = []
                    if target['type'] == 'zone':
                        state = self._state_snapshot()
                        if state['zones'][target['name']].get('occupied_by') is not None:
                            raise RuntimeError(f'{target["name"]} became occupied before place.')
                        x, y = self.scene['zones'][target['name']]
                    else:
                        x, y = float(target['x']), float(target['y'])
                        state = self._state_snapshot()
                        if not self._is_temp_position_free(
                                x, y, state, excluded_object=step['object']):
                            raise RuntimeError('Temporary target was occupied before place.')
                        for candidate in self.scene.get('temporary_candidates', []):
                            candidate_x, candidate_y = map(float, candidate)
                            if (abs(candidate_x - x) < 1e-6
                                    and abs(candidate_y - y) < 1e-6):
                                continue
                            if self._is_temp_position_free(
                                    candidate_x, candidate_y, state,
                                    excluded_object=step['object']):
                                fallback_positions.append((candidate_x, candidate_y))
                    placed_at = self.skills.place(
                        step['object'], x, y, fallback_positions=fallback_positions)
                    if placed_at is None:
                        raise RuntimeError(f'place({step["object"]}) failed in MoveIt/gripper.')
                    x, y = placed_at
                    if target['type'] == 'position':
                        target = {'type': 'position', 'x': x, 'y': y}
                    placed = self._wait_for_state(
                        lambda latest: self._camera_confirms_place(
                            latest, step['object'], target, x, y),
                        timeout=3.0,
                        description=f'{step["object"]} placed at {target}',
                    )
                    if placed is None:
                        raise RuntimeError(f'Camera did not confirm place({step["object"]}).')
                    if not self.skills.sync_planning_scene(placed):
                        raise RuntimeError('Could not update MoveIt with the camera-confirmed place.')
                elif skill == 'home':
                    if not self.skills.home():
                        raise RuntimeError('home() failed in MoveIt.')
                else:
                    raise RuntimeError(f'Unknown skill {skill}.')
            except Exception as exc:
                self.get_logger().error(f'Plan stopped at step {index}: {exc}')
                return False

        final_state = self._wait_for_state(
            lambda latest: latest['zones'][goal['zone']].get('occupied_by') == goal['object'],
            timeout=2.0,
            description=f'{goal["object"]} in {goal["zone"]}',
        )
        if final_state is None:
            self.get_logger().error('Task ended without camera confirmation of the goal.')
            return False
        print('\nTASK SUCCESS: camera confirms the requested zone occupancy.\n')
        return True

    def _is_temp_position_free(self, x, y, state, excluded_object=None):
        clearance = float(self.scene['workspace']['object_clearance'])
        for name, obj in state.get('objects', {}).items():
            if name == excluded_object:
                continue
            if ((float(obj['x']) - x) ** 2 + (float(obj['y']) - y) ** 2) ** 0.5 < clearance:
                return False
        for center in self.scene['zones'].values():
            if ((float(center[0]) - x) ** 2 + (float(center[1]) - y) ** 2) ** 0.5 \
                    < self.scene['vision']['zone_radius'] + 0.025:
                return False
        return True

    def _camera_confirms_place(self, state, object_name, target, x, y):
        observation = state.get('objects', {}).get(object_name)
        if observation is None:
            return False
        if target['type'] == 'zone':
            return state['zones'][target['name']].get('occupied_by') == object_name
        tolerance = 0.07
        return (
            ((float(observation['x']) - x) ** 2 + (float(observation['y']) - y) ** 2) ** 0.5
            <= tolerance
            and float(observation['z']) <= self.scene['workspace']['table_z'] + 0.10
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SkillPlanningExecutor()
    except Exception as exc:
        print(f'Executor startup failed: {exc}', file=sys.stderr)
        if rclpy.ok():
            rclpy.shutdown()
        return
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    print('\nUR3 LLM Skill Planner. Gõ lệnh tiếng Anh hoặc tiếng Việt; exit để thoát.')
    print('Ví dụ: Put the red cube in Zone B.\n')
    try:
        while rclpy.ok():
            try:
                command = input('ur3> ').strip()
            except EOFError:
                break
            if command.lower() in ('exit', 'quit', 'q'):
                break
            if command:
                node.run_command(command)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
