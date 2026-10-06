import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    package_share = get_package_share_directory('ur3_llm')
    description = os.path.join(package_share, 'urdf', 'ur3_llm.urdf.xacro')
    semantic = os.path.join(package_share, 'config', 'ur3_llm.srdf')
    controllers = os.path.join(package_share, 'config', 'moveit_controllers.yaml')
    kinematics = os.path.join(package_share, 'config', 'kinematics.yaml')
    joint_limits = os.path.join(package_share, 'config', 'joint_limits.yaml')

    moveit_config = (
        MoveItConfigsBuilder('ur3_llm', package_name='ur3_llm')
        .robot_description(
            file_path=description,
            mappings={
                'ur_type': 'ur3',
                'simulation_controllers': os.path.join(
                    package_share, 'config', 'ros2_controllers.yaml'),
            },
        )
        .robot_description_semantic(file_path=semantic)
        .robot_description_kinematics(file_path=kinematics)
        .joint_limits(file_path=joint_limits)
        .trajectory_execution(
            file_path=controllers,
            moveit_manage_controllers=False,
        )
        .planning_pipelines(default_planning_pipeline='ompl', pipelines=['ompl'])
        .to_moveit_configs()
    )
    move_group_parameters = {
        'use_sim_time': True,
        'publish_robot_description_semantic': True,
        'allow_trajectory_execution': True,
        'capabilities': '',
        'disable_capabilities': '',
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'monitor_dynamics': False,
    }
    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), move_group_parameters],
    )
    return LaunchDescription([move_group])
