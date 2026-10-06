import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('ur3_llm')
    gazebo_share = get_package_share_directory('gazebo_ros')
    urdf_path = os.path.join(package_share, 'urdf', 'ur3_llm.urdf.xacro')
    world_path = os.path.join(package_share, 'worlds', 'ur3_sorting.world')
    controller_yaml = os.path.join(package_share, 'config', 'ros2_controllers.yaml')
    scene_path = os.path.join(package_share, 'config', 'scene.yaml')
    with open(scene_path, encoding='utf-8') as scene_file:
        scene = yaml.safe_load(scene_file)

    xacro_command = Command([
        'xacro ', urdf_path,
        ' ur_type:=ur3 simulation_controllers:=', controller_yaml,
    ])
    robot_description = {
        'robot_description': ParameterValue(xacro_command, value_type=str),
        'use_sim_time': True,
    }
    zone_names = list(scene['zones'])
    zone_centers = [value for zone in zone_names for value in scene['zones'][zone]]

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'world': world_path,
            'verbose': 'true',
            'gui': LaunchConfiguration('gui'),
        }.items(),
    )
    state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'publish_frequency': 50.0}],
    )
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_ur3_llm',
        output='screen',
        arguments=['-entity', 'ur3_llm', '-topic', 'robot_description'],
    )
    vision = Node(
        package='ur3_llm',
        executable='vision_node',
        name='camera_perception',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'rgb_topic': '/overhead_camera/camera/image_raw',
            'depth_topic': '/overhead_camera/camera/depth/image_raw',
            'camera_info_topic': '/overhead_camera/camera/camera_info',
            'objects_topic': '/vision/objects',
            'debug_image_topic': '/vision/debug',
            'base_frame': 'base_link',
            'zone_names': zone_names,
            'zone_centers_xy': zone_centers,
            'zone_radius': scene['vision']['zone_radius'],
            'min_area': scene['vision']['min_area'],
            'depth_patch': scene['vision']['depth_patch'],
            'min_depth_m': scene['vision']['min_depth_m'],
            'max_depth_m': scene['vision']['max_depth_m'],
            'show_window': ParameterValue(
                LaunchConfiguration('camera_window'), value_type=bool),
            'window_name': 'UR3 RGB-D Camera (Top View)',
        }],
    )
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, 'launch', 'moveit.launch.py')))

    state_broadcaster = Node(
        package='controller_manager', executable='spawner',
        arguments=[
            'joint_state_broadcaster', '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '60.0',
        ],
        output='screen')
    arm_controller = Node(
        package='controller_manager', executable='spawner',
        arguments=[
            'joint_trajectory_controller', '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '60.0',
        ],
        output='screen')
    gripper_controller = Node(
        package='controller_manager', executable='spawner',
        arguments=[
            'gripper_controller', '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '60.0',
        ],
        output='screen')
    # Do not shut down Gazebo from a controller-spawner exit event. A missing
    # controller should be diagnosable in the launch terminal while the world
    # and camera stay alive.
    start_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[state_broadcaster, arm_controller, gripper_controller],
        ))

    return LaunchDescription([
        DeclareLaunchArgument(
            'gui', default_value='true', description='Show the Gazebo client GUI.'),
        DeclareLaunchArgument(
            'camera_window', default_value='true',
            description='Show the annotated OpenCV camera window.'),
        gazebo,
        state_publisher,
        start_controllers,
        spawn_robot,
        vision,
        moveit,
    ])
