from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'ur3_llm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', '.setup_assistant']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UR3 LLM Lab',
    maintainer_email='duong2005@todo.todo',
    description='Camera-grounded skill planning and execution for UR3 in Gazebo.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vision_node = ur3_llm.vision_node:main',
            'executor = ur3_llm.executor:main',
        ],
    },
)
