from setuptools import setup

setup(name='collision_avoidance',
      version='0.0.1',
      install_requires=['gym'],  # And any other dependencies foo needs
      packages=['collision_avoidance','collision_avoidance.envs']
)