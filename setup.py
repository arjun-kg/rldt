from setuptools import setup

setup(name='rldt',
      version='0.1.2',
      packages=["rldt","rldt.envs"],
      install_requires=['gym']  # And any other dependencies we need
)