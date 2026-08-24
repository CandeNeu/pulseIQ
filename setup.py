from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='PulseIQ',
      version="0.0.10",
      description="PulseIQ",
      license="MIT",
      author="Le Wagon",
      author_email="contact@lewagon.org",
      install_requires=requirements,
      packages=find_packages())
