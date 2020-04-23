import os
import subprocess
import setuptools

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

setuptools.setup(
    name="EAST",
    version="0.0.0",
    description="EAST Text Detection",
    # The project's main homepage.
    url="https://github.com/malcolmgreaves/EAST",
    # Author details
    author="Malcolm Greaves",
    author_email="greaves.malcolm+oss@gmail.com",
    # Choose your license
    license="Apache License 2.0",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache License 2.0",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="text detection, computer vision, machine learning, convolutional deep neural network, bounding box, torch",
    # packages=setuptools.find_packages(exclude=[]),
    packages=["lanms"],
    install_requires=[],
    include_package_data=True,
)
