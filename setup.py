#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
import os
from setuptools import setup, find_packages


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name="multiresolutionfit",
    license="Apache License 2.0",
    version='1.0.0',
    author='Eduardo S. Pereira',
    author_email='pereira.somoza@gmail.com',
    packages=find_packages("src"),
    package_dir={"":"src"},
    description="Multi Resolution Fit.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/duducosmos/multiresolutionfit",
    include_package_data=True,
    install_requires=["numpy",
                      "numba",
                      "opencv-python",
                      "progressbar2"
                      ]
)
