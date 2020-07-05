"""Packaging settings"""

from pathlib import Path
from setuptools import setup, find_packages

from proflogit import (
    __author__,
    __email__,
    __version__,
)


base_directory = Path(__file__).parent.resolve()
long_desc = base_directory.joinpath("README.md").read_text()
required_packages = [
    lib
    for lib in base_directory.joinpath("requirements.txt").read_text().split("\n")
    if lib
]


setup(
    name="proflogit",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="Profit maximizing classifier for predictive churn modeling.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/estripling/proflogit",
    python_requires=">= 3.6",
    packages=find_packages(),
    install_requires=required_packages,
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
