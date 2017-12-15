"""Packaging settings."""


from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

try:
    from setuptools import Command, find_packages, setup
except ImportError:
    from distutils.core import Command, setup, find_packages

from proflogit import __author__
from proflogit import __email__
from proflogit import __version__
from proflogit import __license__


this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.rst'), encoding='utf-8') as file:
    long_description = file.read()

required_packages = [i.strip() for i in open('requirements.txt').readlines()]


class RunTests(Command):
    """Run all tests."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(
            [
                'py.test',
                '--cov=proflogit',
                '--cov-report=term-missing',
            ]
        )
        raise SystemExit(errno)

setup(
    name='proflogit',
    version=__version__,
    description=(
        'Profit maximizing classifier for predictive churn modeling'
    ),
    long_description=long_description,
    author=__author__,
    author_email=__email__,
    license=__license__,
    classifiers=[
        'Natural Language :: English',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='churn profit maximizing logistic regression genetic algorithm',
    packages=find_packages(exclude=['docs', 'tests*']),
    install_requires=required_packages,
    extras_require={
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    cmdclass={'test': RunTests},
)
