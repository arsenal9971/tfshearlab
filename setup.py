"""Setup script for adler.
Installation command::
    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages

setup(
    name='arsenal9971',

    version='0.0.0',

    description='tfShearlab',

    url='https://github.com/arsenal9971/tfshearlab',

    author='Hector Andrade Loarca',
    author_email='andrade@math.tu-berlin.de',

    license='GPLv3+',

    packages=find_packages(exclude=['*test*']),
    package_dir={'tfshearlab': 'tfshearlab'},

    install_requires=['numpy', 'demandimport']
)
