
#
# Copyright 2015, 2021 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import re
import io
from setuptools import setup

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('volspy/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

setup(
    name="volspy",
    description="volumetric image visualization using vispy",
    version=__version__,
    packages=["volspy"],
    scripts=["bin/volspy-viewer"],
    requires=["vispy", "numpy", "tifffile"],
    maintainer_email="support@misd.isi.edu",
    license='(new) BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ])
