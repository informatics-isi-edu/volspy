
#
# Copyright 2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from distutils.core import setup

setup(
    name="volspy",
    description="volumetric image visualization using vispy",
    version="0.1-prerelease",
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
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ])
