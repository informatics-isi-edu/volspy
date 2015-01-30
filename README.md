# Volspy: Volumetric image visualization using vispy

[Volspy](http://github.com/informatics-isi-edu/volspy) is an
interactive volume visualization tool. Volspy is being developed to
support research involving 3D fluorescence microscopy imaging, but may
be applicable to other 3D images as well.


## Status

Volspy is experimental software that is subject to frequent changes in
direction depending on the needs of the authors and their scientific
collaborators.

## Using Volspy

Volspy has two usage scenarios:

1. A framework for volumetric data processing tools, where a basic
  interactive volume rendering capability can complement custom
  data-processing tools.
2. A standalone viewer application for quickly inspecting several
  common 3D image formats.

### Prerequisites

Volspy is developed primarily on Linux with Python 2.7 but also tested
on Mac OSX. It has several requirements:

- [Vispy](http://vispy.org) visualization library to access OpenGL
  GPUs.  A recent development version is needed, including
  high-precision texture format features merged into the
  [vispy/master](https://github.com/vispy/vispy) branch on 2015-01-30.
- [Numpy](http://www.numpy.org) numerical library to process
  N-dimensional data.
- [Tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) for
  access to OME-TIFF and LSM microscopy file formats.
- [NiBabel](http://nipy.org/nibabel) for access to additional
  neuroimaging file formats such as NifTI.

The file-reading part of Volspy can tolerate a missing Tifffile or
NiBabel prerequisite if you do not need to read those types of files.

### Installation

1. Check out the development code from GitHub.
2. Install with `python setup.py install`.

## Help and Contact

Please direct questions and comments to the [project issue
tracker](https://github.com/informatics-isi-edu/volspy/issues) at
GitHub.

## License

Volspy is made available as open source under the (new) BSD
License. Please see [LICENSE
file](https://github.com/informatics-isi-edu/volspy/blob/master/LICENSE)
for more information.

## About Us

Volspy is developed in the [Informatics
group](http://www.isi.edu/research_groups/informatics/home) at the
[USC Information Sciences Institute](http://www.isi.edu).  The
computer science researchers involved are:

* Karl Czajkowski
* Carl Kesselman

