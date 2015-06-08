# Volspy: Volumetric image visualization using vispy

[Volspy](http://github.com/informatics-isi-edu/volspy) is an
interactive volume visualization tool. Volspy is being developed to
support research involving 3D fluorescence microscopy imaging, but may
be applicable to other 3D images as well.

## Screenshots

<img src="http://www.isi.edu/~karlcz/sample-data/volspy-shot1.png" />

<img src="http://www.isi.edu/~karlcz/sample-data/volspy-shot2.png" />

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

### Viewing an Image

1. Obtain a sample 1-4 channel 3D TIFF image such as:
   http://www.isi.edu/~karlcz/sample-data/zebra-d19-03b-D.ome.tiff
   **Warning: this is a large 886 MB file!**
  - A 1-channel image will be interpreted as gray.
  - A 2-channel image will be interpreted as red-green.
  - A 3-channel image will be interpreted as red-green-blue.
  - A 4-channel image will be interpreted as red-green-blue-alpha.
  - Any N-channel image with more than 4 will be interpreted one channel at a time (as gray) with a command to step to the next channel.
2. Launch the viewer `volspy-viewer zebra-d19-03b-D.ome.tiff`
3. Interact with the viewer
  - Press the `ESC` key when you have had enough.
  - Press the `?` key to get UI help printed to console output.
  - Click and drag volume with primary mouse button to rotate.
  - Press arrow keys with shift modifer to induce continuous rotation.
  - Press number keys `1` to `9` to change intensity gain and with shift modifier to get reciprocal gain.
  - Press keys `f` and `F` to adjust the floor-level image intensity that is mapped to black.
  - Press `b` key to cycle through color blending modes:
    - Partial transparency
    - Additive blend
    - Maximum intensity projection
  - Press `c` key to cycle through channels on images with more than 4 channels.
  - Click and drag vertically with secondary mouse button to drag a
    slicing plane through the volume. The plane is perpindicular to
    the viewing axis and its depth is controlled with the vertical
    mouse position.
  - Click and drag vertically with the tertiary mouse button to drag a
    clipping plane through the volume.

Do not be alarmed by the copious diagnostic outputs streaming out on
the console. Did we mention this is experimental code?

### Environment Parameters

Several environment variables can be set to modify the behavior of the `volspy-viewer` tool on a run-by-run basis:

- `ZYX_SLICE` selects a grid-aligned region of interest to view from the original image grid, e.g. `0:10,100:200,50:800` selects a region of interest where Z<10, 100<=Y<200, and 50<=X<800. (Default slice contains the whole image.)
- `ZYX_VIEW_GRID` changes the desired rendering grid spacing. Set a preferred ZYX micron spacing, e.g. `0.5,0.5,0.5` which the program will try to approximate using integer bin-averaging of source voxels but it will only reduce grid resolution and never increase it. NOTE: Y and X values should be equal to avoid artifacts with current renderer. (Default grid is 0.25, 0.25, 0.25 micron.)
- `MAX_3D_TEXTURE_WIDTH` sets a limit to the per-dimension size of the volume cube loaded into an OpenGL texture. If the viewing grid is too large to fit, it will be bin-averaged by factors of 2 into a multi-resolution pyramid with limited pan/zoom control in the viewing application to load different subsets of data onto the GPU. (Default is `768`.)
- `ZNOISE_PERCENTILE` enables a sensor noise estimation by calculating the Nth percentile value along the Z axis, e.g. `ZNOISE_PERCENTILE_5` estimates a 2D noise image as the 5th percentile value across the Z stack, and subtracts that noise image from every slice in the stack as a pre-filtering step. *WARNING*: use of this feature causes the entire image to be loaded into RAM, causing a significantly higher minimum RAM size for runs with large input images. (Default is no noise estimate.)
  - `ZNOISE_ZERO_LEVEL` controls a lower value clamp for the pre-filtered data when percentile filtering is enabled. (Default is `0`.)

The `ZYX_SLICE`, `ZYX_VIEW_GRID`, and `MAX_3D_TEXTURE_WIDTH` parameters have different but inter-related effects on the scope of the volumetric visualization.

1. The `ZYX_VIEW_GRID` can control down-sampling of voxels in arbitrary integer ratios, e.g. to set a preferred grid resolution that can differentiate features of a given size without wasting additional storage space on irrelevant small-scale details. This can save overall RAM required to store the processed volume data by reducing the global image size. The down-sampling occurs incrementally as each sub-block is processed by the block-decomposed processing pipeline.
1. The `ZYX_SLICE` can arbitrarily discard voxels and thus reduce the final volume size, though discarded voxels may be temporarily present in RAM and require additional memory allocation at that time.
1. The `MAX_3D_TEXTURE_WIDTH` can avoid allocating oversized OpenGL 3D textures which would either cause a runtime error or unacceptable performance on a given hardware implementation. This can save overall texture RAM required to store the volume data on the GPU, but actually increases the host RAM requirements since it generates a multi-resolution pyramid on the host from which different 3D texture blocks are retrieved dynamically.

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

