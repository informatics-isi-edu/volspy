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
  common 3D image formats based on TIFF.

### Prerequisites

Volspy is developed primarily on Linux with Python 2.7 but also tested
on Mac OSX. It has several requirements:

- [Vispy](http://vispy.org) visualization library to access OpenGL GPUs.  A compatible development version is needed, such as [karlcz/vispy](https://github.com/karlcz/vispy) which is a clean branch of the upstream [vispy/vispy](https://github.com/vispy/vispy) master.
- [Numpy](http://www.numpy.org) numerical library to process
  N-dimensional data.
- [Tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) for access to OME-TIFF and LSM microscopy file formats as well as basic 3D TIFF files containing a stack of 2D pages.

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
  - Click and drag volume with secondary mouse button to translate (pan).
  - Scroll the vertical scroll wheel to move a clipping or slicing plane up and down the viewing axis.
  - Press `SPACE` key to enter or exit slicing mode:
    - Entry to slicing mode repositions slice plane to intersect origin. Use shift modifier to retain current clip distance.
    - Entry to clipping mode repositions clip plane to near clipping distance. Use shift modifier to retain current slice distance.
  - Press number keys `1` to `9` to change intensity gain and with shift modifier to get reciprocal gain.
  - Press keys `f` and `F` to adjust the floor-level image intensity that is mapped to black.
  - Press `b` key to cycle through color blending modes:
    - Partial transparency
    - Additive blend
    - Maximum intensity projection
  - Press `c` key to cycle through channels on images with more than 4 channels.

Do not be alarmed by the copious diagnostic outputs streaming out on
the console. Did we mention this is experimental code?

### Environment Parameters

Several environment variables can be set to modify the behavior of the `volspy-viewer` tool on a run-by-run basis:

- `VIEW_CHANNEL` specifies an integer channel number in range 0 to N-1 inclusive for N channel images, switching the viewer into single-channel mode and with the specified channel loaded initially. The `c` key can then be used to cycle through channels if desired. This mode is entered automatically for images with more than 4 channels.
- `VOXEL_SAMPLE` selects volume rendering texture sampling modes from `nearest` or `linear` (default for unspecified or unrecognized values).
- `ZYX_SLICE` selects a grid-aligned region of interest to view from the original image grid, e.g. `0:10,100:200,50:800` selects a region of interest where Z<10, 100<=Y<200, and 50<=X<800. (Default slice contains the whole image.)
- `ZYX_VIEW_GRID` changes the desired rendering grid spacing. Set a preferred ZYX micron spacing, e.g. `0.5,0.5,0.5` which the program will try to approximate using integer bin-averaging of source voxels but it will only reduce grid resolution and never increase it. NOTE: Y and X values should be equal to avoid artifacts with current renderer. (Default grid is 0.25, 0.25, 0.25 micron.)
- `ZNOISE_PERCENTILE` enables a sensor noise estimation by calculating the Nth percentile value along the Z axis, e.g. `ZNOISE_PERCENTILE=5` estimates a 2D noise image as the 5th percentile value across the Z stack, and subtracts that noise image from every slice in the stack as a pre-filtering step. *WARNING*: use of this feature causes the entire image to be loaded into RAM, causing a significantly higher minimum RAM size for runs with large input images. (Default is no noise estimate.)
  - `ZNOISE_ZERO_LEVEL` controls a lower value clamp for the pre-filtered data when percentile filtering is enabled. (Default is `0`.)

The `ZYX_SLICE` and `ZYX_VIEW_GRID` parameters have different but inter-related effects on the scope of the volumetric visualization.

1. The `ZYX_VIEW_GRID` can control down-sampling of voxels in arbitrary integer ratios, e.g. to set a preferred grid resolution that can differentiate features of a given size without wasting additional storage space on irrelevant small-scale details. This can save overall RAM required to store the processed volume data by reducing the global image size. The down-sampling occurs incrementally as each sub-block is processed by the block-decomposed processing pipeline.
1. The `ZYX_SLICE` can arbitrarily discard voxels and thus reduce the final volume size, though discarded voxels may be temporarily present in RAM and require additional I/O bandwidth.

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

