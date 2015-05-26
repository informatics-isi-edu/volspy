
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from collections import namedtuple

import numpy as np
import tifffile
from tifffile import lazyattr
from xml.dom import minidom

ImageMetadata = namedtuple('ImageMetadata', ['x_microns', 'y_microns', 'z_microns', 'axes'])


def plane_distance(p, plane):
    """Return signed distance to plane of point."""
    x, y, z = p
    A, B, C, D = plane
    return A*x + B*y + C*z + D


def bin_reduce(data, axes_s):
    """Reduce ndarray data via bin-averaging for specified per-axis bin sizes.

       For a 3-D input data with shape (D, H, W) and axes_s of [s1, s2,
       s3], the output value result[0, 0, 0] will contain the average
       of all input values in slice data[0:s1, 0:s2, 0:s3] and the result
       will have shape (D/s1, H/s2, W/s3).

       The implementation is optimized for large ndarray data with
       relatively small bin sizes.  It makes a number of Python
       operations proportional to the product of per-axis bin sizes,
       but minimizes the elementwise data access and arithmetic.

       This function converts input data to float32 for averaging and
       it is the caller's responsibility to convert back to a desired
       type.

    """
    d1 = data

    # sort axes by stride distance to optimize for locality
    # doesn't seem to make much difference on modern systems...
    axes = [ (axis, d1.strides[axis]) for axis in range(d1.ndim) ]
    axes.sort(key=lambda p: p[1])

    assert len(axes_s) == data.ndim

    # reduce one axis at a time to shrink work for subsequent axes
    for axis in map(lambda p: p[0], axes):
        s = axes_s[axis]

        if s == 1:
            # skip useless copying for non-reducing axis
            continue

        # accumulate s-strided subsets that belong to each bin
        a = d1[ 
            tuple(
                [ slice(None) for i in range(axis) ]
                + [ slice(0, 1-s, s) ]
                + [ slice(None) for i in range(d1.ndim - axis - 1) ]
            )
        ].astype(np.float32, copy=False)
                
        for step in range(1, s):
            a += d1[ 
                tuple(
                    [ slice(None) for i in range(axis) ]
                    + [ slice(step, step < s and 1-s+step or None, s) ]
                    + [ slice(None) for i in range(d1.ndim - axis - 1) ]
                )
            ]

        # compute single-axis bin averages from accumulation
        d1 = a * (1./s)

    return d1

class TiffLazyNDArray (object):
    """Lazy wrapper for large TIFF image stacks.

       Supports some basic ND-array compatibility for data access,
       with an intended use case of sub-block decomposition of large
       TIFF stacks, where it is not desirable to hold the entire image
       array in RAM at one time.

       Slicing via the usual __getitem__ interface will perform
       memmapped file I/O to build and return an actual numpy
       ND-array.

       Basic min/max methods will stream through the whole image file
       while only buffering one page of image data at a time.

    """

    def __init__(self, src, transposition=None):
        """Wrap an image source given by filename or an existing tifffile.TiffFile instance."""
        if type(src) in [str, unicode]:
            self.tf = tifffile.TiffFile(src)
        elif isinstance(src, tifffile.TiffFile):
            self.tf = src
        elif isinstance(src, TiffLazyNDArray):
            self.tf = src.tf

        tfimg = self.tf.series[0]
        page0 = tfimg.pages[0]

        self.dtype = tfimg.dtype
        self.tf_shape = tfimg.shape
        self.ndim = len(self.tf_shape)
        self.tf_axes = tfimg.axes
        
        if transposition:
            assert isinstance(transposition, tuple)
            for a in transposition:
                assert isinstance(a, int), "transposition axes must be integer indexes"
                if a < 0 or a >= self.ndim:
                    raise IndexError("transposition axis %d out of range [0,%d)" % self.ndim)
            assert len(set(transposition)) == len(transposition), "each transposed axis can be referenced at most once"
            assert len(transposition) == self.ndim, "number of transposed dimensions must match original"
            self.transposition = transposition
        else:
            self.transposition = tuple(range(self.ndim))
            
        self.page_shape = page0.shape
        self.page_ndim = len(self.page_shape)

        self.stack_ndim = self.ndim - self.page_ndim
        self.stack_shape = tfimg.shape[0:self.stack_ndim]

        assert reduce(lambda a,b: a*b, self.stack_shape, 1) == len(tfimg.pages), "TIFF page count mismatch to stack shape"
        assert tfimg.shape[self.stack_ndim:] == page0.shape, "TIFF page packing structure not understood"

        if self.tf.is_ome:
            # get OME-TIFF XML metadata
            p = list(self.tf)[0]
            d = minidom.parseString(p.tags['image_description'].value)
            a = dict(d.getElementsByTagName('Pixels')[0].attributes.items())
            p = None
            d = None
            assert len(self.tf.series) == 1

            self.micron_spacing = (
                float(a['PhysicalSizeZ']),
                float(a['PhysicalSizeY']),
                float(a['PhysicalSizeX'])
            )

        elif self.tf.is_lsm:
            # get LSM metadata
            lsmi = None
            for page in self.tf:
                if page.is_lsm:
                    lsmi = page.cz_lsm_info

            assert lsmi is not None

            self.micron_spacing = (
                lsmi.voxel_size_z * 10**6,
                lsmi.voxel_size_y * 10**6,
                lsmi.voxel_size_x * 10**6
            )

    def __getitem__(self, key):
        assert isinstance(key, tuple)
        assert len(key) >= self.ndim

        tfimg = self.tf.series[0]
        # determine slicing plan
        output_plan = [] # [(tf_axis, size, in_slice, out_slice)...]
        i = 0
        for elem in key:
            if elem is None:
                # inject fake dimension
                tf_axis = None
                buf_axis = None
                size = 1
                out_slice = slice(0,1)
                in_slice = None
            else:
                tf_axis = self.transposition[i]
                tf_size = self.tf_shape[tf_axis]
                i += 1
                if isinstance(elem, int):
                    # collapse dimension
                    size = None
                    out_slice = None
                    assert elem >= 0, "negative indexing not supported"
                    if elem >= tf_size:
                        raise IndexError()
                    in_slice = elem
                elif isinstance(elem, slice):
                    assert elem.step is None, "only default stepping is supported"
                    # sliced dimension
                    start = elem.start or 0
                    if elem.stop is None:
                        stop = tf_size
                    else:
                        stop = elem.stop
                    assert start >= 0, "negative indexing not supported"
                    assert stop >= 0, "negative indexing not supported"
                    assert start < stop, "empty slicing not supported"
                    size = stop - start
                    out_slice = slice(0, size)
                    in_slice = slice(start, stop)
            output_plan.append((tf_axis, size, in_slice, out_slice))

        # skip fake dimensions for intermediate buffer
        buffer_plan = [p for p in output_plan if p[0] is not None]

        # input will be untransposed with dimension in TIFF order
        input_plan = list(buffer_plan)
        input_plan.sort(key=lambda p: p[0])
        assert len(input_plan) == self.ndim

        # buffer may have fewer dimensions than input slicing due to integer keys
        buffer_shape = tuple(p[1] for p in input_plan if p[1] is not None)
        buffer = np.empty(buffer_shape, self.dtype)

        # generate page-by-page slicing
        stack_plan = input_plan[0:self.stack_ndim]
        page_plan = input_plan[self.stack_ndim:]

        def generate_io_slices(stack_plan, page_plan):
            if stack_plan:
                tf_axis, size, in_slice, out_slice = stack_plan[0]
                if isinstance(in_slice, slice):
                    for x in range(in_slice.start, in_slice.stop):
                        for outslc, inslc in generate_io_slices(stack_plan[1:], page_plan):
                            yield ((x - in_slice.start,) + outslc, (x,) + inslc)
                elif isinstance(in_slice, int):
                    for outslc, inslc in generate_io_slices(stack_plan[1:], page_plan):
                        yield (outslc, (in_slice,) + inslc)
                else:
                    assert False
            else:
                yield tuple(p[3] for p in page_plan), tuple(p[2] for p in page_plan)
                
        stack_spans = [
            reduce(lambda a,b: a*b, self.stack_shape[i+1:], 1)
            for i in range(self.stack_ndim)
        ]

        # perform actual pixel I/O
        for out_slicing, in_slicing in generate_io_slices(stack_plan, page_plan):
            page = sum(map(lambda c, s: c*s, in_slicing[0:self.stack_ndim], stack_spans))
            page_slice = in_slicing[self.stack_ndim:]
            buffer[out_slicing] = tfimg.pages[page].asarray(memmap=True)[page_slice]
            
        # apply current transposition to buffered dimensions
        i = 0
        buffer_axes = dict()
        for tf_axis, size, in_slice, out_slice in input_plan:
            if size is not None:
                buf_axis = i
                i += 1
            else:
                buf_axis = None
            buffer_axes[tf_axis] = buf_axis
        buffer_transposition = tuple(buffer_axes[a] for a in self.transposition if buffer_axes[a] is not None)
        buffer = buffer.transpose(buffer_transposition)
        
        # inject fake dimensions before returning
        out_slicing = []
        for tf_axis, size, in_slice, out_slice in output_plan:
            if tf_axis is None:
                out_slicing.append(None)
            elif size is not None:
                out_slicing.append(slice(None))
        return buffer[tuple(out_slicing)]
        
    def transpose(self, *transposition):
        for a in transposition:
            assert isinstance(a, int), "transposition axes must be integer indexes"
            if a < 0 or a >= self.ndim:
                raise IndexError("transposition axis %d out of range [0,%d)" % (a, self.ndim))
        transposition = map(lambda a: self.transposition[a], transposition)
        return TiffLazyNDArray(self, tuple(transposition))

    @property
    def shape(self):
        return tuple(self.tf_shape[a] for a in self.transposition)

    @property
    def axes(self):
        return ''.join(self.tf_axes[a] for a in self.transposition)

    @lazyattr
    def min_max(self):
        amin = None
        amax = None
        tfimg = self.tf.series[0]
        for tfpage in tfimg.pages:
            p = tfpage.asarray(memmap=True)
            pmin = float(p.min())
            pmax = float(p.max())
            if amin is not None:
                amin = min(amin, pmin)
            else:
                amin = pmin
            if amax is not None:
                amax = max(amax, pmax)
            else:
                amax = pmax
        return (amin, amax)
                
    def max(self):
        return self.min_max[1]
            
    def min(self):
        return self.min_max[0]



def load_tiff(fname):
    """Load named file using TIFF reader, returning (data, metadata).

       Keep temporarily for backward-compatibility...
    """
    data = TiffLazyNDArray(fname)
    z_microns, y_microns, x_microns = data.micron_spacing
    md = ImageMetadata(x_microns, y_microns, z_microns, data.axes)
    # configure our preferred axes ordering
    data = data.transpose(*[d for d in map(data.axes.find, 'TCZYX') if d >= 0])
    
    projection = []

    if 'T' in data.axes and data.shape[0] == 1:
        projection.append(0) # remove trivial T dimension

    if 'C' not in data.axes:
        projection.append(None) # add trivial C dimension
    elif projection:
        projection.append(slice(None))

    if projection:
        projection += [slice(None) for d in 'ZYX']
        data = data[tuple(projection)]
        
    return data, md

def load_image(fname):
    """Load named file, returning (data, metadata).

       Keep temporarily for backward-compatibility...
    """
    return load_tiff(fname)
