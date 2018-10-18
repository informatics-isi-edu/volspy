
#
# Copyright 2014-2017 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from collections import namedtuple
import os
import numpy as np
import tifffile
from tifffile import lazyattr
from xml.dom import minidom
from functools import reduce

ImageMetadata = namedtuple('ImageMetadata', ['x_microns', 'y_microns', 'z_microns', 'axes'])


def plane_distance(p, plane):
    """Return signed distance to plane of point."""
    x, y, z = p
    A, B, C, D = plane
    return A*x + B*y + C*z + D

def clamp(x, x_min, x_max):
    return max(x_min, min(x, x_max))

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
    for axis in [p[0] for p in axes]:
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
        ].astype(np.float32, copy=True)
        
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

    def __init__(self, src, _output_plan=None):
        """Wrap an image source given by filename or an existing tifffile.TiffFile instance."""
        if isinstance(src, str):
            self.tf = tifffile.TiffFile(src)
        elif isinstance(src, tifffile.TiffFile):
            self.tf = src
        elif isinstance(src, TiffLazyNDArray):
            self.tf = src.tf

        tfimg = self.tf.series[0]
        page0 = tfimg.pages[0]

        self.dtype = tfimg.dtype
        self.tf_shape = tfimg.shape
        self.tf_axes = tfimg.axes
        
        self.stack_ndim = len(tfimg.shape) - len(page0.shape)
        self.stack_shape = tfimg.shape[0:self.stack_ndim]
        print("TIFF %s %s %s, page0 %s, stack %s, axes %s?" % (tfimg.shape, tfimg.axes, tfimg.dtype, page0.shape, self.stack_shape, tfimg.axes))
        assert reduce(lambda a,b: a*b, self.stack_shape, 1) == len(tfimg.pages), "TIFF page count %s inconsistent with expected stack shape %s" % (len(tfimg.pages), self.stack_shape)
        assert tfimg.shape[self.stack_ndim:] == page0.shape, "TIFF page packing structure not understood"

        if _output_plan:
            self.output_plan = _output_plan
        else:
            self.output_plan = [
                (a, slice(0, self.tf_shape[a], 1), slice(0, self.tf_shape[a], 1))
                for a in range(len(tfimg.shape))
            ]

        if isinstance(src, TiffLazyNDArray):
            # preserve existing metadata
            self.micron_spacing = src.micron_spacing
        elif self.tf.is_ome:
            # get OME-TIFF XML metadata
            p = self.tf.pages[0]
            try:
                s = p.tags['ImageDescription'].value
            except KeyError:
                # older behavior of tifffile
                s = p.tags['image_description']
            d = minidom.parseString(s)
            a = dict(list(d.getElementsByTagName('Pixels')[0].attributes.items()))
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
            for page in self.tf.pages:
                if page.is_lsm:
                    lsmi = page.cz_lsm_info

            assert lsmi is not None

            self.micron_spacing = (
                lsmi.voxel_size_z * 10**6,
                lsmi.voxel_size_y * 10**6,
                lsmi.voxel_size_x * 10**6
            )

    def _plan_slicing(self, key):
        assert isinstance(key, tuple)
        tfimg = self.tf.series[0]
        output_plan = [
            (tf_axis, in_slice, out_slice)
            for tf_axis, in_slice, out_slice in self.output_plan
            if out_slice is None and in_slice is not None
        ]

        current_plan = [ # FIFO of dimensions projected by key
            (tf_axis, in_slice, out_slice)
            for tf_axis, in_slice, out_slice in self.output_plan
            if out_slice is not None
        ]
        
        for elem in key:
            if elem is None:
                # inject fake output dimension
                tf_axis = None
                out_slice = slice(0,1,1)
                in_slice = None
            else:
                tf_axis, in_slice, out_slice = current_plan.pop(0)
                if isinstance(elem, int):
                    # collapse projected dimension
                    if elem < 0:
                        elem += out_slice.stop
                    if elem >= out_slice.stop or elem < 0:
                        raise IndexError('index %d out of range [0,%d)' % (elem, out_slice.stop))
                    if isinstance(in_slice, slice):
                        in_slice = elem + in_slice.start
                    else:
                        continue
                    out_slice = None
                elif isinstance(elem, slice):
                    # modify sliced dimension
                    if elem.step is None:
                        step = 1
                    else:
                        step = elem.step
                    assert step > 0, "only positive stepping is supported"
                    if elem.start is None:
                        start = 0
                    elif elem.start < 0:
                        start = elem.start + out_slice.stop
                    else:
                        start = elem.start
                    if elem.stop is None:
                        stop = out_slice.stop
                    elif elem.stop < 0:
                        stop = elem.stop + out_slice.stop
                    else:
                        stop = elem.stop
                    start = max(min(start, out_slice.stop), 0)
                    stop = max(min(stop, out_slice.stop), 0)
                    assert start < stop, "empty slicing not supported"
                    if isinstance(in_slice, slice):
                        in_slice = slice(in_slice.start + start, in_slice.start + stop, in_slice.step * step)
                        w = in_slice.stop - in_slice.start
                        w = w//in_slice.step + (w%in_slice.step and 1 or 0)
                        out_slice = slice(0,w,1)
                    else:
                        in_slice = None
                        out_slice = slice(0,1,1)
            output_plan.append((tf_axis, in_slice, out_slice))

        assert not current_plan, "slicing key must project all image dimensions"
            
        return output_plan
            
    def __getitem__(self, key):
        tfimg = self.tf.series[0]
        output_plan = self._plan_slicing(key)
        
        # skip fake dimensions for intermediate buffer
        buffer_plan = [
            (tf_axis, in_slice, out_slice)
            for tf_axis, in_slice, out_slice in output_plan
            if in_slice is not None
        ]

        # input will be untransposed with dimension in TIFF order
        input_plan = list(buffer_plan)
        input_plan.sort(key=lambda p: p[0])
        assert len(input_plan) == len(tfimg.shape)
        
        # buffer may have fewer dimensions than input slicing due to integer keys
        buffer_shape = tuple([
            out_slice.stop
            for tf_axis, in_slice, out_slice in input_plan
            if isinstance(in_slice, slice)
        ])
        buffer_axes = [
            tf_axis
            for tf_axis, in_slice, out_slice in input_plan
            if isinstance(in_slice, slice)
        ]
        buffer = np.empty(buffer_shape, self.dtype)

        # generate page-by-page slicing
        stack_plan = input_plan[0:self.stack_ndim]
        page_plan = input_plan[self.stack_ndim:]
        
        def generate_io_slices(stack_plan, page_plan):
            if stack_plan:
                tf_axis, in_slice, out_slice = stack_plan[0]
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
                yield tuple(p[2] for p in page_plan if p[2] is not None), tuple(p[1] for p in page_plan)
                
        stack_spans = [
            reduce(lambda a,b: a*b, self.stack_shape[i+1:], 1)
            for i in range(self.stack_ndim)
        ]

        # perform actual pixel I/O
        for out_slicing, in_slicing in generate_io_slices(stack_plan, page_plan):
            page = sum(map(lambda c, s: c*s, in_slicing[0:self.stack_ndim], stack_spans))
            page_slice = in_slicing[self.stack_ndim:]
            try:
                buffer[out_slicing] = tfimg.pages[page].asarray(out='memmap')[page_slice]
            except TypeError as e:
                # try older tifffile memmap interface
                buffer[out_slicing] = tfimg.pages[page].asarray(memmap=True)[page_slice]
            
        # apply current transposition to buffered dimensions
        buffer_axis = dict([(buffer_axes[d], d) for d in range(len(buffer_axes))])
        transposition = [
            buffer_axis[tf_axis]
            for tf_axis, in_slice, out_slice in output_plan
            if isinstance(in_slice, slice)
        ]
        buffer = buffer.transpose(tuple(transposition))
        
        out_slicing = [
            in_slice is not None and out_slice or in_slice
            for tf_axis, in_slice, out_slice in output_plan
            if isinstance(in_slice, slice) or in_slice is None
        ]
        return buffer[tuple(out_slicing)]
        
    def transpose(self, *transposition):
        output_plan = [
            (tf_axis, in_slice, out_slice)
            for tf_axis, in_slice, out_slice in self.output_plan
            if out_slice is None
        ]
        current_plan = [ # FIFO of dimensions projected by key
            (tf_axis, in_slice, out_slice)
            for tf_axis, in_slice, out_slice in self.output_plan
            if out_slice is not None
        ]

        for d in transposition:
            assert current_plan[d] is not None, "transpose cannot repeat same dimension"
            p = current_plan[d]
            current_plan[d] = None
            output_plan.append(p)

        assert len([p for p in current_plan if p is not None]) == 0, "transpose must include dimensions"
        return TiffLazyNDArray(self, output_plan)

    def lazyget(self, key):
        output_plan = self._plan_slicing(key)
        return TiffLazyNDArray(self, output_plan)

    def force(self):
        return self[tuple(slice(None) for d in self.shape)]
    
    @property
    def ndim(self):
        return len([p for p in self.output_plan if p[2] is not None])
    
    @property
    def shape(self):
        return tuple(p[2].stop for p in self.output_plan if p[2] is not None)

    @property
    def axes(self):
        return ''.join(p[0] is not None and self.tf_axes[p[0]] or 'Q' for p in self.output_plan if p[2] is not None)

    @property
    def strides(self):
        plan = [(p[0], p[2].stop) for p in self.output_plan if p[2] is not None]
        plan = [(i,) + plan[i] for i in range(len(plan))]
        plan.sort(key=lambda p: p[1])
        strides = []
        for i in range(len(plan)):
            strides.append((plan[i][0], reduce(lambda a, b: a*b, [p[2] for p in plan[i+1:]], 1)))
        strides.sort(key=lambda p: p[0])
        strides = [p[1] for p in strides]
        return strides
        
    @lazyattr
    def min_max(self):
        amin = None
        amax = None
        tfimg = self.tf.series[0]
        for tfpage in tfimg.pages:
            try:
                p = tfpage.asarray(out='memmap')
            except TypeError as e:
                # try older tifffile api
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

def canonicalize(data):
    """Restructure to preferred TCZYX or CZYX form..."""
    data = data.transpose(*[d for d in map(data.axes.find, 'TCIZYX') if d >= 0])
    projection = []

    if 'T' in data.axes and data.shape[0] == 1:
        projection.append(0) # remove trivial T dimension

    if 'C' not in data.axes:
        projection.append(None) # add trivial C dimension
    elif projection:
        projection.append(slice(None))

    if projection:
        projection += [slice(None) for d in 'ZYX']
        data = data.lazyget(tuple(projection))
        
    return data

def load_tiff(fname):
    """Load named file using TIFF reader, returning (data, metadata).

       Keep temporarily for backward-compatibility...
    """
    data = TiffLazyNDArray(fname)
    try:
        data = canonicalize(data)
    except Exception as e:
        print(e)
        # special case for raw TIFF (not LSM, not OME)
        if data.ndim == 3:
            data = data[(None,slice(None),slice(None),slice(None))] # add fake color dimension
        elif data.ndim == 4 and data.shape[3] < 4:
            data = data.transpose(3,0,1,2) # transpose color

    try:
        z_microns, y_microns, x_microns = data.micron_spacing
        md = ImageMetadata(x_microns, y_microns, z_microns, data.axes)
    except AttributeError as e:
        print('got error %s fetching metadata during load_tiff' % e)
        md = None
    return data, md

def load_image(fname):
    """Load named file, returning (data, metadata).

       Keep temporarily for backward-compatibility...
    """
    return load_tiff(fname)

class wrapper (np.ndarray):
    """Subtype to allow extra attributes"""
    pass

def load_and_mangle_image(fname):
    """Load and mangle TIFF image file.

       Arguments:
         fname: LSM or OME-TIFF input file name

       Environment parameters:
         ZYX_SLICE: selects ROI within full image
         ZYX_IMAGE_GRID: overrides image grid step metadata
         ZNOISE_PERCENTILE: see source
         ZNOISE_ZERO_LEVEL: see source

       Results tuple fields:
         image
         meta
         slice_origin
    """
    I, meta = load_image(fname)

    try:
        voxel_size = tuple(map(float, os.getenv('ZYX_IMAGE_GRID').split(",")))
        print("ZYX_IMAGE_GRID environment forces image grid of %s micron." % (voxel_size,))
        assert len(voxel_size) == 3
    except:
        try:
            voxel_size = I.micron_spacing
            print("Using detected %s micron image grid." % (voxel_size,))
        except AttributeError:
            print("ERROR: could not determine image grid spacing. Use ZYX_IMAGE_GRID=Z,Y,X to override.")
            raise

    meta = ImageMetadata(voxel_size[2], voxel_size[1], voxel_size[0], I.axes)
    setattr(I, 'micron_spacing', voxel_size)

    # temporary pre-processing hacks to investigate XY-correlated sensor artifacts...
    try:
        ntile = int(os.getenv('ZNOISE_PERCENTILE'))
        I = I.force().astype(np.float32)
        zerofield = np.percentile(I, ntile, axis=1)
        print('Image %d percentile value over Z-axis ranges [%f,%f]' % (ntile, zerofield.min(), zerofield.max()))
        I -= zerofield
        print('Image offset by %d percentile XY value to new range [%f,%f]' % (ntile, I.min(), I.max()))
        zero = float(os.getenv('ZNOISE_ZERO_LEVEL', 0))
        I = I * (I >= 0.)
        print('Image clamped to range [%f,%f]' % (I.min(), I.max()))
    except:
        pass

    I = I.transpose(1,2,3,0)

    # allow user to select a bounding box region of interest
    bbox = os.getenv('ZYX_SLICE')
    slice_origin = (0, 0, 0)
    if bbox:
        bbox = bbox.split(",")
        assert len(bbox) == 3, "ZYX_SLICE must have comma-separated slices for 3 axes Z,Y,X"

        def parse_axis(slc_s, axis_len):
            bounds = slc_s.split(":")
            assert len(bounds) == 2, "ZYX_SLICE must have colon-separated START:STOP pairs for each axis"

            if bounds[0] != '':
                assert int(bounds[0]) >= 0, "ZYX_SLICE START values must be 0 or greater or empty string"
                assert int(bounds[0]) < (axis_len-2), "ZYX_SLICE START values must be less than axis length - 2"
                bounds[0] = int(bounds[0])
            else:
                bounds[0] = 0

            if bounds[1] != '':
                assert int(bounds[1]) >= bounds[0], "ZYX_SLICE STOP values must be greater than START or empty string"
                bounds[1] = int(bounds[1])
            else:
                bounds[1] = axis_len

            return slice(bounds[0], bounds[1])

        bbox = tuple([
            parse_axis(bbox[d], I.shape[d])
            for d in range(3)
        ]) + (slice(None),)
        I = I.lazyget(bbox)
        slice_origin = tuple([
            slc.start or 0
            for slc in bbox[0:3]
        ])

    if I.shape[2] % 16:
        # trim for 16-pixel row alignment
        slc = tuple([
            slice(None),
            slice(None),
            slice(0,I.shape[2]//16*16),
            slice(None)
        ])
        if hasattr(I, 'lazyget'):
            I = I.lazyget(slc)
        else:
            I = I[slc]

    if isinstance(I, np.ndarray):
        # temporarily maintain micron_spacing after munging above...
        I2 = wrapper(shape=I.shape, dtype=I.dtype)
        I2[:,:,:,:] = I[:,:,:,:]
        I = I2
        setattr(I, 'micron_spacing', voxel_size)

    return I, meta, slice_origin
