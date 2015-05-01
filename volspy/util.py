
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

from collections import namedtuple

import numpy as np

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


def load_tiff(fname):
    """Load named file using TIFF reader, returning (data, metadata).

       Data is an ND-array.  Metadata is an ImageMetadata named tuple.

       Raises NotImplementedError if TIFF reader is not available.
       Raises ValueError if file cannot be loaded by TIFF reader.
    """
    try:
        import tifffile
        from xml.dom import minidom
    except Exception, te:
        raise NotImplementedError(str(te))

    tf = tifffile.TiffFile(fname, multifile=False)
    data = None
    md = None

    def canonicalize(data, axes, flipY=True):
        """Restructure to our preferred CZYX format."""
        print data.shape, axes, data.dtype, data.strides, data.nbytes

        pos = axes.find('T')
        if pos >= 0:
            # remove useless time axis
            if data.shape[pos] == 1:
                data = data[
                    tuple(
                        [ slice(None) for i in range(pos) ]
                        + [ 0 ]
                        + [ slice(None) for i in range(pos+1, data.ndim) ]
                    )
                ]
                axes = axes[0:pos] + axes[pos+1:]
            else:
                raise NotImplementedError('Unexpected TIFF with T axis length %d' % data.shape[pos])

        pos = axes.find('C')
        if pos >= 1:
            # move color axis to first position
            data = data.transpose(
                *tuple(
                    [ pos ]
                    + [ i for i in range(data.ndim) if i != pos ]
                )
            )
            axes = axes[pos] + ''.join([
                axes[i] for i in range(data.ndim) if i != pos
            ])
        elif pos == 0:
            pass
        elif data.ndim == 3:
            data = data[None,...]
            axes = 'C' + axes
        else:
            raise NotImplementedError('Unexpected %d-dimension TIFF without C axis.' % data.ndim)

        # flip Y axis to match Fiji user expectation
        pos = axes.find('Y')
        if pos >= 0:
            if flipY:
                data = data[
                    tuple(
                        [ slice(None) for i in range(pos) ]
                        + [ slice(None, None, -1) ]
                        + [ slice(None) for i in range(pos+1, data.ndim) ]
                    )
                ]
        else:
            raise NotImplementedError('Unexpected TIFF without Y axis.')

        print data.shape, axes, data.dtype, data.strides, data.nbytes
        return data, axes

    if tf.is_ome:
        # get OME-TIFF XML metadata
        p = list(tf)[0]
        d = minidom.parseString(p.tags['image_description'].value)
        a = dict(d.getElementsByTagName('Pixels')[0].attributes.items())
        p = None
        d = None

        assert len(tf.series) == 1
        data = tf.asarray()
        axes = tf.series[0]['axes']
        tf.close()
        tf = None

        data, axes = canonicalize(data, axes)

        md = ImageMetadata(
            float(a['PhysicalSizeX']),
            float(a['PhysicalSizeY']),
            float(a['PhysicalSizeZ']),
            axes
        )

    elif tf.is_lsm:
        # get LSM metadata
        lsmi = None
        for page in tf:
            if page.is_lsm:
                lsmi = page.cz_lsm_info

        assert lsmi is not None

        data = tf.asarray()
        axes = tf.series[0]['axes']

        data, axes = canonicalize(data, axes)

        md = ImageMetadata(
            lsmi.voxel_size_x * 10**6,
            lsmi.voxel_size_y * 10**6,
            lsmi.voxel_size_z * 10**6,
            axes
            )

    else:
        # plain old tiff?
        data = tf.asarray()
        axes = tf.series[0]['axes']
        md = None
        print axes, data.shape
        if data.ndim == 4 and data.shape[3] in range(1, 5):
            data = data.transpose(3,0,1,2)
        elif data.ndim == 3:
            data = data[None,:,:,:]

    assert data is not None
    return data, md

def load_nii(fname):
    """Load named file using NiFTi reader, returning (data, metadata).

       Data is an ND-array.  Metadata is an ImageMetadata named tuple.

       Raises NotImplementedError if NiFTi reader is not available.
       Raises ValueError if file cannot be loaded by NiFTi reader.
    """
    
    try:
        import nibabel
    except:
        raise te

    img = nibabel.load(fname)
    data = img.get_data()
    grid_spacing_mm = img.header.get_zooms()
            
    assert data.ndim == 3
    data = data[None,...] # add single-channel axis
    assert len(grid_spacing_mm) == 3

    md = ImageMetadata(
        grid_spacing_mm[0] * 1000,
        grid_spacing_mm[1] * 1000,
        grid_spacing_mm[2] * 1000,
        None
    )

    print "Loaded %s %s" % (data.shape, data.dtype)
    return data, md

def load_image(fname):
    """Load named file, returning (data, metadata).

       Data is an ND-array.  Metadata is an ImageMetadata named tuple.

       Raises ValueError if file cannot be loaded by any available reader.
    """
    for f in load_tiff, load_nii:
        try:
            return f(fname)
        except ValueError, te:
            continue
        except NotImplementedError, te:
            print "Notice: %s" % te
            continue

    raise ValueError('No data loader worked for %s' % fname)

