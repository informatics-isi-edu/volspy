
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

"""Volume data support.

Volume data is loaded as an ND-array with axes ZYXC and shape
(D,H,W,C) using available volumetric image file readers.

The ImageCropper class encapsulates the state necessary to:

  1. load a volume image file

  2. produce a reduced-resolution pyramid in case the image is too
     large for the volume rendering environment (whether due to
     hardware, software, or user preference)

  3. prepare 3D texture data

     a. of an appropriate texture storage format

     b. at an appropriate resolution in the hierarchy

     c. potentially cropped

  4. prepare 3D bounding box geometry

To perform its functions, the ImageCropper requires some information
about the viewing conditions, such as zoom-level and optional data
origin offset to allow panning within a zoomed and cropped
high-resolution volume.

The ImageCropper also understands image file metadata returned by the
image reader to configure voxel aspect ratio for a spatial
interpretation of the volume data.

"""

import numpy as np
import math

from vispy import gloo

from .util import load_image, bin_reduce
from .geometry import make_cube_clipped

class ImageCropper (object):

    def __init__(self, filename, maxdim=None, reform_data=None):
        if reform_data is None:
            reform_data = lambda x, meta: x

        I, self.meta = load_image(filename)

        self.maxdim = maxdim

        # interleave channels
        I = I.transpose(1,2,3,0)

        # correct aspect ratio if available
        if self.meta is not None:
            self.Zaspect = self.meta.z_microns / self.meta.x_microns
        else: 
            self.Zaspect = 1.0

        I = reform_data(I, self.meta)
        
        # store as base of pyramid
        self.pyramid = [ I ]
        self._extend_pyramid()
        for img in self.pyramid:
            print img.shape, img.dtype

        self.last_zoom = None
        self.last_origin = None
        self.last_channels = None
        self.origin = None
        self.channels = None
        self.set_view()
        self.last_level = None

    def min_pixel_step_size(self, zoom=1.0, outtexture=None):
        zoom_power = max(0, int(math.floor( math.log(zoom, 2.0) )))
        level = max(-1 - zoom_power, - len(self.pyramid))
        
        if outtexture is not None:
            D, H, W, C = outtexture.shape
        else:
            D, H, W, C = self.pyramid[level].shape

        span = max(W, H, D*self.Zaspect)

        return 1./span

    def downsample_power(self, x):
        if self.maxdim:
            return math.ceil( math.log(x/self.maxdim, 2.0) )
        else:
            return 0

    def next_pow2(self, x):
        return int(2**( math.ceil( math.log(x, 2.0) ) ))

    def _extend_pyramid(self):
        p = max([ 0 ] + map(self.downsample_power, self.pyramid[0].shape[0:3]))
        while p > 0:
            # reduce by half
            self.pyramid.append(
                bin_reduce(
                    self.pyramid[-1], 
                    [2, 2, 2, 1]
                ).astype(
                    self.pyramid[-1].dtype, 
                    copy=False
                )
            )
            p -= 1
        print 'pyramid %d levels deep' % len(self.pyramid)

    def set_view(self, zoom=1.0, origin=[0,0,0], anti_view=None, channels=None):
        self.zoom = zoom
        self.origin = tuple(origin)
        if anti_view is not None:
            self.anti_view = anti_view
        if channels is not None:
            # use caller-specified sequence of channels
            assert type(channels) is tuple
            assert len(channels) <= 4
            self.channels = channels
        else:
            # default to first N channels u to 4 for RGBA direct mapping
            self.channels = tuple(range(0, min(self.pyramid[0].shape[3], 4)))
        for c in self.channels:
            assert c >= 0
            assert c < self.pyramid[0].shape[3]

    def _get_texture3d_shape3(self, level=None):
        if level is None:
            level = self.last_level
        return map(lambda x: min(int(x), int(self.maxdim)), self.pyramid[level].shape[0:3])

    def _get_texture3d_format(self):
        I0 = self.pyramid[0]
        nc = len(self.channels)

        if I0.dtype == np.uint8:
            bps = 1
        elif I0.dtype == np.uint16 or I0.dtype == np.int16:
            bps = 2
        else:
            assert I0.dtype == np.float16 or I0.dtype == np.float32
            #bps = 2
            bps = 4

        print (nc, bps)
        return {
            (1,1): ('luminance', 'red'),
            (1,2): ('luminance', 'r16f'),
            (1,4): ('luminance', 'r16f'),
            (2,1): ('rg', 'rg'),
            (2,2): ('rg', 'rg32f'),
            (2,4): ('rg', 'rg32f'),
            (3,1): ('rgb', 'rgb'),
            (3,2): ('rgb', 'rgb16f'),
            (3,4): ('rgb', 'rgb16f'),
            (4,1): ('rgba', 'rgba'),
            (4,2): ('rgba', 'rgba16f'),
            (4,4): ('rgba', 'rgba16f')
        }[(nc, bps)]

    def get_texture3d(self, outtexture=None):
        """Pack N-channel image data into R, RG, RGB, RGBA Texture3D using self.channels projection.

           outtexture:
             None:     allocate new Texture3D
             not None: use existing Texture3D

           sets data in outtexture and returns the texture.
        """

        zoom_power = max(0, int(math.floor( math.log(self.zoom, 2.0) )))
        level = max(-1 - zoom_power, - len(self.pyramid))
        I0 = self.pyramid[level]
        D0, H0, W0, C0 = I0.shape

        # choose size for texture data
        D, H, W = self._get_texture3d_shape3(level)
        C = len(self.channels)

        if outtexture is None:
            print 'allocating texture3D'
            format, internalformat = self._get_texture3d_format()
            print format, internalformat
            outtexture = gloo.Texture3D(shape=(D, H, W, C), format=format, internalformat=internalformat)
        elif self.last_level == level \
             and self.last_origin == self.origin \
             and self.last_channels == self.channels:
            # don't need to reload texture data
            print 'reusing texture', self.last_level, level, self.last_origin, self.origin
            return outtexture
        else:
            print 'regenerating texture', self.last_level, level, self.last_origin, self.origin

        print 'using level', len(self.pyramid) + level
        print (D, H, W, C), '<-', I0.shape, list(self.channels), I0.dtype

        # offsets to center data in texture + border pad

        # requested origin position... only matters if src larger than dst per axis
        z, y, x = self.origin or (0, 0, 0)

        posfactor = 2**abs(2+level)  # level ranges -1 ... -N

        z *= posfactor
        y *= posfactor
        x *= posfactor

        def calculate_slices(L, L0, origin, axisname, pad=0):
            """Return out_slice, in_slice"""
            if (L-pad) >= L0:
                assert (L-pad) == L0
                out_base, in_base, offset = pad, 0, 0
            else:
                out_base = pad
                in_base = (L0 - L + pad)/2
                offset = ( ((origin < 0) and -1 or 1) # sign
                           * min(abs(origin), in_base) # clamp offset to available padding
                           )
            slices = slice(out_base, out_base+L-1), slice(in_base+offset, in_base+L-1+offset)
            print 'axis %s:' % axisname, 'origin %d (requested) %d (final)' % (origin, offset), slices[0], '<-', slices[1]
            return slices

        # find subtexture indices and input origin
        oKK, iKK = calculate_slices(D, D0, z, 'Z', pad=1)
        oJJ, iJJ = calculate_slices(H, H0, y, 'Y', pad=1)
        oII, iII = calculate_slices(W, W0, x, 'X', pad=1)

        # subset to load into texture
        # normalize for OpenGL [0,1.0] or [0,2**N-1] and zero black-level
        maxval = I0.max()
        minval = I0.min()
        scale = 1.0/(float(maxval) - float(minval))
        if I0.dtype == np.uint8 or I0.dtype == np.int8:
            tmpout = np.zeros((D+1, H+1, W+1, C), dtype=np.uint8)
            scale *= float(2**8-1)
        else:
            assert I0.dtype == np.float16 or I0.dtype == np.float32 or I0.dtype == np.uint16 or I0.dtype == np.int16
            tmpout = np.zeros((D+1, H+1, W+1, C), dtype=np.uint16 )
            scale *= (2.0**16-1)

        # pack selected channels into texture
        for i in range(C):
            tmpout[oKK,oJJ,oII,i] = (I0[iKK,iJJ,iII,self.channels[i]].astype(np.float32) - minval) * scale

        self.last_level = level
        self.last_zoom = self.zoom
        self.last_origin = self.origin
        self.last_channels = self.channels
        outtexture.set_data(tmpout)
        return outtexture

    def make_cube_clipped(self, dataplane=None):
        """Generate cube clipped against plane equation 4-tuple.
        
            Excludes semi-space beneath plane, i.e. with negative plane
            distance.  Omitting plane produces regular unclipped cube.
        """
        shape = self._get_texture3d_shape3()
            
        return make_cube_clipped(shape, self.Zaspect, 2**(1+self.last_level), dataplane)
        

