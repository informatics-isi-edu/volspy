
#
# Copyright 2014-2017 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

"""Volume data support.

Volume data is loaded as an ND-array with axes ZYXC and shape
(D,H,W,C) using available volumetric image file readers.

The ImageCropper class encapsulates the state necessary to:

  1. load a volume image file

  2. prepare 3D texture data of an appropriate texture storage format

  3. prepare 3D bounding box geometry

The ImageCropper also understands image file metadata returned by the
image reader to configure voxel aspect ratio for a spatial
interpretation of the volume data.

"""

import os
import numpy as np
import math

from vispy import gloo

from .util import load_and_mangle_image, bin_reduce
from .geometry import make_cube_clipped

class ImageManager (object):

    def __init__(self, filename, reform_data=None):
        I, self.meta, self.slice_origin = load_and_mangle_image(filename)

        voxel_size = I.micron_spacing

        try:
            view_grid_microns = tuple(map(float, os.getenv('ZYX_VIEW_GRID').split(",")))
            assert len(view_grid_microns) == 3
        except:
            view_grid_microns = (0.25, 0.25, 0.25)
        print("Goal is %s micron view grid. Override with ZYX_VIEW_GRID='float,float,float'" % (view_grid_microns,))

        view_reduction = tuple(map(lambda vs, ps: max(int(ps/vs), 1), voxel_size, view_grid_microns))
        print("Using %s view reduction factor on %s image grid." % (view_reduction, voxel_size))
        print("Final %s micron view grid after reduction." % (tuple(map(lambda vs, r: vs*r, voxel_size, view_reduction)),))

        if reform_data is not None:
            I = reform_data(I, self.meta, view_reduction)

        voxel_size = list(map(lambda a, b: a*b, voxel_size, view_reduction))
        self.Zaspect = voxel_size[0] / voxel_size[2]

        self.data = I
        self.last_channels = None
        self.channels = None
        self.set_view()

    def min_pixel_step_size(self, outtexture=None):
        if outtexture is not None:
            D, H, W, C = outtexture.shape
        else:
            D, H, W, C = self.data.shape

        span = max(W, H, D*self.Zaspect)

        return 1./span

    def set_view(self, anti_view=None, channels=None):
        if anti_view is not None:
            self.anti_view = anti_view
        if channels is not None:
            # use caller-specified sequence of channels
            assert type(channels) is tuple
            assert len(channels) <= 4
            self.channels = channels
        else:
            # default to first N channels u to 4 for RGBA direct mapping
            self.channels = tuple(range(0, min(self.data.shape[3], 4)))
        for c in self.channels:
            assert c >= 0
            assert c < self.data.shape[3]

    def _get_texture3d_format(self):
        I0 = self.data
        nc = len(self.channels)

        if I0.dtype == np.uint8:
            bps = 1
        elif I0.dtype == np.uint16 or I0.dtype == np.int16:
            bps = 2
        else:
            assert I0.dtype == np.float16 or I0.dtype == np.float32
            bps = 2
            #bps = 4

        print((nc, bps))
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
        I0 = self.data

        # choose size for texture data
        D, H, W = self.data.shape[0:3]
        C = len(self.channels)

        if outtexture is None:
            format, internalformat = self._get_texture3d_format()
            print('allocating texture3D', (D, H, W, C), internalformat)
            outtexture = gloo.Texture3D(shape=(D, H, W, C), format=format, internalformat=internalformat)
        elif self.last_channels == self.channels:
            print('reusing texture')
            return outtexture
        else:
            print('regenerating texture')

        print((D, H, W, C), '<-', I0.shape, list(self.channels), I0.dtype)

        # normalize for OpenGL [0,1.0] or [0,2**N-1] and zero black-level
        maxval = float(I0.max())
        minval = float(I0.min())
        if maxval > minval:
            scale = 1.0/(maxval - minval)
        else:
            scale = 1.0
        if I0.dtype == np.uint8 or I0.dtype == np.int8:
            tmpout = np.zeros((D, H, W, C), dtype=np.uint8)
            scale *= float(2**8-1)
        else:
            assert I0.dtype == np.float16 or I0.dtype == np.float32 or I0.dtype == np.uint16 or I0.dtype == np.int16
            tmpout = np.zeros((D, H, W, C), dtype=np.uint16 )
            scale *= (2.0**16-1)

        # pack selected channels into texture
        for i in range(C):
            tmpout[:,:,:,i] = (I0[:,:,:,self.channels[i]].astype(np.float32) - minval) * scale

        self.last_channels = self.channels
        outtexture.set_data(tmpout)
        return outtexture

    def make_cube_clipped(self, dataplane=None):
        """Generate cube clipped against plane equation 4-tuple.
        
            Excludes semi-space beneath plane, i.e. with negative plane
            distance.  Omitting plane produces regular unclipped cube.
        """
        shape = self.data.shape[0:3]
        return make_cube_clipped(shape, self.Zaspect, 2, dataplane)
        

