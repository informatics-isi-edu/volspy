
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

"""Volume geometry support.

A volume dataset is a 3D array with ZYX axes.  The shape of the array
is (D,H,W) i.e. depth, height, width number of grid steps.  The
spatial interpretation is a scalar field filling a 3D box, with
aspect-ratio adjustments for the Y and Z axes relative to the X axes.

To support rendering methods, we generate geometry for the volume
bounding box in an origin-centered, unit-cube coordinate system.
Taking into account data grid aspect ratio and data array shape, the
longest spatial dimension of the volume is fit to unit-length, i.e.
spanning from -0.5 to 0.5.

The generated geometry can be scaled by a zoom-factor (making the
box's longest span be length zoom rather than length 1.0, and spanning
-zoom/2 to zoom/2).  The box can also be clipped by a clipping plane,
expressed by a plane equation in the origin-centered, unit-cube
coordinate system.  Zooming is applied before clipping, so a higher
zoom can cause more of the volume to be excluded by the clip.

In practice, this means an application may need to keep track of the
unit-cube to world transform and apply the inverse transform to a
plane in order to request an appropriately clipped volume box.

"""

import numpy as np

from vispy.geometry import create_cube

from .util import plane_distance


def make_cube_clipped(shape, Zaspect, zoom, plane=None):
    """Generate cube clipped against plane equation 4-tuple (A,B,C,D).

       Excludes semi-space beneath plane, i.e. with negative plane
       distance.  Omitting plane produces regular unclipped cube. The
       clipped cube may have zero corners and edges if it falls
       completely beneath the plane.

       Returns (vertices, faces, cutfaces):
         -- vertices is an array suitable for use as a vertex buffer
         -- faces is an array of triangle-strip vertex indices
         -- cutfaces is an array of triangle-strip vertex indices

       The cutfaces triangle strip is a subset of the faces triangle
       strip that only includes the face embedded within the clipping
       plane (if any).
    """
    # shape is number of voxels in each dimension
    D, H, W = shape

    # span is max length in XY pixel units, correcting for Zaspect
    # (X:Y must have 1:1 aspect, Zaspect is Z:X ratio)
    span = float(max(W,H,D*Zaspect))

    # fit shape into unit cube centered on origin w/ zoom 1.0
    # zoom allows caller to shrink or grow unit cube

    # make box to outline volumetric image region

    # actual box shape
    bW = zoom * W/span
    bH = zoom * H/span
    bD = zoom * D/span * Zaspect

    # halved box shape for portion of box in each octant
    hW = bW/2.
    hH = bH/2.
    hD = bD/2.

    cube_verts = np.zeros(20, dtype=[ 
                ('position', np.float32, 3), 
                ('color', np.float32, 4) 
                ])
    
    X = 5 # dummy value to be reset by clipping
    cube_verts['position'] = np.array(
        [
            [ -hW, -hH, -hD ], [ hW, -hH, -hD ], [ hW, hH, -hD ], [ -hW, hH, -hD ], # back corners
            [ -hW, -hH,  hD ], [ hW, -hH,  hD ], [ hW, hH,  hD ], [ -hW, hH,  hD ], # front corners
            [ X, X,  X ], [  X, X,  X ], [  X,  X,  X ], [ X,  X,  X ], # back edge-cuts
            [ X, X,  X ], [  X, X,  X ], [  X,  X,  X ], [ X,  X,  X ], # front edge-cuts
            [ X, X,  X ], [  X, X,  X ], [  X,  X,  X ], [ X,  X,  X ]  # middle edge-cuts
            ]
        )

    # colormap cube for regular 8 corners
    for i in range(8):
        for axis in range(3):
            if cube_verts['position'][i,axis] > 0.:
                cube_verts['color'][i,axis] = 1.0
            else:
                cube_verts['color'][i,axis] = 0.0
        cube_verts['color'][i, 3] = 1.0

    # in CCW winding
    # corners, then edge-cuts (half-step ahead)
    cube_quads = [
        ( ( 0, 3, 2, 1 ), ( 11, 10,  9,  8 ) ),  # back
        ( ( 4, 5, 6, 7 ), ( 12, 13, 14, 15 ) ), # front
        ( ( 5, 1, 2, 6 ), ( 16,  9, 17, 13 ) ),  # right
        ( ( 0, 4, 7, 3 ), ( 19, 15, 18, 11 ) ), # left
        ( ( 0, 1, 5, 4 ), (  8, 16, 12, 19 ) ), # bottom
        ( ( 2, 3, 7, 6 ), ( 10, 18, 14, 17 ) )   # top
        ]

    # check each cube corner against clip plane
    corner_clipped = [ 
        # plane is None means no clipping
        plane is not None and plane_distance( c, plane ) > 0.
        for c in cube_verts['position'][0:8]
        ]

    cubeclipped = set([ i for i in range(8) if corner_clipped[i] ])

    face_triangles = []
    cutface_triangles = []
    face_cutpoint_lists = []

    def face_roll(face):
        assert len(face) == 4
        return face[1:4] + face[0:1]

    def edge_clip(v1, cut, v2):
        # choose v1 or cut with v2 lookahead
        if v1 not in cubeclipped:
            yield v1
            if v2 in cubeclipped:
                yield cut
        elif v2 not in cubeclipped:
            yield cut

    def edge_cutpoints(v1, cut, v2):
        # choose v1 or cut with v2 lookahead
        if v1 not in cubeclipped:
            yield None
            if v2 in cubeclipped:
                yield (v1, cut, v2)
        elif v2 not in cubeclipped:
            yield (v1, cut, v2)

    def outline_clip(face, cuts):
        rolled = face_roll(face)
        for i in range(4):
            for v in edge_clip(face[i], cuts[i], rolled[i]):
                yield v

    def outline_cutpoints(face, cuts):
        rolled = face_roll(face)
        for i in range(4):
            for cp in edge_cutpoints(face[i], cuts[i], rolled[i]):
                yield cp

    def cutpoint_solve(v1, cut, v2):
        # update geometry and colormap
        p1 = cube_verts['position'][v1]
        p2 = cube_verts['position'][v2]
        c1 = cube_verts['color'][v1]
        c2 = cube_verts['color'][v2]

        # find cutpoint bisection of edge
        d1 = abs(plane_distance(p1, plane))
        d2 = abs(plane_distance(p2, plane))
        cutratio = d1 / (d1 + d2)

        # interpolate cutpoint position and color
        cube_verts['position'][cut] = p1 + (p2 - p1) * cutratio
        cube_verts['color'][cut][0:3] = c1[0:3] + (c2[0:3] - c1[0:3]) * cutratio
        cube_verts['color'][cut][3] = 1.0

    def build_polygon(outline):
        # tesselate one polygon in CCW winding order
        mode = len(outline)

        if mode == 0:
            # face is absent
            pass
        elif mode == 3:
            # basic triangle
            face_triangles.extend( outline )
        elif mode == 4:
            # quadragonal
            face_triangles.extend( outline[0:3] )
            face_triangles.extend( outline[2:4] + outline[0:1] )
        elif mode == 5:
            # pentagonal
            face_triangles.extend( outline[0:3] )
            face_triangles.extend( outline[2:4] + outline[0:1] )
            face_triangles.extend( outline[3:5] + outline[0:1] )
        elif mode == 6:
            # hexagonal
            a, b, c, d, e, f = outline
            face_triangles.extend([
                a, b, c,
                c, d, e,
                e, f, a,
                a, c, e
            ])
        else:
            raise ValueError('%d vertex polygon %s not supported' % (mode, outline))

    for face, cuts in cube_quads:
        outline = list(outline_clip(face, cuts))
        build_polygon( outline )
        cplist = list(outline_cutpoints(face, cuts))

        if cplist and cplist[0] and cplist[-1]:
            cplist = [ cplist[-1], cplist[0] ]
        else:
            cplist = [ cp for cp in cplist if cp ]

        if cplist:
            assert len(cplist) == 2
            face_cutpoint_lists.append( cplist )
        
    # update cutpoint geometry where needed
    cutpoints_done = set()
    for cplist in face_cutpoint_lists:
        for v1, cut, v2 in cplist:
            if cut not in cutpoints_done:
                cutpoint_solve(v1, cut, v2)
                cutpoints_done.add(cut)

    # generate final cut-face 
    sides = [ 
        [
            cp[1] # just cut index
            for cp in cplist 
            ]
        for cplist in face_cutpoint_lists
        ]

    if not sides:
        # no cut-face
        pass
    else:
        prev_top = len(face_triangles)

        # each side's cplist gives two cuts in side's CCW winding
        # reverse to get our CCW winding order
        sides = [ side[::-1] for side in sides ]

        # build CCW winding corner list
        outline = list(sides[0])
        next_corner = dict([ tuple(side) for side in sides[1:] ])

        while True:
            prev = outline[-1]
            try:
                next = next_corner[prev]
            except KeyError:
                #print sides, outline, next_corner
                raise
            if next in outline:
                break
            else:
                outline.append(next)
        
        build_polygon(outline)

        cutface_triangles.extend( face_triangles[prev_top:] )
    
    try:
        return cube_verts, np.array(face_triangles, dtype=np.uint32), np.array(cutface_triangles, dtype=np.uint32)
    except:
        print(face_triangles, cutface_triangles)
        raise

