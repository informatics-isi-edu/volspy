
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

"""Volume rendering support.

A stateful VolumeRenderer implementing the canonical two-pass
ray-casting algorithm, with optional clipping and slicing tools and a
near-clipping plane for valid rendering even when the camera moves
inside the volume bounding box.

"""

import numpy as np

from vispy.util.transforms import rotate, perspective, ortho
from vispy import gloo

import os


# downsample volume to fit environmental limit?
maxtexsize = float(os.environ.get('MAX_3D_TEXTURE_WIDTH', 768))

# center on origin and change box aspect ratio to match image
cube_model = np.eye(4, dtype=np.float32)
cube_anti_model = np.eye(4, dtype=np.float32)

# turn volume to match experience with Fiji
rotate(cube_model, 180, 1, 0, 0)
rotate(cube_anti_model, -180, 1, 0, 0)


def _make_port(): 
    # make quad to cover viewport
    port_verts = np.zeros(4, dtype=[ ('position', np.float32, 3), ('texcoord', np.float32, 2) ])
    port_verts['position'] = np.array(
        [
            [ -1, -1, -1 ], [  1, -1, -1 ],
            [ -1,  1, -1 ], [  1,  1, -1 ]
            ]
        )
    port_verts['texcoord'] = np.array(
        [
            [ 0, 0 ], [  1, 0 ],
            [ 0, 1 ], [  1, 1 ]
            ]
        )

    port_verts = gloo.VertexBuffer(port_verts)
    port_faces = gloo.IndexBuffer(np.array([ 0, 1, 3,  0, 3, 2 ], dtype=np.uint32))
    port_model = np.eye(4, dtype=np.float32)

    return port_verts, port_faces, port_model


class VolumeProgram (gloo.Program):

    vert_shader = """
// Uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

// Attributes
attribute vec3 position;
attribute vec2 texcoord;

varying vec2 v_texcoord;

// Main
void main (void)
{
    vec4 pos;
    pos = u_projection * u_view * u_model * vec4(position,1.0);
    gl_Position = pos;
    v_texcoord = texcoord;
}
"""

    port_verts, port_face_indices, port_model = _make_port()

    def __init__(self, fragment_shader, vol_texture, num_channels, entry_texture, gain=1.0):
        gloo.Program.__init__(self, self.vert_shader, fragment_shader)

        self['u_data_texture'] = vol_texture
        self['u_projection'] = ortho(-.5, .5, -.5, .5, -100, 100)
        self.bind(self.port_verts)
        self['u_model'] = self.port_model
        self['u_view'] = np.eye(4, dtype=np.float32)
        self['u_entry_texture'] = entry_texture
        self['u_gain'] = gain
        self['u_numchannels'] = num_channels

    def draw(self):
        gloo.Program.draw(self, 'triangles', self.port_face_indices)
        
_color_uniforms = """
uniform int u_numchannels;
uniform float u_gain;
uniform float u_floorlvl;
"""

_color_repacker = """
    col_smp = col_packed_smp;
    if (u_numchannels == 1) {
       col_smp.g = col_packed_smp.r;
       col_smp.b = col_packed_smp.r;
    }
"""

# compute (clamped) linear amplified RGB color
_color_gain = """
       col_smp = clamp( u_gain * (col_smp - u_floorlvl), 0.0, 1.0);
"""

# compute alpha as (clamped) linear function of RGB
_linear_alpha = """
       col_smp.a = clamp(
          clamp((col_smp.r + col_smp.g + col_smp.b) / 2.0, 0.0, 0.75),
          0.0, 
          1.0
       );
"""

# accumulate voxels with alpha transparency (front-to-back)
_transparent_blend = """
       col_acc += (1.0 - col_acc.a) * col_smp * col_smp.a;
"""

# accumulate voxels with simple addition
_additive_blend = """
       col_acc = clamp(col_acc + col_smp * 0.01, 0.0, 1.0);
"""

# accumulate voxels with maximum intensity projection
_maxintensity_blend = """
       col_acc = max( col_acc, col_smp );
"""

class VolumeSliceProgram (VolumeProgram):

    @staticmethod
    def frag_shader(
        uniforms=None,
        colorunpack=None,
        colorxfer=None,
        **kwargs
        ):
        """Return GLSL fragment shader for volume slicer.

           Optional arguments override rendering code via GLSL code
           fragments inserted into ray-cast loop.

           uniforms:

              Declare uniforms used by shader fragments. When None
              (default), use _color_uniforms global GLSL fragment.

           colorunpack: 

              Interpret col_packed_smp vec4, extracted from texture3d
              sampler, and populate col_smp vec4 with color to be used
              by ray-caster.  When None (default), use _color_repacker
              global GLSL fragment.

           colorxfer:

              Interpret col_smp vec4 and update as per RGB color
              transfer function.  When None (default), use
              _linear_color global GLSL fragment.

        """
        if uniforms is None:
            uniforms = _color_uniforms
        if colorunpack is None:
            colorunpack = _color_repacker
        if colorxfer is None:
            colorxfer = _color_gain
        return """
uniform sampler3D u_data_texture;
uniform sampler2D u_entry_texture;
%(uniforms)s
varying vec2 v_texcoord;

void main()
{
    vec4 col_smp;
    vec4 col_packed_smp;
    vec4 texcoord;

    texcoord = texture2D(u_entry_texture, v_texcoord);
    col_packed_smp = texture3D(u_data_texture, texcoord.xyz / texcoord.w);
%(repack)s
%(colorxfer)s

    gl_FragColor = col_smp;
}

""" % dict(
            uniforms=uniforms,
            repack=colorunpack,
            colorxfer=colorxfer
            )

    def __init__(self, vol_texture, num_channels, entry_texture, gain=1.0, frag_glsl_parts=None):
        if frag_glsl_parts is None:
            frag_glsl_parts = dict()
        VolumeProgram.__init__(self, self.frag_shader(**frag_glsl_parts), vol_texture, num_channels, entry_texture, gain=1.0)

class VolumeRayCastProgram (VolumeProgram):

    @staticmethod
    def frag_shader(
        uniforms=None,
        colorunpack=None,
        colorxfer=None,
        alphastmt=None,
        blendstmt=None,
        **kwargs
        ):
        """Return GLSL fragment shader for volume ray-caster.

           Optional arguments override rendering code via GLSL code
           fragments inserted into ray-cast loop.

           uniforms:

              Declare uniforms used by shader fragments. When None
              (default), use _color_uniforms global GLSL fragment.

           colorunpack: 

              Interpret col_packed_smp vec4, extracted from texture3d
              sampler, and populate col_smp vec4 with color to be used
              by ray-caster.  When None (default), use _color_repacker
              global GLSL fragment.

           colorxfer:

              Interpret col_smp vec4 and update as per RGB color
              transfer function.  When None (default), use
              _linear_color global GLSL fragment.

           alphastmt:

              Interpret col_smp vec4 and adjust col_smp.a channel
              prior to ray-cast integration of voxel.  When None
              (default), use _linear_alpha global GLSL fragment.

           blendstmt:

              Accumulate col_smp vec4 into col_acc vec4 accumulator to
              perform ray-cast integration.  When None (default), use
              _transparent_blend global GLSL fragment.
        """
        if uniforms is None:
            uniforms = _color_uniforms
        if colorunpack is None:
            colorunpack = _color_repacker
        if colorxfer is None:
            colorxfer = _color_gain
        if alphastmt is None:
            alphastmt = _linear_alpha
        if blendstmt is None:
            blendstmt = _transparent_blend
        return """
uniform sampler3D u_data_texture;
uniform sampler2D u_entry_texture;
uniform sampler2D u_exit_texture;
%(uniforms)s
varying vec2 v_texcoord;

float rand(vec3 co)
{
    float a = 12.9898;
    float b = 78.233;
    float c = 43758.5453;
    vec3 co2 = co - vec3(0.5, 0.5, 0.5);
    float dt= dot(2.0 * co2.xyz, vec3(a,b,7.0));
    float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

void main()
{
    vec4 col_acc = vec4(0,0,0,0);
    float cast_len;
    float ray_len;
    float step_len;
    vec2 f_pos;
    vec4 col_packed_smp;
    vec4 col_smp;
    vec4 entry;
    vec4 exit;
    vec4 step;
    vec4 texcoord;

    f_pos = v_texcoord;

    entry = vec4(texture2D(u_entry_texture, f_pos).xyz, 1.0);
    exit = vec4(texture2D(u_exit_texture, f_pos).xyz, 1.0);

    step = 2.0 * normalize(exit - entry) / %(maxtexsize)d.0;
    step_len = length(step);
    ray_len = length(exit - entry) - step_len;

    texcoord = entry + rand(entry.xyz) * step;
    cast_len = step_len;

    for (int s = 0; s < %(maxtexsize)d; s++)
    {
       if (cast_len > ray_len || entry == exit || col_acc.a > 1.0)
         break;

       col_packed_smp = texture3D(u_data_texture, texcoord.xyz / texcoord.w);

%(repack)s
%(colorxfer)s
%(alpha)s
%(blendstmt)s
       
       texcoord += step;
       cast_len += step_len;
    }

    col_acc.a = 1.0;
    gl_FragColor = col_acc;
}

""" % dict(
            maxtexsize=maxtexsize * 2.0, 
            uniforms=uniforms,
            repack=colorunpack,
            alpha=alphastmt,
            colorxfer=colorxfer,
            blendstmt=blendstmt
            )

    def __init__(self, vol_texture, num_channels, entry_texture, exit_texture, gain=1.0, frag_glsl_parts=None):
        if frag_glsl_parts is None:
            frag_glsl_parts = dict()
        self.frag_shader = VolumeRayCastProgram.frag_shader(**frag_glsl_parts)
        VolumeProgram.__init__(self, self.frag_shader, vol_texture, num_channels, entry_texture, gain)
        self['u_exit_texture'] = exit_texture


class PolyhedronProgram (gloo.Program):

    vert_shader = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 position;
attribute vec4 color;
varying vec4 v_color;

void main()
{
   v_color = color;
   gl_Position = u_projection * u_view * u_model * vec4(position,1.0);
}
"""

    frag_shader = """
varying vec4 v_color;
void main()
{
   gl_FragColor = v_color;
}
"""

    def __init__(self, view, model):
        gloo.Program.__init__(self, self.vert_shader, self.frag_shader)
        self['u_view'] = view
        self['u_model'] = model

    def draw(self, faces):
        gloo.Program.draw(self, 'triangles', faces)


class VolumeRenderer (object):

    def __init__(self, vol_cropper, vol_texture, num_channels, vol_view, fbo_size=(1024, 1024), zoom=1.0, frag_glsl_dicts=None, vol_interp='linear'):
        self.vol_cropper = vol_cropper

        self.vol_texture = vol_texture
        self.vol_texture.interpolation = vol_interp
        self.vol_texture.wrapping = 'clamp_to_edge'

        cube_verts, cube_faces, cut_face = self.vol_cropper.make_cube_clipped()
        self.cube_verts = gloo.VertexBuffer(cube_verts)
        self.volume_faces = gloo.IndexBuffer(cube_faces)
        self.slice_faces = gloo.IndexBuffer(cut_face)

        self.fbo_viewport = (0, 0) + fbo_size
        #fbo_format = 'rgba32f'
        fbo_format = 'rgba16'
        self.entry_texture = gloo.Texture2D(shape=(fbo_size + (4,)), internalformat=fbo_format)
        self.exit_texture = gloo.Texture2D(shape=(fbo_size + (4,)), internalformat=fbo_format)

        self.entry_texture.interpolation = 'linear'
        self.exit_texture.interpolation = 'linear'
    
        if frag_glsl_dicts is None:
            # supply different ray blending math
            frag_glsl_dicts = [
                dict(blendstmt=blendstmt, desc=desc)
                for blendstmt, desc in [
                    (_transparent_blend, 'Linear transparency blend.'),
                    (_additive_blend, 'Additive blend.'),
                    (_maxintensity_blend, 'Maximum-intensity projection.')
                    ]
                ]

        # build slicers and ray casters with GLSL code dictionaries
        self.prog_vol_slicers = map(
            lambda parts: VolumeSliceProgram(self.vol_texture, num_channels, self.entry_texture, zoom, parts),
            frag_glsl_dicts
            )
        self.prog_ray_casters = map(
            lambda parts: VolumeRayCastProgram(self.vol_texture, num_channels, self.entry_texture, self.exit_texture, zoom, parts), 
            frag_glsl_dicts
            )

        self.frag_glsl_dicts = frag_glsl_dicts

        self.color_mode = 0

        self.prog_boundary = PolyhedronProgram(vol_view, cube_model)
        self.prog_boundary.bind(self.cube_verts)
        
        self.fbo_entry = gloo.FrameBuffer(self.entry_texture)
        self.fbo_exit = gloo.FrameBuffer(self.exit_texture)
        self.anti_view = None
        
    def set_color_mode(self, i=None):
        if i is None:
            self.color_mode = (self.color_mode + 1) % len(self.prog_ray_casters)
        else:
            self.color_mode = i % len(self.prog_ray_casters)

        print 'color mode %d %s' % (self.color_mode, self.frag_glsl_dicts[self.color_mode].get('desc', ''))

    def set_clip_plane(self, view_plane):
        """Set clipping plane in view coordinate space.
        
           view_plane is (A,B,C,D) plane equation and clipping will
           exclude volume in negative half-space, i.e. with negative
           plane distance.  A value of None disables clipping.
        """
        # transform (A,B,C) as (X,Y,Z,1)
        model_plane = np.dot(
            cube_anti_model,
            np.dot(
                np.array(view_plane[0:3] + [1.], dtype=np.float32),
                self.anti_view
                )
            )
        model_plane = model_plane[0:4]/model_plane[3] # drop W
        model_plane[0:3] = model_plane[0:3]/np.linalg.norm(model_plane[0:3]) # find new A,B,C
        model_plane[3] = view_plane[3]  # copy D back into place
        cube_verts, cube_faces, cut_face = self.vol_cropper.make_cube_clipped(model_plane)
        self.cube_verts.set_data(cube_verts)
        self.volume_faces.set_data(cube_faces, copy=True)
        self.slice_faces.set_data(cut_face, copy=True)

    def set_vol_projection(self, projection):
        self.prog_boundary['u_projection'] = projection

    def set_uniform(self, name, value):
        for prog in self.prog_vol_slicers + self.prog_ray_casters:
            prog[name] = value
        
    def set_vol_view(self, view, anti_view):
        self.prog_boundary['u_view'] = view
        self.anti_view = anti_view

    def draw_volume(self, viewport, color_mask=(True, True, True, True)):
        gloo.set_color_mask(True, True, True, True)

        with self.fbo_entry:
            # draw the ray entry map via front-faces
            gloo.set_clear_color('black')
            gloo.set_viewport(* self.fbo_viewport )
            gloo.set_cull_face(mode='back')
            gloo.clear(color=True, depth=False)
            gloo.set_state(blend=False, depth_test=False, cull_face=True)
            self.prog_boundary.draw(self.volume_faces)
            
        with self.fbo_exit:
            # draw the ray exit map via back-faces
            gloo.set_clear_color('black')
            gloo.set_viewport(* self.fbo_viewport )
            gloo.set_cull_face(mode='front')
            gloo.clear(color=True, depth=False)
            gloo.set_state(blend=False, depth_test=False, cull_face=True)
            self.prog_boundary.draw(self.volume_faces)

        # cast rays based on entry/exit textures
        gloo.set_color_mask(* color_mask)
        gloo.set_clear_color('black')
        gloo.set_viewport(* viewport)
        gloo.set_cull_face(mode='back')
        gloo.clear(color=True, depth=False)
        gloo.set_state(blend=False, depth_test=False, cull_face=True)
        self.prog_ray_casters[self.color_mode].draw()

    def draw_slice(self, viewport, color_mask=(True, True, True, True)):
        gloo.set_color_mask(True, True, True, True)

        with self.fbo_entry:
            # draw the ray entry map via front-faces
            gloo.set_clear_color('black')
            gloo.set_viewport(* self.fbo_viewport )
            gloo.set_cull_face(mode='back')
            gloo.clear(color=True, depth=False)
            gloo.set_state(blend=False, depth_test=False, cull_face=True)
            self.prog_boundary.draw(self.slice_faces)
            
        # slice based on entry texture
        gloo.set_color_mask(* color_mask)
        gloo.set_clear_color('black')
        gloo.set_viewport(* viewport)
        gloo.set_cull_face(mode='back')
        gloo.set_state(blend=False, depth_test=False, cull_face=True)
        gloo.clear(color=True, depth=False)
        self.prog_vol_slicers[self.color_mode].draw()

