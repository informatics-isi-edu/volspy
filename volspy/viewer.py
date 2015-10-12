
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import sys
import os
import numpy as np

import datetime

from vispy.util.transforms import perspective, ortho
from vispy import gloo
from vispy import app

from .data import ImageManager
from .render import maxtexsize, VolumeRenderer, rotate, translate, scale
from .util import bin_reduce, clamp

#gloo.gl.use_gl('pyopengl debug')


def keydoc(details):
    def helper(original_method):
        original_method._keydocs = details
        return original_method
    return helper

_default_view = np.eye(4, dtype=np.float32)
_default_anti_view = np.eye(4, dtype=np.float32)

_rotations = os.getenv('VIEW_ROTATE')
if _rotations:
    _rotations = map(float, _rotations.split(','))
    assert len(_rotations) == 3, "VIEW_ROTATE_XYZ must be euler rotations about static X, Y, Z in degrees"

    rotate(*(_default_view, _rotations[0]) + (1, 0, 0))
    rotate(*(_default_view, _rotations[1]) + (0, 1, 0))
    rotate(*(_default_view, _rotations[2]) + (0, 0, 1))

    rotate(*(_default_anti_view, 0 - _rotations[2]) + (0, 0, 1))
    rotate(*(_default_anti_view, 0 - _rotations[1]) + (0, 1, 0))
    rotate(*(_default_anti_view, 0 - _rotations[0]) + (1, 0, 0))

class Canvas(app.Canvas):

    def _reform_image(self, I, meta, view_reduction):
        return bin_reduce(I, view_reduction + (1,))

    _frag_glsl_dicts = None
    _pick_glsl_index = None
    _vol_interp = {
        'nearest': 'nearest',
        'linear': 'linear'
    }.get(os.getenv('VOXEL_SAMPLE', '').lower(), 'linear')

    def __init__(self, filename, reset=True):
        app.Canvas.__init__(
            self, #vsync=True,
            keys='interactive',
            title='%s %s' % (os.path.basename(sys.argv[0]).replace('-viewer', ''), os.path.basename(filename)),
            )

        self.vol_cropper = ImageManager(filename, self._reform_image)
        nc = self.vol_cropper.data.shape[3]
        try:
            channel = int(os.getenv('VIEW_CHANNEL'))
        except:
            channel = None

        if channel is not None and channel >= 0 and channel < nc:
            print "Starting single-channel mode with user-specified channel %d of %d total channels" % (channel, nc)
            self.vol_channels = (channel,)
        elif nc > 4:
            print "%d channel image encountered, switching to single-channel mode" % nc
            self.vol_channels = (0,)
        else:
            print "%d channel image encountered, using direct %d-channel mapping" % (nc, nc)
            self.vol_channels = None
        self.vol_cropper.set_view(channels=self.vol_channels)
        self.vol_texture = self.vol_cropper.get_texture3d()
        self.vol_zoom = 1.0

        W = self.vol_texture.shape[2]
        self.size = W, W
        self.prev_size = self.size
        self.perspective = True
        if self.vol_channels is not None:
            nc = len(self.vol_channels)
        else:
            nc = self.vol_cropper.data.shape[3]
        self.volume_renderer = VolumeRenderer(
            self.vol_cropper,
            self.vol_texture,
            nc,
            _default_view.copy(), # view
            (int(maxtexsize * 4), int(maxtexsize * 4)), # fbo_size
            frag_glsl_dicts=self._frag_glsl_dicts,
            pick_glsl_index=self._pick_glsl_index,
            vol_interp=self._vol_interp
            )

        self.toggle_color_mode.__func__._keydocs = {
            'B': 'Cycle through color blending modes %s. Reverse cycle with shift key.' % [
                frag.get('desc', 'undocumented')
                for frag in self.volume_renderer.frag_glsl_dicts
            ]
        }
        
        self._timer = None

        self.fps_t0 = datetime.datetime.now()
        self.fps_count = 0
        
        self.mouse_button_offset = 0
        gloo.set_clear_color('black')

        self.frame = 0

        # to allow over-riding by subclasses
        self.drag_button_handlers = {
            1: lambda distance, delta, pos1, basis: self._mouse_drag_rotation(distance, delta),
            2: lambda distance, delta, pos1, basis: self._mouse_drag_translation(delta)
        }

        self.end_drag_handlers = [
            self._end_drag_xform,
            ]

        self.key_press_handlers = dict(
            [
                ('P', self.toggle_projection),
                ('B', self.toggle_color_mode),
                ('C', self.toggle_channel),
                ('Z', self.adjust_zoom),
                ('R', self.r_key),
                ('F', self.adjust_floor_level),
                ('=', self.reorient),
                ('Space', self.toggle_slicing),
                ('?', self.help)
                ]
            + [ (k, self.adjust_gain) for k in 'G1234567890!@#$%^&*()' ]
            + [ 
                (k, self.adjust_rotate) 
                for k in [ 'Left', 'Right', 'Up', 'Down', '[', ']', '{', '}' ]
                ]
            )
            
        self.viewport1 = (0, 0) + self.size

        if reset:
            self.reset_ui()

    def help(self, event=None):
        """Show brief help text for UI."""
        
        handlers = self.key_press_handlers.items()
        handlers.sort(key=lambda i: (len(i[0]), ord(i[0][0]), i))
        print """
Keyboard UI:

key 'ESC': exit the application."""
        for key, handler in handlers:
            if hasattr(handler, '_keydocs'):
                doc = handler._keydocs.get(key, handler.__doc__)
            else:
                doc = handler.__doc__
            if doc is not False:
                print "key '%s': %s" % (key, doc)

        print """
for some keyboard commands, adding the 'Alt' modifier allows a finer
adjustment step than the regular key combinations (with or without
'Shift' modifier)!

Mouse UI:

Button 1 drag: Rotate rendered volume around origin.
Button 2 drag: Pan (translate) rendered volume relative to origin.
Vertical scroll: Move the clipping or slicing plane up and down the view axis.

Resize viewing window using native window-manager controls.
"""
        

    def reload_data(self):
        self.vol_cropper.set_view(channels=self.vol_channels)
        self.vol_cropper.get_texture3d(self.vol_texture)
        self.update()

    def reorient(self, event):
        """Reorient to view down Z axis; or Y axis with 'Control' modifier; or X axis with 'Alt' modifier."""
        self.xform = _default_view.copy()
        self.anti_xform = _default_anti_view.copy()

        if 'Control' in event.modifiers:
            axis = (1, 0, 0)
            plane = 'XZ'
        elif 'Alt' in event.modifiers:
            axis = (0, 1, 0)
            plane = 'YZ'
        else:
            self.update_view()
            plane = 'XY'
            print 'Viewing %s planes.' % plane
            return
        
        rotate(*(self.xform, 90) + axis)
        rotate(*(self.anti_xform, -90) + axis)
        self.update_view()
        print 'Viewing %s planes.' % plane

    def r_key(self, event):
        """Freeze/unfreeze mouse-based reorientation; or reset UI controls with 'Control' modifier."""
        if 'Control' in event.modifiers:
            self.reset_ui(event)
        else:
            self.toggle_drag_reorientation()

    def toggle_drag_reorientation(self):
        self.drag_reorient_enabled = not self.drag_reorient_enabled
        print "Mouse-based reorientation %s." % (self.drag_reorient_enabled and 'ENABLED' or 'DISABLED')
            
    def reset_ui(self, event=None):
        """Reset UI controls to startup settings."""
        print 'reset_ui'

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        self.drag_reorient_enabled = True
        self.view = None
        
        self.xform = _default_view.copy()
        self.anti_xform = _default_anti_view.copy()
        
        self.scale = np.eye(4, dtype=np.float32)
        self.anti_scale = np.eye(4, dtype=np.float32)

        self.gain = 1.0
        self.zoom = 1.0
        self.floorlvl = 0.1

        self.perspective = False
        self.toggle_projection()
        self.volume_renderer.set_color_mode(0)

        self.clip_distance = -1.96

        self.drag_xform = None
        self.drag_anti_xform = None

        self.slice_mode = False

        self.reload_data()
        self.update_view()
        self.volume_renderer.set_uniform('u_gain', self.gain)
        self.volume_renderer.set_uniform('u_floorlvl', self.floorlvl)

    def toggle_color_mode(self, event=None):
        """Cycle through color blending modes."""
        # toggles w/o optional index
        self.volume_renderer.set_color_mode(reverse=event is not None and 'Shift' in event.modifiers)
        self.update()

    def toggle_channel(self, event=None):
        """Cycle through image channels when in single-channel mode."""
        nc = self.vol_cropper.data.shape[3]
        if self.vol_channels is not None:
            c = self.vol_channels[0]
            self.vol_channels = ((c+1)%nc,)
            self.reload_data()
            print "viewing channel %d of %d (zero-based)" % (c, nc)

    def toggle_projection(self, event=None):
        """Toggle between perspective and orthonormal projection modes."""
        self.perspective = not self.perspective
        if self.perspective:
            self.volume_renderer.set_vol_projection(perspective(60, 1., 100, 0))
        else:
            self.volume_renderer.set_vol_projection(ortho(-1, 1, -1, 1, -1000, 1000))

    def adjust_zoom(self, event):
        """Increase ('Z') or decrease ('z') rendered zoom-level."""
        if 'Shift' in event.modifiers:
            self.zoom *= 1.25
        else:
            self.zoom *= 1./1.25
        self.scale = np.eye(4, dtype=np.float32)
        self.anti_scale = np.eye(4, dtype=np.float32)
        scale(self.scale, self.zoom, self.zoom, self.zoom)
        scale(self.anti_scale, 1./self.zoom, 1./self.zoom, 1./self.zoom) 
        # allow cropper to adjust volume for mip-levels
        self.reload_data()
        self.update_view()
        self.update()
        print 'adjust_zoom', self.zoom

    @keydoc(dict(
            [ (k, "Set gain to %d or 1/%d (with 'Shift')." % (int(k), int(k))) for k in '123456789' ]
            + [ ('0', "set gain to 10 or 1/10 (with 'Shift').") ]
            + [ (k, False) for k in '!@#$%^&*()' ]
            ))
    def adjust_gain(self, event):
        """Increase ('G') or decrease ('g') linear gain of rendered data."""
        print 'adjust_gain'
        def numkey(key):
            for i in range(10):
                if key == '%d' % i:
                    return i or 10  # replace 0 with 10
            return False

        shift_keys = {
            '!': 1, '@': 2, '#': 3, '$': 4, '%': 5,
            '^': 6, '&': 7, '*': 8, '(': 9, ')': 10
            }

        if numkey(event.key):
            gain = float(numkey(event.key))
            if 'Shift' in event.modifiers:
                self.gain = 1/gain
            else:
                self.gain = gain
        elif event.key == 'G':
            if 'Shift' in event.modifiers:
                self.gain *= 1.25
            else:
                self.gain *= 1./1.25
        elif event.key in shift_keys:
            self.gain = 1/float(shift_keys[event.key])
        else:
            print 'event %s not understood as gain adjustment' % event

        self.volume_renderer.set_uniform('u_gain', self.gain)
        self.update()
        print 'gain set to %.2f' % self.gain

    def adjust_floor_level(self, event):
        """Increase ('F') or decrease ('f') floor-level used in alpha-transfer function."""
        if 'Shift' in event.modifiers:
            self.floorlvl += 0.005
        else:
            self.floorlvl -= 0.005
        self.volume_renderer.set_uniform('u_floorlvl', self.floorlvl)
        self.update()
        print 'floor level set to %.2f' % self.floorlvl

    def on_key_press(self, event):
        handler = self.key_press_handlers.get(event.key)

        if handler:
            handler(event)
            self.update()
        elif event.key in ['Shift', 'Escape', 'Alt', 'Control']:
            pass
        else:
            print 'no handler for key %s' % event.key

    def on_resize(self, event):
        width, height = event.size

        if self.prev_size == event.size:
            return

        print "window resize", event.size
        self.prev_size = event.size

        if float(width) / float(height) > 1.0:
            self.viewport1 = 0, (height - width)/2, width, width # final ray-casts
        else:
            self.viewport1 = (width - height)/2, 0, height, height # final ray-casts

        self.update()

    @keydoc({
            'Up': "Decrease rotation about X viewing axis.",
            'Down': "Increase rotation about X viewing axis.",
            'Left': "Decrease rotation about Y viewing axis.",
            'Right': "Increase rotation about Y viewing axis.",
            '[': "Increase rotation about Z viewing axis.",
            ']': "Decrease rotation about Z viewing axis.",
            '{': False,
            '}': False
            })
    def adjust_rotate(self, event):
        """Adjust rotation of rendered data."""
        sign = {'Up':-1, 'Left':-1, ']':-1, '}':-1}.get(event.key, 1)

        print 'adjust_rotate %s' % event

        # just apply a single small rotation increment
        axis = {
            'Up': (1, 0, 0), 
            'Down': (1, 0, 0), 
            'Left': (0, 1, 0),
            'Right': (0, 1, 0),
            '[': (0, 0, 1),
            ']': (0, 0, 1)
        }[event.key]

        angle = 2 * sign

        rotate(*(self.xform, angle) + axis)
        rotate(*(self.anti_xform, -angle) + axis)

        self.update_view()

    def _mouse_drag_translation(self, delta):
        prev_xform = self.drag_xform

        self.drag_xform = np.eye(4, dtype=np.float32)
        translate(self.drag_xform, delta[0]/self.zoom, -delta[1]/self.zoom, 0)

        self.drag_anti_xform = np.eye(4, dtype=np.float32)
        translate(self.drag_anti_xform, -delta[0]/self.zoom, delta[1]/self.zoom, 0)

        if prev_xform is None \
                or(self.drag_xform != prev_xform).any():
            self.update_view()
            self.update()
        
    def _mouse_drag_rotation(self, distance, delta):
        prev_rot = self.drag_xform

        self.drag_xform = np.eye(4, dtype=np.float32)
        rotate(self.drag_xform, distance * 180, delta[1], delta[0], 0)

        self.drag_anti_xform = np.eye(4, dtype=np.float32)
        rotate(self.drag_anti_xform, - distance * 180, delta[1], delta[0], 0)

        if prev_rot is None \
                or(self.drag_xform != prev_rot).any():
            self.update_view()
            self.update()

    def on_mouse_wheel(self, event):
        """Adjust clip/slice place distance with vertical scroll wheel.

           The clip range is [-1.96, 1.96] to account for viewing a [-1, 1]
           normalized cube with its longest diagonal perpindicular to
           the screen, and being able to clip past the near and far
           corners.

           Divide the range into basis number of steps, where basis is
           current window size so higher resolution renders have finer
           depth precision.

           Invert the sign of the wheel delta since we interpret
           scrolling "down" as plunging the clip plane deeper into the
           image.

        """
        basis = float(min(*self.size))
        prev_clip = self.clip_distance

        self.clip_distance = clamp(self.clip_distance - event.delta[1]/basis, -1.96, 1.96)
        
        if self.clip_distance != prev_clip:
            print 'scroll %s, clip_distance %s' % (event.delta, self.clip_distance)
            self.update_view()

    def toggle_slicing(self, event):
        """(Space key) Toggle volume and slicing modes."""
        if not self.slice_mode:
            self.slice_mode = True
            if 'Shift' not in event.modifiers:
                # set to center for usability?
                self.clip_distance = 0
        else:
            self.slice_mode = False
            if 'Shift' not in event.modifiers:
                self.clip_distance = -1.96
                
        self.update_view()
        
    def on_mouse_move(self, event):
        if event.is_dragging and self.drag_reorient_enabled:
            pos0 = np.array(event.press_event.pos, dtype=np.float32)
            pos1 = np.array(event.pos, dtype=np.float32)
            delta = pos1 - pos0

            basis = float(min(*self.size))
            delta = delta / basis

            distance = np.linalg.norm(delta)

            if event.button == 0:
                # hack to work around strange older CentOS system
                self.mouse_button_offset =  1

            if (event.button + self.mouse_button_offset) in self.drag_button_handlers:
                self.drag_button_handlers[
                    (event.button + self.mouse_button_offset)
                    ](distance, delta, pos1, basis)
            else:
                print 'unrecognized mouse button %d' % event.button

    def _end_drag_xform(self):
        if self.drag_xform is not None:
            self.xform[...] = np.dot(self.xform, self.drag_xform)
            self.anti_xform[...] = np.dot(self.drag_anti_xform, self.anti_xform)

            self.drag_xform = None
            self.drag_anti_xform = None

            self.update_view()

    def on_mouse_release(self, event):
        if event.is_dragging:
            # let handlers sort out whether their mode was active
            for handler in self.end_drag_handlers:
                handler()

    def update_view(self, on_timer=False):

        s = self.vol_cropper.min_pixel_step_size(outtexture=self.vol_texture)

        prev_view = self.view
        view = _default_view.copy()

        view[...] = np.dot(view, self.xform)
        if self.drag_xform is not None:
            view[...] = np.dot(view, self.drag_xform)
        view[...] = np.dot(view, self.scale)
        translate(view, 0., 0., -1.97) # matched to 60 degree fov

        anti_view = _default_anti_view.copy()
        if self.drag_xform is not None:
            anti_view[...] = np.dot(anti_view, self.drag_anti_xform)
        anti_view[...] = np.dot(anti_view, self.anti_xform)

        self.anti_view = anti_view
        self.volume_renderer.set_vol_view(view, anti_view)
        self.volume_renderer.set_clip_plane([0, 0, 1, max(self.clip_distance, -0.866 / self.zoom)])
        self.vol_cropper.set_view(anti_view, self.vol_channels)

        if prev_view is None \
                or (view != prev_view).any():
            self.update()

        return view

    def on_timer(self, event):
        self.update_view(True)

    def on_draw(self, event, color_mask=(True, True, True, True), pick=None, on_pick=None):
        if self.fps_count >= 10:
            t1 = datetime.datetime.now()
            print "%f FPS" % (10.0 / (t1 - self.fps_t0).total_seconds())
            self.fps_t0 = t1
            self.fps_count = 1
        else:
            self.fps_count += 1

        gloo.set_viewport(* self.viewport1 )
        #print 'draw %d' % self.frame
        self.frame += 1
        if self.slice_mode:
            return self.volume_renderer.draw_slice(self.viewport1, color_mask=color_mask, pick=pick, on_pick=on_pick)
        else:
            return self.volume_renderer.draw_volume(self.viewport1, color_mask=color_mask, pick=pick, on_pick=on_pick)

