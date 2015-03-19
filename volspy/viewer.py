
#
# Copyright 2014-2015 University of Southern California
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#

import numpy as np

import datetime

from vispy.util.transforms import scale, translate, rotate, perspective, ortho
from vispy import gloo
from vispy import app

from .data import ImageCropper
from .render import maxtexsize, VolumeRenderer


#gloo.gl.use_gl('pyopengl debug')


def keydoc(details):
    def helper(original_method):
        original_method._keydocs = details
        return original_method
    return helper
        
class Canvas(app.Canvas):

    def _reform_image(self, I, meta):
        return I[:,:,:,0:3]

    _frag_glsl_dicts = None
    _vol_interp = 'linear'

    def __init__(self, filename, reset=True):
        app.Canvas.__init__(
            self, #vsync=True,
            keys='interactive'
            )

        self.vol_cropper = ImageCropper(filename, maxtexsize, self._reform_image)
        self.vol_texture = self.vol_cropper.get_texture3d()
        self.vol_zoom = 1.0
        self.origin = [0, 0, 0] # Z, Y, X

        W = self.vol_texture.shape[2]
        self.size = W, W
        self.prev_size = self.size
        self.perspective = True
        self.volume_renderer = VolumeRenderer(
            self.vol_cropper,
            self.vol_texture,
            self.vol_cropper.pyramid[0].shape[-1],
            np.eye(4, dtype=np.float32), # view
            (int(maxtexsize * 4), int(maxtexsize * 4)), # fbo_size
            frag_glsl_dicts=self._frag_glsl_dicts,
            vol_interp=self._vol_interp
            )

        self._timer = None

        self.fps_t0 = datetime.datetime.now()
        self.fps_count = 0
        
        self.mouse_button_offset = 0
        gloo.set_clear_color('black')

        self.frame = 0

        # to allow over-riding by subclasses
        self.drag_button_handlers = {
            1: lambda distance, delta, pos1, basis: self._mouse_drag_rotation(distance, delta),
            2: lambda distance, delta, pos1, basis: self._mouse_drag_slicing(pos1, basis),
            3: lambda distance, delta, pos1, basis: self._mouse_drag_clipping(pos1, basis)
            }

        self.end_drag_handlers = [
            self._end_drag_rotation,
            self._end_drag_slicing,
            self._end_drag_clipping
            ]

        self.key_press_handlers = dict(
            [
                ('Q', self.quit),
                ('P', self.toggle_projection),
                ('B', self.toggle_color_mode),
                ('Z', self.adjust_zoom),
                ('R', self.reset_ui),
                ('F', self.adjust_floor_level),
                ('?', self.help)
                ]
            + [ (k, self.adjust_gain) for k in 'G1234567890!@#$%^&*()' ]
            + [ 
                (k, self.adjust_rotate) 
                for k in [ 'Left', 'Right', 'Up', 'Down', '[', ']', '{', '}' ]
                ]
            + [ (k, self.adjust_data_position) for k in 'IJK' ]
            )
            
        self.viewport1 = (0, 0) + self.size

        if reset:
            self.reset_ui()

    def quit(self, event=None):
        """Quit application (same as 'ESC')."""
        self.close()

    def help(self, event=None):
        """Show brief help text for UI."""
        
        handlers = self.key_press_handlers.items()
        handlers.sort(key=lambda i: (len(i[0]), ord(i[0][0]), i))
        print """
Keyboard UI:
"""
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

Button 1 drag: Adjust orientation of rendered volume.
Button 2 drag: Move front clipping-plane along Z viewing axis.
Button 3 drag: Move slicing plane along Z viewing axis.

Resize viewing window using native window-manager controls.
"""
        

    def reload_data(self):
        self.vol_cropper.set_view(self.zoom, self.origin)
        self.vol_cropper.get_texture3d(self.vol_texture)
        self.update()

    def reset_ui(self, event=None):
        """Reset UI controls to startup settings."""
        
        print 'reset_ui'

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        self.vol_origin = [ 0, 0, 0 ]
        self.view = None
        self.rotation = np.eye(4, dtype=np.float32)
        self.anti_rotation = np.eye(4, dtype=np.float32)
        self.scale = np.eye(4, dtype=np.float32)
        self.anti_scale = np.eye(4, dtype=np.float32)

        self.gain = 8.0
        self.zoom = 1.0
        self.floorlvl = 0.1

        self.perspective = False
        self.toggle_projection()
        self.volume_renderer.set_color_mode(0)

        self.clip_distance = -1.96

        # demo-mode rotation about named axes...
        self.auto_rotate_X_angle = 0.0
        self.auto_rotate_Y_angle = 0.0
        self.auto_rotate_Z_angle = 0.0
        self.auto_rotate_X_speed = 0
        self.auto_rotate_Y_speed = 0
        self.auto_rotate_Z_speed = 0

        self.drag_rotation = None
        self.drag_anti_rotation = None

        self.slice_mode = False

        self.reload_data()
        self.update_view()
        self.volume_renderer.set_uniform('u_gain', self.gain)
        self.volume_renderer.set_uniform('u_floorlvl', self.floorlvl)

    def toggle_color_mode(self, event=None):
        """Cycle through color blending modes."""
        self.volume_renderer.set_color_mode() # toggles w/o optional argument
        self.update()

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

    @keydoc({
            'I': "Advance ('I') or retreat ('i') zoomed viewing center along data X axis.",
            'J': "Advance ('J') or retreat ('j') zoomed viewing center along data Y axis.",
            'K': "Advance ('K') or retreat ('k') zoomed viewing center along data Z axis."
            })
    def adjust_data_position(self, event):
        """Adjust center of viewing region when zoomed in on data larger than 3D texture size."""

        axis = dict(K=0, J=1, I=2)[event.key]

        if 'Shift' in event.modifiers:
            self.origin[axis] += 10
        else:
            self.origin[axis] -= 10

        print 'origin offset', tuple(self.origin)
        self.reload_data()

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
        elif event.key in ['Shift', 'Escape', 'Alt']:
            pass
        else:
            print 'no handler for key %s' % event.key

    def on_resize(self, event):
        width, height = event.size

        if self.prev_size == event.size:
            return

        print "window resize", event.size
        #self.size = event.size
        self.prev_size = event.size

        if float(width) / float(height) > 1.0:
            self.viewport1 = 0, (height - width)/2, width, width # final ray-casts
        else:
            self.viewport1 = (width - height)/2, 0, height, height # final ray-casts

        self.update()

    @keydoc({
            'Up': "Decrease rotation (or speed with 'Shift') about X viewing axis.",
            'Down': "Increase rotation (or speed with 'Shift') about X viewing axis.",
            'Left': "Decrease rotation (or speed with 'Shift') about Y viewing axis.",
            'Right': "Increase rotation (or speed with 'Shift') about Y viewing axis.",
            '[': "Increase rotation (or speed with 'Shift') about Z viewing axis.",
            ']': "Decrease rotation (or speed with 'Shift') about Z viewing axis.",
            '{': False,
            '}': False
            })
    def adjust_rotate(self, event):
        """Adjust rotation of rendered data."""
        sign = {'Up':-1, 'Left':-1, ']':-1, '}':-1}.get(event.key, 1)

        print 'adjust_rotate %s' % event

        if event.key == 'S':
            # stop auto-rotate
            self.auto_rotate_X_angle = 0.0
            self.auto_rotate_Y_angle = 0.0
            self.auto_rotate_Z_angle = 0.0
            self.auto_rotate_X_speed = 0
            self.auto_rotate_Y_speed = 0
            self.auto_rotate_Z_speed = 0

            if self._timer is not None:
                self._timer.stop()
                self._timer = None

        elif 'Shift' in event.modifiers:
            # adjust auto-rotation speed in small increments
            if event.key in ('Right', 'Left'):
                self.auto_rotate_Y_speed += sign
            elif event.key in ('Up', 'Down'):
                self.auto_rotate_X_speed += sign
            elif event.key in ('[', ']', '{', '}'):
                self.auto_rotate_Z_speed += sign

            if self.auto_rotate_Z_speed != 0.0 \
                    or self.auto_rotate_Y_speed != 0.0 \
                    or self.auto_rotate_X_speed != 0.0:
                if self._timer is None:
                    print 'starting timer'
                    self._timer = app.Timer('auto', connect=self.on_timer, start=True)
            else:
                if self._timer is not None:
                    print 'stopping timer'
                    self._timer.stop()
                    self._timer = None
        else:
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

            rotate(*(self.rotation, angle) + axis)
            rotate(*(self.anti_rotation, -angle) + axis)

            self.update_view()

    def _mouse_drag_rotation(self, distance, delta):
        prev_rot = self.drag_rotation

        self.drag_rotation = np.eye(4, dtype=np.float32)
        rotate(self.drag_rotation, distance * 180, delta[1], delta[0], 0)

        self.drag_anti_rotation = np.eye(4, dtype=np.float32)
        rotate(self.drag_anti_rotation, - distance * 180, delta[1], delta[0], 0)

        if prev_rot is None \
                or(self.drag_rotation != prev_rot).any():
            self.update_view()
            self.update()

    def _mouse_drag_clipping(self, pos1, basis):
        prev_clip = self.clip_distance
        self.clip_distance = 1.8 * (pos1[1] / basis - .5)
        if prev_clip != self.clip_distance:
            self.update_view()

    def _mouse_drag_slicing(self, pos1, basis):
        self._mouse_drag_clipping(pos1, basis)
        prev_slice = self.slice_mode
        self.slice_mode = True
        if prev_slice != self.slice_mode:
            self.update_view()

    def on_mouse_move(self, event):
        if event.is_dragging:
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

    def _end_drag_rotation(self):
        if self.drag_rotation is not None:
            self.rotation[...] = np.dot(self.rotation, self.drag_rotation)
            self.anti_rotation[...] = np.dot(self.drag_anti_rotation, self.anti_rotation)

            # merge auto-rotation into base rotation to keep stable order w.r.t. drag
            self.rotation[...] = np.dot(self.rotation, self.auto_rotation)
            self.anti_rotation[...] = np.dot(self.auto_anti_rotation, self.anti_rotation)
            self.auto_rotate_X_angle = 0.0
            self.auto_rotate_Y_angle = 0.0
            self.auto_rotate_Z_angle = 0.0
        
            self.drag_rotation = None
            self.drag_anti_rotation = None

            self.update_view()

    def _end_drag_clipping(self):
        prev_clip = self.clip_distance
        self.clip_distance = -0.866 / self.zoom
        if prev_clip != self.clip_distance:
            self.update_view()
            self.update()

    def _end_drag_slicing(self):
        if self.slice_mode:
            self.slice_mode = False
            self.update_view()
            self.update()

    def on_mouse_release(self, event):
        if event.is_dragging:
            # let handlers sort out whether their mode was active
            for handler in self.end_drag_handlers:
                handler()

    def update_view(self, on_timer=False):

        self.auto_rotation = np.eye(4, dtype=np.float32)
        self.auto_anti_rotation = np.eye(4, dtype=np.float32)

        angles = []
        axes = []

        if on_timer:
            self.auto_rotate_X_angle += 0.1 * self.auto_rotate_X_speed
            self.auto_rotate_Y_angle += 0.1 * self.auto_rotate_Y_speed
            self.auto_rotate_Z_angle += 0.1 * self.auto_rotate_Z_speed

        angles.append(self.auto_rotate_X_angle)
        axes.append( (1, 0, 0) )

        angles.append(self.auto_rotate_Y_angle)
        axes.append( (0, 1, 0) )

        angles.append(self.auto_rotate_Z_angle)
        axes.append( (0, 0, 1) )

        for angle, axis in zip(angles, axes):
            rotate(*(self.auto_rotation, angle) + axis)
            
        angles.reverse()
        axes.reverse()

        for angle, axis in zip(angles, axes):
            rotate(*(self.auto_anti_rotation, -angle) + axis)

        X, Y, Z = self.vol_origin
        s = self.vol_cropper.min_pixel_step_size(zoom=1.0, outtexture=self.vol_texture)

        prev_view = self.view
        view = np.eye(4, dtype=np.float32)

        # allow subclasses to translate data origin
        translate(view, -X*s, -Y*s, -Z*s)
        view[...] = np.dot(view, self.rotation)
        if self.drag_rotation is not None:
            view[...] = np.dot(view, self.drag_rotation)
        view[...] = np.dot(view, self.auto_rotation)
        view[...] = np.dot(view, self.scale)
        translate(view, 0., 0., -1.97) # matched to 60 degree fov

        anti_view = np.eye(4, dtype=np.float32)
        anti_origin = np.eye(4, dtype=np.float32)
        anti_view[...] = np.dot(anti_view, self.auto_anti_rotation)
        if self.drag_rotation is not None:
            anti_view[...] = np.dot(anti_view, self.drag_anti_rotation)
        anti_view[...] = np.dot(anti_view, self.anti_rotation)
        translate(anti_origin, X*s, Y*s, Z*s)
        anti_view[...] = np.dot(anti_view, anti_origin)

        self.anti_view = anti_view
        self.volume_renderer.set_vol_view(view, anti_view)
        self.volume_renderer.set_clip_plane([0, 0, 1, max(self.clip_distance, -0.866 / self.zoom)])
        self.vol_cropper.set_view(self.zoom, self.origin, anti_view)

        if prev_view is None \
                or (view != prev_view).any():
            self.update()

        return view

    def on_timer(self, event):
        self.update_view(True)

    def on_draw(self, event, color_mask=(True, True, True, True)):
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
            self.volume_renderer.draw_slice(self.viewport1, color_mask=color_mask)
        else:
            self.volume_renderer.draw_volume(self.viewport1, color_mask=color_mask)

