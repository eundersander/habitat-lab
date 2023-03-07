#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes

# must call this before importing habitat or magnum! avoids EGL_BAD_ACCESS error on some platforms
import sys

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import abc
import math
import time

import magnum as mn
import numpy as np
from magnum.platform.glfw import Application


class SimWrapper:
    @abc.abstractmethod
    def sim_update(self, dt):
        pass


class RenderWrapper:
    @abc.abstractmethod
    def post_sim_update(self, keyframe):
        pass

    @abc.abstractmethod
    def render_update(self, dt):
        pass

    @abc.abstractmethod
    def unproject(self, viewport_pos):
        pass


# based on https://docs.unity3d.com/ScriptReference/Input.html
class InputWrapper:
    KeyNS = Application.KeyEvent.Key

    def __init__(self):
        self._key_held = set()
        self._mouse_button_held = set()
        self._mouse_position = [0, 0]

        self._key_down = set()
        self._key_up = set()
        self._mouse_button_down = set()
        self._mouse_button_up = set()
        self._relative_mouse_position = [0, 0]
        self._mouse_scroll_offset = 0
        self._mouse_ray = None

    def validate_key(key):
        assert isinstance(key, Application.KeyEvent.Key)

    def get_key(self, key):
        InputWrapper.validate_key(key)
        return key in self._key_held

    def get_key_down(self, key):
        InputWrapper.validate_key(key)
        return key in self._key_down

    def get_key_up(self, key):
        InputWrapper.validate_key(key)
        return key in self._key_up

    def validate_mouse_button(mouse_button):
        assert isinstance(mouse_button, Application.MouseEvent.Button)

    def get_mouse_button(self, mouse_button):
        InputWrapper.validate_mouse_button(mouse_button)
        return mouse_button in self._mouse_button_held

    def get_mouse_button_down(self, mouse_button):
        InputWrapper.validate_mouse_button(mouse_button)
        return mouse_button in self._mouse_button_down

    def get_mouse_button_up(self, mouse_button):
        InputWrapper.validate_mouse_button(mouse_button)
        return mouse_button in self._mouse_button_up

    @property
    def mouse_position(self):
        return self._mouse_position

    @property
    def relative_mouse_position(self):
        return self._relative_mouse_position

    @property
    def mouse_scroll_offset(self):
        return self._mouse_scroll_offset

    @property
    def mouse_ray(self):
        return self._mouse_ray

    # Key/button up/down is only True on the frame it occurred. Mouse relative position is
    # relative to its position at the start of frame.
    def on_frame_end(self):
        self._key_down.clear()
        self._key_up.clear()
        self._mouse_button_down.clear()
        self._mouse_button_up.clear()
        self._relative_mouse_position = [0, 0]
        self._mouse_scroll_offset = 0


class InputDriverApplication(Application):
    def __init__(self, config):
        super().__init__(config)
        self._input_wrappers = []

    def add_input_wrapper(self, input_wrapper):
        self._input_wrappers.append(input_wrapper)

    def key_press_event(self, event: Application.KeyEvent) -> None:
        key = event.key
        InputWrapper.validate_key(key)
        for wrapper in self._input_wrappers:
            # If the key is already held, this is a repeat press event and we should
            # ignore it.
            if key not in wrapper._key_held:
                wrapper._key_held.add(key)
                wrapper._key_down.add(key)

    def key_release_event(self, event: Application.KeyEvent) -> None:
        key = event.key
        InputWrapper.validate_key(key)
        for wrapper in self._input_wrappers:
            wrapper._key_held.remove(key)
            wrapper._key_up.add(key)

    def mouse_press_event(self, event: Application.MouseEvent) -> None:
        mouse_button = event.button
        InputWrapper.validate_mouse_button(mouse_button)
        for wrapper in self._input_wrappers:
            wrapper._mouse_button_held.add(mouse_button)
            wrapper._mouse_button_down.add(mouse_button)

    def mouse_release_event(self, event: Application.MouseEvent) -> None:
        mouse_button = event.button
        InputWrapper.validate_mouse_button(mouse_button)
        for wrapper in self._input_wrappers:
            wrapper._mouse_button_held.remove(mouse_button)
            wrapper._mouse_button_up.add(mouse_button)

    def mouse_scroll_event(self, event: Application.MouseEvent) -> None:
        # shift+scroll is forced into x direction on mac, seemingly at OS level,
        # so use both x and y offsets.
        scroll_mod_val = (
            event.offset.y
            if abs(event.offset.y) > abs(event.offset.x)
            else event.offset.x
        )

        for wrapper in self._input_wrappers:
            # accumulate
            wrapper._mouse_scroll_offset += scroll_mod_val

    def get_mouse_position(
        self, mouse_event_position: mn.Vector2i
    ) -> mn.Vector2i:
        """
        This function will get a screen-space mouse position appropriately
        scaled based on framebuffer size and window size.  Generally these would be
        the same value, but on certain HiDPI displays (Retina displays) they may be
        different.
        """
        scaling = mn.Vector2i(self.framebuffer_size) / mn.Vector2i(
            self.window_size
        )
        return mouse_event_position * scaling

    def mouse_move_event(self, event: Application.MouseMoveEvent) -> None:
        mouse_pos = self.get_mouse_position(event.position)
        for wrapper in self._input_wrappers:
            wrapper._mouse_position = mouse_pos

    def update_mouse_ray(self, unproject_fn):
        for wrapper in self._input_wrappers:
            wrapper._mouse_ray = unproject_fn(wrapper._mouse_position)
            # print(wrapper._mouse_position, " -> [", wrapper._mouse_ray.origin, wrapper._mouse_ray.direction, "]")


class GuiAppWrapper(InputDriverApplication):
    def __init__(self, width, height):
        configuration = self.Configuration()
        configuration.title = "title goes here"
        configuration.size = (width, height)  # todo
        super().__init__(configuration)

        self._sim_input = InputWrapper()
        self.add_input_wrapper(self._sim_input)

        self._sim_wrapper = None
        self._render_wrapper = None
        self._sim_time = None

    def get_sim_input(self):
        return self._sim_input

    def set_sim_and_render_wrappers(self, sim_wrapper, render_wrapper):
        assert isinstance(sim_wrapper, SimWrapper)
        assert isinstance(render_wrapper, RenderWrapper)
        self._sim_wrapper = sim_wrapper
        self._render_wrapper = render_wrapper

        # temp hack
        self._sim_wrapper._debug_line_render = (
            self._render_wrapper._replay_renderer.debug_line_render(0)
        )

    def draw_event(self):
        max_sim_updates_per_render = 1
        sim_dt = 1 / 20.0  # todo

        curr_time = time.time()
        if self._sim_time is None:
            num_sim_updates = 1
            self._sim_time = curr_time
            self._last_draw_event_time = curr_time
            self._debug_counter = 0
            self._debug_timer = curr_time
        else:
            # elapsed = curr_time - self._last_draw_event_time
            elapsed_since_last_sim_update = curr_time - self._sim_time
            num_sim_updates = int(
                math.floor(elapsed_since_last_sim_update / sim_dt)
            )
            num_sim_updates = min(num_sim_updates, max_sim_updates_per_render)
            self._debug_counter += num_sim_updates
            self._sim_time += sim_dt * num_sim_updates
            self._last_draw_event_time = curr_time
            # if self._debug_counter >= 50:
            #     elapsed = curr_time - self._debug_timer
            #     sps = self._debug_counter / elapsed
            #     print("sps: ", sps)
            #     self._debug_timer = curr_time
            #     self._debug_counter = 0

        for _ in range(num_sim_updates):
            post_sim_update_dict = self._sim_wrapper.sim_update(sim_dt)
            self._sim_input.on_frame_end()
            self._render_wrapper.post_sim_update(post_sim_update_dict)

        render_dt = 1 / 60.0  # todo
        self._render_wrapper.render_update(render_dt)

        # todo: also update when mouse moves
        self.update_mouse_ray(self._render_wrapper.unproject)

        time.sleep(0.03)  # temp hack

        # render_wrapper should have rendered to mn.gl.default_framebuffer
        self.swap_buffers()
        self.redraw()

        time.sleep(0.03)  # temp hack


# todo: better name
class ImageDrawer:
    def __init__(self, max_width=1024, max_height=1024):
        size = mn.Vector2i(max_width, max_height)

        # pre-allocate texture and framebuffer
        self.texture = mn.gl.Texture2D()
        self.texture.set_storage(1, mn.gl.TextureFormat.RGBA8, size)
        # color.set_sub_image(0, (0, 0), image)
        self.framebuffer = mn.gl.Framebuffer(mn.Range2Di((0, 0), size))
        self.framebuffer.attach_texture(
            mn.gl.Framebuffer.ColorAttachment(0), self.texture, 0
        )

    def draw(self, pixel_data, dest_x, dest_y):
        import torch  # lazy import; avoid torch dependency at file scope

        if isinstance(pixel_data, (np.ndarray, torch.Tensor)):
            assert len(pixel_data.shape) == 3 and (
                pixel_data.shape[2] == 3 or pixel_data.shape[2] == 4
            )
            assert (
                pixel_data.dtype == np.uint8 or pixel_data.dtype == torch.uint8
            )
            # todo: catch case where storage is not 0-dim-major?
            self.draw_bytearray(
                bytearray(pixel_data),
                pixel_data.shape[0],
                pixel_data.shape[1],
                pixel_data.shape[2],
                dest_x,
                dest_y,
            )
        else:
            raise TypeError(
                "Type "
                + type(pixel_data)
                + " isn't yet supported by ImageDrawer. You should add it!"
            )

    def draw_bytearray(
        self,
        bytearray_pixel_data,
        width,
        height,
        bytes_per_pixel,
        dest_x,
        dest_y,
    ):
        # see max_width, max_height in constructor
        assert width <= self.texture.image_size(0)[0]
        assert height <= self.texture.image_size(0)[1]

        assert len(bytearray_pixel_data) == width * height * bytes_per_pixel
        assert bytes_per_pixel == 3 or bytes_per_pixel == 4

        size = mn.Vector2i(width, height)
        image = mn.ImageView2D(
            mn.PixelFormat.RGBA8_UNORM
            if bytes_per_pixel == 4
            else mn.PixelFormat.RGB8_UNORM,
            size,
            bytearray_pixel_data,
        )
        self.texture.set_sub_image(0, (0, 0), image)

        dest_coord = mn.Vector2i(dest_x, dest_y)
        mn.gl.AbstractFramebuffer.blit(
            self.framebuffer,
            mn.gl.default_framebuffer,
            mn.Range2Di((0, 0), size),
            mn.Range2Di(dest_coord, dest_coord + size),
            mn.gl.FramebufferBlit.COLOR,
            mn.gl.FramebufferBlitFilter.NEAREST,
        )


class TestSimRenderWrapper(SimWrapper, RenderWrapper):
    def __init__(self):
        self.image_drawer = ImageDrawer(max_width=1024, max_height=1024)
        self.dest_x = 100
        self.dest_y = 50

    def sim_update(self, dt):
        return None

    def post_sim_update(self, keyframe):
        pass

    def render_update(self, dt):
        # self.dest_x += 1
        # self.dest_y += 1

        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )
        mn.gl.default_framebuffer.bind()

        width = 256
        height = 768
        raw_pixel_data = []
        for row in range(height):
            tmp = row // 3
            raw_pixel_data.extend(
                [tmp, 0, 0, 0, 0, tmp, tmp, 0] * (width // 2)
                if (row % 2)
                else [0, tmp, 0, 0, tmp, tmp, tmp, 0] * (width // 2)
            )
        # raw_pixel_data = [255, 0, 0, 0, 0, 255, 255, 0] * (width // 2) * height
        bytes_per_pixel = 4
        arr = np.array(raw_pixel_data, dtype=np.uint8)
        arr = arr.reshape((width, height, bytes_per_pixel))

        self.image_drawer.draw(arr, self.dest_x, self.dest_y)


if __name__ == "__main__":
    gui_app_wrapper = GuiAppWrapper(1024, 1024)
    test_wrapper = TestSimRenderWrapper()
    gui_app_wrapper.set_sim_and_render_wrappers(test_wrapper, test_wrapper)
    gui_app_wrapper.exec()
