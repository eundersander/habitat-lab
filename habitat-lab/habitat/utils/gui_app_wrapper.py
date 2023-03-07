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


class GuiAppWrapper(Application):
    def __init__(self, width, height):
        configuration = self.Configuration()
        configuration.title = "title goes here"
        configuration.size = (width, height)  # todo
        Application.__init__(self, configuration)

        self._sim_wrapper = None
        self._render_wrapper = None

    def set_sim_and_render_wrappers(self, sim_wrapper, render_wrapper):
        assert isinstance(sim_wrapper, SimWrapper)
        assert isinstance(render_wrapper, RenderWrapper)
        self._sim_wrapper = sim_wrapper
        self._render_wrapper = render_wrapper

    def draw_event(self):
        sim_dt = 1 / 60.0  # todo
        keyframe = self._sim_wrapper.sim_update(sim_dt)

        self._render_wrapper.post_sim_update(keyframe)

        render_dt = 1 / 60.0  # todo
        self._render_wrapper.render_update(render_dt)

        # render_wrapper should have rendered to mn.gl.default_framebuffer
        self.swap_buffers()
        self.redraw()


class TestSimRenderWrapper(SimWrapper, RenderWrapper):
    def sim_update(self, dt):
        return None

    def post_sim_update(self, keyframe):
        pass

    def render_update(self, dt):
        pass


if __name__ == "__main__":
    test_wrapper = TestSimRenderWrapper()

    gui_app_wrapper = GuiAppWrapper(test_wrapper, test_wrapper)
    gui_app_wrapper.exec()
