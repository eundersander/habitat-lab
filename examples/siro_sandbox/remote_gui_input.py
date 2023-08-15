#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from server.interprocess_record import get_queued_client_states
from server.average_rate_tracker import AverageRateTracker

import magnum as mn

from habitat.gui.gui_input import GuiInput

class RemoteGuiInput:
    def __init__(self):

        self._recent_client_states = []
        self._debug_line_render = None  # will be set later by user code

        self._receive_rate_tracker = AverageRateTracker(2.0)

        # temp hack
        self._key_map = {
            "q": GuiInput.KeyNS.Q,
            "w": GuiInput.KeyNS.W,
            "e": GuiInput.KeyNS.E,
            "a": GuiInput.KeyNS.A,
            "s": GuiInput.KeyNS.S,
            "d": GuiInput.KeyNS.D,
            "f": GuiInput.KeyNS.F,
        }

        # temp map VR button to key
        self._button_map = {
            '0': GuiInput.KeyNS.ZERO
        }

        self._gui_input = GuiInput()

    def get_gui_input(self):
        return self._gui_input

    def get_history_length(self):
        return 4

    def get_history_timestep(self):
        return 1 / 60

    def get_head_pose(self, history_index=0):

        if history_index >= len(self._recent_client_states):
            return None, None

        avatar_root_json = self._recent_client_states[history_index]["avatar"]["root"]
        pos_json = avatar_root_json["position"]
        pos = mn.Vector3(pos_json[0], pos_json[1], pos_json[2])
        rot_json = avatar_root_json["rotation"]
        rot_quat = mn.Quaternion(mn.Vector3(rot_json[0], rot_json[1], rot_json[2]), rot_json[3])

        return pos, rot_quat

    def get_hand_pose(self, hand_idx, history_index=0):

        if history_index >= len(self._recent_client_states):
            return None, None

        client_state = self._recent_client_states[history_index]
        assert "hands" in client_state["avatar"]
        hands_json = client_state["avatar"]["hands"]
        assert hand_idx >= 0 and hand_idx < len(hands_json)

        hand_json = hands_json[hand_idx]
        pos_json = hand_json["position"]
        # pos = mn.Vector3(pos_json["_x"], pos_json["_y"], pos_json["_z"])
        pos = mn.Vector3(pos_json[0], pos_json[1], pos_json[2])
        rot_json = hand_json["rotation"]
        # rot_quat = mn.Quaternion(mn.Vector3(rot_json["_x"], rot_json["_y"], rot_json["_z"]), rot_json["_w"])
        rot_quat = mn.Quaternion(mn.Vector3(rot_json[0], rot_json[1], rot_json[2]), rot_json[3])

        return pos, rot_quat

    def update_input_state_from_remote_client_states(self, client_states):

        if not len(client_states):
            return

        # gather all recent keyDown and keyUp events
        for client_state in client_states:
            # Beware client_state input has dicts of bools (unlike GuiInput, which uses sets)
            assert "input" in client_state
            input_json = client_state["input"]
            for key in input_json["keyDown"]:
                if key not in self._key_map:
                    print(f"key {key} not mapped!")
                    continue
                if input_json["keyDown"][key]:
                    self._gui_input._key_down.add(self._key_map[key])
            for key in input_json["keyUp"]:
                if key not in self._key_map:
                    print(f"key {key} not mapped!")
                    continue
                if input_json["keyUp"][key]:
                    self._gui_input._key_up.add(self._key_map[key])

            for button in input_json["buttonDown"]:
                if button not in self._button_map:
                    print(f"button {button} not mapped!")
                    continue
                if input_json["buttonDown"][button]:
                    self._gui_input._key_down.add(self._button_map[button])
            for button in input_json["buttonUp"]:
                if button not in self._button_map:
                    print(f"key {button} not mapped!")
                    continue
                if input_json["buttonUp"][button]:
                    self._gui_input._key_up.add(self._button_map[button])

        # todo: think about ambiguous GuiInput states (key-down and key-up events in the same
        # frame and other ways that keyHeld, keyDown, and keyUp can be inconsistent.
        client_state = client_states[-1]
        self._gui_input._key_held.clear()
        for key in input_json["keyHeld"]:
            if key not in self._key_map:
                print(f"key {key} not mapped!")
                continue
            if input_json["keyHeld"][key]:
                self._gui_input._key_held.add(self._key_map[key])

        for button in input_json["buttonHeld"]:
            if button not in self._button_map:
                print(f"button {button} not mapped!")
                continue
            if input_json["buttonHeld"][button]:
                self._gui_input._key_held.add(self._button_map[button])

    def debug_visualize_client(self):

        if not self._debug_line_render:
            return

        if not len(self._recent_client_states):
            return
        
        if True:
            pos, rot_quat = self.get_head_pose()
            trans = mn.Matrix4.from_(rot_quat.to_matrix(), pos)
            half_size = 0.25
            self._debug_line_render.push_transform(trans)
            # self._debug_line_render.draw_box(mn.Vector3(-half_size, -half_size, -half_size), 
            #     mn.Vector3(half_size, half_size, half_size),
            #     mn.Color3(151, 206, 222) / 255)
            color0 = mn.Color3(255, 255, 255) / 255
            color1 = mn.Color4(255, 255, 255, 0) / 255
            size = 0.5
            # draw a frustum (forward is z-)
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0), mn.Vector3(size, size, -size),
                color0, color1)
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0), mn.Vector3(-size, size, -size),
                color0, color1)
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0), mn.Vector3(size, -size, -size),
                color0, color1)
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0), mn.Vector3(-size, -size, -size),
                color0, color1)

            self._debug_line_render.pop_transform()

        hand_colors = (mn.Color3(255, 0, 0) / 255, mn.Color3(0, 0, 255) / 255)
        for hand_idx in range(2):
            hand_pos, hand_rot_quat = self.get_hand_pose(hand_idx)
            trans =  mn.Matrix4.from_(hand_rot_quat.to_matrix(), hand_pos)
            half_size = 0.1
            self._debug_line_render.push_transform(trans)
            self._debug_line_render.draw_box(mn.Vector3(-half_size, -half_size, -half_size), 
                mn.Vector3(half_size, half_size, half_size),
                hand_colors[hand_idx])
            pointer_len = 0.5
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0), mn.Vector3(0, 0, -pointer_len),
                color0, color1)

            self._debug_line_render.pop_transform()


    def update(self):

        client_states = get_queued_client_states() 
        self._receive_rate_tracker.increment(len(client_states))

        if len(client_states) > self.get_history_length():
            client_states = client_states[-self.get_history_length():]

        for client_state in client_states:
            self._recent_client_states.insert(0, client_state)
            if len(self._recent_client_states) == self.get_history_length() + 1:
                self._recent_client_states.pop()

        self.update_input_state_from_remote_client_states(client_states)

        self.debug_visualize_client()

    def on_frame_end(self):

        self._gui_input.on_frame_end()

