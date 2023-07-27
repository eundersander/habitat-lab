#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from server.interprocess_record import send_keyframe_to_networking_thread, get_queued_client_states
from server.average_rate_tracker import AverageRateTracker

import magnum as mn

from habitat.gui.gui_input import GuiInput

class CubeTest:
    def __init__(self, sim):

        self._base_angle = 0
        self._step_count = 0

        self._num_cubes = 20

        self._cubes = []

        obj_template_mgr = sim.get_object_template_manager()
        assert obj_template_mgr.get_num_templates() > 0
        rigid_obj_mgr = sim.get_rigid_object_manager()
        obj_handle_list = obj_template_mgr.get_template_handles("003_cracker_box")
        assert len(obj_handle_list) == 1
        obj_handle = obj_handle_list[0]

        start_trans = mn.Vector3(0, 0.09, 0)
        
        for i in range(self._num_cubes):

            cube_obj = rigid_obj_mgr.add_object_by_template_handle(obj_handle)
            cube_obj.translation = start_trans + mn.Vector3(0, i * 0.20, 0.0)
            self._cubes.append(cube_obj)

        self._hands = []
        num_hands = 2
        for _ in range(num_hands):
            cube_obj = rigid_obj_mgr.add_object_by_template_handle(obj_handle)
            cube_obj.translation = mn.Vector3(0, 0, 0)
            cube_obj.collidable = False
            self._hands.append(cube_obj)

        self._latest_client_state = None
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

        self._remote_gui_input = GuiInput()

        self._grasp_cube_idx = None
        self._grasp_hand_idx = None

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
                    self._remote_gui_input._key_down.add(self._key_map[key])
            for key in input_json["keyUp"]:
                if key not in self._key_map:
                    print(f"key {key} not mapped!")
                    continue
                if input_json["keyUp"][key]:
                    self._remote_gui_input._key_up.add(self._key_map[key])

            for button in input_json["buttonDown"]:
                if button not in self._button_map:
                    print(f"button {button} not mapped!")
                    continue
                if input_json["buttonDown"][button]:
                    self._remote_gui_input._key_down.add(self._button_map[button])
            for button in input_json["buttonUp"]:
                if button not in self._button_map:
                    print(f"key {button} not mapped!")
                    continue
                if input_json["buttonUp"][button]:
                    self._remote_gui_input._key_up.add(self._button_map[button])

        # todo: think about ambiguous GuiInput states (key-down and key-up events in the same
        # frame and other ways that keyHeld, keyDown, and keyUp can be inconsistent.
        client_state = client_states[-1]
        self._remote_gui_input._key_held.clear()
        for key in input_json["keyHeld"]:
            if key not in self._key_map:
                print(f"key {key} not mapped!")
                continue
            if input_json["keyHeld"][key]:
                self._remote_gui_input._key_held.add(self._key_map[key])

        for button in input_json["buttonHeld"]:
            if button not in self._button_map:
                print(f"button {button} not mapped!")
                continue
            if input_json["buttonHeld"][button]:
                self._remote_gui_input._key_held.add(self._button_map[button])

    def update_hand_poses(self):

        if not self._latest_client_state:
            return

        assert "avatar" in self._latest_client_state
        assert "hands" in self._latest_client_state["avatar"]
        hands_json = self._latest_client_state["avatar"]["hands"]
        assert len(hands_json) == len(self._hands)

        for hand_idx, hand_json in enumerate(hands_json):
            pos_json = hand_json["position"]
            # pos = mn.Vector3(pos_json["_x"], pos_json["_y"], pos_json["_z"])
            pos = mn.Vector3(pos_json[0], pos_json[1], pos_json[2])
            rot_json = hand_json["rotation"]
            # rot_quat = mn.Quaternion(mn.Vector3(rot_json["_x"], rot_json["_y"], rot_json["_z"]), rot_json["_w"])
            rot_quat = mn.Quaternion(mn.Vector3(rot_json[0], rot_json[1], rot_json[2]), rot_json[3])

            hand_obj = self._hands[hand_idx]
            hand_obj.translation = pos
            hand_obj.rotation = rot_quat

    def debug_visualize_client(self):

        if not self._debug_line_render:
            return

        if not self._latest_client_state:
            return

        assert "avatar" in self._latest_client_state
        assert "root" in self._latest_client_state["avatar"]
        
        if True:
            avatar_root_json = self._latest_client_state["avatar"]["root"]
            pos_json = avatar_root_json["position"]
            pos = mn.Vector3(pos_json[0], pos_json[1], pos_json[2])
            rot_json = avatar_root_json["rotation"]
            rot_quat = mn.Quaternion(mn.Vector3(rot_json[0], rot_json[1], rot_json[2]), rot_json[3])

            trans =  mn.Matrix4.from_(rot_quat.to_matrix(), pos)

            half_size = 0.25
            self._debug_line_render.push_transform(trans)
            self._debug_line_render.draw_box(mn.Vector3(-half_size, -half_size, -half_size), 
                mn.Vector3(half_size, half_size, half_size),
                mn.Color3(151, 206, 222) / 255)
            self._debug_line_render.pop_transform()


    def update_grasp(self):
        if self._remote_gui_input.get_key_down(GuiInput.KeyNS.F) or \
            self._remote_gui_input.get_key_down(GuiInput.KeyNS.ZERO):

            if self._grasp_cube_idx:
                # ensure vel is zero so that the object falls reasonably
                cube_obj = self._cubes[self._grasp_cube_idx]
                cube_obj.angular_velocity = mn.Vector3(0, 0, 0)
                cube_obj.linear_velocity = mn.Vector3(0, 0, 0)

                self._grasp_cube_idx = None
                self._grasp_hand_idx = None

            else:
                # attempt grasp
                hand_idx = 0
                query_pos = self._hands[hand_idx].translation

                # find nearest cube to hand 0
                min_dist = 0.75
                min_idx = -1
                for cube_idx, cube in enumerate(self._cubes):
                    dist = (query_pos - cube.translation).length()
                    if dist < min_dist:
                        min_idx = cube_idx
                        min_dist = dist

                if min_idx != -1:
                    print(f"grabbing cube {min_idx}!")
                    self._grasp_cube_idx = min_idx
                    self._grasp_hand_idx = hand_idx

        if self._grasp_cube_idx:
            assert self._grasp_cube_idx != -1

            cube_obj = self._cubes[self._grasp_cube_idx]
            hand_obj = self._hands[self._grasp_hand_idx]
            # do vertical displacement so we can see both hand and cube
            cube_obj.translation = hand_obj.translation + mn.Vector3(0, 0.1, 0.0)
            cube_obj.rotation = hand_obj.rotation
            cube_obj.angular_velocity = mn.Vector3(0, 0, 0)
            cube_obj.linear_velocity = mn.Vector3(0, 0, 0)

    def pre_step(self):

        client_states = get_queued_client_states() 
        self._receive_rate_tracker.increment(len(client_states))
        if len(client_states):    
            self._latest_client_state = client_states[-1]

        self.update_input_state_from_remote_client_states(client_states)

        self.update_hand_poses()

        self.debug_visualize_client()

    def post_step(self):

        # # Define circle parameters
        # center_x = 0
        # center_z = 0
        # radius = 2
        # dt = 1 / 10

        # angular_inc = math.pi * 2 * 0.1 * dt
        # angular_offset = 2 * math.pi / num_cubes

        self.update_grasp()

        cube_poses = []
        for cube_obj in self._cubes:
            # # Calculate the angle for each cube
            # angle = self._base_angle + i * angular_offset

            # # Calculate cube position in the circle
            # cube_x = center_x + radius * math.cos(angle)
            # cube_y = 1 + i * 0.1  # Different Y height for each cube
            # cube_z = center_z + radius * math.sin(angle)

            trans = cube_obj.translation
            rot = cube_obj.rotation

            # Update cube pose
            cube_pose = {
                "rotation": {"x": rot.vector.x, "y": rot.vector.y, "z": rot.vector.z, "w": rot.scalar},
                "translation": {"x": trans.x, "y": trans.y, "z": trans.z}
            }
            cube_poses.append(cube_pose)

        keyframe = {"keyframe_index": self._step_count, "cube_poses": cube_poses}

        # # Update the angles for the next iteration
        # self._base_angle += angular_inc
        self._step_count += 1

        send_keyframe_to_networking_thread(keyframe)

        # self._step_count += 1
        # if self._step_count % 10 == 0:
        #     print(f"step_count: {self._step_count}")        

        self._remote_gui_input.on_frame_end()

