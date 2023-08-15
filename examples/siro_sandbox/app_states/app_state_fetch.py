
from typing import Any, Dict, List, Set, Tuple
import magnum as mn
import numpy as np
from controllers import GuiHumanoidController
from habitat.gui.gui_input import GuiInput
from magnum.platform.glfw import Application
import habitat_sim
from habitat.gui.text_drawer import TextOnScreenAlignment
from habitat_sim.physics import MotionType


def _evaluate_cubic_bezier(ctrl_pts, t):
    assert len(ctrl_pts) == 4
    weights = (
        pow(1 - t, 3),
        3 * t * pow(1 - t, 2),
        3 * pow(t, 2) * (1 - t),
        pow(t, 3),
    )

    result = weights[0] * ctrl_pts[0]
    for i in range(1, 4):
        result += weights[i] * ctrl_pts[i]

    return result    

class AppStateFetch:

    def __init__(self, args, config, env, sim, gui_input, remote_gui_input, gui_agent_ctrl, compute_action_and_step_env, end_episode, get_metrics):

        self._env = env
        self._sim = sim
        self._gui_input = gui_input
        self._remote_gui_input = remote_gui_input
        self.gui_agent_ctrl = gui_agent_ctrl
        self._compute_action_and_step_env = compute_action_and_step_env
        self._end_episode = end_episode
        self._get_metrics = get_metrics

        # cache items from config; config is expensive to access at runtime
        self._end_on_success = config.habitat.task.end_on_success
        self._obj_succ_thresh = config.habitat.task.obj_succ_thresh
        self._success_measure_name = config.habitat.task.success_measure

        # lookat offset yaw (spin left/right) and pitch (up/down)
        # to enable camera rotation and pitch control
        self._first_person_mode = args.first_person_mode
        if self._first_person_mode:
            self._lookat_offset_yaw = 0.0
            self._lookat_offset_pitch = float(
                mn.Rad(mn.Deg(20.0))
            )  # look slightly down
            self._min_lookat_offset_pitch = (
                -max(min(np.radians(args.max_look_up_angle), np.pi / 2), 0)
                + 1e-5
            )
            self._max_lookat_offset_pitch = (
                -min(max(np.radians(args.min_look_down_angle), -np.pi / 2), 0)
                - 1e-5
            )
        else:
            # (computed from previously hardcoded mn.Vector3(0.5, 1, 0.5).normalized())
            self._lookat_offset_yaw = 0.785
            self._lookat_offset_pitch = 0.955
            self._min_lookat_offset_pitch = -np.pi / 2 + 1e-5
            self._max_lookat_offset_pitch = np.pi / 2 - 1e-5

        self._can_grasp_place_threshold = args.can_grasp_place_threshold

        self._viz_anim_fraction = 0.0

        self.cam_transform = None
        self.cam_zoom_dist = 1.0
        self._max_zoom_dist = 50.0
        self._min_zoom_dist = 0.02

        self._debug_line_render = None  # will be set later by user code
        self._text_drawer = None  # will be set later by user code

        self._num_iter_episodes: int = len(self._env.episode_iterator.episodes)  # type: ignore
        self._num_episodes_done: int = 0

        self._held_target_obj_idx = None
        self._fetch_target_obj_idx = None

    @staticmethod
    def _to_zero_2pi_range(radians):
        """Helper method to properly clip radians to [0, 2pi] range."""
        return (
            (2 * np.pi) - ((-radians) % (2 * np.pi))
            if radians < 0
            else radians % (2 * np.pi)
        )
        
    def get_sim(self):
        return self._sim

    def get_fetch_goal(self):

        if self._fetch_target_obj_idx is not None:
            obj = self._get_target_rigid_object(self._fetch_target_obj_idx)
            return obj.translation
        else:
            return None


    def _update_grasping_and_set_act_hints(self):

        hand_idx = 1  # temp hard-coded to right hand

        gui_input = self._remote_gui_input.get_gui_input()
        is_button_down = gui_input.get_key_down(GuiInput.KeyNS.F) or gui_input.get_key_down(GuiInput.KeyNS.ZERO)
        is_button_up = gui_input.get_key_up(GuiInput.KeyNS.F) or gui_input.get_key_up(GuiInput.KeyNS.ZERO)

        if self._held_target_obj_idx is not None:

            if is_button_up:
                obj = self._get_target_rigid_object(self._held_target_obj_idx)
                obj.motion_type = MotionType.DYNAMIC
                obj.collidable = True

                history_len = self._remote_gui_input.get_history_length()
                assert history_len >= 2
                pos1, _ = self._remote_gui_input.get_hand_pose(hand_idx, history_index=0)
                pos0, _ = self._remote_gui_input.get_hand_pose(hand_idx, history_index=history_len - 1)
                if pos0 and pos1:
                    vel = (pos1 - pos0) / (self._remote_gui_input.get_history_timestep() * history_len)
                    obj.linear_velocity = vel
                else:
                    obj.linear_velocity = mn.Vector3(0, 0, 0)

                self._fetch_target_obj_idx = self._held_target_obj_idx
                self._held_target_obj_idx = None

        else:
            # check for new grasp and call gui_agent_ctrl.set_act_hints
            if self._held_target_obj_idx is None:

                if is_button_down:
                    translation, _ = self._remote_gui_input.get_hand_pose(hand_idx)
                    assert translation
                    min_dist = self._can_grasp_place_threshold
                    min_i = None
                    for i in range(len(self._target_obj_ids)):

                        this_target_pos = self._get_target_object_position(i)
                        offset = this_target_pos - translation
                        dist_xz = offset.length()
                        if dist_xz < min_dist:
                            min_dist = dist_xz
                            min_i = i

                    if min_i is not None:
                        self._held_target_obj_idx = min_i
                        self._fetch_target_obj_idx = None
                        # todo: save a relative transform

                        obj = self._get_target_rigid_object(self._held_target_obj_idx)
                        obj.motion_type = MotionType.KINEMATIC
                        obj.collidable = False


        if self._held_target_obj_idx is not None:

            obj = self._get_target_rigid_object(self._held_target_obj_idx)

            hand_pos, hand_rot_quat = self._remote_gui_input.get_hand_pose(hand_idx)
            assert hand_pos
            obj.translation = hand_pos
            obj.rotation = hand_rot_quat
            obj.angular_velocity = mn.Vector3(0, 0, 0)
            obj.linear_velocity = mn.Vector3(0, 0, 0)

        if self._fetch_target_obj_idx is not None:
            obj = self._get_target_rigid_object(self._fetch_target_obj_idx)
            self._fetch_goal = obj.translation

            trans = mn.Matrix4.from_(obj.rotation.to_matrix(), obj.translation)
            half_size = 0.1
            self._debug_line_render.push_transform(trans)
            self._debug_line_render.draw_box(mn.Vector3(-half_size, -half_size, -half_size), 
                mn.Vector3(half_size, half_size, half_size),
                mn.Color3(1, 1, 0))
            self._debug_line_render.pop_transform()

        else:
            self._fetch_goal = None

        # walk_dir = (
        #     self._viz_and_get_humanoid_walk_dir()
        #     if not self._first_person_mode
        #     else None
        # )

        # Unlike the rearrange app state, in the fetch state, the gui agent doesn't 
        # actually grasp and drop objects. Instead, we snap the held object to
        # the VR hand pose.
        # drop_pos = None
        # grasp_object_id = None

        # self.gui_agent_ctrl.set_act_hints(
        #     walk_dir, grasp_object_id, drop_pos, self.lookat_offset_yaw
        # )

    def _get_target_rigid_object(self, target_obj_idx):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        object_id = self._target_obj_ids[target_obj_idx]
        return rom.get_object_by_id(object_id)

    def _get_target_object_position(self, target_obj_idx):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        object_id = self._target_obj_ids[target_obj_idx]
        return rom.get_object_by_id(object_id).translation

    def _get_target_object_positions(self):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        return np.array(
            [
                rom.get_object_by_id(obj_id).translation
                for obj_id in self._target_obj_ids
            ]
        )

    def _is_target_object_at_goal_position(self, target_obj_idx):
        this_target_pos = self._get_target_object_position(target_obj_idx)
        end_radius = self._obj_succ_thresh
        return (
            this_target_pos - self._goal_positions[target_obj_idx]
        ).length() < end_radius

    def _get_grasped_objects_idxs(self):
        sim = self.get_sim()
        agents_mgr = sim.agents_mgr

        grasped_objects_idxs = []
        for agent_idx in range(len(agents_mgr._all_agent_data)):
            # todo: should gui_agent_ctrl._agent_idx be public?
            if agent_idx == self.get_gui_controlled_agent_index():
                continue
            grasp_mgr = agents_mgr._all_agent_data[agent_idx].grasp_mgr
            if grasp_mgr.is_grasped:
                grasped_objects_idxs.append(
                    sim.scene_obj_ids.index(grasp_mgr.snap_idx)
                )
                
        return grasped_objects_idxs



    def get_gui_controlled_agent_index(self):
        return self.gui_agent_ctrl._agent_idx

    def _draw_nav_hint_from_agent(self, end_pos, end_radius, color):
        agent_idx = self.get_gui_controlled_agent_index()
        assert agent_idx is not None
        art_obj = (
            self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
        )
        agent_pos = art_obj.transformation.translation
        # get forward_dir from FPS camera yaw, not art_obj.transformation
        # (the problem with art_obj.transformation is that it includes a "wobble"
        # introduced by the walk animation)
        transformation = self.cam_transform or art_obj.transformation
        forward_dir = transformation.transform_vector(-mn.Vector3(0, 0, 1))
        forward_dir[1] = 0
        forward_dir = forward_dir.normalized()

        self._draw_nav_hint(
            agent_pos,
            forward_dir,
            end_pos,
            end_radius,
            color,
            self._viz_anim_fraction,
        )

    def _get_agent_translation(self):
        assert isinstance(self.gui_agent_ctrl, GuiHumanoidController)
        return (
            self.gui_agent_ctrl._humanoid_controller.obj_transform_base.translation
        )

    def _get_agent_feet_height(self):
        assert isinstance(self.gui_agent_ctrl, GuiHumanoidController)
        base_offset = (
            self.gui_agent_ctrl.get_articulated_agent().params.base_offset
        )
        agent_feet_translation = self._get_agent_translation() + base_offset
        return agent_feet_translation[1]


    def _viz_and_get_humanoid_walk_dir(self):
        path_color = mn.Color3(0, 153 / 255, 255 / 255)
        path_endpoint_radius = 0.12

        ray = self._gui_input.mouse_ray

        floor_y = 0.15  # hardcoded to ReplicaCAD

        if not ray or ray.direction.y >= 0 or ray.origin.y <= floor_y:
            return None

        dist_to_floor_y = (ray.origin.y - floor_y) / -ray.direction.y
        target_on_floor = ray.origin + ray.direction * dist_to_floor_y

        agent_idx = self.get_gui_controlled_agent_index()
        art_obj = (
            self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
        )
        robot_root = art_obj.transformation

        pathfinder = self.get_sim().pathfinder
        snapped_pos = pathfinder.snap_point(target_on_floor)
        snapped_start_pos = robot_root.translation
        snapped_start_pos.y = snapped_pos.y

        path = habitat_sim.ShortestPath()
        path.requested_start = snapped_start_pos
        path.requested_end = snapped_pos
        found_path = pathfinder.find_path(path)

        if not found_path or len(path.points) < 2:
            return None

        path_points = []
        for path_i in range(0, len(path.points)):
            adjusted_point = mn.Vector3(path.points[path_i])
            # first point in path is at wrong height
            if path_i == 0:
                adjusted_point.y = mn.Vector3(path.points[path_i + 1]).y
            path_points.append(adjusted_point)

        self._debug_line_render.draw_path_with_endpoint_circles(
            path_points, path_endpoint_radius, path_color
        )

        if (self._gui_input.get_mouse_button(GuiInput.MouseNS.RIGHT)) and len(
            path.points
        ) >= 2:
            walk_dir = mn.Vector3(path.points[1]) - mn.Vector3(path.points[0])
            return walk_dir

        return None                    

    @property
    def lookat_offset_yaw(self):
        return self._to_zero_2pi_range(self._lookat_offset_yaw)

    @property
    def lookat_offset_pitch(self):
        return self._lookat_offset_pitch


    def _camera_pitch_and_yaw_mouse_control(self):
        enable_mouse_control = self._gui_input.get_key(GuiInput.KeyNS.R)

        if enable_mouse_control:
            # update yaw and pitch by scale * mouse relative position delta
            scale = 1 / 50
            self._lookat_offset_yaw += (
                scale * self._gui_input.relative_mouse_position[0]
            )
            self._lookat_offset_pitch += (
                scale * self._gui_input.relative_mouse_position[1]
            )
            self._lookat_offset_pitch = np.clip(
                self._lookat_offset_pitch,
                self._min_lookat_offset_pitch,
                self._max_lookat_offset_pitch,
            )    

    def _draw_nav_hint(
        self, start_pos, start_dir, end_pos, end_radius, color, anim_fraction
    ):
        assert isinstance(start_pos, mn.Vector3)
        assert isinstance(start_dir, mn.Vector3)
        assert isinstance(end_pos, mn.Vector3)

        bias_weight = 0.5
        biased_dir = (
            start_dir + (end_pos - start_pos).normalized() * bias_weight
        ).normalized()

        start_dir_weight = min(4.0, (end_pos - start_pos).length() / 2)
        ctrl_pts = [
            start_pos,
            start_pos + biased_dir * start_dir_weight,
            end_pos,
            end_pos,
        ]

        steps_per_meter = 10
        pad_meters = 1.0
        alpha_ramp_dist = 1.0
        num_steps = max(
            2,
            int(
                ((end_pos - start_pos).length() + pad_meters) * steps_per_meter
            ),
        )

        prev_pos = None
        for step_idx in range(num_steps):
            t = step_idx / (num_steps - 1) + anim_fraction * (
                1 / (num_steps - 1)
            )
            pos = _evaluate_cubic_bezier(ctrl_pts, t)

            if (pos - end_pos).length() < end_radius:
                break

            if step_idx > 0:
                alpha = min(1.0, (pos - start_pos).length() / alpha_ramp_dist)

                radius = 0.05
                num_segments = 12
                # todo: use safe_normalize
                normal = (pos - prev_pos).normalized()
                color_with_alpha = mn.Color4(color)
                color_with_alpha[3] *= alpha
                self._debug_line_render.draw_circle(
                    pos, radius, color_with_alpha, num_segments, normal
                )
            prev_pos = pos            

    def _next_episode_exists(self):
        return self._num_episodes_done < self._num_iter_episodes - 1

    def _get_controls_text(self):

        controls_str: str = ""
        controls_str += "ESC: exit\n"
        controls_str += "R + drag: rotate camera\n"
        controls_str += "Scroll: zoom\n"

        # hack: showing status here in controls_str
        if self._remote_gui_input:
            controls_str += f"receive rate: {self._remote_gui_input._receive_rate_tracker.get_smoothed_rate():.1f}\n"

        return controls_str            

    @property
    def _env_task_complete(self):
        return (
            self._end_on_success and self._get_metrics()[self._success_measure_name]
        )

    def _env_episode_active(self) -> bool:
        return True

    def _get_status_text(self):
        status_str = ""

        assert self._env_episode_active()

        # these are instructions to send to the remote client
        # if self._held_target_obj_idx is not None:
        #     status_str += "Throw the object!"
        # else:
        #     status_str += "Pick up an object!"

        # center align the status_str
        max_status_str_len = 50
        status_str = "/n".join(
            line.center(max_status_str_len) for line in status_str.split("/n")
        )

        return status_str        


    def _update_help_text(self):
        if True:
            controls_str = self._get_controls_text()
            if len(controls_str) > 0:
                self._text_drawer.add_text(
                    controls_str, TextOnScreenAlignment.TOP_LEFT
                )

            status_str = self._get_status_text()
            if len(status_str) > 0:
                self._text_drawer.add_text(
                    status_str,
                    TextOnScreenAlignment.TOP_CENTER,
                    text_delta_x=-280,
                    text_delta_y=-50,
                )
  

    def _create_camera_lookat(self) -> Tuple[mn.Vector3, mn.Vector3]:
        # agent_idx = self.get_gui_controlled_agent_index()
        # if agent_idx is None:
        #     self._free_camera_lookat_control()
        #     lookat = self.lookat
        # else:
        #     art_obj = (
        #         self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
        #     )
        #     robot_root = art_obj.transformation
        #     lookat = robot_root.translation + mn.Vector3(0, 1, 0)

        head_pos, head_rot_quat = self._remote_gui_input.get_head_pose()
        lookat = head_pos if head_pos else mn.Vector3(0, 0, 0)
        lookat.y = 0.5

        offset = mn.Vector3(
            np.cos(self.lookat_offset_yaw) * np.cos(self.lookat_offset_pitch),
            np.sin(self.lookat_offset_pitch),
            np.sin(self.lookat_offset_yaw) * np.cos(self.lookat_offset_pitch),
        )

        return (lookat + offset.normalized() * self.cam_zoom_dist, lookat)                

    def get_num_agents(self):
        return len(self.get_sim().agents_mgr._all_agent_data)


    def _record_task_state(self):
        agent_states = []
        for agent_idx in range(self.get_num_agents()):
            art_obj = (
                self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
            )
            rotation_quat = mn.Quaternion.from_matrix(
                art_obj.transformation.rotation()
            )
            rotation_list = list(rotation_quat.vector) + [rotation_quat.scalar]
            pos = art_obj.transformation.translation

            # snap_idx = (
            #     self.get_sim()
            #     .agents_mgr._all_agent_data[agent_idx]
            #     .grasp_mgr.snap_idx
            # )

            agent_states.append(
                {
                    "position": pos,
                    "rotation_xyzw": rotation_list,
                    # "grasp_mgr_snap_idx": snap_idx,
                }
            )

        self._step_recorder.record("agent_states", agent_states)

        self._step_recorder.record(
            "target_object_positions", self._get_target_object_positions()
        )


    def sim_update(self, dt, post_sim_update_dict):

        # if self._remote_gui_input:
        #     self._remote_gui_input.update()

        if self._gui_input.get_key_down(GuiInput.KeyNS.ESC):
            self._end_episode()
            post_sim_update_dict["application_exit"] = True

        if self._gui_input.get_key_down(GuiInput.KeyNS.M):
            self._end_episode(do_reset=True)

        # _viz_anim_fraction goes from 0 to 1 over time and then resets to 0
        viz_anim_speed = 2.0
        self._viz_anim_fraction = (
            self._viz_anim_fraction + dt * viz_anim_speed
        ) % 1.0

        self._update_grasping_and_set_act_hints()

        self._compute_action_and_step_env()

        if self._gui_input.mouse_scroll_offset != 0:
            zoom_sensitivity = 0.07
            if self._gui_input.mouse_scroll_offset < 0:
                self.cam_zoom_dist *= (
                    1.0
                    + -self._gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            else:
                self.cam_zoom_dist /= (
                    1.0 + self._gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            self.cam_zoom_dist = mn.math.clamp(
                self.cam_zoom_dist,
                self._min_zoom_dist,
                self._max_zoom_dist,
            )

        self._camera_pitch_and_yaw_mouse_control()

        lookat = self._create_camera_lookat()
        self.cam_transform = mn.Matrix4.look_at(
            lookat[0], lookat[1], mn.Vector3(0, 1, 0)
        )    
        
        # self.cam_transform is set to new value after
        # self._sim_update_controlling_agent(dt) or self._sim_update_tutorial(dt)
        post_sim_update_dict["cam_transform"] = self.cam_transform


    def on_environment_reset(self):
        # self._obs = self._env.reset()
        # self._metrics = self._env.get_metrics()
        # self.ctrl_helper.on_environment_reset()
        self._held_target_obj_idx = None
        self._num_remaining_objects = None  # resting, not at goal location yet
        self._num_busy_objects = None  # currently held by non-gui agents

        sim = self.get_sim()
        temp_ids, goal_positions_np = sim.get_targets()
        self._target_obj_ids = [
            sim._scene_obj_ids[temp_id] for temp_id in temp_ids
        ]
        self._goal_positions = [mn.Vector3(pos) for pos in goal_positions_np]

        self._num_episodes_done += 1

        # self._sandbox_state = (
        #     SandboxState.TUTORIAL
        #     if args.show_tutorial
        #     else SandboxState.CONTROLLING_AGENT
        # )
        # self._tutorial: Tutorial = (
        #     generate_tutorial(
        #         self.get_sim(),
        #         self.get_gui_controlled_agent_index(),
        #         self._create_camera_lookat(),
        #     )
        #     if args.show_tutorial
        #     else None
        # )        