
from typing import Any, Dict, List, Set, Tuple
import magnum as mn
import numpy as np
from controllers import GuiHumanoidController
from habitat.gui.gui_input import GuiInput
from magnum.platform.glfw import Application
import habitat_sim
from habitat.gui.text_drawer import TextOnScreenAlignment


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

class AppStateRearrange:

    def __init__(self, args, config, env, sim, gui_input, gui_agent_ctrl, compute_action_and_step_env, end_episode, get_metrics):

        self._env = env
        self._sim = sim
        self._gui_input = gui_input
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

        self._cursor_style = None
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

    def _update_grasping_and_set_act_hints(self):

        end_radius = self._obj_succ_thresh

        drop_pos = None
        grasp_object_id = None

        if self._held_target_obj_idx is not None:
            color = mn.Color3(0, 255 / 255, 0)  # green
            goal_position = self._goal_positions[self._held_target_obj_idx]
            self._debug_line_render.draw_circle(
                goal_position, end_radius, color, 24
            )

            self._draw_nav_hint_from_agent(
                mn.Vector3(goal_position), end_radius, color
            )
            # draw can place area
            can_place_position = mn.Vector3(goal_position)
            can_place_position[1] = self._get_agent_feet_height()
            self._debug_line_render.draw_circle(
                can_place_position,
                self._can_grasp_place_threshold,
                mn.Color3(255 / 255, 255 / 255, 0),
                24,
            )

            if self._gui_input.get_key_down(GuiInput.KeyNS.SPACE):
                translation = self._get_agent_translation()
                dist_to_obj = np.linalg.norm(goal_position - translation)
                if dist_to_obj < self._can_grasp_place_threshold:
                    self._held_target_obj_idx = None
                    drop_pos = goal_position
        else:
            # check for new grasp and call gui_agent_ctrl.set_act_hints
            if self._held_target_obj_idx is None:
                assert not self.gui_agent_ctrl.is_grasped
                # pick up an object
                if self._gui_input.get_key_down(GuiInput.KeyNS.SPACE):
                    translation = self._get_agent_translation()

                    min_dist = self._can_grasp_place_threshold
                    min_i = None
                    for i in range(len(self._target_obj_ids)):
                        if self._is_target_object_at_goal_position(i):
                            continue

                        this_target_pos = self._get_target_object_position(i)
                        # compute distance in xz plane
                        offset = this_target_pos - translation
                        offset.y = 0
                        dist_xz = offset.length()
                        if dist_xz < min_dist:
                            min_dist = dist_xz
                            min_i = i

                    if min_i is not None:
                        self._held_target_obj_idx = min_i
                        grasp_object_id = self._target_obj_ids[
                            self._held_target_obj_idx
                        ]

        walk_dir = (
            self._viz_and_get_humanoid_walk_dir()
            if not self._first_person_mode
            else None
        )

        self.gui_agent_ctrl.set_act_hints(
            walk_dir, grasp_object_id, drop_pos, self.lookat_offset_yaw
        )

        return drop_pos


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

    def _update_task(self):
        end_radius = self._obj_succ_thresh

        grasped_objects_idxs = self._get_grasped_objects_idxs()
        self._num_remaining_objects = 0
        self._num_busy_objects = len(grasped_objects_idxs)

        # draw nav_hint and target box
        for i in range(len(self._target_obj_ids)):
            # object is grasped
            if i in grasped_objects_idxs:
                continue

            color = mn.Color3(255 / 255, 128 / 255, 0)  # orange
            if self._is_target_object_at_goal_position(i):
                continue

            self._num_remaining_objects += 1

            if self._held_target_obj_idx is None:
                this_target_pos = self._get_target_object_position(i)
                box_half_size = 0.15
                box_offset = mn.Vector3(
                    box_half_size, box_half_size, box_half_size
                )
                self._debug_line_render.draw_box(
                    this_target_pos - box_offset,
                    this_target_pos + box_offset,
                    color,
                )

                if True:  # not self.is_free_camera_mode():
                    self._draw_nav_hint_from_agent(
                        mn.Vector3(this_target_pos), end_radius, color
                    )
                    # draw can grasp area
                    can_grasp_position = mn.Vector3(this_target_pos)
                    can_grasp_position[1] = self._get_agent_feet_height()
                    self._debug_line_render.draw_circle(
                        can_grasp_position,
                        self._can_grasp_place_threshold,
                        mn.Color3(255 / 255, 255 / 255, 0),
                        24,
                    )


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

    def _update_cursor_style(self):
        do_update_cursor = False
        if self._cursor_style is None:
            self._cursor_style = Application.Cursor.ARROW
            do_update_cursor = True
        else:
            if (
                self._first_person_mode
                and self._gui_input.get_mouse_button_down(GuiInput.MouseNS.LEFT)
            ):
                # toggle cursor mode
                self._cursor_style = (
                    Application.Cursor.HIDDEN_LOCKED
                    if self._cursor_style == Application.Cursor.ARROW
                    else Application.Cursor.ARROW
                )
                do_update_cursor = True

        return do_update_cursor

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


    def _camera_pitch_and_yaw_wasd_control(self):
        # update yaw and pitch using ADIK keys
        cam_rot_angle = 0.1

        if self._gui_input.get_key(GuiInput.KeyNS.I):
            self._lookat_offset_pitch -= cam_rot_angle
        if self._gui_input.get_key(GuiInput.KeyNS.K):
            self._lookat_offset_pitch += cam_rot_angle
        self._lookat_offset_pitch = np.clip(
            self._lookat_offset_pitch,
            self._min_lookat_offset_pitch,
            self._max_lookat_offset_pitch,
        )
        if self._gui_input.get_key(GuiInput.KeyNS.A):
            self._lookat_offset_yaw -= cam_rot_angle
        if self._gui_input.get_key(GuiInput.KeyNS.D):
            self._lookat_offset_yaw += cam_rot_angle

    def _camera_pitch_and_yaw_mouse_control(self):
        enable_mouse_control = (
            self._first_person_mode
            and self._cursor_style == Application.Cursor.HIDDEN_LOCKED
        ) or (
            not self._first_person_mode
            and self._gui_input.get_key(GuiInput.KeyNS.R)
        )

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
        def get_grasp_release_controls_text():
            if self._held_target_obj_idx is not None:
                return "Spacebar: put down\n"
            else:
                return "Spacebar: pick up\n"

        controls_str: str = ""
        controls_str += "ESC: exit\n"
        if self._next_episode_exists():
            controls_str += "M: next episode\n"

        if self._env_episode_active():
            if self._first_person_mode:
                # controls_str += "Left-click: toggle cursor\n"  # make this "unofficial" for now
                controls_str += "I, K: look up, down\n"
                controls_str += "A, D: turn\n"
                controls_str += "W, S: walk\n"
                controls_str += get_grasp_release_controls_text()
            # third-person mode
            else:
                controls_str += "R + drag: rotate camera\n"
                controls_str += "Right-click: walk\n"
                controls_str += "A, D: turn\n"
                controls_str += "W, S: walk\n"
                controls_str += "Scroll: zoom\n"
                controls_str += get_grasp_release_controls_text()
            # else:
            #     controls_str += "Left-click + drag: rotate camera\n"
            #     controls_str += "A, D: turn camera\n"
            #     controls_str += "W, S: pan camera\n"
            #     controls_str += "O, P: raise or lower camera\n"
            #     controls_str += "Scroll: zoom\n"

        # hack: showing status here in controls_str
        # if self._remote_gui_input:
        #     controls_str += f"receive rate: {self._remote_gui_input._receive_rate_tracker.get_smoothed_rate():.1f}\n"

        return controls_str            

    @property
    def _env_task_complete(self):
        return (
            self._end_on_success and self._get_metrics()[self._success_measure_name]
        )

    def _env_episode_active(self) -> bool:
        """
        Returns True if current episode is active:
        1) not self._env.episode_over - none of the constraints is violated, or
        2) not self._env_task_complete - success measure value is not True
        """
        return not (self._env.episode_over or self._env_task_complete)

    def _get_status_text(self):
        status_str = ""

        assert self._num_remaining_objects is not None
        assert self._num_busy_objects is not None

        if not self._env_episode_active():
            if self._env_task_complete:
                status_str += "Task complete!\n"
            else:
                status_str += "Oops! Something went wrong.\n"
        elif self._held_target_obj_idx is not None:
            # reference code to display object handle
            # sim = self.get_sim()
            # grasp_object_id = sim.scene_obj_ids[
            #     self._held_target_obj_idx
            # ]
            # obj_handle = (
            #     sim.get_rigid_object_manager().get_object_handle_by_id(
            #         grasp_object_id
            #     )
            # )
            status_str += (
                "Place the "
                # + get_pretty_object_name_from_handle(obj_handle)
                + "object"
                + " at its goal location!\n"
            )
        elif self._num_remaining_objects > 0:
            status_str += "Move the remaining {} object{}!".format(
                self._num_remaining_objects,
                "s" if self._num_remaining_objects > 1 else "",
            )
        elif self._num_busy_objects > 0:
            status_str += "Just wait! The robot is moving the last object.\n"
        else:
            # we don't expect to hit this case ever
            status_str += "Oops! Something went wrong.\n"

        # center align the status_str
        max_status_str_len = 50
        status_str = "/n".join(
            line.center(max_status_str_len) for line in status_str.split("/n")
        )

        return status_str        


    def _update_help_text(self):
        # if self._sandbox_state == SandboxState.CONTROLLING_AGENT:
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

            progress_str = f"{self._num_iter_episodes - (self._num_episodes_done + 1)} episodes remaining"
            self._text_drawer.add_text(
                progress_str,
                TextOnScreenAlignment.TOP_RIGHT,
                text_delta_x=370,
            )

        # elif self._sandbox_state == SandboxState.TUTORIAL:
        #     assert False
        #     controls_str = self._tutorial.get_help_text()
        #     if len(controls_str) > 0:
        #         self._text_drawer.add_text(
        #             controls_str, TextOnScreenAlignment.TOP_LEFT
        #         )

        #     tutorial_str = self._tutorial.get_display_text()
        #     if len(tutorial_str) > 0:
        #         self._text_drawer.add_text(
        #             tutorial_str,
        #             TextOnScreenAlignment.TOP_CENTER,
        #             text_delta_x=-280,
        #             text_delta_y=-50,
        #         )        

    def _create_camera_lookat(self) -> Tuple[mn.Vector3, mn.Vector3]:
        agent_idx = self.get_gui_controlled_agent_index()
        if agent_idx is None:
            self._free_camera_lookat_control()
            lookat = self.lookat
        else:
            art_obj = (
                self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
            )
            robot_root = art_obj.transformation
            lookat = robot_root.translation + mn.Vector3(0, 1, 0)

            # temp hack
            # lookat -= mn.Vector3(0, 1.5, 0)

        if self._first_person_mode:
            self.cam_zoom_dist = self._min_zoom_dist
            lookat += 0.075 * robot_root.backward
            lookat -= mn.Vector3(0, 0.2, 0)

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

            snap_idx = (
                self.get_sim()
                .agents_mgr._all_agent_data[agent_idx]
                .grasp_mgr.snap_idx
            )

            agent_states.append(
                {
                    "position": pos,
                    "rotation_xyzw": rotation_list,
                    "grasp_mgr_snap_idx": snap_idx,
                }
            )

        self._step_recorder.record("agent_states", agent_states)

        self._step_recorder.record(
            "target_object_positions", self._get_target_object_positions()
        )

    def _sim_update_controlling_agent(self, dt: float):

        if self._env_episode_active():
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

        # two ways for camera pitch and yaw control for UX comparison:
        # 1) press/hold ADIK keys
        self._camera_pitch_and_yaw_wasd_control()
        # 2) press left mouse button and move mouse
        self._camera_pitch_and_yaw_mouse_control()

        lookat = self._create_camera_lookat()
        self.cam_transform = mn.Matrix4.look_at(
            lookat[0], lookat[1], mn.Vector3(0, 1, 0)
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

        if self._env_episode_active():
            self._update_task()
            self._update_grasping_and_set_act_hints()

        # Navmesh visualization only works in the debug third-person view
        # (--debug-third-person-width), not the main sandbox viewport. Navmesh
        # visualization is only implemented for simulator-rendering, not replay-
        # rendering.
        if self._gui_input.get_key_down(GuiInput.KeyNS.N):
            self._sim.navmesh_visualization = (  # type: ignore
                not self._sim.navmesh_visualization  # type: ignore
            )

        if True:  # self._sandbox_state == SandboxState.CONTROLLING_AGENT:
            self._sim_update_controlling_agent(dt)
        # else:
        #     assert False
            # self._sim_update_tutorial(dt)

        # if self._cube_test:
        #     self._cube_test.pre_step()

        # self.cam_transform is set to new value after
        # self._sim_update_controlling_agent(dt) or self._sim_update_tutorial(dt)
        post_sim_update_dict["cam_transform"] = self.cam_transform

        if self._update_cursor_style():
            post_sim_update_dict["application_cursor"] = self._cursor_style

        # keyframes = (
        #     self.get_sim().gfx_replay_manager.write_incremental_saved_keyframes_to_string_array()
        # )

        # if self._save_gfx_replay_keyframes:
        #     for keyframe in keyframes:
        #         self._recording_keyframes.append(keyframe)

        # if self._args.hide_humanoid_in_gui:
        #     # Hack to hide skinned humanoids in the GUI viewport. Specifically, this
        #     # hides all render instances with a filepath starting with
        #     # "data/humanoids/humanoid_data", by replacing with an invalid filepath.
        #     # Gfx-replay playback logic will print a warning to the terminal and then
        #     # not attempt to render the instance. This is a temp hack until
        #     # skinning is supported in gfx-replay.
        #     for i in range(len(keyframes)):
        #         keyframes[i] = keyframes[i].replace(
        #             '"creation":{"filepath":"data/humanoids/humanoid_data',
        #             '"creation":{"filepath":"invalid_filepath',
        #         )

        # for i in range(len(keyframes)):
        #     if use_simplified_hssd_objects:
        #         keyframes[i] = keyframes[i].replace(
        #             'data/fpss/fphab/objects',
        #             'data/hitl/simplified/fpss/fphab/objects',
        #         )
        #     if use_simplified_ycb_objects:
        #         keyframes[i] = keyframes[i].replace(
        #             'data/objects/ycb/',
        #             'data/hitl/simplified/objects/ycb/',
        #         )
        #     if use_simplified_robot_meshes:
        #         keyframes[i] = keyframes[i].replace(
        #             'data/robots/hab_spot_arm/',
        #             'data/hitl/simplified/robots/hab_spot_arm/',
        #         )
        #     if remove_extra_config_in_model_filepaths:
        #         keyframes[i] = keyframes[i].replace(
        #             'configs/../',
        #             '/',
        #         )
        #     if use_glb_black_list:
        #         for black_item in glb_black_list:
        #             keyframes[i] = keyframes[i].replace(
        #                 black_item,
        #                 black_item + ".invalid_filepath",
        #             )

        # elif use_collision_proxies_for_hssd_objects:
        #     import re
        #     for i in range(len(keyframes)):
        #         pattern = r'(/objects/[^.]+)\.glb'
        #         replacement = r'\1.collider.glb'
        #         keyframes[i] = re.sub(pattern, replacement, keyframes[i])



        # post_sim_update_dict["keyframes"] = keyframes

        # def depth_to_rgb(obs):
        #     converted_obs = np.concatenate(
        #         [obs * 255.0 for _ in range(3)], axis=2
        #     ).astype(np.uint8)
        #     return converted_obs

        # # reference code for visualizing a camera sensor in the app GUI
        # assert set(self._debug_images).issubset(set(self._obs.keys())), (
        #     f"Camera sensors ids: {list(set(self._debug_images).difference(set(self._obs.keys())))} "
        #     f"not in available sensors ids: {list(self._obs.keys())}"
        # )
        # debug_images = (
        #     depth_to_rgb(self._obs[k]) if "depth" in k else self._obs[k]
        #     for k in self._debug_images
        # )
        # post_sim_update_dict["debug_images"] = [
        #     np.flipud(image) for image in debug_images
        # ]

        # self._update_help_text()

        # if self._cube_test:
        #     self._cube_test.post_step()

        # if do_network_server:
        #     for keyframe_json in keyframes:
        #         single_item_array = json.loads(keyframe_json)
        #         assert len(single_item_array) == 1
        #         keyframe_obj = single_item_array
        #         send_keyframe_to_networking_thread(keyframe_obj)

        # return post_sim_update_dict        


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